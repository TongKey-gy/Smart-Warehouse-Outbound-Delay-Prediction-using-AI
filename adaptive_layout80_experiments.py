from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
RESULTS_PATH = ROOT / "logs" / "results.csv"
RUNS_ROOT = ROOT / "outputs" / "adaptive_layout80_runs"
STRATEGY_MODULE_PATH = ROOT / "adaptive_layout80_strategy.py"
BASELINE_EXPERIMENT_NAME = "exp79_bottleneck_blend_layoutid_unweighted_w35"

TRACKED_KEYS = [
    "n_splits",
    "use_layout_info",
    "use_layout_id",
    "add_layout_interaction_features",
    "layout_feature_set",
    "add_delay_risk_features",
    "delay_risk_feature_set",
    "target_weight_strength",
    "secondary_weight",
    "secondary_use_layout_id",
    "secondary_layout_feature_set",
    "secondary_target_weight_mode",
    "secondary_target_weight_strength",
]


@dataclass
class ExperimentOutcome:
    iteration: int
    experiment_name: str
    oof_mae: float
    mae_std: float
    residual_mean: float
    tail_residual_mean: float
    config: dict[str, Any]
    summary_path: str
    experiment_dir: str


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH)


def parse_config(raw: Any) -> dict[str, Any]:
    if pd.isna(raw):
        return {}
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw))


def build_env(config: dict[str, Any], notes: str) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in config.items():
        env[f"TRAIN_{key.upper()}"] = str(value)
    env["TRAIN_IMPROVEMENT_NOTES"] = notes
    env["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
    return env


def read_oof_stats(experiment_dir: Path) -> tuple[float, float]:
    oof = pd.read_csv(experiment_dir / "oof_predictions.csv")
    residual_mean = float(oof["residual"].mean())
    tail_cutoff = float(oof["target"].quantile(0.99))
    tail = oof[oof["target"] >= tail_cutoff]
    tail_residual_mean = float(tail["residual"].mean()) if not tail.empty else residual_mean
    return residual_mean, tail_residual_mean


def get_row(df: pd.DataFrame, experiment_name: str) -> pd.Series:
    rows = df[df["experiment_name"] == experiment_name].sort_values("timestamp")
    if rows.empty:
        raise ValueError(f"Missing experiment: {experiment_name}")
    return rows.iloc[-1]


def config_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    if previous is None:
        return [f"`{key}`={current[key]!r}" for key in TRACKED_KEYS if key in current]
    changes: list[str] = []
    for key in TRACKED_KEYS:
        if previous.get(key) != current.get(key):
            changes.append(f"`{key}`: {previous.get(key)!r} -> {current.get(key)!r}")
    return changes


def write_strategy(
    iteration: int,
    analysis: str,
    proposal: str,
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    changes = config_diff(previous, current)
    lines = [
        '"""Adaptive layout strategy state for the next training run."""',
        "",
        f"ITERATION = {iteration}",
        f"ANALYSIS = {analysis!r}",
        f"PROPOSAL = {proposal!r}",
        "CURRENT_STRATEGY = {",
    ]
    for key in TRACKED_KEYS + ["experiment_name"]:
        if key in current:
            lines.append(f"    {key!r}: {current[key]!r},")
    lines.extend(["}", ""])
    STRATEGY_MODULE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return changes


def run_training(session_dir: Path, iteration: int, config: dict[str, Any], notes: str) -> ExperimentOutcome:
    log_path = session_dir / f"iteration_{iteration:02d}.log"
    print(f"[layout80] iteration {iteration:02d} start: {config['experiment_name']}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[layout80] launching {config['experiment_name']}\n")
        handle.flush()
        process = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=ROOT,
            env=build_env(config, notes),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert process.returncode == 0
    print(f"[layout80] iteration {iteration:02d} finished: {config['experiment_name']}", flush=True)
    df = load_results().sort_values("timestamp")
    row = df[df["experiment_name"] == config["experiment_name"]].iloc[-1]
    experiment_dir = ROOT / str(row["experiment_dir"])
    residual_mean, tail_residual_mean = read_oof_stats(experiment_dir)
    return ExperimentOutcome(
        iteration=iteration,
        experiment_name=str(row["experiment_name"]),
        oof_mae=float(row["oof_mae"]),
        mae_std=float(row["mae_std"]),
        residual_mean=residual_mean,
        tail_residual_mean=tail_residual_mean,
        config=parse_config(row["config_json"]),
        summary_path=str(row["summary_path"]),
        experiment_dir=str(row["experiment_dir"]),
    )


def analyze(last: ExperimentOutcome | None, baseline_mae: float, best_so_far: ExperimentOutcome | None) -> str:
    if last is None:
        return (
            f"기준점 `{BASELINE_EXPERIMENT_NAME}`의 OOF MAE는 {baseline_mae:.6f}다. "
            "이번 탐색은 `submission_80`의 layout-heavy bottleneck blend를 유지하되 layout_info interaction을 더 직접적으로 쓰는 방향이다."
        )
    best_text = ""
    if best_so_far is not None and best_so_far.experiment_name != last.experiment_name:
        best_text = f" 세션 최고는 `{best_so_far.experiment_name}`의 {best_so_far.oof_mae:.6f}다."
    return (
        f"직전 실험 `{last.experiment_name}`의 OOF MAE는 {last.oof_mae:.6f}, fold std는 {last.mae_std:.6f}였다. "
        f"baseline 대비 {last.oof_mae - baseline_mae:+.6f}, mean residual {last.residual_mean:.4f}, "
        f"tail residual mean {last.tail_residual_mean:.4f}다.{best_text}"
    )


def make_config(
    base: dict[str, Any],
    iteration: int,
    layout_feature_set: str,
    secondary_weight: float,
    target_weight_strength: float,
    use_layout_id: bool = False,
    secondary_use_layout_id: bool = True,
    secondary_layout_feature_set: str = "base",
    add_delay_risk_features: bool = False,
    delay_risk_feature_set: str = "base",
    secondary_target_weight_mode: str = "none",
    secondary_target_weight_strength: float = 0.0,
    n_splits: int | None = None,
) -> dict[str, Any]:
    config = dict(base)
    config["add_layout_interaction_features"] = True
    config["layout_feature_set"] = layout_feature_set
    config["secondary_layout_feature_set"] = secondary_layout_feature_set
    config["use_layout_id"] = use_layout_id
    config["secondary_use_layout_id"] = secondary_use_layout_id
    config["secondary_weight"] = secondary_weight
    config["target_weight_strength"] = target_weight_strength
    config["add_delay_risk_features"] = add_delay_risk_features
    config["delay_risk_feature_set"] = delay_risk_feature_set
    config["secondary_target_weight_mode"] = secondary_target_weight_mode
    config["secondary_target_weight_strength"] = secondary_target_weight_strength
    if n_splits is not None:
        config["n_splits"] = n_splits
    config["experiment_name"] = f"adaptive_layout80_{iteration:02d}"
    return config


def choose_config(iteration: int, base: dict[str, Any], history: list[ExperimentOutcome]) -> tuple[dict[str, Any], str]:
    recipes = [
        ("plus_flow", 0.35, 0.20, False, True, "base", False, "base", "none", 0.0, 5, "submission_80의 구조를 유지한 채 flow/layout 상호작용만 먼저 추가해 가장 직접적인 레이아웃 효과를 본다."),
        ("plus_path", 0.35, 0.20, False, True, "base", False, "base", "none", 0.0, 5, "통로 폭과 교차점 대기 중심의 path interaction으로 실제 동선 병목을 더 직접 반영한다."),
        ("plus_density", 0.35, 0.20, False, True, "base", False, "base", "none", 0.0, 5, "면적 대비 robot/pack/storage 밀도 축이 더 유효한지 분리 확인한다."),
        ("plus_hybrid", 0.30, 0.22, False, True, "base", False, "base", "none", 0.0, 5, "지금까지 본 layout 축을 합치고 secondary 비중을 조금 줄여 주모델의 layout signal을 더 살린다."),
        ("plus_hybrid", 0.28, 0.24, True, True, "base", False, "base", "none", 0.0, 5, "주모델에도 layout_id를 열어 layout metadata와 id 조합이 함께 먹히는지 확인한다."),
        ("plus_density", 0.25, 0.24, True, True, "plus_density", False, "base", "none", 0.0, 5, "밀도 축이 나쁘지 않다면 secondary에도 같은 density layout set을 공유해 layout bias를 보강한다."),
        ("plus_flow", 0.25, 0.28, True, True, "base", True, "plus_storage", "none", 0.0, 5, "layout 효과가 어느 정도 확인되면 storage 계열 delay_risk를 얇게 섞어 layout-driven access risk를 보강한다."),
        ("plus_path", 0.22, 0.26, True, False, "base", True, "plus_storage", "none", 0.0, 5, "secondary layout_id 의존을 제거해 path interaction의 일반화가 좋아지는지 본다."),
        ("plus_hybrid", 0.22, 0.28, True, True, "plus_hybrid", True, "plus_storage", "log", 0.02, 5, "가장 layout-heavy한 조합에서 secondary에만 아주 약한 weighting과 layout set 공유를 더한다."),
        ("plus_hybrid", 0.25, 0.28, True, True, "plus_hybrid", True, "plus_storage", "none", 0.0, 7, "마지막으로 가장 유망한 layout-heavy 조합을 더 촘촘한 fold로 재검증한다."),
    ]

    if history:
        best = min(history, key=lambda item: item.oof_mae)
        best_layout = str(best.config.get("layout_feature_set", "base"))
        best_sw = float(best.config.get("secondary_weight", base.get("secondary_weight", 0.35)))
        best_tws = float(best.config.get("target_weight_strength", base.get("target_weight_strength", 0.2)))
        best_use_layout_id = bool(best.config.get("use_layout_id", False))
        if iteration == 4 and best_layout in {"plus_density", "plus_flow", "plus_path"}:
            recipes[3] = (
                "plus_hybrid" if best_layout != "plus_hybrid" else best_layout,
                max(0.25, best_sw - 0.05),
                max(0.22, best_tws),
                best_use_layout_id,
                True,
                "base",
                False,
                "base",
                "none",
                0.0,
                5,
                "현재까지 나은 layout 축들을 합치고 secondary 비중을 소폭 줄여 주모델 신호를 살린다.",
            )
        if iteration == 5 and best_layout in {"plus_density", "plus_hybrid"}:
            recipes[4] = (
                best_layout,
                max(0.25, best_sw - 0.02),
                max(0.24, best_tws),
                True,
                True,
                "base",
                False,
                "base",
                "none",
                0.0,
                5,
                "현재 최고 layout 축에서 주모델 `layout_id`만 추가해 레이아웃 구분력의 실효성을 확인한다.",
            )
        if iteration == 10:
            recipes[9] = (
                best_layout if best_layout != "base" else "plus_hybrid",
                min(0.28, max(0.22, best_sw)),
                max(0.24, best_tws),
                bool(best.config.get("use_layout_id", True)),
                bool(best.config.get("secondary_use_layout_id", True)),
                str(best.config.get("secondary_layout_feature_set", "base")),
                bool(best.config.get("add_delay_risk_features", False)),
                str(best.config.get("delay_risk_feature_set", "base")),
                str(best.config.get("secondary_target_weight_mode", "none")),
                float(best.config.get("secondary_target_weight_strength", 0.0)),
                7,
                "현재 최고 조합을 더 촘촘한 fold로 다시 확인해 layout 신호의 안정성을 본다.",
            )

    recipe = recipes[iteration - 1]
    (
        layout_feature_set,
        secondary_weight,
        target_weight_strength,
        use_layout_id,
        secondary_use_layout_id,
        secondary_layout_feature_set,
        add_delay_risk_features,
        delay_risk_feature_set,
        secondary_target_weight_mode,
        secondary_target_weight_strength,
        n_splits,
        note,
    ) = recipe
    return make_config(
        base=base,
        iteration=iteration,
        layout_feature_set=str(layout_feature_set),
        secondary_weight=float(secondary_weight),
        target_weight_strength=float(target_weight_strength),
        use_layout_id=bool(use_layout_id),
        secondary_use_layout_id=bool(secondary_use_layout_id),
        secondary_layout_feature_set=str(secondary_layout_feature_set),
        add_delay_risk_features=bool(add_delay_risk_features),
        delay_risk_feature_set=str(delay_risk_feature_set),
        secondary_target_weight_mode=str(secondary_target_weight_mode),
        secondary_target_weight_strength=float(secondary_target_weight_strength),
        n_splits=int(n_splits),
    ), note


def append_report(
    report_path: Path,
    iteration: int,
    analysis_text: str,
    proposal: str,
    changes: list[str],
    outcome: ExperimentOutcome,
    baseline_mae: float,
    best_so_far: ExperimentOutcome,
) -> None:
    lines = [
        f"## Iteration {iteration:02d}",
        "",
        "1. 결과 분석",
        f"- {analysis_text}",
        "2. 개선 방향 제안",
        f"- {proposal}",
        "3. 코드 수정",
    ]
    lines.extend(changes or ["- 전략 파일 초기화"])
    lines.extend(
        [
            "4. 다음 실험 실행",
            f"- 실행 실험명: `{outcome.experiment_name}`",
            f"- OOF MAE: `{outcome.oof_mae:.6f}`",
            f"- Fold std: `{outcome.mae_std:.6f}`",
            f"- Mean residual: `{outcome.residual_mean:.4f}`",
            f"- Tail residual mean: `{outcome.tail_residual_mean:.4f}`",
            f"- baseline: `{baseline_mae:.6f}`",
            f"- 세션 최고: `{best_so_far.experiment_name}` / `{best_so_far.oof_mae:.6f}`",
            "",
        ]
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    session_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    report_path = session_dir / "report.md"

    df = load_results()
    baseline_row = get_row(df, BASELINE_EXPERIMENT_NAME)
    baseline_config = parse_config(baseline_row["config_json"])
    baseline_mae = float(baseline_row["oof_mae"])

    report_path.write_text(
        "# Adaptive Layout80 Report\n\n"
        f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- 반복 횟수: `{iterations}`\n"
        f"- 기준 실험: `{BASELINE_EXPERIMENT_NAME}` / `{baseline_mae:.6f}`\n"
        "- 탐색 원칙: `submission_80` bottleneck blend를 유지하고 layout_info interaction과 layout_id 사용 방식을 단계적으로 강화\n\n",
        encoding="utf-8",
    )

    history: list[ExperimentOutcome] = []
    previous_config: dict[str, Any] | None = None
    for iteration in range(1, iterations + 1):
        analysis_text = analyze(history[-1] if history else None, baseline_mae, min(history, key=lambda item: item.oof_mae) if history else None)
        next_config, proposal = choose_config(iteration, baseline_config, history)
        print(f"[layout80] proposal {iteration:02d}: {proposal}", flush=True)
        changes = write_strategy(iteration, analysis_text, proposal, previous_config, next_config)
        outcome = run_training(session_dir, iteration, next_config, proposal)
        history.append(outcome)
        best_so_far = min(history, key=lambda item: item.oof_mae)
        append_report(report_path, iteration, analysis_text, proposal, changes, outcome, baseline_mae, best_so_far)
        previous_config = next_config

    final_best = min(history, key=lambda item: item.oof_mae)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Final Summary\n\n"
            f"- 최고 layout-aware 확장: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )
    print(f"Adaptive layout80 session completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
