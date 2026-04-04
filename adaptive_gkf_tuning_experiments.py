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
RUNS_ROOT = ROOT / "outputs" / "adaptive_gkf_tuning_runs"
STRATEGY_MODULE_PATH = ROOT / "adaptive_gkf_tuning_strategy.py"
BASELINE_EXPERIMENT_NAME = "adaptive_gkf_01"

TRACKED_KEYS = [
    "n_splits",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "target_weight_strength",
    "secondary_weight",
    "secondary_use_layout_id",
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
    improvement_notes: str


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


def write_strategy(iteration: int, analysis: str, proposal: str, previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    changes = config_diff(previous, current)
    lines = [
        '"""Adaptive gkf tuning strategy state for the next training run."""',
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
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=ROOT,
            env=build_env(config, notes),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert process.returncode == 0
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
        improvement_notes=str(row["improvement_notes"]),
    )


def analyze(last: ExperimentOutcome | None, baseline_mae: float, best_so_far: ExperimentOutcome | None) -> str:
    if last is None:
        return (
            f"기준점 `{BASELINE_EXPERIMENT_NAME}`의 OOF MAE는 {baseline_mae:.6f}다. "
            "이번 탐색은 feature set을 고정하고 secondary_weight와 target_weight_strength만 좁게 미세 조정한다."
        )
    best_text = ""
    if best_so_far is not None and best_so_far.experiment_name != last.experiment_name:
        best_text = f" 세션 최고는 `{best_so_far.experiment_name}`의 {best_so_far.oof_mae:.6f}다."
    return (
        f"직전 실험 `{last.experiment_name}`의 OOF MAE는 {last.oof_mae:.6f}, fold std는 {last.mae_std:.6f}였다. "
        f"baseline 대비 {last.oof_mae - baseline_mae:+.6f}, mean residual {last.residual_mean:.4f}, "
        f"tail residual mean {last.tail_residual_mean:.4f}다.{best_text}"
    )


def make_config(base: dict[str, Any], iteration: int, secondary_weight: float, target_weight_strength: float, secondary_use_layout_id: bool, secondary_target_weight_mode: str = "none", secondary_target_weight_strength: float = 0.0) -> dict[str, Any]:
    config = dict(base)
    config["secondary_weight"] = secondary_weight
    config["target_weight_strength"] = target_weight_strength
    config["secondary_use_layout_id"] = secondary_use_layout_id
    config["secondary_target_weight_mode"] = secondary_target_weight_mode
    config["secondary_target_weight_strength"] = secondary_target_weight_strength
    config["experiment_name"] = f"adaptive_gkf_tuning_{iteration:02d}"
    return config


def choose_config(iteration: int, base: dict[str, Any], history: list[ExperimentOutcome]) -> tuple[dict[str, Any], str]:
    recipes = [
        (0.22, 0.34, True, "none", 0.0, "baseline보다 보조모델 비중과 타깃 가중을 동시에 한 단계 낮춰 과적합을 줄인다."),
        (0.20, 0.34, True, "none", 0.0, "직전 결과를 보고 secondary_weight를 더 낮춰 주모델 anchor를 더 강하게 유지한다."),
        (0.22, 0.30, True, "none", 0.0, "target_weight_strength를 더 낮춰 tail overweight가 과했는지 확인한다."),
        (0.30, 0.34, True, "none", 0.0, "반대로 보조모델 비중을 높여 blend 다양성이 도움이 되는지 확인한다."),
        (0.22, 0.40, True, "none", 0.0, "tail residual을 더 줄일 여지가 있는지 higher target weighting을 재확인한다."),
        (0.25, 0.34, True, "none", 0.0, "baseline blend에 더 가까운 위치에서 weight만 줄인 보수적 조합을 확인한다."),
        (0.20, 0.30, True, "none", 0.0, "두 축을 동시에 낮춘 가장 보수적 조합으로 분산 축소를 확인한다."),
        (0.22, 0.34, False, "none", 0.0, "보조모델의 layout_id 의존을 제거해 generalization을 점검한다."),
        (0.22, 0.34, True, "log", 0.02, "보조모델에만 아주 약한 log weighting을 넣어 tail 보정을 미세하게 확인한다."),
        (0.24, 0.32, True, "none", 0.0, "앞선 결과를 절충해 secondary_weight와 target weighting을 중간값으로 맞춘다."),
    ]

    if history:
        best = min(history, key=lambda item: item.oof_mae)
        if iteration == 6 and best.oof_mae < history[0].oof_mae:
            recipes[5] = (best.config["secondary_weight"], 0.32, True, "none", 0.0, "현재 최고 설정 근처에서 target weighting만 0.32로 절충한다.")
        if iteration == 10 and best.oof_mae <= min(item.oof_mae for item in history[:-1]):
            recipes[9] = (
                float(best.config["secondary_weight"]),
                float(best.config["target_weight_strength"]),
                bool(best.config["secondary_use_layout_id"]),
                best.config["secondary_target_weight_mode"],
                float(best.config["secondary_target_weight_strength"]),
                "현재 최고 설정을 마지막으로 한 번 더 좁게 재확인한다.",
            )

    secondary_weight, target_weight_strength, secondary_use_layout_id, secondary_target_weight_mode, secondary_target_weight_strength, note = recipes[iteration - 1]
    return make_config(
        base,
        iteration,
        float(secondary_weight),
        float(target_weight_strength),
        bool(secondary_use_layout_id),
        str(secondary_target_weight_mode),
        float(secondary_target_weight_strength),
    ), note


def append_report(report_path: Path, iteration: int, analysis_text: str, proposal: str, changes: list[str], outcome: ExperimentOutcome, baseline_mae: float, best_so_far: ExperimentOutcome) -> None:
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
        "# Adaptive GKF Tuning Report\n\n"
        f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- 반복 횟수: `{iterations}`\n"
        f"- 기준 실험: `{BASELINE_EXPERIMENT_NAME}` / `{baseline_mae:.6f}`\n"
        "- 탐색 원칙: feature set 고정, secondary_weight와 target_weight_strength만 미세 조정\n\n",
        encoding="utf-8",
    )

    history: list[ExperimentOutcome] = []
    previous_config: dict[str, Any] | None = None
    for iteration in range(1, iterations + 1):
        analysis_text = analyze(history[-1] if history else None, baseline_mae, min(history, key=lambda item: item.oof_mae) if history else None)
        next_config, proposal = choose_config(iteration, baseline_config, history)
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
            f"- 최고 tuning: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )
    print(f"Adaptive gkf tuning completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
