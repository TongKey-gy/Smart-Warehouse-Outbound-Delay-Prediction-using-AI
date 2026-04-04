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
RUNS_ROOT = ROOT / "outputs" / "adaptive_congestion_followup_runs"
STRATEGY_MODULE_PATH = ROOT / "adaptive_congestion_followup_strategy.py"
BASELINE_EXPERIMENT_NAME = "adaptive_gkf_01"
REFERENCE_EXPERIMENT_NAME = "adaptive_congestion_mix_04"

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
    "add_congestion_features",
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


def build_env(config: dict[str, Any], improvement_notes: str) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in config.items():
        env[f"TRAIN_{key.upper()}"] = str(value)
    env["TRAIN_IMPROVEMENT_NOTES"] = improvement_notes
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


def write_strategy(iteration: int, analysis: str, proposal: str, prev: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    changes = config_diff(prev, current)
    lines = [
        '"""Follow-up congestion strategy state for the next training run."""',
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


def make_config(base: dict[str, Any], iteration: int, overrides: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    config.update(overrides)
    config["experiment_name"] = f"adaptive_congestion_followup_{iteration:02d}"
    return config


def analyze(last: ExperimentOutcome | None, baseline_mae: float, reference_mae: float) -> str:
    if last is None:
        return (
            f"기준점 `{BASELINE_EXPERIMENT_NAME}`은 {baseline_mae:.6f}, 현재 congestion 최고 `{REFERENCE_EXPERIMENT_NAME}`는 "
            f"{reference_mae:.6f}다. follow-up은 이 근처 초근접 3점만 다시 확인한다."
        )
    return (
        f"직전 실험 `{last.experiment_name}`의 OOF MAE는 {last.oof_mae:.6f}, fold std는 {last.mae_std:.6f}였다. "
        f"baseline 대비 {last.oof_mae - baseline_mae:+.6f}, congestion reference 대비 {last.oof_mae - reference_mae:+.6f}다."
    )


def choose_config(iteration: int, base: dict[str, Any], history: list[ExperimentOutcome]) -> tuple[dict[str, Any], str]:
    recipes = [
        (
            {
                "num_leaves": 95,
                "min_child_samples": 27,
                "target_weight_strength": 0.30,
                "secondary_weight": 0.23,
                "secondary_use_layout_id": True,
                "secondary_target_weight_mode": "none",
                "secondary_target_weight_strength": 0.0,
            },
            "최고점 `mix_04` 주변에서 secondary_weight만 `0.23`으로 한 칸 올려 블렌드 균형을 미세 조정한다.",
        ),
        (
            {
                "num_leaves": 95,
                "min_child_samples": 27,
                "target_weight_strength": 0.31,
                "secondary_weight": 0.22,
                "secondary_use_layout_id": True,
                "secondary_target_weight_mode": "none",
                "secondary_target_weight_strength": 0.0,
            },
            "첫 결과를 보고 congestion 규제를 아주 조금만 풀기 위해 `target_weight_strength=0.31`로 올린다.",
        ),
        (
            {
                "num_leaves": 95,
                "min_child_samples": 28,
                "target_weight_strength": 0.30,
                "secondary_weight": 0.22,
                "secondary_use_layout_id": False,
                "secondary_target_weight_mode": "none",
                "secondary_target_weight_strength": 0.0,
            },
            "마지막으로 보조 모델 layout_id 의존을 제거한 near-best 조합을 다시 확인해 일반화 편향을 점검한다.",
        ),
    ]

    if history:
        best = min(history, key=lambda item: item.oof_mae)
        if iteration == 2 and best.oof_mae > history[0].oof_mae:
            recipes[1] = (
                {
                    "num_leaves": 95,
                    "min_child_samples": 28,
                    "target_weight_strength": 0.30,
                    "secondary_weight": 0.21,
                    "secondary_use_layout_id": True,
                    "secondary_target_weight_mode": "none",
                    "secondary_target_weight_strength": 0.0,
                },
                "첫 결과가 더 나쁘면 secondary_weight를 `0.21`로 낮춰 보조모델 비중만 더 보수적으로 줄인다.",
            )
        if iteration == 3 and best.oof_mae <= history[0].oof_mae:
            recipes[2] = (
                {
                    "num_leaves": 95,
                    "min_child_samples": 27,
                    "target_weight_strength": 0.30,
                    "secondary_weight": 0.22,
                    "secondary_use_layout_id": True,
                    "secondary_target_weight_mode": "log",
                    "secondary_target_weight_strength": 0.02,
                },
                "직전 개선이 있으면 보조모델에 아주 약한 log weighting을 더해 tail 보정을 미세하게 확인한다.",
            )

    overrides, note = recipes[iteration - 1]
    return make_config(base, iteration, overrides), note


def append_report(
    report_path: Path,
    iteration: int,
    analysis_text: str,
    proposal: str,
    changes: list[str],
    outcome: ExperimentOutcome,
    baseline_mae: float,
    reference_mae: float,
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
            f"- reference: `{reference_mae:.6f}`",
            f"- follow-up 최고: `{best_so_far.experiment_name}` / `{best_so_far.oof_mae:.6f}`",
            "",
        ]
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    session_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    report_path = session_dir / "report.md"

    df = load_results()
    baseline_row = get_row(df, BASELINE_EXPERIMENT_NAME)
    reference_row = get_row(df, REFERENCE_EXPERIMENT_NAME)
    baseline_mae = float(baseline_row["oof_mae"])
    reference_mae = float(reference_row["oof_mae"])
    reference_config = parse_config(reference_row["config_json"])

    report_path.write_text(
        "# Adaptive Congestion Follow-up Report\n\n"
        f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- 반복 횟수: `{iterations}`\n"
        f"- baseline: `{BASELINE_EXPERIMENT_NAME}` / `{baseline_mae:.6f}`\n"
        f"- reference: `{REFERENCE_EXPERIMENT_NAME}` / `{reference_mae:.6f}`\n\n",
        encoding="utf-8",
    )

    history: list[ExperimentOutcome] = []
    previous_config: dict[str, Any] | None = None
    for iteration in range(1, iterations + 1):
        analysis_text = analyze(history[-1] if history else None, baseline_mae, reference_mae)
        next_config, proposal = choose_config(iteration, reference_config, history)
        changes = write_strategy(iteration, analysis_text, proposal, previous_config, next_config)
        outcome = run_training(session_dir, iteration, next_config, proposal)
        history.append(outcome)
        best_so_far = min(history, key=lambda item: item.oof_mae)
        append_report(report_path, iteration, analysis_text, proposal, changes, outcome, baseline_mae, reference_mae, best_so_far)
        previous_config = next_config

    final_best = min(history, key=lambda item: item.oof_mae)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Final Summary\n\n"
            f"- 최고 follow-up: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )
    print(f"Adaptive congestion follow-up completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
