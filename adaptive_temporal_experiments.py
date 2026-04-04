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
STRATEGY_MODULE_PATH = ROOT / "adaptive_temporal_strategy.py"
RUNS_ROOT = ROOT / "outputs" / "adaptive_temporal_runs"
BASELINE_EXPERIMENT_NAME = "adaptive_gkf_01"

TRACKED_KEYS = [
    "validation_type",
    "group_column",
    "use_layout_info",
    "use_layout_id",
    "use_scenario_id",
    "seed",
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
    "objective",
    "objective_alpha",
    "use_log_target",
    "add_capacity_features",
    "add_bottleneck_features",
    "add_temporal_features",
    "add_congestion_features",
    "add_layout_interaction_features",
    "add_delay_risk_features",
    "target_weight_mode",
    "target_weight_strength",
    "blend_secondary_model",
    "secondary_weight",
    "secondary_use_layout_id",
    "secondary_add_capacity_features",
    "secondary_add_bottleneck_features",
    "secondary_add_temporal_features",
    "secondary_target_weight_mode",
    "secondary_target_weight_strength",
    "secondary_seed",
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
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results log: {RESULTS_PATH}")
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
    oof_path = experiment_dir / "oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    residual_mean = float(oof["residual"].mean())
    tail_cutoff = float(oof["target"].quantile(0.99))
    tail = oof[oof["target"] >= tail_cutoff]
    tail_residual_mean = float(tail["residual"].mean()) if not tail.empty else residual_mean
    return residual_mean, tail_residual_mean


def config_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    if previous is None:
        return [f"`{key}`={current[key]!r}" for key in TRACKED_KEYS if key in current]
    changes: list[str] = []
    for key in TRACKED_KEYS:
        if key not in current:
            continue
        old = previous.get(key)
        new = current.get(key)
        if old != new:
            changes.append(f"`{key}`: {old!r} -> {new!r}")
    return changes


def write_strategy_module(
    iteration: int,
    analysis: str,
    proposal: str,
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> list[str]:
    changes = config_diff(previous_config, current_config)
    lines = [
        '"""Adaptive temporal-mix strategy state for the next training run."""',
        "",
        f"ITERATION = {iteration}",
        f"ANALYSIS = {analysis!r}",
        f"PROPOSAL = {proposal!r}",
        "CURRENT_STRATEGY = {",
    ]
    for key in TRACKED_KEYS + ["experiment_name"]:
        if key in current_config:
            lines.append(f"    {key!r}: {current_config[key]!r},")
    lines.extend(["}", ""])
    STRATEGY_MODULE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return changes


def run_training(session_dir: Path, iteration: int, config: dict[str, Any], notes: str) -> ExperimentOutcome:
    log_path = session_dir / f"iteration_{iteration:02d}.log"
    env = build_env(config, notes)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=ROOT,
            env=env,
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


def baseline_row(df: pd.DataFrame) -> pd.Series:
    rows = df[df["experiment_name"] == BASELINE_EXPERIMENT_NAME].sort_values("timestamp")
    if rows.empty:
        raise ValueError(f"Missing baseline experiment: {BASELINE_EXPERIMENT_NAME}")
    return rows.iloc[-1]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def candidate_pool(best_weight: float, prefer_primary: bool, widen: bool) -> list[dict[str, Any]]:
    lower = clamp(best_weight - 0.03, 0.04, 0.20)
    upper = clamp(best_weight + 0.03, 0.05, 0.22)
    widened = clamp(best_weight + 0.06, 0.06, 0.24)
    conservative = clamp(best_weight - 0.04, 0.04, 0.18)

    secondary_first = [
        {"branch": "secondary_only", "temporal_weight": conservative, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.36},
        {"branch": "secondary_only", "temporal_weight": lower, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.36},
        {"branch": "secondary_only", "temporal_weight": best_weight, "secondary_use_layout_id": True, "secondary_target_weight_mode": "log", "secondary_target_weight_strength": 0.04, "target_weight_strength": 0.36},
        {"branch": "secondary_only", "temporal_weight": upper, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.34},
        {"branch": "secondary_only", "temporal_weight": lower, "secondary_use_layout_id": False, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.36},
        {"branch": "secondary_only", "temporal_weight": upper, "secondary_use_layout_id": False, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.34},
        {"branch": "secondary_only", "temporal_weight": best_weight, "secondary_use_layout_id": True, "secondary_target_weight_mode": "sqrt", "secondary_target_weight_strength": 0.06, "target_weight_strength": 0.36},
    ]
    primary_first = [
        {"branch": "primary_only", "temporal_weight": conservative, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.34},
        {"branch": "primary_only", "temporal_weight": lower, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.34},
        {"branch": "primary_only", "temporal_weight": best_weight, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.38},
        {"branch": "primary_only", "temporal_weight": upper, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.32},
    ]

    pool = primary_first + secondary_first if prefer_primary else secondary_first + primary_first
    if widen:
        pool.extend(
            [
                {"branch": "secondary_only", "temporal_weight": widened, "secondary_use_layout_id": False, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.34},
                {"branch": "primary_only", "temporal_weight": widened, "secondary_use_layout_id": True, "secondary_target_weight_mode": "none", "secondary_target_weight_strength": 0.0, "target_weight_strength": 0.32},
            ]
        )
    return pool


def recipe_key(recipe: dict[str, Any]) -> tuple[Any, ...]:
    return (
        recipe["branch"],
        round(float(recipe["temporal_weight"]), 4),
        bool(recipe["secondary_use_layout_id"]),
        recipe["secondary_target_weight_mode"],
        round(float(recipe["secondary_target_weight_strength"]), 4),
        round(float(recipe["target_weight_strength"]), 4),
    )


def config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(config["add_temporal_features"]),
        bool(config["secondary_add_temporal_features"]),
        round(float(config["secondary_weight"]), 4),
        bool(config["secondary_use_layout_id"]),
        config["secondary_target_weight_mode"],
        round(float(config["secondary_target_weight_strength"]), 4),
        round(float(config["target_weight_strength"]), 4),
    )


def make_config(base: dict[str, Any], iteration: int, recipe: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    config["validation_type"] = "group_kfold"
    config["group_column"] = "scenario_id"
    config["n_splits"] = 5
    config["blend_secondary_model"] = True
    config["secondary_add_capacity_features"] = True
    config["secondary_add_bottleneck_features"] = True
    config["secondary_seed"] = 7
    config["target_weight_mode"] = "log"
    config["target_weight_strength"] = recipe["target_weight_strength"]
    config["secondary_target_weight_mode"] = recipe["secondary_target_weight_mode"]
    config["secondary_target_weight_strength"] = recipe["secondary_target_weight_strength"]
    config["secondary_use_layout_id"] = recipe["secondary_use_layout_id"]

    temporal_weight = float(recipe["temporal_weight"])
    if recipe["branch"] == "secondary_only":
        config["add_temporal_features"] = False
        config["secondary_add_temporal_features"] = True
        config["secondary_weight"] = temporal_weight
    else:
        config["add_temporal_features"] = True
        config["secondary_add_temporal_features"] = False
        config["secondary_weight"] = 1.0 - temporal_weight

    config["experiment_name"] = f"adaptive_temporal_mix_{iteration:02d}"
    return config


def describe_recipe(recipe: dict[str, Any]) -> str:
    branch = "보조 모델" if recipe["branch"] == "secondary_only" else "주 모델"
    return (
        f"`adaptive_gkf_01` 본체를 기준으로 유지하고 {branch}에만 temporal을 얹어 "
        f"실효 temporal 비중을 `{recipe['temporal_weight']:.2f}`로 조정한다."
        f" secondary_use_layout_id={recipe['secondary_use_layout_id']}, "
        f"secondary_target_weight_mode={recipe['secondary_target_weight_mode']}로 과적합 여부를 함께 확인한다."
    )


def analyze_outcome(
    last: ExperimentOutcome | None,
    baseline_mae: float,
    best_history: ExperimentOutcome | None,
) -> str:
    if last is None:
        return (
            f"기준점 `{BASELINE_EXPERIMENT_NAME}`의 OOF MAE는 {baseline_mae:.6f}다. "
            "첫 실험은 temporal을 보조 블렌드에만 아주 약하게 섞어 baseline anchor가 유지되는지 확인한다."
        )

    delta_baseline = last.oof_mae - baseline_mae
    best_text = ""
    if best_history is not None and best_history.experiment_name != last.experiment_name:
        best_text = f" 세션 최고는 `{best_history.experiment_name}`의 {best_history.oof_mae:.6f}다."
    return (
        f"직전 실험 `{last.experiment_name}`의 OOF MAE는 {last.oof_mae:.6f}, fold std는 {last.mae_std:.6f}였다."
        f" baseline 대비 변화는 {delta_baseline:+.6f}, 평균 residual은 {last.residual_mean:.4f}, "
        f"tail residual 평균은 {last.tail_residual_mean:.4f}다.{best_text}"
    )


def choose_recipe(
    iteration: int,
    history: list[ExperimentOutcome],
    baseline_config: dict[str, Any],
    baseline_mae: float,
) -> tuple[dict[str, Any], str]:
    if not history:
        recipe = {
            "branch": "secondary_only",
            "temporal_weight": 0.08,
            "secondary_use_layout_id": True,
            "secondary_target_weight_mode": "none",
            "secondary_target_weight_strength": 0.0,
            "target_weight_strength": 0.36,
        }
        return make_config(baseline_config, iteration, recipe), describe_recipe(recipe)

    best_history = min(history, key=lambda item: item.oof_mae)
    best_weight = best_history.config["secondary_weight"]
    if best_history.config.get("add_temporal_features"):
        best_weight = 1.0 - float(best_weight)
    else:
        best_weight = float(best_weight)

    prefer_primary = (
        all(item.oof_mae > baseline_mae + 0.002 for item in history)
        and len(history) >= 4
    )
    widen = len(history) >= 7 and min(item.oof_mae for item in history) > baseline_mae
    tried = {config_key(item.config) for item in history}

    for recipe in candidate_pool(best_weight, prefer_primary=prefer_primary, widen=widen):
        candidate = make_config(baseline_config, iteration, recipe)
        if config_key(candidate) not in tried:
            return candidate, describe_recipe(recipe)

    fallback_recipe = {
        "branch": "secondary_only",
        "temporal_weight": clamp(best_weight + 0.01, 0.04, 0.24),
        "secondary_use_layout_id": False,
        "secondary_target_weight_mode": "none",
        "secondary_target_weight_strength": 0.0,
        "target_weight_strength": 0.34,
    }
    return make_config(baseline_config, iteration, fallback_recipe), describe_recipe(fallback_recipe)


def append_report(
    report_path: Path,
    iteration: int,
    analysis: str,
    proposal: str,
    changes: list[str],
    outcome: ExperimentOutcome,
    baseline_name: str,
    baseline_mae: float,
    best_so_far: ExperimentOutcome,
) -> None:
    lines = [
        f"## Iteration {iteration:02d}",
        "",
        "1. 결과 분석",
        f"- {analysis}",
        "2. 개선 방향 제안",
        f"- {proposal}",
        "3. 코드 수정",
    ]
    if changes:
        lines.extend(f"- {change}" for change in changes)
    else:
        lines.append("- 전략 파일 초기화")
    lines.extend(
        [
            "4. 다음 실험 실행",
            f"- 실행 실험명: `{outcome.experiment_name}`",
            f"- OOF MAE: `{outcome.oof_mae:.6f}`",
            f"- Fold std: `{outcome.mae_std:.6f}`",
            f"- Mean residual: `{outcome.residual_mean:.4f}`",
            f"- Tail residual mean: `{outcome.tail_residual_mean:.4f}`",
            f"- 기준점: `{baseline_name}` / `{baseline_mae:.6f}`",
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
    base_row = baseline_row(df)
    baseline_name = str(base_row["experiment_name"])
    baseline_mae = float(base_row["oof_mae"])
    baseline_config = parse_config(base_row["config_json"])

    report_path.write_text(
        "# Adaptive Temporal Mix Report\n\n"
        f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- 반복 횟수: `{iterations}`\n"
        f"- 기준 실험: `{baseline_name}` / `{baseline_mae:.6f}`\n"
        "- 탐색 원칙: `adaptive_gkf_01` anchor 유지, temporal은 약한 blend 축으로만 주입\n\n",
        encoding="utf-8",
    )

    history: list[ExperimentOutcome] = []
    previous_config: dict[str, Any] | None = None

    for iteration in range(1, iterations + 1):
        last = history[-1] if history else None
        best_history = min(history, key=lambda item: item.oof_mae) if history else None
        analysis = analyze_outcome(last, baseline_mae, best_history)
        next_config, proposal = choose_recipe(iteration, history, baseline_config, baseline_mae)
        changes = write_strategy_module(iteration, analysis, proposal, previous_config, next_config)
        outcome = run_training(session_dir, iteration, next_config, proposal)
        history.append(outcome)
        best_so_far = min(history, key=lambda item: item.oof_mae)
        append_report(report_path, iteration, analysis, proposal, changes, outcome, baseline_name, baseline_mae, best_so_far)
        previous_config = next_config

    final_best = min(history, key=lambda item: item.oof_mae)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Final Summary\n\n"
            f"- 기준 실험: `{baseline_name}` / `{baseline_mae:.6f}`\n"
            f"- 최고 temporal mix: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )

    print(f"Adaptive temporal session completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
