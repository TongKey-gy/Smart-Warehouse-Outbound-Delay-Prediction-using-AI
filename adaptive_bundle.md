# Adaptive Files Bundle

이 파일은 지정한 adaptive 실험/전략 Python 파일들을 한곳에 모은 번들이다.
원본 파일은 유지되며, 이 파일은 참고/공유용이다.

## adaptive_congestion_experiments.py

```python
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
STRATEGY_MODULE_PATH = ROOT / "adaptive_congestion_strategy.py"
RUNS_ROOT = ROOT / "outputs" / "adaptive_congestion_runs"
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
    "add_congestion_features",
    "add_temporal_features",
    "add_delay_risk_features",
    "target_weight_mode",
    "target_weight_strength",
    "blend_secondary_model",
    "secondary_weight",
    "secondary_use_layout_id",
    "secondary_add_capacity_features",
    "secondary_add_bottleneck_features",
    "secondary_add_congestion_features",
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def baseline_row(df: pd.DataFrame) -> pd.Series:
    rows = df[df["experiment_name"] == BASELINE_EXPERIMENT_NAME].sort_values("timestamp")
    if rows.empty:
        raise ValueError(f"Missing baseline experiment: {BASELINE_EXPERIMENT_NAME}")
    return rows.iloc[-1]


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
        '"""Adaptive congestion-mix strategy state for the next training run."""',
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


def make_config(base: dict[str, Any], iteration: int, recipe: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    config["validation_type"] = "group_kfold"
    config["group_column"] = "scenario_id"
    config["n_splits"] = 5
    config["blend_secondary_model"] = True
    config["add_congestion_features"] = True
    config["secondary_add_congestion_features"] = False
    config["secondary_add_capacity_features"] = True
    config["secondary_add_bottleneck_features"] = True
    config["secondary_seed"] = 7
    config["secondary_weight"] = recipe["secondary_weight"]
    config["target_weight_mode"] = "log"
    config["target_weight_strength"] = recipe["target_weight_strength"]
    config["secondary_target_weight_mode"] = recipe["secondary_target_weight_mode"]
    config["secondary_target_weight_strength"] = recipe["secondary_target_weight_strength"]
    config["secondary_use_layout_id"] = recipe["secondary_use_layout_id"]
    config["num_leaves"] = recipe["num_leaves"]
    config["min_child_samples"] = recipe["min_child_samples"]
    config["max_depth"] = recipe["max_depth"]
    config["learning_rate"] = recipe["learning_rate"]
    config["colsample_bytree"] = recipe["colsample_bytree"]
    config["subsample"] = recipe["subsample"]
    config["reg_alpha"] = recipe["reg_alpha"]
    config["reg_lambda"] = recipe["reg_lambda"]
    config["experiment_name"] = f"adaptive_congestion_mix_{iteration:02d}"
    return config


def describe_recipe(recipe: dict[str, Any]) -> str:
    return (
        "`adaptive_gkf_01` 블렌드는 유지한 채 주모델에만 congestion을 추가하고 "
        f"`num_leaves={recipe['num_leaves']}`, `min_child_samples={recipe['min_child_samples']}`, "
        f"`target_weight_strength={recipe['target_weight_strength']:.2f}`로 복잡도를 낮춘다. "
        f"secondary_weight={recipe['secondary_weight']:.2f}, "
        f"secondary_use_layout_id={recipe['secondary_use_layout_id']}로 과적합 완화를 함께 확인한다."
    )


def recipe_key(recipe: dict[str, Any]) -> tuple[Any, ...]:
    return (
        recipe["num_leaves"],
        recipe["min_child_samples"],
        recipe["max_depth"],
        round(float(recipe["target_weight_strength"]), 4),
        round(float(recipe["secondary_weight"]), 4),
        round(float(recipe["learning_rate"]), 4),
        round(float(recipe["colsample_bytree"]), 4),
        round(float(recipe["subsample"]), 4),
        round(float(recipe["reg_alpha"]), 4),
        round(float(recipe["reg_lambda"]), 4),
        recipe["secondary_use_layout_id"],
        recipe["secondary_target_weight_mode"],
        round(float(recipe["secondary_target_weight_strength"]), 4),
    )


def config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config["num_leaves"],
        config["min_child_samples"],
        config["max_depth"],
        round(float(config["target_weight_strength"]), 4),
        round(float(config["secondary_weight"]), 4),
        round(float(config["learning_rate"]), 4),
        round(float(config["colsample_bytree"]), 4),
        round(float(config["subsample"]), 4),
        round(float(config["reg_alpha"]), 4),
        round(float(config["reg_lambda"]), 4),
        bool(config["secondary_use_layout_id"]),
        config["secondary_target_weight_mode"],
        round(float(config["secondary_target_weight_strength"]), 4),
    )


def candidate_pool(best_recipe: dict[str, Any], widen: bool) -> list[dict[str, Any]]:
    leaves = int(best_recipe["num_leaves"])
    min_child = int(best_recipe["min_child_samples"])
    strength = float(best_recipe["target_weight_strength"])
    secondary_weight = float(best_recipe["secondary_weight"])

    pool = [
        {
            **best_recipe,
            "num_leaves": max(95, leaves - 16),
            "min_child_samples": min(30, min_child + 2),
            "target_weight_strength": clamp(strength - 0.02, 0.28, 0.36),
        },
        {
            **best_recipe,
            "num_leaves": leaves,
            "min_child_samples": min(30, min_child + 2),
            "target_weight_strength": clamp(strength, 0.28, 0.36),
            "secondary_weight": clamp(secondary_weight - 0.03, 0.20, 0.30),
        },
        {
            **best_recipe,
            "num_leaves": min(127, leaves + 16),
            "min_child_samples": max(22, min_child - 2),
            "target_weight_strength": clamp(strength + 0.02, 0.28, 0.36),
        },
        {
            **best_recipe,
            "secondary_use_layout_id": False,
            "target_weight_strength": clamp(strength, 0.28, 0.36),
        },
        {
            **best_recipe,
            "secondary_target_weight_mode": "log",
            "secondary_target_weight_strength": 0.04,
            "secondary_weight": clamp(secondary_weight - 0.02, 0.20, 0.30),
        },
        {
            **best_recipe,
            "learning_rate": clamp(float(best_recipe["learning_rate"]) - 0.002, 0.02, 0.03),
            "colsample_bytree": clamp(float(best_recipe["colsample_bytree"]) + 0.03, 0.85, 0.95),
            "subsample": clamp(float(best_recipe["subsample"]) + 0.02, 0.90, 0.96),
        },
        {
            **best_recipe,
            "reg_alpha": clamp(float(best_recipe["reg_alpha"]) + 0.01, 0.02, 0.06),
            "reg_lambda": clamp(float(best_recipe["reg_lambda"]) + 0.01, 0.02, 0.06),
        },
    ]
    if widen:
        pool.extend(
            [
                {
                    **best_recipe,
                    "num_leaves": 111,
                    "min_child_samples": 26,
                    "target_weight_strength": 0.30,
                    "secondary_weight": 0.22,
                    "secondary_use_layout_id": False,
                },
                {
                    **best_recipe,
                    "num_leaves": 95,
                    "min_child_samples": 30,
                    "max_depth": 10,
                    "target_weight_strength": 0.28,
                    "secondary_weight": 0.20,
                },
            ]
        )
    return pool


def analyze_outcome(last: ExperimentOutcome | None, baseline_mae: float, best_history: ExperimentOutcome | None) -> str:
    if last is None:
        return (
            f"기준점 `{BASELINE_EXPERIMENT_NAME}`의 OOF MAE는 {baseline_mae:.6f}다. "
            "첫 실험은 `NEXT_STEPS` 권장값처럼 주모델 congestion만 켜고 트리 용량과 타깃 가중을 약하게 낮춰 과적합 여부를 확인한다."
        )
    delta = last.oof_mae - baseline_mae
    best_text = ""
    if best_history is not None and best_history.experiment_name != last.experiment_name:
        best_text = f" 세션 최고는 `{best_history.experiment_name}`의 {best_history.oof_mae:.6f}다."
    return (
        f"직전 실험 `{last.experiment_name}`의 OOF MAE는 {last.oof_mae:.6f}, fold std는 {last.mae_std:.6f}였다."
        f" baseline 대비 변화는 {delta:+.6f}, 평균 residual은 {last.residual_mean:.4f}, "
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
            "num_leaves": 111,
            "min_child_samples": 24,
            "max_depth": 11,
            "target_weight_strength": 0.32,
            "secondary_weight": 0.25,
            "learning_rate": 0.025,
            "colsample_bytree": 0.85,
            "subsample": 0.9,
            "reg_alpha": 0.03,
            "reg_lambda": 0.03,
            "secondary_use_layout_id": True,
            "secondary_target_weight_mode": "none",
            "secondary_target_weight_strength": 0.0,
        }
        return make_config(baseline_config, iteration, recipe), describe_recipe(recipe)

    best_history = min(history, key=lambda item: item.oof_mae)
    best_recipe = {
        "num_leaves": int(best_history.config["num_leaves"]),
        "min_child_samples": int(best_history.config["min_child_samples"]),
        "max_depth": int(best_history.config["max_depth"]),
        "target_weight_strength": float(best_history.config["target_weight_strength"]),
        "secondary_weight": float(best_history.config["secondary_weight"]),
        "learning_rate": float(best_history.config["learning_rate"]),
        "colsample_bytree": float(best_history.config["colsample_bytree"]),
        "subsample": float(best_history.config["subsample"]),
        "reg_alpha": float(best_history.config["reg_alpha"]),
        "reg_lambda": float(best_history.config["reg_lambda"]),
        "secondary_use_layout_id": bool(best_history.config["secondary_use_layout_id"]),
        "secondary_target_weight_mode": best_history.config["secondary_target_weight_mode"],
        "secondary_target_weight_strength": float(best_history.config["secondary_target_weight_strength"]),
    }

    widen = len(history) >= 7 and min(item.oof_mae for item in history) > baseline_mae
    tried = {config_key(item.config) for item in history}
    for recipe in candidate_pool(best_recipe, widen=widen):
        candidate = make_config(baseline_config, iteration, recipe)
        if config_key(candidate) not in tried:
            return candidate, describe_recipe(recipe)

    fallback = {
        **best_recipe,
        "num_leaves": 95,
        "min_child_samples": 30,
        "max_depth": 10,
        "target_weight_strength": 0.30,
        "secondary_weight": 0.22,
        "secondary_use_layout_id": False,
    }
    return make_config(baseline_config, iteration, fallback), describe_recipe(fallback)


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
        "# Adaptive Congestion Mix Report\n\n"
        f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- 반복 횟수: `{iterations}`\n"
        f"- 기준 실험: `{baseline_name}` / `{baseline_mae:.6f}`\n"
        "- 탐색 원칙: `adaptive_gkf_01` 블렌드 유지, congestion은 주모델에만 매우 약하게 주입\n\n",
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
            f"- 최고 congestion mix: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )

    print(f"Adaptive congestion session completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
```

## adaptive_congestion_followup.py

```python
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
```

## adaptive_congestion_followup_strategy.py

```python
"""Follow-up congestion strategy state for the next training run."""

ITERATION = 3
ANALYSIS = '직전 실험 `adaptive_congestion_followup_02`의 OOF MAE는 9.114007, fold std는 0.237610였다. baseline 대비 +0.006525, congestion reference 대비 +0.006272다.'
PROPOSAL = '직전 개선이 있으면 보조모델에 아주 약한 log weighting을 더해 tail 보정을 미세하게 확인한다.'
CURRENT_STRATEGY = {
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.025,
    'num_leaves': 95,
    'max_depth': 11,
    'min_child_samples': 27,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'add_congestion_features': True,
    'target_weight_strength': 0.3,
    'secondary_weight': 0.22,
    'secondary_use_layout_id': True,
    'secondary_target_weight_mode': 'log',
    'secondary_target_weight_strength': 0.02,
    'experiment_name': 'adaptive_congestion_followup_03',
}
```

## adaptive_congestion_strategy.py

```python
"""Adaptive congestion-mix strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_congestion_mix_09`의 OOF MAE는 9.110202, fold std는 0.237354였다. baseline 대비 변화는 +0.002720, 평균 residual은 3.3566, tail residual 평균은 163.8171다. 세션 최고는 `adaptive_congestion_mix_04`의 9.107735다.'
PROPOSAL = '`adaptive_gkf_01` 블렌드는 유지한 채 주모델에만 congestion을 추가하고 `num_leaves=95`, `min_child_samples=28`, `target_weight_strength=0.30`로 복잡도를 낮춘다. secondary_weight=0.22, secondary_use_layout_id=True로 과적합 완화를 함께 확인한다.'
CURRENT_STRATEGY = {
    'validation_type': 'group_kfold',
    'group_column': 'scenario_id',
    'use_layout_info': True,
    'use_layout_id': False,
    'use_scenario_id': False,
    'seed': 42,
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.023,
    'num_leaves': 95,
    'max_depth': 11,
    'min_child_samples': 28,
    'subsample': 0.92,
    'colsample_bytree': 0.88,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'objective': 'regression',
    'objective_alpha': 0.9,
    'use_log_target': True,
    'add_capacity_features': True,
    'add_bottleneck_features': True,
    'add_congestion_features': True,
    'add_temporal_features': False,
    'add_delay_risk_features': True,
    'target_weight_mode': 'log',
    'target_weight_strength': 0.3,
    'blend_secondary_model': True,
    'secondary_weight': 0.22,
    'secondary_use_layout_id': True,
    'secondary_add_capacity_features': True,
    'secondary_add_bottleneck_features': True,
    'secondary_add_congestion_features': False,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'secondary_seed': 7,
    'experiment_name': 'adaptive_congestion_mix_10',
}
```

## adaptive_delay_risk_experiments.py

```python
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
RUNS_ROOT = ROOT / "outputs" / "adaptive_delay_risk_runs"
STRATEGY_MODULE_PATH = ROOT / "adaptive_delay_risk_strategy.py"
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
    "delay_risk_feature_set",
    "secondary_delay_risk_feature_set",
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
        '"""Adaptive delay-risk strategy state for the next training run."""',
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
    print(f"[delay-risk] iteration {iteration:02d} start: {config['experiment_name']}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[delay-risk] launching {config['experiment_name']}\n")
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
    print(f"[delay-risk] iteration {iteration:02d} finished: {config['experiment_name']}", flush=True)
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
            "이번 탐색은 `adaptive_gkf_01` 구조는 유지하고, 주모델에만 더 정교한 delay-risk feature set을 단계적으로 추가한다."
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
    delay_risk_feature_set: str,
    secondary_weight: float,
    target_weight_strength: float,
    secondary_use_layout_id: bool = True,
    secondary_target_weight_mode: str = "none",
    secondary_target_weight_strength: float = 0.0,
    secondary_delay_risk_feature_set: str = "base",
) -> dict[str, Any]:
    config = dict(base)
    config["delay_risk_feature_set"] = delay_risk_feature_set
    config["secondary_delay_risk_feature_set"] = secondary_delay_risk_feature_set
    config["secondary_weight"] = secondary_weight
    config["target_weight_strength"] = target_weight_strength
    config["secondary_use_layout_id"] = secondary_use_layout_id
    config["secondary_target_weight_mode"] = secondary_target_weight_mode
    config["secondary_target_weight_strength"] = secondary_target_weight_strength
    config["experiment_name"] = f"adaptive_delay_risk_{iteration:02d}"
    return config


def choose_config(iteration: int, base: dict[str, Any], history: list[ExperimentOutcome]) -> tuple[dict[str, Any], str]:
    recipes = [
        ("plus_flow", 0.25, 0.36, True, "none", 0.0, "base", "flow/backlog 계열 delay-risk만 먼저 추가해 가장 보수적인 확장을 확인한다."),
        ("plus_queue", 0.25, 0.36, True, "none", 0.0, "base", "queue/energy 계열 delay-risk로 병목 대기 신호가 더 직접적으로 먹히는지 본다."),
        ("plus_motion", 0.22, 0.38, True, "none", 0.0, "base", "movement friction 계열을 추가하고 weighting을 조금 높여 tail 과소예측을 줄이는지 본다."),
        ("plus_hybrid", 0.22, 0.36, True, "none", 0.0, "base", "flow+queue+motion을 함께 넣은 hybrid risk set으로 상호보완 여부를 확인한다."),
        ("plus_flow", 0.22, 0.40, True, "none", 0.0, "base", "현재까지 나은 방향이면 flow 확장에 stronger weighting을 얹어 tail 개선 폭을 본다."),
        ("plus_storage", 0.25, 0.36, True, "none", 0.0, "base", "storage 접근 리스크 축이 별도 신호를 주는지 분리 확인한다."),
        ("plus_hybrid", 0.20, 0.40, True, "none", 0.0, "base", "hybrid risk set에서 secondary 비중을 줄여 주모델 확장 효과만 더 선명하게 본다."),
        ("plus_flow", 0.22, 0.40, False, "none", 0.0, "base", "보조모델 layout_id 의존을 제거해 risk feature 일반화가 더 좋아지는지 본다."),
        ("plus_flow", 0.22, 0.40, True, "log", 0.02, "base", "보조모델에 아주 약한 log weighting을 넣어 risk 확장과 tail 보정을 결합한다."),
        ("plus_flow", 0.25, 0.40, True, "none", 0.0, "plus_flow", "마지막으로 보조모델에도 같은 flow risk를 얕게 공유해 진짜 역할 분리가 필요한지 확인한다."),
    ]

    if history:
        best = min(history, key=lambda item: item.oof_mae)
        best_set = best.config["delay_risk_feature_set"]
        best_sw = float(best.config["secondary_weight"])
        best_tws = float(best.config["target_weight_strength"])
        if iteration == 5 and best_set == "plus_flow":
            recipes[4] = ("plus_flow", best_sw, 0.40, True, "none", 0.0, "base", "현재 최고가 flow 계열이면 그 축에서 weighting만 더 높여 tail 보정 한계를 확인한다.")
        if iteration == 7 and best_set in {"plus_flow", "plus_hybrid"}:
            recipes[6] = (best_set, 0.20, max(best_tws, 0.38), True, "none", 0.0, "base", "현재 최고 feature set에서 secondary 비중만 더 낮춰 주모델 확장 효과를 극대화한다.")
        if iteration == 10 and best_set == "plus_flow":
            recipes[9] = ("plus_flow", best_sw, best_tws, True, "none", 0.0, "base", "현재 최고 flow 설정을 마지막으로 한 번 더 재검증한다.")

    feature_set, secondary_weight, target_weight_strength, secondary_use_layout_id, secondary_target_weight_mode, secondary_target_weight_strength, secondary_delay_risk_feature_set, note = recipes[iteration - 1]
    return make_config(
        base,
        iteration,
        str(feature_set),
        float(secondary_weight),
        float(target_weight_strength),
        bool(secondary_use_layout_id),
        str(secondary_target_weight_mode),
        float(secondary_target_weight_strength),
        str(secondary_delay_risk_feature_set),
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


def load_outcome(df: pd.DataFrame, experiment_name: str) -> ExperimentOutcome:
    row = get_row(df, experiment_name)
    experiment_dir = ROOT / str(row["experiment_dir"])
    residual_mean, tail_residual_mean = read_oof_stats(experiment_dir)
    return ExperimentOutcome(
        iteration=int(str(experiment_name).rsplit("_", 1)[-1]),
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


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    resume_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    completed_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    session_dir = resume_dir if resume_dir is not None else RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    report_path = session_dir / "report.md"

    df = load_results()
    baseline_row = get_row(df, BASELINE_EXPERIMENT_NAME)
    baseline_config = parse_config(baseline_row["config_json"])
    baseline_mae = float(baseline_row["oof_mae"])

    if completed_iterations == 0 or not report_path.exists():
        report_path.write_text(
            "# Adaptive Delay-Risk Report\n\n"
            f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
            f"- 반복 횟수: `{iterations}`\n"
            f"- 기준 실험: `{BASELINE_EXPERIMENT_NAME}` / `{baseline_mae:.6f}`\n"
            "- 탐색 원칙: `adaptive_gkf_01` 구조 유지, 주모델 delay-risk feature set만 단계적으로 확장\n\n",
            encoding="utf-8",
        )

    history: list[ExperimentOutcome] = []
    for finished_iteration in range(1, completed_iterations + 1):
        history.append(load_outcome(df, f"adaptive_delay_risk_{finished_iteration:02d}"))

    previous_config: dict[str, Any] | None = history[-1].config if history else None
    for iteration in range(completed_iterations + 1, iterations + 1):
        analysis_text = analyze(history[-1] if history else None, baseline_mae, min(history, key=lambda item: item.oof_mae) if history else None)
        next_config, proposal = choose_config(iteration, baseline_config, history)
        print(f"[delay-risk] proposal {iteration:02d}: {proposal}", flush=True)
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
            f"- 최고 delay-risk 확장: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )
    print(f"Adaptive delay-risk session completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
```

## adaptive_delay_risk_strategy.py

```python
"""Adaptive delay-risk strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_delay_risk_09`의 OOF MAE는 9.116918, fold std는 0.234705였다. baseline 대비 +0.009436, mean residual 3.2715, tail residual mean 163.0921다. 세션 최고는 `adaptive_delay_risk_06`의 9.108772다.'
PROPOSAL = '마지막으로 보조모델에도 같은 flow risk를 얕게 공유해 진짜 역할 분리가 필요한지 확인한다.'
CURRENT_STRATEGY = {
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.025,
    'num_leaves': 127,
    'max_depth': 11,
    'min_child_samples': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'delay_risk_feature_set': 'plus_flow',
    'secondary_delay_risk_feature_set': 'plus_flow',
    'target_weight_strength': 0.4,
    'secondary_weight': 0.25,
    'secondary_use_layout_id': True,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'experiment_name': 'adaptive_delay_risk_10',
}
```

## adaptive_experiments.py

```python
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
STRATEGY_MODULE_PATH = ROOT / "adaptive_strategy.py"
RUNS_ROOT = ROOT / "outputs" / "adaptive_runs"

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
    "add_robot_balance_features",
    "add_environment_features",
    "add_workload_features",
    "add_capacity_features",
    "add_bottleneck_features",
    "add_temporal_features",
    "add_congestion_features",
    "add_layout_interaction_features",
    "add_delay_risk_features",
    "target_weight_mode",
    "target_weight_strength",
    "min_prediction",
    "blend_secondary_model",
    "secondary_weight",
    "secondary_use_layout_id",
    "secondary_add_capacity_features",
    "secondary_add_bottleneck_features",
    "secondary_target_weight_mode",
    "secondary_target_weight_strength",
    "secondary_seed",
    "early_stopping_rounds",
    "log_evaluation_period",
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


def select_recent_reference(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    scored = df[df["oof_mae"].notna()].copy()
    scored = scored[scored["cv_strategy"] == "group_kfold"].copy()
    scored = scored.sort_values("timestamp")
    recent = scored.tail(20).copy()
    if recent.empty:
        raise ValueError("No recent group_kfold experiments with oof_mae were found.")
    best_recent = recent.sort_values("oof_mae").iloc[0]
    return best_recent, recent


def get_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return repr(value)


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


def summarize_recent_context(recent: pd.DataFrame) -> str:
    best = recent.sort_values("oof_mae").iloc[0]
    last = recent.iloc[-1]
    return (
        f"최근 group_kfold 실험 {len(recent)}개 기준 최고점은 `{best['experiment_name']}`의 "
        f"OOF MAE {best['oof_mae']:.6f}이고, 가장 최근 실험은 `{last['experiment_name']}`의 "
        f"OOF MAE {last['oof_mae']:.6f}였다."
    )


def analyze_outcome(last: ExperimentOutcome | None, best_so_far: ExperimentOutcome | None, recent_text: str) -> str:
    if last is None:
        return recent_text + " 최근 결과는 capacity/bottleneck + secondary blend 중심이며, tail underprediction을 직접 겨냥한 실험은 부족했다."

    delta_text = ""
    if best_so_far is not None and best_so_far.experiment_name != last.experiment_name:
        delta = last.oof_mae - best_so_far.oof_mae
        delta_text = f" 현재 최고점 대비 차이는 {delta:.6f}다."

    return (
        f"직전 실험 `{last.experiment_name}`의 OOF MAE는 {last.oof_mae:.6f}, fold std는 {last.mae_std:.6f}였다."
        f" 평균 residual은 {last.residual_mean:.4f}, 상위 1% tail residual 평균은 {last.tail_residual_mean:.4f}로"
        f" 고지연 구간 과소예측이 남아 있다.{delta_text}"
    )


def config_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    if previous is None:
        return [f"`{key}`={current[key]}" for key in TRACKED_KEYS if key in current]
    changes: list[str] = []
    for key in TRACKED_KEYS:
        if key not in current:
            continue
        old = previous.get(key)
        new = current.get(key)
        if old != new:
            changes.append(f"`{key}`: {old!r} -> {new!r}")
    return changes


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def objective_for_tail(last: ExperimentOutcome | None) -> tuple[str, float]:
    if last is not None and last.tail_residual_mean > 90.0:
        return "quantile", 0.65
    return "regression", 0.9


def propose_next_config(
    iteration: int,
    best_config: dict[str, Any],
    last: ExperimentOutcome | None,
    best_outcome: ExperimentOutcome | None,
    label_iteration: int,
) -> tuple[dict[str, Any], str]:
    base = dict(best_config if (last is None or (best_outcome and last.oof_mae >= best_outcome.oof_mae)) else last.config)
    objective, alpha = objective_for_tail(last)

    phase = (iteration - 1) % 6
    config = dict(base)
    config["objective"] = objective
    config["objective_alpha"] = alpha
    config["use_log_target"] = True
    config["validation_type"] = "group_kfold"
    config["group_column"] = "scenario_id"
    config["use_layout_info"] = True
    config["n_splits"] = 2
    config["n_estimators"] = min(int(base.get("n_estimators", 1100)), 100)
    config["early_stopping_rounds"] = 10
    config["log_evaluation_period"] = 200
    config["add_capacity_features"] = True
    config["add_bottleneck_features"] = True
    config["blend_secondary_model"] = False
    config["secondary_use_layout_id"] = True
    config["secondary_add_capacity_features"] = True
    config["secondary_add_bottleneck_features"] = True
    config["secondary_seed"] = 7

    if phase == 0:
        config["add_delay_risk_features"] = True
        config["target_weight_mode"] = "log"
        config["target_weight_strength"] = clamp(float(base.get("target_weight_strength", 0.2)) + 0.08, 0.12, 0.38)
        config["secondary_weight"] = 0.18
        reason = "tail underprediction을 줄이기 위해 delay_risk 피처와 stronger log weighting을 우선 적용한다."
    elif phase == 1:
        config["add_delay_risk_features"] = True
        config["add_layout_interaction_features"] = True
        config["target_weight_mode"] = "linear"
        config["target_weight_strength"] = 0.12
        config["secondary_weight"] = 0.18
        reason = "레이아웃 제약과 수요 압력의 상호작용을 함께 반영해 구조적 병목을 더 직접적으로 잡는다."
    elif phase == 2:
        config["add_delay_risk_features"] = True
        config["add_workload_features"] = True
        config["target_weight_mode"] = "sqrt"
        config["target_weight_strength"] = 0.10
        config["blend_secondary_model"] = False
        config["secondary_target_weight_mode"] = "log"
        config["secondary_target_weight_strength"] = 0.08
        config["secondary_weight"] = 0.18
        reason = "delay_risk와 workload를 결합하고 보조 모델도 약하게 가중해 분산과 tail bias를 동시에 줄인다."
    elif phase == 3:
        config["add_delay_risk_features"] = True
        config["add_congestion_features"] = True
        config["num_leaves"] = min(191, int(base.get("num_leaves", 127)) + 16)
        config["max_depth"] = min(13, int(base.get("max_depth", 11)) + 1)
        config["min_child_samples"] = max(10, int(base.get("min_child_samples", 20)) - 4)
        config["n_estimators"] = 140
        config["learning_rate"] = clamp(float(base.get("learning_rate", 0.025)) - 0.003, 0.015, 0.035)
        config["secondary_weight"] = 0.15
        reason = "복잡한 상호작용을 더 담기 위해 트리 용량을 늘리고, congestion 피처를 추가한다."
    elif phase == 4:
        config["add_delay_risk_features"] = True
        config["add_temporal_features"] = True
        config["subsample"] = clamp(float(base.get("subsample", 0.9)) + 0.03, 0.8, 0.98)
        config["colsample_bytree"] = clamp(float(base.get("colsample_bytree", 0.85)) + 0.05, 0.8, 0.98)
        config["reg_alpha"] = clamp(float(base.get("reg_alpha", 0.03)) * 0.5, 0.0, 0.05)
        config["reg_lambda"] = clamp(float(base.get("reg_lambda", 0.03)) * 0.5, 0.0, 0.05)
        config["blend_secondary_model"] = False
        config["secondary_weight"] = 0.18
        reason = "시간대 패턴과 약한 정규화를 통해 고지연 샘플 분리를 더 세밀하게 시도한다."
    else:
        config["add_delay_risk_features"] = True
        config["secondary_target_weight_mode"] = "none"
        config["secondary_target_weight_strength"] = 0.0
        config["secondary_weight"] = 0.10 if last and last.mae_std > 0.25 else 0.22
        config["num_leaves"] = max(95, int(base.get("num_leaves", 127)) - 16)
        config["min_child_samples"] = int(base.get("min_child_samples", 20)) + 6
        reason = "최근 fold 변동성을 완화하기 위해 블렌드 비중을 줄이고 트리를 약간 보수적으로 조정한다."

    if last is not None and last.oof_mae <= (best_outcome.oof_mae if best_outcome else last.oof_mae):
        config["secondary_weight"] = clamp(float(config.get("secondary_weight", 0.2)) + 0.05, 0.05, 0.4)
        reason += " 직전 결과가 최고점이었으므로 해당 방향을 근처 값으로 한 번 더 확장한다."

    if iteration % 5 == 0:
        config["objective"] = "regression"
        config["objective_alpha"] = 0.9

    config["experiment_name"] = f"adaptive_fast_gkf_{label_iteration:02d}"
    return config, reason


def write_strategy_module(
    iteration: int,
    analysis: str,
    proposal: str,
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> list[str]:
    changes = config_diff(previous_config, current_config)
    lines = [
        '"""Adaptive experiment strategy state for the next training run."""',
        "",
        f"ITERATION = {iteration}",
        f"ANALYSIS = {analysis!r}",
        f"PROPOSAL = {proposal!r}",
        "CURRENT_STRATEGY = {",
    ]
    for key in TRACKED_KEYS + ["experiment_name"]:
        if key in current_config:
            lines.append(f"    {key!r}: {format_value(current_config[key])},")
    lines.extend(["}", ""])
    STRATEGY_MODULE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return changes


def run_training(
    session_dir: Path,
    iteration: int,
    config: dict[str, Any],
    improvement_notes: str,
) -> ExperimentOutcome:
    log_path = session_dir / f"iteration_{iteration:02d}.log"
    env = build_env(config, improvement_notes)
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


def append_report(
    report_path: Path,
    iteration: int,
    analysis: str,
    proposal: str,
    changes: list[str],
    outcome: ExperimentOutcome,
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
            f"- 현재 최고 성능: `{best_so_far.experiment_name}` / `{best_so_far.oof_mae:.6f}`",
            "",
        ]
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    session_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    report_path = session_dir / "report.md"

    df = load_results()
    best_recent_row, recent = select_recent_reference(df)
    best_recent_config = parse_config(best_recent_row["config_json"])
    recent_text = summarize_recent_context(recent)
    report_path.write_text(
        "# Adaptive Experiment Report\n\n"
        f"- 세션 시작 시각: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- 반복 횟수: `{iterations}`\n"
        f"- 실험 번호 시작값: `{start_index}`\n"
        "- 탐색 프로토콜: `group_kfold 2-fold`, capped `n_estimators`, mostly single-model search\n"
        f"- 기준 실험: `{best_recent_row['experiment_name']}` / `{best_recent_row['oof_mae']:.6f}`\n"
        f"- 최근 요약: {recent_text}\n\n",
        encoding="utf-8",
    )

    history: list[ExperimentOutcome] = []
    previous_config: dict[str, Any] | None = None

    for iteration in range(1, iterations + 1):
        label_iteration = start_index + iteration - 1
        last = history[-1] if history else None
        best_outcome = min(history, key=lambda item: item.oof_mae) if history else None
        analysis = analyze_outcome(last, best_outcome, recent_text)
        proposal_base_config = best_outcome.config if best_outcome is not None else best_recent_config
        next_config, proposal = propose_next_config(iteration, proposal_base_config, last, best_outcome, label_iteration)
        changes = write_strategy_module(label_iteration, analysis, proposal, previous_config, next_config)
        outcome = run_training(session_dir, label_iteration, next_config, proposal)
        history.append(outcome)
        best_outcome = min(history, key=lambda item: item.oof_mae)
        append_report(report_path, label_iteration, analysis, proposal, changes, outcome, best_outcome)
        previous_config = next_config
        if outcome.oof_mae < float(best_recent_row["oof_mae"]):
            best_recent_row = pd.Series({"experiment_name": outcome.experiment_name, "oof_mae": outcome.oof_mae})
            best_recent_config = outcome.config

    final_best = min(history, key=lambda item: item.oof_mae)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Final Summary\n\n"
            f"- 최고 성능: `{final_best.experiment_name}` / `{final_best.oof_mae:.6f}`\n"
            f"- 최고 실험 디렉터리: `{final_best.experiment_dir}`\n"
            f"- 세부 요약 파일: `{final_best.summary_path}`\n"
        )
    print(f"Adaptive experiment session completed: {session_dir}")
    print(f"Best experiment: {final_best.experiment_name} / OOF MAE={final_best.oof_mae:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
```

## adaptive_gkf_tuning_experiments.py

```python
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
```

## adaptive_gkf_tuning_strategy.py

```python
"""Adaptive gkf tuning strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_gkf_tuning_09`의 OOF MAE는 9.113235, fold std는 0.236388였다. baseline 대비 +0.005753, mean residual 3.3602, tail residual mean 163.8906다. 세션 최고는 `adaptive_gkf_tuning_05`의 9.108616다.'
PROPOSAL = '현재 최고 설정을 마지막으로 한 번 더 좁게 재확인한다.'
CURRENT_STRATEGY = {
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.025,
    'num_leaves': 127,
    'max_depth': 11,
    'min_child_samples': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'target_weight_strength': 0.4,
    'secondary_weight': 0.22,
    'secondary_use_layout_id': True,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'experiment_name': 'adaptive_gkf_tuning_10',
}
```

## adaptive_layout80_experiments.py

```python
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
```

## adaptive_layout80_strategy.py

```python
"""Adaptive layout strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_layout80_09`의 OOF MAE는 9.113797, fold std는 0.236183였다. baseline 대비 -0.006404, mean residual 3.4746, tail residual mean 164.7175다. 세션 최고는 `adaptive_layout80_06`의 9.108352다.'
PROPOSAL = '현재 최고 조합을 더 촘촘한 fold로 다시 확인해 layout 신호의 안정성을 본다.'
CURRENT_STRATEGY = {
    'n_splits': 7,
    'use_layout_info': True,
    'use_layout_id': True,
    'add_layout_interaction_features': True,
    'layout_feature_set': 'plus_density',
    'add_delay_risk_features': False,
    'delay_risk_feature_set': 'base',
    'target_weight_strength': 0.24,
    'secondary_weight': 0.25,
    'secondary_use_layout_id': True,
    'secondary_layout_feature_set': 'plus_density',
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'experiment_name': 'adaptive_layout80_10',
}
```

## adaptive_strategy.py

```python
"""Adaptive experiment strategy state for the next training run."""

ITERATION = 24
ANALYSIS = '직전 실험 `adaptive_fast_gkf_23`의 OOF MAE는 9.885561, fold std는 0.314058였다. 평균 residual은 -0.4789, 상위 1% tail residual 평균은 136.6936로 고지연 구간 과소예측이 남아 있다. 현재 최고점 대비 차이는 0.734613다.'
PROPOSAL = '최근 fold 변동성을 완화하기 위해 블렌드 비중을 줄이고 트리를 약간 보수적으로 조정한다.'
CURRENT_STRATEGY = {
    'validation_type': 'group_kfold',
    'group_column': 'scenario_id',
    'use_layout_info': True,
    'use_layout_id': False,
    'use_scenario_id': False,
    'seed': 42,
    'n_splits': 3,
    'n_estimators': 450,
    'learning_rate': 0.025,
    'num_leaves': 111,
    'max_depth': 11,
    'min_child_samples': 26,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'objective': 'quantile',
    'objective_alpha': 0.65,
    'use_log_target': True,
    'add_robot_balance_features': False,
    'add_environment_features': False,
    'add_workload_features': False,
    'add_capacity_features': True,
    'add_bottleneck_features': True,
    'add_temporal_features': False,
    'add_congestion_features': False,
    'add_layout_interaction_features': False,
    'add_delay_risk_features': True,
    'target_weight_mode': 'log',
    'target_weight_strength': 0.36000000000000004,
    'min_prediction': 0.0,
    'blend_secondary_model': False,
    'secondary_weight': 0.1,
    'secondary_use_layout_id': True,
    'secondary_add_capacity_features': True,
    'secondary_add_bottleneck_features': True,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'secondary_seed': 7,
    'early_stopping_rounds': 20,
    'log_evaluation_period': 200,
    'experiment_name': 'adaptive_fast_gkf_24',
}
```

## adaptive_temporal_experiments.py

```python
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
```

## adaptive_temporal_strategy.py

```python
"""Adaptive temporal-mix strategy state for the next training run."""

ITERATION = 10
ANALYSIS = '직전 실험 `adaptive_temporal_mix_09`의 OOF MAE는 9.118080, fold std는 0.234965였다. baseline 대비 변화는 +0.010598, 평균 residual은 3.2515, tail residual 평균은 163.0346다. 세션 최고는 `adaptive_temporal_mix_01`의 9.116417다.'
PROPOSAL = '`adaptive_gkf_01` 본체를 기준으로 유지하고 보조 모델에만 temporal을 얹어 실효 temporal 비중을 `0.05`로 조정한다. secondary_use_layout_id=False, secondary_target_weight_mode=none로 과적합 여부를 함께 확인한다.'
CURRENT_STRATEGY = {
    'validation_type': 'group_kfold',
    'group_column': 'scenario_id',
    'use_layout_info': True,
    'use_layout_id': False,
    'use_scenario_id': False,
    'seed': 42,
    'n_splits': 5,
    'n_estimators': 1100,
    'learning_rate': 0.025,
    'num_leaves': 127,
    'max_depth': 11,
    'min_child_samples': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.03,
    'objective': 'regression',
    'objective_alpha': 0.9,
    'use_log_target': True,
    'add_capacity_features': True,
    'add_bottleneck_features': True,
    'add_temporal_features': False,
    'add_congestion_features': False,
    'add_layout_interaction_features': False,
    'add_delay_risk_features': True,
    'target_weight_mode': 'log',
    'target_weight_strength': 0.36,
    'blend_secondary_model': True,
    'secondary_weight': 0.05,
    'secondary_use_layout_id': False,
    'secondary_add_capacity_features': True,
    'secondary_add_bottleneck_features': True,
    'secondary_add_temporal_features': True,
    'secondary_target_weight_mode': 'none',
    'secondary_target_weight_strength': 0.0,
    'secondary_seed': 7,
    'experiment_name': 'adaptive_temporal_mix_10',
}
```

