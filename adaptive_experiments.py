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
