from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "lightgbm is required for this baseline. "
        "Install it with 'pip install lightgbm' or './.venv/bin/pip install lightgbm'."
    ) from exc

from prepare import (
    LOGS_DIR,
    OUTPUTS_DIR,
    SUBMISSIONS_DIR,
    ensure_runtime_directories,
    load_prepared_data,
)

BASE_EXCLUDED_FEATURE_COLUMNS = (
    "ID",
    "replenishment_overlap",
    "task_reassign_15m",
)
CONFIG = {
    "experiment_name": "default_groupkfold_bottleneck_blend_v1",
    "validation_type": "group_kfold",
    "group_column": "scenario_id",
    "use_layout_info": True,
    "use_layout_id": False,
    "use_scenario_id": False,
    "seed": 42,
    "n_splits": 5,
    "n_estimators": 1100,
    "learning_rate": 0.025,
    "num_leaves": 127,
    "max_depth": 11,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.03,
    "reg_lambda": 0.03,
    "objective": "regression",
    "use_log_target": True,
    "add_robot_balance_features": False,
    "add_environment_features": False,
    "add_workload_features": False,
    "add_capacity_features": True,
    "add_bottleneck_features": True,
    "add_temporal_features": False,
    "add_congestion_features": False,
    "add_layout_interaction_features": False,
    "target_weight_mode": "log",
    "target_weight_strength": 0.2,
    "min_prediction": 0.0,
    "blend_secondary_model": True,
    "secondary_weight": 0.35,
    "secondary_use_layout_id": True,
    "secondary_add_capacity_features": True,
    "secondary_add_bottleneck_features": True,
    "secondary_target_weight_mode": "none",
    "secondary_target_weight_strength": 0.0,
    "secondary_seed": 7,
    "early_stopping_rounds": 30,
    "log_evaluation_period": 100,
}

README_PATH = Path(__file__).resolve().parent / "README.md"
SUBMISSIONS_LOCAL_DIR = OUTPUTS_DIR / "submissions_local"
EXPERIMENT_LOG_SECTION_TITLE = "## 실험기록"
EXPERIMENT_LOG_START = "<!-- EXPERIMENT_LOG_START -->"
EXPERIMENT_LOG_END = "<!-- EXPERIMENT_LOG_END -->"
EXPERIMENT_LOG_COLUMNS = [
    "실험 번호",
    "저장 파일명 (submission_xx.csv)",
    "원본 파일명",
    "실험 시각",
    "모델/전략",
    "성능 점수",
    "개선 사항",
]
SUBMISSION_ALIAS_PATTERN = re.compile(r"submission_(\d+)\.csv")
ORIGINAL_SUBMISSION_PATTERN = re.compile(r"^(submission_.+?)_(\d{8}_\d{6})\.csv$")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "default_groupkfold_bottleneck_blend_v1"
    validation_type: str = "group_kfold"
    group_column: str = "scenario_id"
    use_layout_info: bool = True
    use_layout_id: bool = False
    use_scenario_id: bool = False
    seed: int = 42
    n_splits: int = 5
    n_estimators: int = 1100
    learning_rate: float = 0.025
    num_leaves: int = 127
    max_depth: int = 11
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.03
    reg_lambda: float = 0.03
    objective: str = "regression"
    use_log_target: bool = True
    add_robot_balance_features: bool = False
    add_environment_features: bool = False
    add_workload_features: bool = False
    add_capacity_features: bool = True
    add_bottleneck_features: bool = True
    add_temporal_features: bool = False
    add_congestion_features: bool = False
    add_layout_interaction_features: bool = False
    target_weight_mode: str = "log"
    target_weight_strength: float = 0.2
    min_prediction: float = 0.0
    blend_secondary_model: bool = True
    secondary_weight: float = 0.35
    secondary_use_layout_id: bool = True
    secondary_add_capacity_features: bool = True
    secondary_add_bottleneck_features: bool = True
    secondary_target_weight_mode: str = "none"
    secondary_target_weight_strength: float = 0.0
    secondary_seed: int = 7
    early_stopping_rounds: int = 30
    log_evaluation_period: int = 100


def _parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> ExperimentConfig:
    values = dict(CONFIG)

    for name, default_value in CONFIG.items():
        env_key = f"TRAIN_{name.upper()}"
        raw_value = os.environ.get(env_key)
        if raw_value is None:
            continue

        if isinstance(default_value, bool):
            values[name] = _parse_bool(raw_value)
        elif isinstance(default_value, int):
            values[name] = int(raw_value)
        elif isinstance(default_value, float):
            values[name] = float(raw_value)
        else:
            values[name] = raw_value

    return ExperimentConfig(**values)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(config: ExperimentConfig, n_estimators: int | None = None) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators or config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_samples=config.min_child_samples,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed,
        objective=config.objective,
        metric="rmse",
        verbose=-1,
    )


def evaluate_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def append_results_log(row: dict[str, object]) -> Path:
    results_path = LOGS_DIR / "results.csv"
    new_frame = pd.DataFrame([row])

    if results_path.exists():
        existing = pd.read_csv(results_path)
        ordered_columns = existing.columns.tolist()
        for column in new_frame.columns:
            if column not in ordered_columns:
                ordered_columns.append(column)
        for column in ordered_columns:
            if column not in existing.columns:
                existing[column] = pd.NA
            if column not in new_frame.columns:
                new_frame[column] = pd.NA
        combined = pd.concat(
            [existing.loc[:, ordered_columns], new_frame.loc[:, ordered_columns]],
            ignore_index=True,
        )
    else:
        combined = new_frame

    combined.to_csv(results_path, index=False)
    return results_path


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return preprocessor.get_feature_names_out().tolist()


def create_experiment_dir(config: ExperimentConfig, timestamp: str) -> Path:
    experiment_dir = OUTPUTS_DIR / "experiments" / f"{timestamp}_{config.experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def ensure_local_submission_dir() -> Path:
    SUBMISSIONS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    return SUBMISSIONS_LOCAL_DIR


def extract_timestamp_from_name(filename: str) -> str | None:
    match = ORIGINAL_SUBMISSION_PATTERN.match(filename)
    if not match:
        return None
    return match.group(2)


def infer_strategy_from_name(filename: str) -> str:
    stem = Path(filename).stem
    timestamp = extract_timestamp_from_name(filename)
    if timestamp is None:
        return stem.removeprefix("submission_")
    return stem[: -(len(timestamp) + 1)].removeprefix("submission_")


def format_experiment_time(timestamp: str | None) -> str:
    if not timestamp:
        return "-"
    return f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"


def format_submission_alias(number: int) -> str:
    return f"submission_{number:02d}.csv"


def read_submission_alias_numbers() -> list[int]:
    alias_numbers: set[int] = set()

    if README_PATH.exists():
        content = README_PATH.read_text(encoding="utf-8")
        alias_numbers.update(int(match) for match in SUBMISSION_ALIAS_PATTERN.findall(content))

    if SUBMISSIONS_LOCAL_DIR.exists():
        for path in SUBMISSIONS_LOCAL_DIR.glob("submission_*.csv"):
            match = re.fullmatch(SUBMISSION_ALIAS_PATTERN, path.name)
            if match:
                alias_numbers.add(int(match.group(1)))

    return sorted(alias_numbers)


def get_next_submission_number() -> int:
    existing_numbers = read_submission_alias_numbers()
    if not existing_numbers:
        return 1
    return max(existing_numbers) + 1


def load_metrics_lookup() -> dict[str, dict[str, str]]:
    metrics_lookup: dict[str, dict[str, str]] = {}

    for metrics_path in sorted((OUTPUTS_DIR / "experiments").glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        submission_path = summary.get("submission_path", "")
        original_name = Path(submission_path).name if submission_path else ""
        if not original_name:
            continue

        score_value = summary.get("oof_mae", summary.get("mae_mean", summary.get("oof_rmse", summary.get("rmse_mean"))))
        score_text = "-" if score_value in (None, "") else f"{float(score_value):.6f}"
        strategy = summary.get("model_name") or summary.get("experiment_name") or infer_strategy_from_name(original_name)

        metrics_lookup[original_name] = {
            "timestamp": summary.get("timestamp", "") or extract_timestamp_from_name(original_name) or "",
            "score": score_text,
            "strategy": str(strategy),
            "improvement": str(summary.get("improvement_notes", "-") or "-"),
        }

    results_path = LOGS_DIR / "results.csv"
    if results_path.exists():
        results_frame = pd.read_csv(results_path)
        for _, row in results_frame.iterrows():
            submission_path = row.get("submission_path")
            if pd.isna(submission_path):
                continue

            original_name = Path(str(submission_path)).name
            if not original_name or original_name in metrics_lookup:
                continue

            raw_score = row.get("oof_mae", row.get("mae_mean", row.get("oof_rmse", row.get("rmse_mean"))))
            score_text = "-" if pd.isna(raw_score) else f"{float(raw_score):.6f}"
            strategy = row.get("model_name", row.get("experiment_name", infer_strategy_from_name(original_name)))
            timestamp = row.get("timestamp")
            metrics_lookup[original_name] = {
                "timestamp": "" if pd.isna(timestamp) else str(timestamp),
                "score": score_text,
                "strategy": str(strategy),
                "improvement": "-" if pd.isna(row.get("improvement_notes")) else str(row.get("improvement_notes")),
            }

    return metrics_lookup


def build_experiment_records() -> list[dict[str, str]]:
    metrics_lookup = load_metrics_lookup()
    original_files = sorted(
        [path.name for path in SUBMISSIONS_DIR.glob("submission_*.csv")],
        key=lambda name: extract_timestamp_from_name(name) or name,
    )
    records: list[dict[str, str]] = []

    for idx, original_name in enumerate(original_files, start=1):
        metrics = metrics_lookup.get(original_name, {})
        timestamp = metrics.get("timestamp") or extract_timestamp_from_name(original_name) or ""
        records.append(
            {
                "실험 번호": str(idx),
                "저장 파일명 (submission_xx.csv)": format_submission_alias(idx),
                "원본 파일명": original_name,
                "실험 시각": format_experiment_time(timestamp),
                "모델/전략": metrics.get("strategy", infer_strategy_from_name(original_name)),
                "성능 점수": metrics.get("score", "-"),
                "개선 사항": metrics.get("improvement", "-"),
            }
        )

    return records


def render_experiment_log_table(records: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(EXPERIMENT_LOG_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(EXPERIMENT_LOG_COLUMNS)) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(record[column] for column in EXPERIMENT_LOG_COLUMNS) + " |")
    return "\n".join(lines)


def update_readme_experiment_log() -> list[dict[str, str]]:
    records = build_experiment_records()
    table = render_experiment_log_table(records)
    section = (
        f"{EXPERIMENT_LOG_SECTION_TITLE}\n\n"
        f"{EXPERIMENT_LOG_START}\n"
        f"{table}\n"
        f"{EXPERIMENT_LOG_END}"
    )

    content = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(EXPERIMENT_LOG_SECTION_TITLE)}\n\n{re.escape(EXPERIMENT_LOG_START)}.*?{re.escape(EXPERIMENT_LOG_END)}",
        re.DOTALL,
    )

    if pattern.search(content):
        updated = pattern.sub(section, content, count=1)
    else:
        updated = content.rstrip() + "\n\n" + section + "\n"

    README_PATH.write_text(updated, encoding="utf-8")
    return records


def archive_submission_locally(source_path: Path) -> tuple[int, Path]:
    submission_number = get_next_submission_number()
    local_dir = ensure_local_submission_dir()
    alias_name = format_submission_alias(submission_number)
    archived_path = local_dir / alias_name
    shutil.copy2(source_path, archived_path)
    return submission_number, archived_path


def save_oof_predictions(
    prepared,
    predictions: pd.Series,
    experiment_dir: Path,
) -> Path:
    oof_frame = pd.DataFrame(
        {
            "row_index": prepared.X_train.index,
            "prediction": predictions.loc[prepared.X_train.index].to_numpy(),
            "target": prepared.y.loc[prepared.X_train.index].to_numpy(),
        }
    )
    if "ID" in prepared.train_df.columns:
        oof_frame.insert(1, "ID", prepared.train_df.loc[prepared.X_train.index, "ID"].to_numpy())
    oof_frame["residual"] = oof_frame["target"] - oof_frame["prediction"]
    output_path = experiment_dir / "oof_predictions.csv"
    oof_frame.to_csv(output_path, index=False)
    return output_path


def save_feature_importance(fold_importances: list[pd.DataFrame], experiment_dir: Path) -> Path | None:
    if not fold_importances:
        return None

    all_importances = pd.concat(fold_importances, ignore_index=True)
    summary = (
        all_importances.groupby("feature_name", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    output_path = experiment_dir / "feature_importance.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def save_experiment_summary(
    prepared,
    config: ExperimentConfig,
    fold_rmse_scores: list[float],
    fold_mae_scores: list[float],
    best_iterations: list[int],
    results_row: dict[str, object],
    experiment_dir: Path,
) -> Path:
    summary_path = experiment_dir / "metrics.json"
    payload = {
        "config": asdict(config),
        "excluded_feature_columns": list(get_excluded_feature_columns(prepared, config)),
        "fold_rmse_scores": fold_rmse_scores,
        "fold_mae_scores": fold_mae_scores,
        "best_iterations": best_iterations,
        "summary": results_row,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary_path


def get_model_best_iteration(model: LGBMRegressor, config: ExperimentConfig) -> int:
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        return config.n_estimators
    return int(best_iteration)


def get_layout_metadata_columns(prepared) -> list[str]:
    return [
        column
        for column in prepared.layout_df.columns
        if column != "layout_id" and column in prepared.feature_columns
    ]


def get_excluded_feature_columns(prepared, config: ExperimentConfig) -> list[str]:
    excluded_columns = list(BASE_EXCLUDED_FEATURE_COLUMNS)
    if not config.use_layout_info:
        excluded_columns.extend(get_layout_metadata_columns(prepared))
    if not config.use_layout_id:
        excluded_columns.append("layout_id")
    if not config.use_scenario_id:
        excluded_columns.append("scenario_id")
    return list(dict.fromkeys(excluded_columns))


def select_training_view(
    prepared,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    excluded_feature_columns = get_excluded_feature_columns(prepared, config)
    selected_columns = [
        column for column in prepared.feature_columns if column not in excluded_feature_columns
    ]
    numeric_columns = [column for column in prepared.numeric_columns if column in selected_columns]
    categorical_columns = [column for column in prepared.categorical_columns if column in selected_columns]
    X_train = prepared.X_train.loc[:, selected_columns].copy()
    X_test = prepared.X_test.loc[:, selected_columns].copy()
    excluded_columns = [column for column in excluded_feature_columns if column in prepared.feature_columns]
    return X_train, X_test, numeric_columns, categorical_columns, excluded_columns


def add_engineered_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()

        if config.add_robot_balance_features:
            if {"robot_active", "robot_idle", "robot_charging"}.issubset(result.columns):
                total_robot_state = result["robot_active"] + result["robot_idle"] + result["robot_charging"] + 1.0
                result["robot_active_ratio"] = result["robot_active"] / total_robot_state
                result["robot_idle_ratio"] = result["robot_idle"] / total_robot_state
                result["robot_charging_ratio"] = result["robot_charging"] / total_robot_state
            if {"battery_mean", "battery_std"}.issubset(result.columns):
                result["battery_stability"] = result["battery_mean"] - result["battery_std"]

        if config.add_environment_features:
            if {"warehouse_temp_avg", "external_temp_c"}.issubset(result.columns):
                result["temp_gap"] = result["warehouse_temp_avg"] - result["external_temp_c"]
            if {"humidity_pct", "warehouse_temp_avg"}.issubset(result.columns):
                result["temp_humidity_interaction"] = result["humidity_pct"] * result["warehouse_temp_avg"]
            if {"air_quality_idx", "co2_level_ppm"}.issubset(result.columns):
                result["air_quality_load"] = result["air_quality_idx"] * result["co2_level_ppm"]

        if config.add_workload_features:
            if {"order_inflow_15m", "staff_on_floor"}.issubset(result.columns):
                result["orders_per_staff"] = result["order_inflow_15m"] / (result["staff_on_floor"] + 1.0)
            if {"order_inflow_15m", "robot_active"}.issubset(result.columns):
                result["orders_per_robot"] = result["order_inflow_15m"] / (result["robot_active"] + 1.0)
            if {"unique_sku_15m", "pick_list_length_avg"}.issubset(result.columns):
                result["sku_pick_pressure"] = result["unique_sku_15m"] * result["pick_list_length_avg"]
            if {"loading_dock_util", "staging_area_util"}.issubset(result.columns):
                result["dock_staging_pressure"] = result["loading_dock_util"] * result["staging_area_util"]

        if config.add_capacity_features:
            if {"order_inflow_15m", "pack_station_count"}.issubset(result.columns):
                result["orders_per_pack_station"] = result["order_inflow_15m"] / (result["pack_station_count"] + 1.0)
            if {"order_inflow_15m", "charger_count"}.issubset(result.columns):
                result["orders_per_charger"] = result["order_inflow_15m"] / (result["charger_count"] + 1.0)
            if {"order_inflow_15m", "loading_dock_util"}.issubset(result.columns):
                result["dock_load_pressure"] = result["order_inflow_15m"] * result["loading_dock_util"]
            if {"order_inflow_15m", "pack_utilization"}.issubset(result.columns):
                result["pack_load_pressure"] = result["order_inflow_15m"] * result["pack_utilization"]
            if {"robot_active", "pack_station_count"}.issubset(result.columns):
                result["robots_per_pack_station"] = result["robot_active"] / (result["pack_station_count"] + 1.0)
            if {"charge_queue_length", "charger_count"}.issubset(result.columns):
                result["charge_queue_per_charger"] = result["charge_queue_length"] / (result["charger_count"] + 1.0)

        if config.add_bottleneck_features:
            if {"blocked_path_15m", "robot_active"}.issubset(result.columns):
                result["blocked_path_per_robot"] = result["blocked_path_15m"] / (result["robot_active"] + 1.0)
            if {"blocked_path_15m", "avg_trip_distance"}.issubset(result.columns):
                result["blocked_path_trip_pressure"] = result["blocked_path_15m"] / (result["avg_trip_distance"] + 1.0)
            if {"outbound_truck_wait_min", "loading_dock_util"}.issubset(result.columns):
                result["truck_wait_per_dock_util"] = result["outbound_truck_wait_min"] / (
                    result["loading_dock_util"] + 0.1
                )
            if {"congestion_score", "robot_active"}.issubset(result.columns):
                result["congestion_per_robot"] = result["congestion_score"] / (result["robot_active"] + 1.0)
            if {"charge_queue_length", "robot_active"}.issubset(result.columns):
                result["charge_queue_per_robot"] = result["charge_queue_length"] / (result["robot_active"] + 1.0)
            if {"label_print_queue", "staff_on_floor"}.issubset(result.columns):
                result["labels_per_staff"] = result["label_print_queue"] / (result["staff_on_floor"] + 1.0)
            if {"order_inflow_15m", "blocked_path_15m"}.issubset(result.columns):
                result["inflow_blocked_pressure"] = result["order_inflow_15m"] * (result["blocked_path_15m"] + 1.0)
            if {"order_inflow_15m", "congestion_score"}.issubset(result.columns):
                result["inflow_congestion_pressure"] = result["order_inflow_15m"] * (
                    result["congestion_score"] + 1.0
                )

        if config.add_temporal_features:
            if "shift_hour" in result.columns:
                shift_angle = 2.0 * np.pi * result["shift_hour"] / 24.0
                result["shift_hour_sin"] = np.sin(shift_angle)
                result["shift_hour_cos"] = np.cos(shift_angle)
                result["is_night_shift"] = result["shift_hour"].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(float)
                result["is_peak_shift"] = result["shift_hour"].isin([14, 15, 16, 17, 18, 19, 20, 21, 22]).astype(float)
            if "day_of_week" in result.columns:
                week_angle = 2.0 * np.pi * result["day_of_week"] / 7.0
                result["day_of_week_sin"] = np.sin(week_angle)
                result["day_of_week_cos"] = np.cos(week_angle)
                result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(float)

        if config.add_congestion_features:
            if {"congestion_score", "avg_trip_distance"}.issubset(result.columns):
                result["trip_congestion_pressure"] = result["congestion_score"] * result["avg_trip_distance"]
            if {"blocked_path_15m", "near_collision_15m"}.issubset(result.columns):
                result["path_disruption"] = result["blocked_path_15m"] + result["near_collision_15m"]
            if {"max_zone_density", "zone_dispersion"}.issubset(result.columns):
                result["zone_density_pressure"] = result["max_zone_density"] * result["zone_dispersion"]
            if {"manual_override_ratio", "congestion_score"}.issubset(result.columns):
                result["override_congestion_pressure"] = (
                    result["manual_override_ratio"] * result["congestion_score"]
                )

        if config.add_layout_interaction_features:
            if {"avg_trip_distance", "layout_compactness"}.issubset(result.columns):
                result["trip_distance_layout_penalty"] = (
                    result["avg_trip_distance"] * (1.0 - result["layout_compactness"])
                )
            if {"pack_station_count", "floor_area_sqm"}.issubset(result.columns):
                result["pack_station_density"] = result["pack_station_count"] / (result["floor_area_sqm"] + 1.0)
            if {"robot_total", "floor_area_sqm"}.issubset(result.columns):
                result["robot_density"] = result["robot_total"] / (result["floor_area_sqm"] + 1.0)
            if {"charger_count", "robot_total"}.issubset(result.columns):
                result["charger_coverage"] = result["charger_count"] / (result["robot_total"] + 1.0)

        return result

    return transform(X_train), transform(X_test)


def build_sample_weights(target: pd.Series, config: ExperimentConfig) -> np.ndarray | None:
    mode = config.target_weight_mode.strip().lower()
    strength = float(config.target_weight_strength)
    if mode == "none" or strength <= 0.0:
        return None

    safe_target = np.clip(target.to_numpy(dtype=float), a_min=0.0, a_max=None)
    if mode == "log":
        weights = 1.0 + strength * np.log1p(safe_target)
    elif mode == "sqrt":
        weights = 1.0 + strength * np.sqrt(safe_target)
    elif mode == "linear":
        scale = float(np.mean(safe_target) + 1.0)
        weights = 1.0 + strength * (safe_target / scale)
    else:
        raise ValueError(
            "CONFIG['target_weight_mode'] must be one of: none, log, sqrt, linear."
        )

    return weights


def make_secondary_config(config: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(
        **{
            **asdict(config),
            "use_layout_id": config.secondary_use_layout_id,
            "add_capacity_features": config.secondary_add_capacity_features,
            "add_bottleneck_features": config.secondary_add_bottleneck_features,
            "target_weight_mode": config.secondary_target_weight_mode,
            "target_weight_strength": config.secondary_target_weight_strength,
            "seed": config.secondary_seed,
        }
    )


def train_single_model(
    prepared,
    config: ExperimentConfig,
) -> tuple[pd.Series, np.ndarray, list[float], list[float], list[pd.DataFrame], list[int], list[str], list[str]]:
    X_train, X_test, numeric_columns, categorical_columns, _ = select_training_view(prepared, config)
    X_train, X_test = add_engineered_features(X_train, X_test, config)
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    _, _, split_iterator = iter_train_cv_splits(prepared, config)
    train_target = np.log1p(prepared.y) if config.use_log_target else prepared.y.copy()
    sample_weights = build_sample_weights(prepared.y, config)

    oof_predictions = pd.Series(index=X_train.index, dtype=float)
    fold_rmse_scores: list[float] = []
    fold_mae_scores: list[float] = []
    fold_importances: list[pd.DataFrame] = []
    best_iterations: list[int] = []

    for fold_idx, train_idx, valid_idx in split_iterator:
        X_fold_train = X_train.loc[train_idx]
        y_fold_train = train_target.loc[train_idx]
        X_fold_valid = X_train.loc[valid_idx]
        y_fold_valid = prepared.y.loc[valid_idx]
        y_fold_valid_train_scale = train_target.loc[valid_idx]

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        X_fold_train_processed = preprocessor.fit_transform(X_fold_train)
        X_fold_valid_processed = preprocessor.transform(X_fold_valid)
        feature_names = get_feature_names(preprocessor)

        model = build_model(config)
        fold_sample_weights = None if sample_weights is None else sample_weights[X_train.index.get_indexer(train_idx)]
        model.fit(
            X_fold_train_processed,
            y_fold_train,
            sample_weight=fold_sample_weights,
            eval_set=[(X_fold_valid_processed, y_fold_valid_train_scale)],
            eval_metric="l1" if config.use_log_target else "mae",
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds),
                lgb.log_evaluation(config.log_evaluation_period),
            ],
        )

        best_iteration = get_model_best_iteration(model, config)
        best_iterations.append(best_iteration)
        fold_pred = model.predict(X_fold_valid_processed, num_iteration=best_iteration)
        if config.use_log_target:
            fold_pred = np.expm1(fold_pred)
        fold_pred = np.clip(fold_pred, a_min=config.min_prediction, a_max=None)
        oof_predictions.loc[valid_idx] = fold_pred
        fold_rmse = evaluate_rmse(y_fold_valid, fold_pred)
        fold_mae = evaluate_mae(y_fold_valid, fold_pred)
        fold_rmse_scores.append(fold_rmse)
        fold_mae_scores.append(fold_mae)
        fold_importances.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "feature_name": feature_names,
                    "importance": model.feature_importances_,
                }
            )
        )
        print(f"Fold {fold_idx}: MAE={fold_mae:.6f} RMSE={fold_rmse:.6f} BEST_ITER={best_iteration}")

    final_n_estimators = max(1, int(round(float(np.mean(best_iterations)))))
    final_preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    X_train_processed = final_preprocessor.fit_transform(X_train)
    X_test_processed = final_preprocessor.transform(X_test)
    final_model = build_model(config, n_estimators=final_n_estimators)
    final_model.fit(X_train_processed, train_target, sample_weight=sample_weights)
    test_pred = final_model.predict(X_test_processed)
    if config.use_log_target:
        test_pred = np.expm1(test_pred)
    test_pred = np.clip(test_pred, a_min=config.min_prediction, a_max=None)

    return (
        oof_predictions,
        test_pred,
        fold_rmse_scores,
        fold_mae_scores,
        fold_importances,
        best_iterations,
        X_train.columns.tolist(),
        [column for column in get_excluded_feature_columns(prepared, config) if column in prepared.feature_columns],
    )


def resolve_cv_groups(prepared, config: ExperimentConfig) -> pd.Series | None:
    if config.validation_type != "group_kfold":
        return None
    if not config.group_column:
        raise ValueError("CONFIG['group_column'] must be set when validation_type is 'group_kfold'.")
    if config.group_column not in prepared.train_df.columns:
        raise ValueError(f"Group column '{config.group_column}' is not available in train data.")
    return prepared.train_df.loc[prepared.X_train.index, config.group_column].copy()


def iter_train_cv_splits(prepared, config: ExperimentConfig) -> tuple[str, int, Iterator[tuple[int, pd.Index, pd.Index]]]:
    index = prepared.X_train.index
    if config.validation_type not in {"group_kfold", "kfold"}:
        raise ValueError("CONFIG['validation_type'] must be either 'group_kfold' or 'kfold'.")

    if config.validation_type == "group_kfold":
        groups = resolve_cv_groups(prepared, config)
        assert groups is not None
        unique_groups = int(groups.nunique(dropna=False))
        n_splits = min(config.n_splits, unique_groups)
        if n_splits < 2:
            raise ValueError("At least two unique groups are required for group_kfold validation.")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(prepared.X_train, prepared.y, groups=groups)
    else:
        n_splits = min(config.n_splits, len(index))
        if n_splits < 2:
            raise ValueError("At least two folds are required for kfold validation.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        split_iter = splitter.split(prepared.X_train, prepared.y)

    def generate() -> Iterator[tuple[int, pd.Index, pd.Index]]:
        for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
            yield fold_idx, index[train_idx], index[valid_idx]

    return config.validation_type, n_splits, generate()


def main() -> None:
    ensure_runtime_directories()
    ensure_local_submission_dir()
    prepared = load_prepared_data()
    config = load_config()
    improvement_notes = os.environ.get("TRAIN_IMPROVEMENT_NOTES", "-").strip() or "-"
    cv_strategy, actual_n_splits, split_iterator = iter_train_cv_splits(prepared, config)
    del split_iterator
    (
        primary_oof,
        primary_test_pred,
        primary_fold_rmse_scores,
        primary_fold_mae_scores,
        primary_fold_importances,
        primary_best_iterations,
        feature_columns,
        excluded_columns,
    ) = train_single_model(prepared, config)

    oof_predictions = primary_oof
    test_pred = primary_test_pred
    fold_rmse_scores = primary_fold_rmse_scores
    fold_mae_scores = primary_fold_mae_scores
    fold_importances = primary_fold_importances
    best_iterations = primary_best_iterations

    blend_notes = ""
    if config.blend_secondary_model:
        secondary_config = make_secondary_config(config)
        print(
            "Training secondary blend model with "
            f"use_layout_id={secondary_config.use_layout_id} "
            f"target_weight_mode={secondary_config.target_weight_mode} "
            f"seed={secondary_config.seed}"
        )
        (
            secondary_oof,
            secondary_test_pred,
            _secondary_fold_rmse_scores,
            _secondary_fold_mae_scores,
            secondary_fold_importances,
            secondary_best_iterations,
            _secondary_feature_columns,
            _secondary_excluded_columns,
        ) = train_single_model(prepared, secondary_config)
        secondary_weight = float(np.clip(config.secondary_weight, 0.0, 1.0))
        primary_weight = 1.0 - secondary_weight
        oof_predictions = (primary_weight * primary_oof) + (secondary_weight * secondary_oof)
        test_pred = (primary_weight * primary_test_pred) + (secondary_weight * secondary_test_pred)
        fold_importances.extend(secondary_fold_importances)
        best_iterations.extend(secondary_best_iterations)
        fold_rmse_scores = []
        fold_mae_scores = []
        _, _, split_iterator = iter_train_cv_splits(prepared, config)
        for _fold_idx, _train_idx, valid_idx in split_iterator:
            fold_pred = oof_predictions.loc[valid_idx]
            y_fold_valid = prepared.y.loc[valid_idx]
            fold_rmse_scores.append(evaluate_rmse(y_fold_valid, fold_pred))
            fold_mae_scores.append(evaluate_mae(y_fold_valid, fold_pred))
        blend_notes = (
            f" | blend_secondary_model weight={secondary_weight:.2f} "
            f"secondary_use_layout_id={secondary_config.use_layout_id} "
            f"secondary_target_weight_mode={secondary_config.target_weight_mode}"
        )

    oof_rmse = evaluate_rmse(prepared.y.loc[oof_predictions.index], oof_predictions.loc[prepared.y.index])
    oof_mae = evaluate_mae(prepared.y.loc[oof_predictions.index], oof_predictions.loc[prepared.y.index])
    final_n_estimators = max(1, int(round(float(np.mean(best_iterations)))))
    print(
        f"CV Summary: MAE={np.mean(fold_mae_scores):.6f}±{np.std(fold_mae_scores):.6f} "
        f"OOF_MAE={oof_mae:.6f} "
        f"RMSE={np.mean(fold_rmse_scores):.6f}±{np.std(fold_rmse_scores):.6f} "
        f"OOF_RMSE={oof_rmse:.6f} "
        f"FINAL_N_ESTIMATORS={final_n_estimators}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = create_experiment_dir(config, timestamp)
    submission = prepared.submission_df.copy()
    submission[prepared.target_column] = test_pred
    submission_path = SUBMISSIONS_DIR / f"submission_{config.experiment_name}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")
    submission_number, archived_submission_path = archive_submission_locally(submission_path)
    print(f"Archived submission copy: {archived_submission_path}")

    oof_path = save_oof_predictions(prepared, oof_predictions, experiment_dir)
    importance_path = save_feature_importance(fold_importances, experiment_dir)

    results_row = {
        "timestamp": timestamp,
        "experiment_name": config.experiment_name,
        "target_column": prepared.target_column,
        "cv_strategy": cv_strategy,
        "group_column": config.group_column if cv_strategy == "group_kfold" else "-",
        "n_splits": actual_n_splits,
        "n_train_rows": len(oof_predictions),
        "n_test_rows": len(prepared.X_test),
        "n_features": len(feature_columns),
        "excluded_features": "|".join(excluded_columns),
        "metric_name": "mae",
        "mae_mean": float(np.mean(fold_mae_scores)),
        "mae_std": float(np.std(fold_mae_scores)),
        "rmse_mean": float(np.mean(fold_rmse_scores)),
        "rmse_std": float(np.std(fold_rmse_scores)),
        "oof_mae": oof_mae,
        "oof_rmse": oof_rmse,
        "best_iteration_mean": float(np.mean(best_iterations)),
        "best_iteration_std": float(np.std(best_iterations)),
        "final_n_estimators": final_n_estimators,
        "model_name": config.experiment_name,
        "improvement_notes": improvement_notes + blend_notes,
        "config_json": json.dumps(asdict(config), sort_keys=True),
        "submission_path": str(submission_path.relative_to(Path.cwd())),
        "submission_alias": format_submission_alias(submission_number),
        "submission_local_path": str(archived_submission_path.relative_to(Path.cwd())),
        "experiment_dir": str(experiment_dir.relative_to(Path.cwd())),
    }
    summary_path = save_experiment_summary(
        prepared,
        config,
        fold_rmse_scores,
        fold_mae_scores,
        best_iterations,
        results_row,
        experiment_dir,
    )
    results_row["summary_path"] = str(summary_path.relative_to(Path.cwd()))
    results_path = append_results_log(results_row)
    records = update_readme_experiment_log()

    print(f"Updated log: {results_path}")
    print(f"Updated README experiment log with {len(records)} records")
    print(f"Saved OOF predictions: {oof_path}")
    if importance_path is not None:
        print(f"Saved feature importance: {importance_path}")
    print(f"Saved experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
