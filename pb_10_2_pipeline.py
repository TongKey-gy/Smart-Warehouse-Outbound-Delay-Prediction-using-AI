from __future__ import annotations

import copy
import json
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
except Exception as exc:
    raise ImportError(
        "lightgbm, xgboost, catboost 가 필요합니다. "
        "필요하면 `pip install lightgbm xgboost catboost` 후 다시 실행하세요."
    ) from exc

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "open"

CONFIG = {
    "project": {"name": "smart_warehouse_delay_prediction_ipynb", "seed": 42},
    "paths": {
        "train_csv": DATA_DIR / "train.csv",
        "test_csv": DATA_DIR / "test.csv",
        "train_layout_csv": DATA_DIR / "layout_info.csv",
        "test_layout_csv": DATA_DIR / "layout_info.csv",
        "sample_submission_csv": DATA_DIR / "sample_submission.csv",
        "model_path": ROOT_DIR / "outputs" / "pb_10_2_pipeline" / "model_bundle.joblib",
        "cv_output_json": ROOT_DIR / "outputs" / "pb_10_2_pipeline" / "cv_results.json",
        "prediction_output_csv": ROOT_DIR / "outputs" / "pb_10_2_pipeline" / "submission.csv",
    },
    "data": {
        "id_col": "ID",
        "layout_col": "layout_id",
        "scenario_col": "scenario_id",
        "target_col": "avg_delay_minutes_next_30m",
    },
    "features": {
        "use_layout": True,
        "use_layout_type_onehot": True,
        "use_missing_indicators": True,
        "sequence_cols": [
            "order_inflow_15m", "unique_sku_15m", "robot_active", "robot_idle",
            "robot_charging", "battery_mean", "battery_std", "low_battery_ratio",
            "charge_queue_length", "avg_charge_wait", "congestion_score",
            "max_zone_density", "blocked_path_15m", "near_collision_15m",
            "fault_count_15m", "avg_recovery_time", "task_reassign_15m",
            "replenishment_overlap", "pack_utilization", "loading_dock_util",
            "staging_area_util", "label_print_queue",
        ],
        "baseline_expanding_cols": [
            "order_inflow_15m", "unique_sku_15m", "avg_items_per_order",
            "urgent_order_ratio", "heavy_item_ratio", "cold_chain_ratio",
            "sku_concentration", "bulk_order_ratio", "avg_trip_distance",
            "network_latency_ms", "air_quality_idx", "barcode_read_success_rate",
            "hvac_power_kw", "ambient_noise_db", "inventory_turnover_rate",
            "safety_score_monthly", "scanner_error_rate", "wms_response_time_ms",
            "backorder_ratio",
        ],
    },
    "training": {
        "clip_prediction_min": 0.0,
        "early_stopping": {"enabled": True, "rounds": 80},
        "sample_weight": {
            "enabled": True,
            "q90_bonus": 0.15,
            "q95_bonus": 0.30,
            "q99_bonus": 0.60,
            "late_time_alpha": 0.08,
        },
        "cv": {
            "run": True,
            "seen_layout_folds": 5,
            "unseen_layout_folds": 5,
            "weight_seen": 0.6,
            "weight_unseen": 0.4,
            "auto_weight_from_hybrid_cv": True,
        },
        "models": [
            {
                "name": "lgbm_l1_raw",
                "family": "lightgbm",
                "enabled": True,
                "seeds": [42],
                "weight": 0.32,
                "target_transform": "none",
                "params": {
                    "objective": "mae", "n_estimators": 700, "learning_rate": 0.03,
                    "num_leaves": 96, "max_depth": -1, "min_child_samples": 80,
                    "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.85,
                    "reg_alpha": 0.1, "reg_lambda": 1.5, "verbosity": -1,
                },
            },
            {
                "name": "lgbm_huber_log",
                "family": "lightgbm",
                "enabled": True,
                "seeds": [42],
                "weight": 0.24,
                "target_transform": "log1p",
                "params": {
                    "objective": "huber", "alpha": 0.9, "n_estimators": 700,
                    "learning_rate": 0.03, "num_leaves": 128, "max_depth": -1,
                    "min_child_samples": 60, "subsample": 0.9, "subsample_freq": 1,
                    "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 1.0,
                    "verbosity": -1,
                },
            },
            {
                "name": "xgb_abs_raw",
                "family": "xgboost",
                "enabled": True,
                "seeds": [42],
                "weight": 0.22,
                "target_transform": "none",
                "params": {
                    "objective": "reg:absoluteerror", "n_estimators": 700,
                    "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 6.0,
                    "subsample": 0.9, "colsample_bytree": 0.85, "reg_lambda": 1.5,
                    "reg_alpha": 0.05, "tree_method": "hist", "verbosity": 0,
                },
            },
            {
                "name": "catboost_mae_log",
                "family": "catboost",
                "enabled": True,
                "seeds": [42],
                "weight": 0.22,
                "target_transform": "log1p",
                "params": {
                    "loss_function": "MAE", "iterations": 900, "learning_rate": 0.03,
                    "depth": 8, "l2_leaf_reg": 5.0, "bootstrap_type": "Bernoulli",
                    "subsample": 0.9, "allow_writing_files": False, "verbose": False,
                },
            },
        ],
    },
}

FAST_DEV_SETTINGS = {"enabled": False, "max_scenarios": 1500}
pd.set_option("display.max_columns", 200)

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "smart_warehouse") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def safe_divide(numerator: Any, denominator: Any) -> pd.Series:
    num = pd.Series(numerator, copy=False, dtype="float64")
    den = pd.Series(denominator, copy=False, dtype="float64").replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def hybrid_score(seen_mae: float, unseen_mae: float, seen_weight: float = 0.6, unseen_weight: float = 0.4) -> float:
    return float(seen_weight * seen_mae + unseen_weight * unseen_mae)


def normalize_weights(weights: Sequence[float]) -> list[float]:
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return [1.0 / len(w)] * len(w)
    return (w / total).tolist()


def save_joblib(obj: Any, path: str | Path, compress: int = 3) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path, compress=compress)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def transform_target(y: Any, name: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if name == "none":
        return y
    if name == "log1p":
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    raise ValueError(name)


def inverse_transform_target(y: Any, name: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if name == "none":
        return y
    if name == "log1p":
        return np.expm1(y)
    raise ValueError(name)


def build_sample_weight(y: pd.Series, time_idx: pd.Series | None, cfg: Mapping[str, Any]) -> np.ndarray:
    sw_cfg = cfg.get("sample_weight", {})
    if not sw_cfg.get("enabled", False):
        return np.ones(len(y), dtype=np.float32)
    y = pd.Series(y).astype(float)
    w = np.ones(len(y), dtype=np.float32)
    q90 = float(np.nanquantile(y, 0.90))
    q95 = float(np.nanquantile(y, 0.95))
    q99 = float(np.nanquantile(y, 0.99))
    w += sw_cfg.get("q90_bonus", 0.0) * (y >= q90).astype(np.float32)
    w += sw_cfg.get("q95_bonus", 0.0) * (y >= q95).astype(np.float32)
    w += sw_cfg.get("q99_bonus", 0.0) * (y >= q99).astype(np.float32)
    if time_idx is not None:
        time_idx = pd.Series(time_idx).astype(float)
        max_time = max(float(time_idx.max()), 1.0)
        w += sw_cfg.get("late_time_alpha", 0.0) * (time_idx / max_time).to_numpy(dtype=np.float32)
    return w.astype(np.float32)


def make_unseen_layout_folds(df: pd.DataFrame, layout_col: str, target_col: str, n_splits: int, seed: int):
    stats = df.groupby(layout_col)[target_col].mean().rename("layout_target_mean").reset_index()
    try:
        stats["bin"] = pd.qcut(
            stats["layout_target_mean"].rank(method="first"),
            q=min(n_splits, len(stats)),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        stats["bin"] = 0
    rng = np.random.default_rng(seed)
    assign = {}
    for _, grp in stats.groupby("bin"):
        layouts = grp[layout_col].tolist()
        rng.shuffle(layouts)
        for idx, layout_id in enumerate(layouts):
            assign[layout_id] = idx % n_splits
    folds = []
    for fold_id in range(n_splits):
        val_mask = df[layout_col].map(assign).eq(fold_id).to_numpy()
        folds.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return folds


def make_seen_layout_folds(train_df: pd.DataFrame, test_df: pd.DataFrame, layout_col: str, scenario_col: str, n_splits: int, seed: int):
    shared_layouts = sorted(set(train_df[layout_col].unique()) & set(test_df[layout_col].unique()))
    shared = train_df[train_df[layout_col].isin(shared_layouts)][[layout_col, scenario_col]].drop_duplicates()
    rng = np.random.default_rng(seed)
    fold_by_scenario = {}
    for _, grp in shared.groupby(layout_col):
        scenarios = grp[scenario_col].tolist()
        rng.shuffle(scenarios)
        for idx, scenario_id in enumerate(scenarios):
            fold_by_scenario[scenario_id] = idx % n_splits
    folds = []
    for fold_id in range(n_splits):
        val_mask = train_df[scenario_col].map(fold_by_scenario).eq(fold_id).fillna(False).to_numpy()
        folds.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return folds


def sample_scenarios_for_fast_run(train_df: pd.DataFrame, max_scenarios: int, scenario_col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenarios = train_df[scenario_col].drop_duplicates().tolist()
    rng.shuffle(scenarios)
    keep = set(scenarios[:max_scenarios])
    return train_df[train_df[scenario_col].isin(keep)].reset_index(drop=True)

SEQ_COLS = [
    "order_inflow_15m", "unique_sku_15m", "robot_active", "robot_idle", "robot_charging",
    "battery_mean", "battery_std", "low_battery_ratio", "charge_queue_length",
    "avg_charge_wait", "congestion_score", "max_zone_density", "blocked_path_15m",
    "near_collision_15m", "fault_count_15m", "avg_recovery_time", "task_reassign_15m",
    "replenishment_overlap", "pack_utilization", "loading_dock_util",
    "staging_area_util", "label_print_queue",
]

BASELINE_COLS = [
    "order_inflow_15m", "unique_sku_15m", "avg_items_per_order", "urgent_order_ratio",
    "heavy_item_ratio", "cold_chain_ratio", "sku_concentration", "bulk_order_ratio",
    "avg_trip_distance", "network_latency_ms", "air_quality_idx",
    "barcode_read_success_rate", "hvac_power_kw", "ambient_noise_db",
    "inventory_turnover_rate", "safety_score_monthly", "scanner_error_rate",
    "wms_response_time_ms", "backorder_ratio",
]

DYN_COLS = [
    "battery_mean", "low_battery_ratio", "robot_charging", "charge_queue_length",
    "avg_charge_wait", "congestion_score", "max_zone_density", "blocked_path_15m",
    "near_collision_15m", "fault_count_15m", "avg_recovery_time", "task_reassign_15m",
    "replenishment_overlap", "pack_utilization", "loading_dock_util", "staging_area_util",
]


@dataclass
class FeatureBuilder:
    config: Mapping[str, Any]
    layout_type_categories_: list[str] = field(default_factory=list)
    feature_columns_: list[str] = field(default_factory=list)

    @property
    def id_col(self): return self.config["data"]["id_col"]

    @property
    def layout_col(self): return self.config["data"]["layout_col"]

    @property
    def scenario_col(self): return self.config["data"]["scenario_col"]

    @property
    def target_col(self): return self.config["data"]["target_col"]

    def fit(self, train_df: pd.DataFrame, layout_df: pd.DataFrame):
        self.layout_type_categories_ = sorted(layout_df["layout_type"].dropna().astype(str).unique().tolist())
        feat = self._build(train_df, layout_df)
        self.feature_columns_ = feat.columns.tolist()
        return self

    def fit_transform(self, train_df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df, layout_df)
        return self.transform(train_df, layout_df)

    def transform(self, df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
        feat = self._build(df, layout_df)
        if self.feature_columns_:
            return feat.reindex(columns=self.feature_columns_, fill_value=0.0)
        return feat

    def _build(self, df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["__row_order__"] = np.arange(len(df))
        df["__id_num__"] = df[self.id_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
        df = df.sort_values([self.scenario_col, "__id_num__", "__row_order__"]).reset_index(drop=True)

        if self.config["features"].get("use_layout", True):
            df = df.merge(layout_df.copy(), on=self.layout_col, how="left", validate="m:1")

        grp_key = df[self.scenario_col]
        grp = df.groupby(self.scenario_col, sort=False)

        exclude_raw = {self.target_col, self.id_col, self.scenario_col, "__row_order__", "__id_num__"}
        orig_cols = [c for c in df.columns if c not in exclude_raw]
        orig_num = [c for c in orig_cols if pd.api.types.is_numeric_dtype(df[c])]

        df["time_idx"] = grp.cumcount().astype(np.int16)
        df["time_frac"] = (df["time_idx"] / 24.0).astype(np.float32)
        df["time_remaining"] = (24 - df["time_idx"]).astype(np.int16)
        df["time_idx_sq"] = (df["time_frac"] ** 2).astype(np.float32)
        df["is_early_phase"] = (df["time_idx"] <= 5).astype(np.int8)
        df["is_mid_phase"] = ((df["time_idx"] >= 6) & (df["time_idx"] <= 15)).astype(np.int8)
        df["is_late_phase"] = (df["time_idx"] >= 16).astype(np.int8)

        if self.config["features"].get("use_missing_indicators", True):
            for col in orig_num:
                if df[col].isna().any():
                    df[f"{col}__is_missing"] = df[col].isna().astype(np.int8)
            df["n_missing_all_raw"] = df[orig_num].isna().sum(axis=1).astype(np.int16)
            dyn = [c for c in DYN_COLS if c in df.columns]
            df["n_missing_dynamic_raw"] = df[dyn].isna().sum(axis=1).astype(np.int16) if dyn else 0
            df["missing_ratio_all_raw"] = (df["n_missing_all_raw"] / max(len(orig_num), 1)).astype(np.float32)

        def add_ratio(name, a, b):
            if a in df.columns and b in df.columns:
                df[name] = safe_divide(df[a], df[b])

        if {"floor_area_sqm", "ceiling_height_m"}.issubset(df.columns):
            df["warehouse_volume_proxy"] = safe_divide(df["floor_area_sqm"] * df["ceiling_height_m"], 1.0)
        if {"intersection_count", "floor_area_sqm"}.issubset(df.columns):
            df["intersection_density"] = safe_divide(df["intersection_count"], df["floor_area_sqm"])
        if {"pack_station_count", "floor_area_sqm"}.issubset(df.columns):
            df["pack_station_density"] = safe_divide(df["pack_station_count"], df["floor_area_sqm"])
        if {"charger_count", "floor_area_sqm"}.issubset(df.columns):
            df["charger_density"] = safe_divide(df["charger_count"], df["floor_area_sqm"])
        if {"robot_total", "floor_area_sqm"}.issubset(df.columns):
            df["robot_density_layout"] = safe_divide(df["robot_total"], df["floor_area_sqm"])
        if {"intersection_count", "aisle_width_avg"}.issubset(df.columns):
            df["movement_friction_layout"] = safe_divide(df["intersection_count"], df["aisle_width_avg"])
        if {"layout_compactness", "zone_dispersion"}.issubset(df.columns):
            df["layout_compactness_x_dispersion"] = (df["layout_compactness"] * df["zone_dispersion"]).astype(float)
        if {"one_way_ratio", "intersection_count", "aisle_width_avg"}.issubset(df.columns):
            df["one_way_friction"] = (df["one_way_ratio"] * safe_divide(df["intersection_count"], df["aisle_width_avg"])).astype(float)
        if {"fire_sprinkler_count", "floor_area_sqm"}.issubset(df.columns):
            df["sprinkler_density"] = safe_divide(df["fire_sprinkler_count"], df["floor_area_sqm"])
        if {"emergency_exit_count", "floor_area_sqm"}.issubset(df.columns):
            df["exit_density"] = safe_divide(df["emergency_exit_count"], df["floor_area_sqm"])

        if {"robot_active", "robot_idle", "robot_charging"}.issubset(df.columns):
            df["robot_total_state"] = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
            df["robot_total_gap"] = df["robot_total_state"] - df["robot_total"]
            df["robot_active_share_state"] = safe_divide(df["robot_active"], df["robot_total_state"])
            df["robot_idle_share_state"] = safe_divide(df["robot_idle"], df["robot_total_state"])
            df["robot_charging_share_state"] = safe_divide(df["robot_charging"], df["robot_total_state"])
            df["charging_to_active_ratio"] = safe_divide(df["robot_charging"], df["robot_active"])
            df["idle_to_active_ratio"] = safe_divide(df["robot_idle"], df["robot_active"])

        add_ratio("inflow_per_robot", "order_inflow_15m", "robot_total")
        add_ratio("inflow_per_pack_station", "order_inflow_15m", "pack_station_count")
        add_ratio("unique_sku_per_robot", "unique_sku_15m", "robot_total")
        add_ratio("unique_sku_per_pack_station", "unique_sku_15m", "pack_station_count")
        add_ratio("charge_queue_per_charger", "charge_queue_length", "charger_count")
        add_ratio("charging_per_charger", "robot_charging", "charger_count")
        add_ratio("congestion_per_width", "congestion_score", "aisle_width_avg")
        add_ratio("zone_density_per_width", "max_zone_density", "aisle_width_avg")
        add_ratio("order_per_sqm", "order_inflow_15m", "floor_area_sqm")
        add_ratio("robot_per_sqm", "robot_total", "floor_area_sqm")
        add_ratio("dock_pressure", "order_inflow_15m", "staff_on_floor")
        add_ratio("label_queue_per_pack_station", "label_print_queue", "pack_station_count")
        add_ratio("robot_active_per_intersection", "robot_active", "intersection_count")
        add_ratio("congestion_per_active", "congestion_score", "robot_active")
        add_ratio("density_per_active", "max_zone_density", "robot_active")
        add_ratio("fault_per_active", "fault_count_15m", "robot_active")
        add_ratio("collision_per_active", "near_collision_15m", "robot_active")
        add_ratio("blocked_per_active", "blocked_path_15m", "robot_active")
        add_ratio("inflow_per_charger", "order_inflow_15m", "charger_count")
        add_ratio("pack_station_per_robot", "pack_station_count", "robot_total")
        add_ratio("charger_per_robot", "charger_count", "robot_total")
        add_ratio("inflow_per_aisle_width", "order_inflow_15m", "aisle_width_avg")

        if {"robot_charging", "charge_queue_length", "charger_count"}.issubset(df.columns):
            df["charge_pressure"] = safe_divide(df["robot_charging"] + df["charge_queue_length"], df["charger_count"])
        if {"order_inflow_15m", "avg_package_weight_kg"}.issubset(df.columns):
            df["demand_mass"] = (df["order_inflow_15m"] * df["avg_package_weight_kg"]).astype(float)
            if "robot_total" in df.columns:
                df["demand_mass_per_robot"] = safe_divide(df["demand_mass"], df["robot_total"])
            if "pack_station_count" in df.columns:
                df["demand_mass_per_pack_station"] = safe_divide(df["demand_mass"], df["pack_station_count"])
        if {"order_inflow_15m", "avg_trip_distance"}.issubset(df.columns):
            df["trip_load"] = (df["order_inflow_15m"] * df["avg_trip_distance"]).astype(float)
            if "robot_total" in df.columns:
                df["trip_load_per_robot"] = safe_divide(df["trip_load"], df["robot_total"])
        if {"order_inflow_15m", "unique_sku_15m"}.issubset(df.columns):
            df["complexity_load"] = (df["order_inflow_15m"] * df["unique_sku_15m"]).astype(float)
            if "pack_station_count" in df.columns:
                df["complexity_load_per_pack"] = safe_divide(df["complexity_load"], df["pack_station_count"])
        if {"congestion_score", "low_battery_ratio"}.issubset(df.columns):
            df["congestion_x_lowbat"] = (df["congestion_score"] * df["low_battery_ratio"]).astype(float)
        if {"low_battery_ratio", "robot_active"}.issubset(df.columns):
            df["battery_pressure"] = (df["low_battery_ratio"] * df["robot_active"]).astype(float)
        if {"charge_queue_length", "avg_charge_wait"}.issubset(df.columns):
            df["queue_wait_pressure"] = (df["charge_queue_length"] * df["avg_charge_wait"]).astype(float)
        if {"loading_dock_util", "pack_utilization"}.issubset(df.columns):
            df["dock_pack_pressure"] = (df["loading_dock_util"] * df["pack_utilization"]).astype(float)
        if {"staging_area_util", "pack_utilization"}.issubset(df.columns):
            df["staging_pack_pressure"] = (df["staging_area_util"] * df["pack_utilization"]).astype(float)
        if {"avg_recovery_time", "fault_count_15m"}.issubset(df.columns):
            df["recovery_x_fault"] = (df["avg_recovery_time"] * df["fault_count_15m"]).astype(float)
        if {"near_collision_15m", "blocked_path_15m"}.issubset(df.columns):
            df["collision_x_blocked"] = (df["near_collision_15m"] * df["blocked_path_15m"]).astype(float)
        if {"charge_pressure", "congestion_score"}.issubset(df.columns):
            df["charge_pressure_x_congestion"] = (df["charge_pressure"] * df["congestion_score"]).astype(float)
        if {"inflow_per_pack_station", "charge_pressure"}.issubset(df.columns):
            df["inflow_pack_x_charge_pressure"] = (df["inflow_per_pack_station"] * df["charge_pressure"]).astype(float)
        if "battery_mean" in df.columns:
            df["battery_mean_below_44"] = np.clip(44.0 - df["battery_mean"], a_min=0.0, a_max=None).astype(float)
        if "charge_pressure" in df.columns:
            df["charge_pressure_above_1_36"] = np.clip(df["charge_pressure"] - 1.36, a_min=0.0, a_max=None).astype(float)
        if "pack_utilization" in df.columns:
            df["pack_utilization_sq"] = df["pack_utilization"].astype(float) ** 2
        if "loading_dock_util" in df.columns:
            df["loading_dock_util_sq"] = df["loading_dock_util"].astype(float) ** 2
        if "staging_area_util" in df.columns:
            df["staging_area_util_sq"] = df["staging_area_util"].astype(float) ** 2

        def add_onset(value_col: str, prefix: str):
            if value_col not in df.columns:
                return
            positive = df[value_col].fillna(0).gt(0).astype(bool)
            t = df["time_idx"].where(positive)
            first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
            prev = positive.groupby(grp_key).shift(1, fill_value=False).astype(bool)
            df[f"{prefix}_ever_started_so_far"] = first.notna().astype(np.int8)
            df[f"{prefix}_start_idx_so_far"] = first.fillna(-1).astype(np.int16)
            df[f"{prefix}_started_now"] = (positive & ~prev).astype(np.int8)
            df[f"{prefix}_started_early_so_far"] = (first <= 5).fillna(False).astype(np.int8)
            df[f"{prefix}_steps_since_start"] = np.where(first.notna(), (df["time_idx"] - first).astype(float), -1.0).astype(np.float32)

        add_onset("robot_charging", "charging")
        add_onset("charge_queue_length", "queue")

        for col in self.config["features"].get("sequence_cols") or SEQ_COLS:
            if col not in df.columns:
                continue
            lag1 = grp[col].shift(1)
            lag2 = grp[col].shift(2)
            df[f"{col}__lag1"] = lag1
            df[f"{col}__lag2"] = lag2
            df[f"{col}__diff1"] = df[col] - lag1
            lg = lag1.groupby(grp_key)
            roll_mean = lg.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            roll_max = lg.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"{col}__rollmean3_prev"] = roll_mean
            df[f"{col}__rollmax3_prev"] = roll_max
            df[f"{col}__dev_rollmean3_prev"] = df[col] - roll_mean

        for col in self.config["features"].get("baseline_expanding_cols") or BASELINE_COLS:
            if col not in df.columns:
                continue
            prev = grp[col].shift(1)
            exp_mean = prev.groupby(grp_key).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{col}__expmean_prev"] = exp_mean
            df[f"{col}__delta_expmean_prev"] = df[col] - exp_mean

        if self.config["features"].get("use_layout_type_onehot", True) and "layout_type" in df.columns:
            cats = self.layout_type_categories_ or sorted(df["layout_type"].dropna().astype(str).unique().tolist())
            dummies = pd.get_dummies(
                pd.Categorical(df["layout_type"].astype(str), categories=cats),
                prefix="layout_type",
                dummy_na=False,
            ).astype(np.int8)
            df = pd.concat([df, dummies], axis=1)

        exclude = {
            self.id_col, self.scenario_col, self.layout_col, self.target_col,
            "layout_type", "__row_order__", "__id_num__",
        }
        feat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_object_dtype(df[c])]
        feat = df[feat_cols + ["__row_order__"]].sort_values("__row_order__").drop(columns="__row_order__").reset_index(drop=True)
        for col in feat.columns:
            if pd.api.types.is_bool_dtype(feat[col]):
                feat[col] = feat[col].astype(np.int8)
            else:
                feat[col] = feat[col].astype(np.float32)
        return feat

@dataclass
class TrainedModelEntry:
    name: str
    family: str
    estimator: Any
    target_transform: str
    weight: float
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame, clip_prediction_min: float = 0.0) -> np.ndarray:
        pred = self.estimator.predict(X)
        pred = inverse_transform_target(pred, self.target_transform)
        pred = np.asarray(pred, dtype=float)
        if clip_prediction_min is not None:
            pred = np.clip(pred, a_min=clip_prediction_min, a_max=None)
        return pred


@dataclass
class EnsembleBundle:
    feature_builder: FeatureBuilder
    models: list[TrainedModelEntry]
    clip_prediction_min: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Bundle does not contain trained models.")
        weights = normalize_weights([m.weight for m in self.models])
        preds = [m.predict(X, clip_prediction_min=self.clip_prediction_min) for m in self.models]
        return np.average(np.vstack(preds), axis=0, weights=np.asarray(weights, dtype=float))

    def predict_from_dataframe(self, df: pd.DataFrame, layout_df: pd.DataFrame) -> np.ndarray:
        X = self.feature_builder.transform(df, layout_df)
        return self.predict_features(X)


def flatten_model_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for spec in config["training"]["models"]:
        if not spec.get("enabled", True):
            continue
        seeds = spec.get("seeds", [config["project"]["seed"]])
        base_weight = float(spec.get("weight", 1.0))
        per_seed_weight = base_weight / max(len(seeds), 1)
        for seed in seeds:
            specs.append(
                {
                    "name": f"{spec['name']}__seed{seed}",
                    "base_name": spec["name"],
                    "family": spec["family"],
                    "seed": int(seed),
                    "target_transform": spec.get("target_transform", "none"),
                    "weight": per_seed_weight,
                    "params": dict(spec.get("params", {})),
                }
            )
    return specs


def build_estimator(family: str, params: Mapping[str, Any], seed: int):
    params = dict(params)
    family = family.lower()
    if family == "lightgbm":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        return LGBMRegressor(**params)
    if family == "xgboost":
        params.setdefault("random_state", seed)
        params.setdefault("tree_method", "hist")
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", 0)
        return XGBRegressor(**params)
    if family == "catboost":
        params.setdefault("random_seed", seed)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostRegressor(**params)
    raise ValueError(f"Unsupported family: {family}")


def fit_model_entry(
    spec: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train,
    sample_weight=None,
    X_valid: pd.DataFrame | None = None,
    y_valid=None,
    early_stopping_cfg: Mapping[str, Any] | None = None,
) -> TrainedModelEntry:
    target_transform = spec.get("target_transform", "none")
    y_tr = transform_target(y_train, target_transform)
    y_va = transform_target(y_valid, target_transform) if y_valid is not None else None
    estimator = build_estimator(spec["family"], spec.get("params", {}), spec["seed"])
    family = spec["family"].lower()
    fit_kwargs = {}
    early_stopping_cfg = early_stopping_cfg or {"enabled": False, "rounds": 0}

    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    if family == "lightgbm":
        callbacks = []
        if X_valid is not None and y_va is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_va)]
            fit_kwargs["eval_metric"] = "l1"
            if early_stopping_cfg.get("enabled", False):
                from lightgbm import early_stopping as lgb_early_stopping
                callbacks.append(lgb_early_stopping(stopping_rounds=int(early_stopping_cfg["rounds"]), verbose=False))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        estimator.fit(X_train, y_tr, **fit_kwargs)

    elif family == "xgboost":
        callbacks = []
        if X_valid is not None and y_va is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_va)]
            fit_kwargs["verbose"] = False
            if early_stopping_cfg.get("enabled", False):
                from xgboost.callback import EarlyStopping
                callbacks.append(EarlyStopping(rounds=int(early_stopping_cfg["rounds"]), save_best=True, maximize=False))
        if callbacks:
            estimator.set_params(callbacks=callbacks)
        estimator.fit(X_train, y_tr, **fit_kwargs)
        if callbacks:
            estimator.set_params(callbacks=None)

    elif family == "catboost":
        fit_kwargs["verbose"] = False
        if X_valid is not None and y_va is not None:
            fit_kwargs["eval_set"] = (X_valid, y_va)
            fit_kwargs["use_best_model"] = True
            if early_stopping_cfg.get("enabled", False):
                fit_kwargs["early_stopping_rounds"] = int(early_stopping_cfg["rounds"])
        estimator.fit(X_train, y_tr, **fit_kwargs)
    else:
        raise ValueError(family)

    return TrainedModelEntry(
        name=spec["name"],
        family=spec["family"],
        estimator=estimator,
        target_transform=target_transform,
        weight=float(spec.get("weight", 1.0)),
        seed=int(spec["seed"]),
        params=dict(spec.get("params", {})),
    )


def get_best_iteration_value(estimator, spec) -> int | None:
    family = str(spec["family"]).lower()
    if family == "lightgbm":
        value = getattr(estimator, "best_iteration_", None)
    elif family == "xgboost":
        value = getattr(estimator, "best_iteration", None)
    elif family == "catboost":
        try:
            value = estimator.get_best_iteration()
        except Exception:
            value = None
    else:
        value = None
    if value is None or int(value) < 0:
        return None
    return int(value) + 1


def shrink_model_specs_for_fast_run(config: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    for spec in config["training"]["models"]:
        family = spec["family"].lower()
        if family in {"lightgbm", "xgboost"}:
            spec["params"]["n_estimators"] = min(int(spec["params"].get("n_estimators", 700)), 160)
        elif family == "catboost":
            spec["params"]["iterations"] = min(int(spec["params"].get("iterations", 900)), 200)
    config["training"]["cv"]["run"] = False
    return config


def apply_recommended_iterations(model_specs, cv_results, logger):
    if not cv_results or not cv_results.get("recommended_iterations"):
        return model_specs
    adjusted_specs = copy.deepcopy(model_specs)
    for spec in adjusted_specs:
        rec = cv_results["recommended_iterations"].get(spec["name"])
        if not rec:
            continue
        family = str(spec["family"]).lower()
        if family in {"lightgbm", "xgboost"}:
            original = int(spec["params"].get("n_estimators", rec))
            spec["params"]["n_estimators"] = max(20, min(original, int(rec)))
            logger.info(f"Final {spec['name']} iterations set to {spec['params']['n_estimators']}.")
        elif family == "catboost":
            original = int(spec["params"].get("iterations", rec))
            spec["params"]["iterations"] = max(20, min(original, int(rec)))
            logger.info(f"Final {spec['name']} iterations set to {spec['params']['iterations']}.")
    return adjusted_specs


def run_cv(train_df, test_df, X, y, model_specs, config, logger):
    layout_col = config["data"]["layout_col"]
    scenario_col = config["data"]["scenario_col"]
    target_col = config["data"]["target_col"]
    seed = int(config["project"]["seed"])
    clip = float(config["training"].get("clip_prediction_min", 0.0))
    cv_cfg = config["training"]["cv"]
    early_stopping_cfg = dict(config["training"].get("early_stopping", {}))

    splits = {
        "seen_layout": make_seen_layout_folds(
            train_df, test_df, layout_col, scenario_col, int(cv_cfg["seen_layout_folds"]), seed
        ),
        "unseen_layout": make_unseen_layout_folds(
            train_df, layout_col, target_col, int(cv_cfg["unseen_layout_folds"]), seed
        ),
    }

    results = {"splits": {}, "model_hybrid_scores": {}, "auto_weights": {}, "recommended_iterations": {}}

    for split_name, split_list in splits.items():
        pred_by_model = {spec["name"]: np.full(len(train_df), np.nan, dtype=float) for spec in model_specs}
        val_mask = np.zeros(len(train_df), dtype=bool)
        split_res = {"models": {}, "ensemble": {}}

        for fold_id, (tr_idx, va_idx) in enumerate(split_list, start=1):
            logger.info(f"{split_name} / fold {fold_id} / {len(split_list)}")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            val_mask[va_idx] = True
            time_idx_tr = X_tr["time_idx"] if "time_idx" in X_tr.columns else None
            sample_weight = build_sample_weight(y_tr, time_idx_tr, config["training"])

            for spec in model_specs:
                entry = fit_model_entry(
                    spec,
                    X_tr,
                    y_tr,
                    sample_weight=sample_weight,
                    X_valid=X_va,
                    y_valid=y_va,
                    early_stopping_cfg=early_stopping_cfg,
                )
                pred = entry.predict(X_va, clip_prediction_min=clip)
                fold_mae = mae(y_va, pred)
                pred_by_model[spec["name"]][va_idx] = pred
                split_res["models"].setdefault(spec["name"], {"fold_mae": [], "best_iterations": []})
                split_res["models"][spec["name"]]["fold_mae"].append(fold_mae)
                best_iteration = get_best_iteration_value(entry.estimator, spec)
                if best_iteration is not None:
                    split_res["models"][spec["name"]]["best_iterations"].append(best_iteration)
                logger.info(
                    f"{spec['name']} val_mae={fold_mae:.6f}" +
                    (f" best_iter={best_iteration}" if best_iteration is not None else "")
                )

        for spec in model_specs:
            name = spec["name"]
            split_res["models"][name]["mae"] = mae(y.iloc[val_mask], pred_by_model[name][val_mask])
            split_res["models"][name]["weight"] = float(spec["weight"])

        weights = np.asarray([spec["weight"] for spec in model_specs], dtype=float)
        weights = weights / weights.sum()
        ensemble_pred = np.zeros(val_mask.sum(), dtype=float)
        for weight, spec in zip(weights, model_specs):
            ensemble_pred += weight * pred_by_model[spec["name"]][val_mask]

        split_res["ensemble"]["mae"] = mae(y.iloc[val_mask], ensemble_pred)
        split_res["ensemble"]["weights"] = {spec["name"]: float(weight) for spec, weight in zip(model_specs, weights)}
        results["splits"][split_name] = split_res
        logger.info(f"{split_name} ensemble MAE: {split_res['ensemble']['mae']:.6f}")

    seen_weight = float(cv_cfg["weight_seen"])
    unseen_weight = float(cv_cfg["weight_unseen"])
    for spec in model_specs:
        name = spec["name"]
        seen_mae = results["splits"]["seen_layout"]["models"][name]["mae"]
        unseen_mae = results["splits"]["unseen_layout"]["models"][name]["mae"]
        results["model_hybrid_scores"][name] = hybrid_score(seen_mae, unseen_mae, seen_weight, unseen_weight)
        best_values = []
        for split_name in results["splits"]:
            best_values.extend(results["splits"][split_name]["models"][name].get("best_iterations", []))
        if best_values:
            results["recommended_iterations"][name] = int(round(float(np.mean(best_values))))

    results["ensemble_hybrid_score"] = hybrid_score(
        results["splits"]["seen_layout"]["ensemble"]["mae"],
        results["splits"]["unseen_layout"]["ensemble"]["mae"],
        seen_weight,
        unseen_weight,
    )

    if cv_cfg.get("auto_weight_from_hybrid_cv", False):
        inverse = []
        for spec in model_specs:
            inverse.append(1.0 / max(results["model_hybrid_scores"][spec["name"]], 1e-6))
        inverse = np.asarray(inverse, dtype=float)
        inverse = inverse / inverse.sum()
        results["auto_weights"] = {spec["name"]: float(weight) for spec, weight in zip(model_specs, inverse)}

    return results


def fit_final_bundle(X, y, feature_builder, model_specs, config, cv_results, logger):
    clip = float(config["training"].get("clip_prediction_min", 0.0))
    time_idx = X["time_idx"] if "time_idx" in X.columns else None
    sample_weight = build_sample_weight(y, time_idx, config["training"])
    early_stopping_cfg = {"enabled": False, "rounds": 0}

    if cv_results and cv_results.get("auto_weights"):
        for spec in model_specs:
            spec["weight"] = float(cv_results["auto_weights"][spec["name"]])
    model_specs = apply_recommended_iterations(model_specs, cv_results, logger)

    models = []
    for spec in model_specs:
        logger.info(f"Fitting final model: {spec['name']}")
        models.append(
            fit_model_entry(
                spec,
                X,
                y,
                sample_weight=sample_weight,
                early_stopping_cfg=early_stopping_cfg,
            )
        )

    return EnsembleBundle(
        feature_builder=feature_builder,
        models=models,
        clip_prediction_min=clip,
        metadata={"project": config["project"]["name"], "cv_results": cv_results},
    )


def load_competition_data(config: Mapping[str, Any]):
    train_df = pd.read_csv(config["paths"]["train_csv"])
    test_df = pd.read_csv(config["paths"]["test_csv"])
    train_layout_df = pd.read_csv(config["paths"]["train_layout_csv"])
    test_layout_df = pd.read_csv(config["paths"]["test_layout_csv"])
    sample_submission_df = pd.read_csv(config["paths"]["sample_submission_csv"])
    return train_df, test_df, train_layout_df, test_layout_df, sample_submission_df


def run_full_pipeline(config: Mapping[str, Any] | None = None, fast_dev: Mapping[str, Any] | None = None):
    config = copy.deepcopy(config or CONFIG)
    fast_dev = fast_dev or FAST_DEV_SETTINGS
    logger = get_logger("ipynb_pipeline")
    set_seed(int(config["project"]["seed"]))

    if fast_dev.get("enabled", False):
        config = shrink_model_specs_for_fast_run(config)
        logger.info("Fast dev mode enabled.")

    train_df, test_df, train_layout_df, test_layout_df, sample_submission_df = load_competition_data(config)
    if fast_dev.get("enabled", False):
        train_df = sample_scenarios_for_fast_run(
            train_df,
            int(fast_dev.get("max_scenarios", 1500)),
            config["data"]["scenario_col"],
            int(config["project"]["seed"]),
        )

    feature_builder = FeatureBuilder(config)
    y = train_df[config["data"]["target_col"]].copy()
    X = feature_builder.fit_transform(train_df, train_layout_df)
    model_specs = flatten_model_specs(config)

    logger.info(f"Train features: {X.shape[0]:,} rows x {X.shape[1]:,} cols")

    cv_results = None
    if config["training"]["cv"].get("run", True):
        cv_results = run_cv(train_df, test_df, X, y, model_specs, config, logger)
        logger.info(f"Hybrid ensemble CV score: {cv_results['ensemble_hybrid_score']:.6f}")
        save_json(config["paths"]["cv_output_json"], cv_results)

    bundle = fit_final_bundle(X, y, feature_builder, model_specs, config, cv_results, logger)
    save_joblib(bundle, config["paths"]["model_path"], compress=3)
    logger.info(f"Saved model bundle to: {config['paths']['model_path']}")

    pred = bundle.predict_from_dataframe(test_df, test_layout_df)
    sample_submission_df[config["data"]["target_col"]] = pred
    output_csv = Path(config["paths"]["prediction_output_csv"])
    ensure_dir(output_csv.parent)
    sample_submission_df.to_csv(output_csv, index=False)
    logger.info(f"Saved submission to: {output_csv}")

    return {
        "config": config,
        "bundle": bundle,
        "cv_results": cv_results,
        "submission": sample_submission_df,
        "submission_path": output_csv,
        "model_path": Path(config["paths"]["model_path"]),
    }

def main() -> None:
    print("ROOT_DIR:", ROOT_DIR)
    print("Train CSV:", CONFIG["paths"]["train_csv"])
    print("Test CSV:", CONFIG["paths"]["test_csv"])
    print("Sample Submission:", CONFIG["paths"]["sample_submission_csv"])
    print("Model Output:", CONFIG["paths"]["model_path"])
    print("Submission Output:", CONFIG["paths"]["prediction_output_csv"])
    artifacts = run_full_pipeline(config=CONFIG, fast_dev=FAST_DEV_SETTINGS)
    print(artifacts["submission"].head())


if __name__ == "__main__":
    main()
