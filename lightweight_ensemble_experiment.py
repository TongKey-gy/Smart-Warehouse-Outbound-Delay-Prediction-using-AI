from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

from prepare import OUTPUTS_DIR, ensure_runtime_directories, load_prepared_data
from train import (
    ExperimentConfig,
    add_engineered_features,
    build_model,
    build_preprocessor,
    evaluate_mae,
    evaluate_rmse,
    get_excluded_feature_columns,
    get_model_best_iteration,
    get_feature_names,
    iter_train_cv_splits,
    load_config,
    make_seen_layout_split_indices,
    make_unseen_layout_split_indices,
    select_training_view,
)


def build_hist_model(config: ExperimentConfig) -> HistGradientBoostingRegressor:
    max_leaf_nodes = max(31, min(int(config.num_leaves), 127))
    min_samples_leaf = max(20, int(config.min_child_samples))
    return HistGradientBoostingRegressor(
        loss="least_absolute_deviation",
        learning_rate=max(0.03, float(config.learning_rate)),
        max_iter=max(250, min(int(config.n_estimators), 500)),
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max(6, min(int(config.max_depth), 12)),
        min_samples_leaf=min_samples_leaf,
        l2_regularization=max(0.0, float(config.reg_lambda)),
        random_state=int(config.seed),
        early_stopping=False,
    )


def prepare_feature_matrices(prepared, config: ExperimentConfig):
    X_train, X_test, numeric_columns, categorical_columns, _ = select_training_view(prepared, config)
    X_train, X_test = add_engineered_features(X_train, X_test, config)
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    return X_train, X_test, numeric_columns, categorical_columns


def run_dual_model_oof(
    prepared,
    config: ExperimentConfig,
    split_indices: list[tuple[pd.Index, pd.Index]],
) -> dict[str, object]:
    X_train, X_test, numeric_columns, categorical_columns = prepare_feature_matrices(prepared, config)
    train_target = np.log1p(prepared.y) if config.use_log_target else prepared.y.copy()

    lgb_oof = pd.Series(index=X_train.index, dtype=float)
    hist_oof = pd.Series(index=X_train.index, dtype=float)
    lgb_best_iterations: list[int] = []
    hist_feature_names: list[str] | None = None

    for fold_idx, (train_idx, valid_idx) in enumerate(split_indices, start=1):
        X_fold_train = X_train.loc[train_idx]
        X_fold_valid = X_train.loc[valid_idx]
        y_fold_train = train_target.loc[train_idx]

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        X_train_processed = preprocessor.fit_transform(X_fold_train)
        X_valid_processed = preprocessor.transform(X_fold_valid)
        hist_feature_names = get_feature_names(preprocessor)

        lgb_model = build_model(config)
        lgb_model.fit(
            X_train_processed,
            y_fold_train,
            eval_set=[(X_valid_processed, train_target.loc[valid_idx])],
            eval_metric="l1" if config.use_log_target else "mae",
            callbacks=[],
        )
        best_iteration = get_model_best_iteration(lgb_model, config)
        lgb_best_iterations.append(best_iteration)
        lgb_pred = lgb_model.predict(X_valid_processed, num_iteration=best_iteration)
        if config.use_log_target:
            lgb_pred = np.expm1(lgb_pred)
        lgb_oof.loc[valid_idx] = np.clip(lgb_pred, a_min=config.min_prediction, a_max=None)

        hist_model = build_hist_model(config)
        hist_model.fit(X_train_processed, prepared.y.loc[train_idx])
        hist_pred = hist_model.predict(X_valid_processed)
        hist_oof.loc[valid_idx] = np.clip(hist_pred, a_min=config.min_prediction, a_max=None)

        print(
            f"Fold {fold_idx}: "
            f"LGB_MAE={evaluate_mae(prepared.y.loc[valid_idx], lgb_oof.loc[valid_idx]):.6f} "
            f"HIST_MAE={evaluate_mae(prepared.y.loc[valid_idx], hist_oof.loc[valid_idx]):.6f}"
        )

    final_preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    X_train_processed = final_preprocessor.fit_transform(X_train)
    X_test_processed = final_preprocessor.transform(X_test)

    final_lgb = build_model(config, n_estimators=max(1, int(round(float(np.mean(lgb_best_iterations))))))
    final_lgb.fit(X_train_processed, train_target)
    lgb_test_pred = final_lgb.predict(X_test_processed)
    if config.use_log_target:
        lgb_test_pred = np.expm1(lgb_test_pred)
    lgb_test_pred = np.clip(lgb_test_pred, a_min=config.min_prediction, a_max=None)

    final_hist = build_hist_model(config)
    final_hist.fit(X_train_processed, prepared.y)
    hist_test_pred = np.clip(final_hist.predict(X_test_processed), a_min=config.min_prediction, a_max=None)

    return {
        "lgb_oof": lgb_oof,
        "hist_oof": hist_oof,
        "lgb_test_pred": lgb_test_pred,
        "hist_test_pred": hist_test_pred,
        "feature_names": hist_feature_names or [],
        "excluded_columns": [column for column in get_excluded_feature_columns(prepared, config) if column in prepared.feature_columns],
        "final_lgb_n_estimators": max(1, int(round(float(np.mean(lgb_best_iterations))))),
    }


def choose_hist_weight(y_true: pd.Series, lgb_oof: pd.Series, hist_oof: pd.Series) -> tuple[float, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    valid_index = y_true.index
    for hist_weight in np.linspace(0.0, 0.5, 11):
        blend = (1.0 - hist_weight) * lgb_oof.loc[valid_index] + hist_weight * hist_oof.loc[valid_index]
        candidates.append(
            {
                "hist_weight": float(hist_weight),
                "mae": evaluate_mae(y_true.loc[valid_index], blend),
                "rmse": evaluate_rmse(y_true.loc[valid_index], blend.to_numpy()),
            }
        )
    best = min(candidates, key=lambda item: item["mae"])
    return float(best["hist_weight"]), candidates


def blend_predictions(lgb_pred, hist_pred, hist_weight: float):
    return (1.0 - hist_weight) * np.asarray(lgb_pred, dtype=float) + hist_weight * np.asarray(hist_pred, dtype=float)


def save_outputs(prepared, config: ExperimentConfig, summary: dict[str, object], submission_pred: np.ndarray) -> Path:
    run_dir = OUTPUTS_DIR / "lightweight_ensemble" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    submission = prepared.submission_df.copy()
    submission[prepared.target_column] = submission_pred
    submission_path = run_dir / f"submission_{config.experiment_name}.csv"
    submission.to_csv(submission_path, index=False)

    return summary_path


def main() -> None:
    ensure_runtime_directories()
    prepared = load_prepared_data()
    config = load_config()

    if config.blend_secondary_model:
        print("Ignoring train.py secondary blend path for lightweight ensemble experiment.")

    split_indices = [(train_idx, valid_idx) for _fold_idx, train_idx, valid_idx in iter_train_cv_splits(prepared, config)[2]]
    main_result = run_dual_model_oof(prepared, config, split_indices)
    hist_weight, weight_grid = choose_hist_weight(prepared.y, main_result["lgb_oof"], main_result["hist_oof"])
    main_oof = blend_predictions(main_result["lgb_oof"], main_result["hist_oof"], hist_weight)
    main_test_pred = blend_predictions(main_result["lgb_test_pred"], main_result["hist_test_pred"], hist_weight)

    seen_splits = make_seen_layout_split_indices(prepared, config)
    unseen_splits = make_unseen_layout_split_indices(prepared, config)

    seen_mae = None
    unseen_mae = None
    if len(seen_splits) >= 2:
        seen_result = run_dual_model_oof(prepared, config, seen_splits)
        seen_blend = blend_predictions(seen_result["lgb_oof"], seen_result["hist_oof"], hist_weight)
        seen_mae = evaluate_mae(prepared.y.loc[seen_result["lgb_oof"].dropna().index], seen_blend[~np.isnan(seen_blend)])
    if len(unseen_splits) >= 2:
        unseen_result = run_dual_model_oof(prepared, config, unseen_splits)
        unseen_blend = blend_predictions(unseen_result["lgb_oof"], unseen_result["hist_oof"], hist_weight)
        unseen_mae = evaluate_mae(prepared.y.loc[unseen_result["lgb_oof"].dropna().index], unseen_blend[~np.isnan(unseen_blend)])

    hybrid_score = None
    if seen_mae is not None and unseen_mae is not None:
        hybrid_score = float(config.hybrid_seen_weight) * float(seen_mae) + float(config.hybrid_unseen_weight) * float(unseen_mae)

    summary = {
        "config": asdict(config),
        "ensemble": "lightgbm_histgradientboosting",
        "hist_weight": hist_weight,
        "weight_grid": weight_grid,
        "main_oof_mae": evaluate_mae(prepared.y, main_oof),
        "main_oof_rmse": evaluate_rmse(prepared.y, main_oof),
        "seen_layout_oof_mae": seen_mae,
        "unseen_layout_oof_mae": unseen_mae,
        "hybrid_layout_score": hybrid_score,
        "final_lgb_n_estimators": main_result["final_lgb_n_estimators"],
    }
    summary_path = save_outputs(prepared, config, summary, main_test_pred)

    print(f"Chosen hist weight: {hist_weight:.2f}")
    print(f"Main OOF MAE: {summary['main_oof_mae']:.6f}")
    if seen_mae is not None:
        print(f"Seen layout OOF MAE: {seen_mae:.6f}")
    if unseen_mae is not None:
        print(f"Unseen layout OOF MAE: {unseen_mae:.6f}")
    if hybrid_score is not None:
        print(f"Hybrid layout score: {hybrid_score:.6f}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
