from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from prepare import LOGS_DIR, SUBMISSIONS_DIR, ensure_runtime_directories, iter_cv_splits, load_prepared_data


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    learning_rate: float = 0.05
    max_depth: int = 6
    max_iter: int = 300
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0


def build_model(config: ExperimentConfig, numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
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
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_iter=config.max_iter,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def append_results_log(row: dict[str, object]) -> Path:
    results_path = LOGS_DIR / "results.csv"
    frame = pd.DataFrame([row])
    if results_path.exists():
        frame.to_csv(results_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(results_path, index=False)
    return results_path


def main() -> None:
    ensure_runtime_directories()
    prepared = load_prepared_data()
    config = ExperimentConfig()

    fold_metrics: list[dict[str, float]] = []
    for fold_idx, train_idx, valid_idx in iter_cv_splits(prepared):
        model = build_model(config, prepared.numeric_columns, prepared.categorical_columns)
        X_fold_train = prepared.X_train.loc[train_idx]
        y_fold_train = prepared.y.loc[train_idx]
        X_fold_valid = prepared.X_train.loc[valid_idx]
        y_fold_valid = prepared.y.loc[valid_idx]

        model.fit(X_fold_train, y_fold_train)
        fold_pred = model.predict(X_fold_valid)
        metrics = evaluate_predictions(y_fold_valid, fold_pred)
        fold_metrics.append(metrics)
        print(
            f"Fold {fold_idx}: "
            f"RMSE={metrics['rmse']:.6f} "
            f"MAE={metrics['mae']:.6f} "
            f"R2={metrics['r2']:.6f}"
        )

    rmse_scores = [metric["rmse"] for metric in fold_metrics]
    mae_scores = [metric["mae"] for metric in fold_metrics]
    r2_scores = [metric["r2"] for metric in fold_metrics]
    print(
        f"CV Summary: RMSE={np.mean(rmse_scores):.6f}±{np.std(rmse_scores):.6f} "
        f"MAE={np.mean(mae_scores):.6f} "
        f"R2={np.mean(r2_scores):.6f}"
    )

    final_model = build_model(config, prepared.numeric_columns, prepared.categorical_columns)
    final_model.fit(prepared.X_train, prepared.y)
    test_pred = final_model.predict(prepared.X_test)

    submission = prepared.submission_df.copy()
    submission[prepared.target_column] = test_pred
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = SUBMISSIONS_DIR / f"submission_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")

    results_row = {
        "timestamp": timestamp,
        "target_column": prepared.target_column,
        "cv_strategy": prepared.cv_strategy,
        "n_splits": prepared.n_splits,
        "n_train_rows": len(prepared.X_train),
        "n_test_rows": len(prepared.X_test),
        "n_features": len(prepared.feature_columns),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "r2_mean": float(np.mean(r2_scores)),
        "model_name": "HistGradientBoostingRegressor",
        "submission_path": str(submission_path.relative_to(Path.cwd())),
    }
    results_path = append_results_log(results_row)
    print(f"Updated log: {results_path}")


if __name__ == "__main__":
    main()
