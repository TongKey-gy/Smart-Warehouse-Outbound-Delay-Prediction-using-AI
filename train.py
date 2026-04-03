from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
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
    iter_cv_splits,
    load_prepared_data,
)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "baseline_lightgbm_v1"
    random_state: int = 42
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 50
    log_evaluation_period: int = 100

        if field_type is int:
            values[name] = int(raw_value)
        elif field_type is float:
            values[name] = float(raw_value)
        else:
            values[name] = raw_value

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


def build_model(config: ExperimentConfig) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        verbose=-1,
    )


def evaluate_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
    config: ExperimentConfig,
    fold_metrics: list[dict[str, float]],
    results_row: dict[str, object],
    experiment_dir: Path,
) -> Path:
    summary_path = experiment_dir / "metrics.json"
    payload = {
        "config": asdict(config),
        "fold_metrics": fold_metrics,
        "summary": results_row,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    ensure_runtime_directories()
    ensure_local_submission_dir()
    prepared = load_prepared_data()
    config = load_config_from_env(ExperimentConfig())
    improvement_notes = os.environ.get("TRAIN_IMPROVEMENT_NOTES", "-").strip() or "-"
    X_train, X_test, numeric_columns, categorical_columns, excluded_columns = select_training_view(prepared)

    oof_predictions = pd.Series(index=X_train.index, dtype=float)
    fold_rmse_scores: list[float] = []
    fold_importances: list[pd.DataFrame] = []
    best_iterations: list[int] = []

    oof_predictions = pd.Series(index=prepared.X_train.index, dtype=float)
    fold_metrics: list[dict[str, float]] = []
    fold_importances: list[pd.DataFrame] = []

    for fold_idx, train_idx, valid_idx in iter_cv_splits(prepared):
        X_fold_train = prepared.X_train.loc[train_idx]
        y_fold_train = prepared.y.loc[train_idx]
        X_fold_valid = X_train.loc[valid_idx]
        y_fold_valid = prepared.y.loc[valid_idx]

        preprocessor = build_preprocessor(prepared.numeric_columns, prepared.categorical_columns)
        X_fold_train_processed = preprocessor.fit_transform(X_fold_train)
        X_fold_valid_processed = preprocessor.transform(X_fold_valid)
        feature_names = get_feature_names(preprocessor)

        model = build_model(config)
        model.fit(
            X_fold_train_processed,
            y_fold_train,
            eval_set=[(X_fold_valid_processed, y_fold_valid)],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds),
                lgb.log_evaluation(config.log_evaluation_period),
            ],
        )

        fold_pred = model.predict(X_fold_valid_processed, num_iteration=model.best_iteration_)
        oof_predictions.loc[valid_idx] = fold_pred
        metrics = evaluate_predictions(y_fold_valid, fold_pred)
        fold_metrics.append(metrics)
        fold_importances.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "feature_name": feature_names,
                    "importance": model.feature_importances_,
                }
            )
        )
        print(
            f"Fold {fold_idx}: RMSE={fold_rmse:.6f} BEST_ITER={best_iteration}"
        )

    rmse_scores = [metric["rmse"] for metric in fold_metrics]
    mae_scores = [metric["mae"] for metric in fold_metrics]
    r2_scores = [metric["r2"] for metric in fold_metrics]
    oof_mae = float(mean_absolute_error(prepared.y.loc[oof_predictions.index], oof_predictions.loc[prepared.y.index]))
    print(
        f"CV Summary: RMSE={np.mean(rmse_scores):.6f}±{np.std(rmse_scores):.6f} "
        f"MAE={np.mean(mae_scores):.6f} "
        f"R2={np.mean(r2_scores):.6f} "
        f"OOF_MAE={oof_mae:.6f}"
    )

    final_preprocessor = build_preprocessor(prepared.numeric_columns, prepared.categorical_columns)
    X_train_processed = final_preprocessor.fit_transform(prepared.X_train)
    X_test_processed = final_preprocessor.transform(prepared.X_test)
    final_model = build_model(config)
    final_model.fit(X_train_processed, prepared.y)
    test_pred = final_model.predict(X_test_processed)

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

    oof_path = save_oof_predictions(prepared, oof_predictions, experiment_dir)
    importance_path = save_feature_importance(fold_importances, experiment_dir)

    results_row = {
        "timestamp": timestamp,
        "experiment_name": config.experiment_name,
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
        "model_name": config.experiment_name,
        "submission_path": str(submission_path.relative_to(Path.cwd())),
        "submission_alias": format_submission_alias(submission_number),
        "submission_local_path": str(archived_submission_path.relative_to(Path.cwd())),
        "experiment_dir": str(experiment_dir.relative_to(Path.cwd())),
    }
    summary_path = save_experiment_summary(config, fold_metrics, results_row, experiment_dir)
    results_path = append_results_log(results_row)

    print(f"Updated log: {results_path}")
    print(f"Saved OOF predictions: {oof_path}")
    if importance_path is not None:
        print(f"Saved feature importance: {importance_path}")
    print(f"Saved experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
