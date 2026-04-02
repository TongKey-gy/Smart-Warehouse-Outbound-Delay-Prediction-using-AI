from __future__ import annotations

import logging
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_URL = "https://drive.google.com/file/d/1_9SO7hGWSgiP7vGZfV095VH13OLBYwFG/view?usp=drive_link"
DATA_URL = os.environ.get("OPEN_DATA_URL", DEFAULT_DATA_URL)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "open"))).resolve()
DATA_ZIP_PATH = ROOT / "open.zip"
CACHE_DIR = ROOT / "cache"
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
EXPECTED_FILES = (
    "train.csv",
    "test.csv",
    "layout_info.csv",
    "sample_submission.csv",
)

logger = logging.getLogger("prepare")


@dataclass(frozen=True)
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    layout_df: pd.DataFrame
    submission_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None
    target_column: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    cv_strategy: str
    n_splits: int


def configure_logging() -> None:
    if logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_runtime_directories() -> None:
    for path in (CACHE_DIR, LOGS_DIR, OUTPUTS_DIR, SUBMISSIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def extract_google_drive_file_id(url: str) -> str:
    patterns = (
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Unable to extract a Google Drive file id from '{url}'.")


def validate_data_dir(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    missing = [name for name in EXPECTED_FILES if not (data_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        expected_path = data_dir.resolve()
        raise FileNotFoundError(
            f"Missing dataset files in '{expected_path}'. "
            f"Expected: {missing_text}. "
            "The dataset bootstrap expects open/train.csv, open/test.csv, "
            "open/layout_info.csv, and open/sample_submission.csv."
        )
    return {name: data_dir / name for name in EXPECTED_FILES}


def validate_extracted_dataset(data_dir: Path = DATA_DIR) -> None:
    paths = validate_data_dir(data_dir)
    for name, path in paths.items():
        if path.stat().st_size == 0:
            raise ValueError(f"Extracted dataset file '{name}' is empty: '{path}'.")


def download_with_gdown(url: str, destination: Path) -> None:
    logger.info("Downloading dataset with gdown: %s -> %s", url, destination)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed. Install it with 'pip install gdown' and rerun prepare.py."
        ) from exc

    output = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
    if output is None or not destination.exists():
        raise RuntimeError("gdown did not produce the expected zip file.")


def download_with_urllib(url: str, destination: Path) -> None:
    file_id = extract_google_drive_file_id(url)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    logger.info("Falling back to direct download: %s -> %s", direct_url, destination)
    try:
        with urllib.request.urlopen(direct_url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Direct download failed for '{direct_url}': {exc}") from exc

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("Direct download produced an empty zip file.")


def ensure_zip_downloaded(url: str = DATA_URL, destination: Path = DATA_ZIP_PATH) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Using existing zip file: %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_with_gdown(url, destination)
    except Exception as gdown_error:
        logger.warning("gdown download failed: %s", gdown_error)
        if destination.exists():
            destination.unlink()
        try:
            download_with_urllib(url, destination)
        except Exception as direct_error:
            if destination.exists():
                destination.unlink()
            raise RuntimeError(
                "Failed to download dataset from Google Drive via both gdown and direct download. "
                f"gdown error: {gdown_error} | direct download error: {direct_error}"
            ) from direct_error

    logger.info("Dataset zip is ready: %s (%.2f MB)", destination, destination.stat().st_size / (1024 * 1024))
    return destination


def extract_dataset(zip_path: Path = DATA_ZIP_PATH, data_dir: Path = DATA_DIR) -> None:
    logger.info("Extracting dataset: %s -> %s", zip_path, data_dir)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(ROOT)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded zip file is invalid: '{zip_path}'.") from exc

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Extraction completed but '{data_dir}' was not created. "
            "Check whether the zip file contains the open/ directory."
        )


def ensure_dataset_available(data_dir: Path = DATA_DIR, url: str = DATA_URL) -> None:
    if data_dir.exists():
        logger.info("Dataset directory already exists. Skipping download and extraction: %s", data_dir)
        validate_extracted_dataset(data_dir)
        return

    zip_path = ensure_zip_downloaded(url=url, destination=DATA_ZIP_PATH)
    extract_dataset(zip_path=zip_path, data_dir=data_dir)
    validate_extracted_dataset(data_dir)
    logger.info("Dataset bootstrap completed successfully.")


def load_raw_frames(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dataset_available(data_dir)
    paths = validate_data_dir(data_dir)
    train_df = pd.read_csv(paths["train.csv"])
    test_df = pd.read_csv(paths["test.csv"])
    layout_df = pd.read_csv(paths["layout_info.csv"])
    submission_df = pd.read_csv(paths["sample_submission.csv"])
    return train_df, test_df, layout_df, submission_df


def detect_target_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame | None = None,
) -> str:
    candidates = [column for column in train_df.columns if column not in test_df.columns]
    if not candidates:
        raise ValueError("Unable to detect target column because train/test columns are identical.")
    if len(candidates) == 1:
        return candidates[0]

    if submission_df is not None:
        submission_candidates = [column for column in submission_df.columns if column != "ID"]
        matches = [column for column in candidates if column in submission_candidates]
        if len(matches) == 1:
            return matches[0]

    numeric_candidates = [column for column in candidates if pd.api.types.is_numeric_dtype(train_df[column])]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Unable to detect target column automatically. "
        f"Candidates found: {candidates}"
    )


def merge_layout_info(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "layout_id" not in train_df.columns or "layout_id" not in test_df.columns:
        return train_df.copy(), test_df.copy()
    if "layout_id" not in layout_df.columns:
        raise ValueError("layout_info.csv must contain a 'layout_id' column.")

    merged_train = train_df.merge(layout_df, on="layout_id", how="left", validate="m:1")
    merged_test = test_df.merge(layout_df, on="layout_id", how="left", validate="m:1")
    return merged_train, merged_test


def choose_cv_strategy(groups: pd.Series | None, n_rows: int) -> tuple[str, int]:
    if groups is not None:
        unique_groups = int(groups.nunique(dropna=False))
        if unique_groups >= 2:
            return "group_kfold", min(5, unique_groups)
    return "kfold", min(5, n_rows) if n_rows > 1 else 1


def build_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_features = train_df.drop(columns=[target_column])
    feature_columns = [column for column in train_features.columns if column in test_df.columns]
    X_train = train_features.loc[:, feature_columns].copy()
    X_test = test_df.loc[:, feature_columns].copy()
    return X_train, X_test, feature_columns


def load_prepared_data(data_dir: Path = DATA_DIR) -> PreparedData:
    ensure_runtime_directories()
    train_raw, test_raw, layout_df, submission_df = load_raw_frames(data_dir)
    target_column = detect_target_column(train_raw, test_raw, submission_df)
    train_df, test_df = merge_layout_info(train_raw, test_raw, layout_df)
    X_train, X_test, feature_columns = build_feature_frames(train_df, test_df, target_column)
    y = train_df[target_column].copy()
    groups = train_df["scenario_id"].copy() if "scenario_id" in train_df.columns else None
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    cv_strategy, n_splits = choose_cv_strategy(groups, len(X_train))

    return PreparedData(
        train_df=train_df,
        test_df=test_df,
        layout_df=layout_df,
        submission_df=submission_df,
        X_train=X_train,
        X_test=X_test,
        y=y,
        groups=groups,
        target_column=target_column,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
    )


def iter_cv_splits(prepared: PreparedData) -> Iterator[tuple[int, pd.Index, pd.Index]]:
    index = prepared.X_train.index
    if prepared.n_splits < 2:
        raise ValueError("At least two folds are required for cross validation.")

    if prepared.cv_strategy == "group_kfold":
        splitter = GroupKFold(n_splits=prepared.n_splits)
        split_iter = splitter.split(prepared.X_train, prepared.y, groups=prepared.groups)
    else:
        splitter = KFold(n_splits=prepared.n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(prepared.X_train, prepared.y)

    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        yield fold_idx, index[train_idx], index[valid_idx]


def summarize_prepared_data(prepared: PreparedData) -> dict[str, object]:
    return {
        "train_rows": len(prepared.train_df),
        "test_rows": len(prepared.test_df),
        "layout_rows": len(prepared.layout_df),
        "target_column": prepared.target_column,
        "feature_count": len(prepared.feature_columns),
        "numeric_feature_count": len(prepared.numeric_columns),
        "categorical_feature_count": len(prepared.categorical_columns),
        "cv_strategy": prepared.cv_strategy,
        "n_splits": prepared.n_splits,
        "has_groups": prepared.groups is not None,
    }


def main() -> None:
    configure_logging()
    try:
        prepared = load_prepared_data()
    except Exception as exc:
        logger.exception("prepare.py failed: %s", exc)
        raise SystemExit(1) from exc

    summary = summarize_prepared_data(prepared)
    logger.info("Data preparation check completed.")
    for key, value in summary.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
