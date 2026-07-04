"""
Data Loader — Breast Cancer Diagnosis
========================================
Handles all data loading, validation, splitting, and persistence.
Every dataset operation flows through this module.

Functions:
    load_breast_cancer_data  — Load raw data from sklearn
    validate_data            — Validate DataFrame shape, nulls, distributions
    split_data               — Stratified train/test split
    save_data                — Persist raw + split data to disk with hashing
    load_processed_data      — Reload previously saved splits
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ─── Project path setup ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    DATA_CONFIG,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ensure_directories,
)
from src.utils import (
    get_logger,
    timer,
    validate_dataframe,
    compute_file_hash,
    save_json,
    load_json,
)

# ─── Module-level setup ────────────────────────────────────────────────────────
ensure_directories()
logger = get_logger(__name__)


# ─── Public API ─────────────────────────────────────────────────────────────────

@timer
def load_breast_cancer_data() -> pd.DataFrame:
    """
    Load the Wisconsin Breast Cancer dataset from sklearn.

    The dataset contains 569 samples with 30 numeric features computed from
    digitised images of fine-needle aspirates of breast masses.  A ``target``
    column is appended (0 = Malignant, 1 = Benign).

    Returns:
        pd.DataFrame: DataFrame with 30 feature columns and 1 target column.
    """
    logger.info("Loading breast cancer dataset from sklearn …")

    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df[DATA_CONFIG["target_column"]] = bunch.target

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {df.shape[1] - 1}  |  Samples: {df.shape[0]}")

    # Log class distribution
    class_counts = df[DATA_CONFIG["target_column"]].value_counts().sort_index()
    for label_idx, count in class_counts.items():
        class_name = DATA_CONFIG["class_names"][label_idx]
        pct = count / len(df) * 100
        logger.info(f"  Class {label_idx} ({class_name}): {count} ({pct:.1f}%)")

    return df


@timer
def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the breast cancer DataFrame for completeness and integrity.

    Checks performed:
      • Shape matches expected samples / features from DATA_CONFIG
      • Null and infinite value detection
      • Duplicate row detection
      • Class distribution sanity check (both classes present)

    Args:
        df: DataFrame returned by :func:`load_breast_cancer_data`.

    Returns:
        dict: Validation result with keys ``is_valid``, ``shape``, ``issues``, etc.

    Raises:
        ValueError: If a critical validation check fails.
    """
    logger.info("Validating dataset …")

    # Total columns = features + target
    expected_total_cols = DATA_CONFIG["expected_n_features"] + 1

    validation = validate_dataframe(
        df,
        expected_rows=DATA_CONFIG["expected_n_samples"],
        expected_cols=expected_total_cols,
        name="BreastCancerDataset",
    )

    # ── Class-distribution check ────────────────────────────────────────────
    target_col = DATA_CONFIG["target_column"]
    if target_col not in df.columns:
        validation["issues"].append(f"Missing target column: '{target_col}'")
        validation["is_valid"] = False
        logger.error(f"Target column '{target_col}' not found in DataFrame")
    else:
        unique_classes = sorted(df[target_col].unique())
        expected_classes = list(range(len(DATA_CONFIG["class_names"])))
        if unique_classes != expected_classes:
            msg = (
                f"Expected classes {expected_classes}, "
                f"got {unique_classes}"
            )
            validation["issues"].append(msg)
            validation["is_valid"] = False
            logger.warning(msg)
        else:
            class_counts = df[target_col].value_counts().sort_index()
            validation["class_distribution"] = {
                DATA_CONFIG["class_names"][idx]: int(cnt)
                for idx, cnt in class_counts.items()
            }
            logger.info(
                f"Class distribution: {validation['class_distribution']}"
            )

    if validation["is_valid"]:
        logger.info("✅ Data validation PASSED")
    else:
        logger.warning(
            f"⚠️ Data validation FAILED — {len(validation['issues'])} issue(s)"
        )

    return validation


@timer
def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split.

    Uses ``DATA_CONFIG['test_size']`` and ``DATA_CONFIG['random_state']``
    so that every run produces identical splits.

    Args:
        df: Full dataset with features and target column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    target_col = DATA_CONFIG["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(
        f"Splitting data: test_size={DATA_CONFIG['test_size']}, "
        f"random_state={DATA_CONFIG['random_state']}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y,
    )

    # ── Log split sizes ─────────────────────────────────────────────────────
    logger.info(f"Train set: {X_train.shape[0]} samples  ({X_train.shape[1]} features)")
    logger.info(f"Test  set: {X_test.shape[0]} samples  ({X_test.shape[1]} features)")

    # ── Log class distribution per split ────────────────────────────────────
    for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
        counts = y_split.value_counts().sort_index()
        dist_str = "  |  ".join(
            f"{DATA_CONFIG['class_names'][idx]}: {cnt} ({cnt / len(y_split) * 100:.1f}%)"
            for idx, cnt in counts.items()
        )
        logger.info(f"  {split_name} class dist → {dist_str}")

    return X_train, X_test, y_train, y_test


@timer
def save_data(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Persist raw data and train/test splits to disk.

    Saves:
      • ``data/raw/breast_cancer_raw.csv``
      • ``data/processed/X_train.csv``
      • ``data/processed/X_test.csv``
      • ``data/processed/y_train.csv``
      • ``data/processed/y_test.csv``

    Also computes MD5 hashes of every file and updates
    ``data/data_registry.json``.

    Args:
        df: Full raw DataFrame.
        X_train, X_test: Feature splits.
        y_train, y_test: Label splits.

    Returns:
        dict: Updated data-registry payload.
    """
    logger.info("Saving data to disk …")

    # ── Save raw data ───────────────────────────────────────────────────────
    raw_path = os.path.join(RAW_DATA_DIR, "breast_cancer_raw.csv")
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw data → {raw_path}")

    # ── Save processed splits ───────────────────────────────────────────────
    splits = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": y_train,
        "y_test.csv": y_test,
    }

    saved_files = {}
    for filename, data in splits.items():
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        if isinstance(data, pd.Series):
            data.to_csv(filepath, index=False, header=True)
        else:
            data.to_csv(filepath, index=False)
        saved_files[filename] = filepath
        logger.info(f"Saved split → {filepath}")

    # ── Compute file hashes ─────────────────────────────────────────────────
    all_files = {"breast_cancer_raw.csv": raw_path, **saved_files}
    hashes = {}
    for name, path in all_files.items():
        file_hash = compute_file_hash(path)
        hashes[name] = file_hash
        logger.info(f"  Hash [{name}]: {file_hash}")

    # ── Update data registry ────────────────────────────────────────────────
    registry_path = os.path.join(DATA_DIR, "data_registry.json")

    if os.path.exists(registry_path):
        registry = load_json(registry_path)
    else:
        registry = {"version": "0.1.0", "created": datetime.now().strftime("%Y-%m-%d"), "entries": []}

    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "save_data",
        "files": {name: {"path": path, "md5": hashes[name]} for name, path in all_files.items()},
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "test_size": DATA_CONFIG["test_size"],
        "random_state": DATA_CONFIG["random_state"],
    }
    registry["entries"].append(entry)

    save_json(registry, registry_path)
    logger.info(f"Updated data registry → {registry_path}")

    return registry


@timer
def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load previously saved train/test splits from ``PROCESSED_DATA_DIR``.

    Expects the files written by :func:`save_data`:
      • ``X_train.csv``, ``X_test.csv``, ``y_train.csv``, ``y_test.csv``

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        FileNotFoundError: If any expected split file is missing.
    """
    logger.info("Loading processed data from disk …")

    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    paths = {}
    for fname in required_files:
        fpath = os.path.join(PROCESSED_DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Processed file not found: {fpath}. Run save_data() first."
            )
        paths[fname] = fpath

    X_train = pd.read_csv(paths["X_train.csv"])
    X_test = pd.read_csv(paths["X_test.csv"])
    y_train = pd.read_csv(paths["y_train.csv"]).squeeze("columns")
    y_test = pd.read_csv(paths["y_test.csv"]).squeeze("columns")

    logger.info(f"Loaded X_train: {X_train.shape}")
    logger.info(f"Loaded X_test:  {X_test.shape}")
    logger.info(f"Loaded y_train: {y_train.shape}")
    logger.info(f"Loaded y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ─── Main (self-test) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  DATA LOADER — Self-Test")
    print("=" * 70)

    # Step 1: Load
    print("\n[1/5] Loading data …")
    df = load_breast_cancer_data()
    print(f"      Shape: {df.shape}")

    # Step 2: Validate
    print("\n[2/5] Validating data …")
    val = validate_data(df)
    print(f"      Valid: {val['is_valid']}")
    if val["issues"]:
        for issue in val["issues"]:
            print(f"      ⚠ {issue}")

    # Step 3: Split
    print("\n[3/5] Splitting data …")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Step 4: Save
    print("\n[4/5] Saving data …")
    registry = save_data(df, X_train, X_test, y_train, y_test)
    print(f"      Registry entries: {len(registry['entries'])}")

    # Step 5: Reload
    print("\n[5/5] Reloading processed data …")
    X_tr2, X_te2, y_tr2, y_te2 = load_processed_data()

    # Verify round-trip
    assert X_tr2.shape == X_train.shape, "X_train shape mismatch after reload"
    assert X_te2.shape == X_test.shape, "X_test shape mismatch after reload"
    assert np.array_equal(y_tr2.values, y_train.values), "y_train mismatch after reload"
    assert np.array_equal(y_te2.values, y_test.values), "y_test mismatch after reload"
    print("      ✅ Round-trip verification passed")

    print("\n" + "=" * 70)
    print("  ALL SELF-TESTS PASSED ✅")
    print("=" * 70)
