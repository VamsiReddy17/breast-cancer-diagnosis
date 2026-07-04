"""
Tests for Data Loader — Breast Cancer Diagnosis
=================================================
Validates data loading, integrity, and splitting.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import DATA_CONFIG
from src.data_loader import load_breast_cancer_data, validate_data, split_data


class TestDataLoader:
    """Tests for the data loading module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load data once for all tests."""
        self.df = load_breast_cancer_data()

    def test_data_loads_successfully(self):
        """Data should load as a non-empty DataFrame."""
        assert self.df is not None
        assert isinstance(self.df, pd.DataFrame)
        assert len(self.df) > 0

    def test_correct_sample_count(self):
        """Should have exactly 569 samples."""
        assert len(self.df) == DATA_CONFIG["expected_n_samples"]

    def test_correct_feature_count(self):
        """Should have 30 features + 1 target = 31 columns."""
        # 30 features + 1 target column
        assert len(self.df.columns) == DATA_CONFIG["expected_n_features"] + 1

    def test_target_column_exists(self):
        """Target column must exist."""
        assert DATA_CONFIG["target_column"] in self.df.columns

    def test_target_has_two_classes(self):
        """Target should have exactly 2 classes (0 and 1)."""
        unique_classes = self.df[DATA_CONFIG["target_column"]].nunique()
        assert unique_classes == 2

    def test_no_null_values(self):
        """Dataset should have no null values."""
        assert self.df.isnull().sum().sum() == 0

    def test_no_infinite_values(self):
        """Dataset should have no infinite values."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_count = self.df[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
        assert inf_count == 0

    def test_all_features_numeric(self):
        """All feature columns should be numeric."""
        feature_cols = [c for c in self.df.columns if c != DATA_CONFIG["target_column"]]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(self.df[col]), f"{col} is not numeric"

    def test_class_distribution(self):
        """Both classes should have samples."""
        class_counts = self.df[DATA_CONFIG["target_column"]].value_counts()
        assert all(count > 0 for count in class_counts.values)

    def test_validation_passes(self):
        """Validation should pass for clean data."""
        result = validate_data(self.df)
        assert result["is_valid"] is True

    def test_split_preserves_samples(self):
        """Train + test samples should equal total samples."""
        X_train, X_test, y_train, y_test = split_data(self.df)
        total = len(X_train) + len(X_test)
        expected_features = DATA_CONFIG["expected_n_features"]
        assert total == DATA_CONFIG["expected_n_samples"]
        assert X_train.shape[1] == expected_features
        assert X_test.shape[1] == expected_features

    def test_split_is_stratified(self):
        """Class proportions should be similar in train and test."""
        X_train, X_test, y_train, y_test = split_data(self.df)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        # Stratified split should keep ratios within 5%
        assert abs(train_ratio - test_ratio) < 0.05

    def test_split_is_reproducible(self):
        """Same random_state should produce identical splits."""
        X1, _, y1, _ = split_data(self.df)
        X2, _, y2, _ = split_data(self.df)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
