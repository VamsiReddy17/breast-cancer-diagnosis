"""
Tests for Models — Breast Cancer Diagnosis
============================================
Validates model training, predictions, and evaluation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import MODEL_CONFIG, DATA_CONFIG
from src.model_training import get_models, train_single_model
from src.evaluation import evaluate_model


class TestModels:
    """Tests for model training and evaluation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Prepare a small scaled dataset for quick tests."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=DATA_CONFIG["test_size"],
            random_state=DATA_CONFIG["random_state"],
            stratify=y
        )

        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        self.X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        self.y_train = y_train
        self.y_test = y_test

    def test_get_models_returns_all(self):
        """get_models should return all 6 model configs."""
        models = get_models()
        assert len(models) >= 5  # At least 5 (XGBoost may not be installed)
        expected_names = [
            "Logistic Regression", "Random Forest", "SVM", "KNN", "MLP"
        ]
        for name in expected_names:
            assert name in models, f"Missing model: {name}"

    def test_each_model_has_correct_structure(self):
        """Each model entry should have (model_instance, param_grid)."""
        models = get_models()
        for name, (model, param_grid) in models.items():
            assert model is not None, f"{name}: model is None"
            assert isinstance(param_grid, dict), f"{name}: param_grid not dict"
            assert len(param_grid) > 0, f"{name}: param_grid is empty"

    def test_logistic_regression_trains(self):
        """Logistic Regression should train without errors."""
        models = get_models()
        model, param_grid = models["Logistic Regression"]
        # Use minimal grid for speed
        small_grid = {"C": [1.0], "penalty": ["l2"], "solver": ["lbfgs"],
                      "max_iter": [1000], "class_weight": ["balanced"]}
        best_model, best_params, _ = train_single_model(
            "Logistic Regression", model, small_grid,
            self.X_train, self.y_train
        )
        assert best_model is not None
        preds = best_model.predict(self.X_test)
        assert len(preds) == len(self.y_test)

    def test_random_forest_trains(self):
        """Random Forest should train without errors."""
        models = get_models()
        model, _ = models["Random Forest"]
        small_grid = {"n_estimators": [50], "max_depth": [5],
                      "min_samples_split": [2], "class_weight": ["balanced"]}
        best_model, _, _ = train_single_model(
            "Random Forest", model, small_grid,
            self.X_train, self.y_train
        )
        assert best_model is not None
        assert hasattr(best_model, "feature_importances_")

    def test_predictions_are_binary(self):
        """All model predictions should be 0 or 1."""
        models = get_models()
        model, _ = models["Logistic Regression"]
        small_grid = {"C": [1.0], "penalty": ["l2"], "solver": ["lbfgs"],
                      "max_iter": [1000], "class_weight": ["balanced"]}
        best_model, _, _ = train_single_model(
            "Logistic Regression", model, small_grid,
            self.X_train, self.y_train
        )
        preds = best_model.predict(self.X_test)
        unique_preds = set(preds)
        assert unique_preds.issubset({0, 1})

    def test_evaluation_returns_all_metrics(self):
        """Evaluation should return all configured metrics."""
        models = get_models()
        model, _ = models["Logistic Regression"]
        small_grid = {"C": [1.0], "penalty": ["l2"], "solver": ["lbfgs"],
                      "max_iter": [1000], "class_weight": ["balanced"]}
        best_model, _, _ = train_single_model(
            "Logistic Regression", model, small_grid,
            self.X_train, self.y_train
        )
        metrics = evaluate_model(best_model, self.X_test, self.y_test, "LR")
        expected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for m in expected_metrics:
            assert m in metrics, f"Missing metric: {m}"
            assert 0.0 <= metrics[m] <= 1.0, f"{m} out of range: {metrics[m]}"

    def test_accuracy_above_baseline(self):
        """Model accuracy should beat random guessing (>50%)."""
        models = get_models()
        model, _ = models["Logistic Regression"]
        small_grid = {"C": [1.0], "penalty": ["l2"], "solver": ["lbfgs"],
                      "max_iter": [1000], "class_weight": ["balanced"]}
        best_model, _, _ = train_single_model(
            "Logistic Regression", model, small_grid,
            self.X_train, self.y_train
        )
        metrics = evaluate_model(best_model, self.X_test, self.y_test, "LR")
        assert metrics["accuracy"] > 0.5, "Accuracy below random baseline!"

    def test_reproducibility(self):
        """Same random_state should produce identical results."""
        models = get_models()
        model1, _ = models["Logistic Regression"]
        model2, _ = models["Logistic Regression"]
        small_grid = {"C": [1.0], "penalty": ["l2"], "solver": ["lbfgs"],
                      "max_iter": [1000], "class_weight": ["balanced"]}

        best1, _, _ = train_single_model(
            "LR", model1, small_grid, self.X_train, self.y_train
        )
        best2, _, _ = train_single_model(
            "LR", model2, small_grid, self.X_train, self.y_train
        )

        preds1 = best1.predict(self.X_test)
        preds2 = best2.predict(self.X_test)
        np.testing.assert_array_equal(preds1, preds2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
