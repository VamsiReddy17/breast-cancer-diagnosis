"""
Tests for Pipeline — Breast Cancer Diagnosis
===============================================
End-to-end smoke test for the full pipeline.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    FIGURES_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR, ensure_directories
)


class TestPipelineOutputs:
    """
    Tests that validate pipeline outputs exist after a run.
    Run the pipeline first: python src/pipeline.py
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure directories exist."""
        ensure_directories()

    def test_directories_exist(self):
        """All required directories should exist."""
        dirs = [FIGURES_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]
        for d in dirs:
            assert os.path.isdir(d), f"Directory missing: {d}"

    def test_figures_generated(self):
        """After pipeline run, figures directory should have plots."""
        if not os.listdir(FIGURES_DIR):
            pytest.skip("No figures found — run pipeline first")
        png_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
        assert len(png_files) > 0, "No PNG files in figures directory"

    def test_results_generated(self):
        """After pipeline run, results directory should have CSVs."""
        if not os.listdir(RESULTS_DIR):
            pytest.skip("No results found — run pipeline first")
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
        assert len(csv_files) > 0, "No CSV files in results directory"

    def test_models_saved(self):
        """After pipeline run, models directory should have .pkl files."""
        if not os.listdir(MODELS_DIR):
            pytest.skip("No models found — run pipeline first")
        pkl_files = [f for f in os.listdir(MODELS_DIR)
                     if f.endswith(".pkl") or f.endswith(".joblib")]
        assert len(pkl_files) > 0, "No model files in models directory"

    def test_log_file_exists(self):
        """Pipeline log file should exist after a run."""
        log_file = os.path.join(LOGS_DIR, "pipeline.log")
        if not os.path.exists(log_file):
            pytest.skip("Log file not found — run pipeline first")
        assert os.path.getsize(log_file) > 0, "Log file is empty"

    def test_pipeline_summary_exists(self):
        """Pipeline run summary JSON should exist."""
        summary_path = os.path.join(RESULTS_DIR, "pipeline_run_summary.json")
        if not os.path.exists(summary_path):
            pytest.skip("Summary not found — run pipeline first")
        assert os.path.getsize(summary_path) > 0, "Summary file is empty"


class TestPipelineImports:
    """Tests that all pipeline modules can be imported."""

    def test_import_data_loader(self):
        """data_loader module should import without errors."""
        from src.data_loader import load_breast_cancer_data, validate_data, split_data

    def test_import_eda(self):
        """eda module should import without errors."""
        from src.eda import run_eda

    def test_import_feature_engineering(self):
        """feature_engineering module should import without errors."""
        from src.feature_engineering import run_feature_engineering

    def test_import_model_training(self):
        """model_training module should import without errors."""
        from src.model_training import get_models, train_all_models

    def test_import_evaluation(self):
        """evaluation module should import without errors."""
        from src.evaluation import evaluate_all_models

    def test_import_pipeline(self):
        """pipeline module should import without errors."""
        from src.pipeline import run_pipeline

    def test_import_utils(self):
        """utils module should import without errors."""
        from src.utils import get_logger, timer, safe_execute


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
