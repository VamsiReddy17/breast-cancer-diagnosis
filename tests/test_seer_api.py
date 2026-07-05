"""
Tests for SEER API Routes — Breast Cancer Diagnosis
===================================================
Validates SEER clinical endpoints and prediction logic.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api import app

client = TestClient(app)


def test_get_seer_data_paginated():
    """Verify that get_seer_data retrieves paginated results."""
    response = client.get("/api/seer/data?page=1&size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["size"] == 5
    assert len(data["data"]) == 5
    assert "target" in data["data"][0]


def test_get_seer_data_search():
    """Verify search filter functions correctly on SEER."""
    response = client.get("/api/seer/data?page=1&size=5&search=White")
    assert response.status_code == 200


def test_get_seer_random_sample():
    """Verify SEER random sample prefill returns valid structure."""
    response = client.get("/api/seer/data/random")
    assert response.status_code == 200
    sample = response.json()
    assert "target" in sample
    assert "Age" in sample


def test_get_seer_eda_details():
    """Verify SEER EDA endpoints return lists of summaries and plots."""
    response = client.get("/api/seer/eda")
    assert response.status_code == 200
    res = response.json()
    assert "statistical_summary" in res
    assert "plots" in res
    assert len(res["plots"]) > 0


def test_get_seer_models_details():
    """Verify SEER model metrics comparisons return correctly."""
    response = client.get("/api/seer/models")
    assert response.status_code == 200
    res = response.json()
    assert "models" in res
    assert "overall_plots" in res
    assert len(res["models"]) > 0


def test_seer_prediction_endpoint():
    """Test SEER prediction endpoint with mock request."""
    # Fetch a real sample to use for testing
    sample_response = client.get("/api/seer/data/random")
    assert sample_response.status_code == 200
    sample = sample_response.json()
    
    # Extract target and ID
    sample.pop("target", None)
    sample.pop("id", None)

    # Test for each model
    for model in ["KNN", "Logistic Regression", "SVM", "Random Forest", "MLP"]:
        payload = {
            "model_name": model,
            "features": sample
        }
        response = client.post("/api/seer/predict", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert "prediction" in res
        assert "class_label" in res
        assert "confidence" in res
        assert res["prediction"] in [0, 1]
        assert res["class_label"] in ["Dead", "Alive"]
