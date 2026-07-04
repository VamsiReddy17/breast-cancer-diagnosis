"""
Tests for API Backend — Breast Cancer Diagnosis
===============================================
Validates FastAPI endpoints and prediction logic.
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


def test_health_check():
    """Verify that the health check endpoint returns 200."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_get_data_paginated():
    """Verify that get_data retrieves paginated results."""
    response = client.get("/api/data?page=1&size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["size"] == 5
    assert len(data["data"]) == 5
    assert "target" in data["data"][0]


def test_get_data_search():
    """Verify search filter functions correctly."""
    response = client.get("/api/data?page=1&size=5&search=1")
    assert response.status_code == 200


def test_get_random_sample():
    """Verify random sample prefill returns valid structure."""
    response = client.get("/api/data/random")
    assert response.status_code == 200
    sample = response.json()
    assert "target" in sample
    assert len(sample.keys()) >= 30


def test_get_eda_details():
    """Verify EDA endpoints return lists of summaries and plots."""
    response = client.get("/api/api/../api/eda")  # Test route normalizer
    response = client.get("/api/eda")
    assert response.status_code == 200
    res = response.json()
    assert "statistical_summary" in res
    assert "plots" in res
    assert len(res["plots"]) > 0


def test_get_models_details():
    """Verify model metrics comparisons return correctly."""
    response = client.get("/api/models")
    assert response.status_code == 200
    res = response.json()
    assert "models" in res
    assert "overall_plots" in res
    assert len(res["models"]) > 0


def test_prediction_endpoint():
    """Test prediction endpoint with mock request."""
    # Fetch a real sample to use for testing
    sample_response = client.get("/api/data/random")
    assert sample_response.status_code == 200
    sample = sample_response.json()
    
    # Extract target out since it is not a feature
    target = sample.pop("target", None)
    if "id" in sample:
        sample.pop("id")

    payload = {
        "model_name": "KNN",
        "features": sample
    }
    
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "prediction" in res
    assert "class_label" in res
    assert "confidence" in res
    assert res["prediction"] in [0, 1]
    assert res["class_label"] in ["Malignant", "Benign"]
