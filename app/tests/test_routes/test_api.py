"""Tests for API routes."""
import pytest
import json
from unittest.mock import patch

SAMPLE_EVALUATIONS = {
    "models": [
        {
            "model_id": "model_001",
            "model_name": "Logistic Regression",
            "version": "1.0",
            "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85},
        },
        {
            "model_id": "model_002",
            "model_name": "Random Forest",
            "version": "1.0",
            "metrics": {"accuracy": 0.88, "precision": 0.86, "recall": 0.90, "f1_score": 0.88},
        },
    ]
}


@pytest.fixture
def mock_evaluations():
    """Patch evaluation functions at the import site in api.py."""
    def _get_model_evaluation(model_id=None):
        models = SAMPLE_EVALUATIONS["models"]
        if model_id is None:
            return models[0] if models else {}
        return next((m for m in models if m["model_id"] == model_id), {})

    with patch("app.routes.api.load_all_evaluations", return_value=SAMPLE_EVALUATIONS), \
         patch("app.routes.api.get_model_evaluation", side_effect=_get_model_evaluation), \
         patch("app.routes.api.get_model_artifacts", return_value={"images": {}}):
        yield


class TestApiModelsEndpoint:
    """Test GET /api/models endpoint."""

    def test_list_models_success(self, client, mock_evaluations):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.get_json()
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["model_id"] == "model_001"
        assert data["models"][0]["model_name"] == "Logistic Regression"

    def test_list_models_response_structure(self, client, mock_evaluations):
        response = client.get("/api/models")
        data = response.get_json()
        for model in data["models"]:
            assert "model_id" in model
            assert "model_name" in model
            assert "version" in model
            assert "metrics" in model

    def test_list_models_empty(self, client):
        with patch("app.routes.api.load_all_evaluations", return_value={"models": []}):
            response = client.get("/api/models")
            assert response.status_code == 400
            data = response.get_json()
            assert data["models"] == []

    def test_api_returns_json(self, client, mock_evaluations):
        response = client.get("/api/models")
        assert response.content_type == "application/json"


class TestApiModelEvaluationEndpoint:
    """Test GET /api/models/<model_id>/evaluation endpoint."""

    def test_get_model_evaluation_by_id(self, client, mock_evaluations):
        response = client.get("/api/models/model_001/evaluation")
        assert response.status_code == 200
        data = response.get_json()
        assert data["model_id"] == "model_001"
        assert data["metrics"]["accuracy"] == 0.85

    def test_get_model_evaluation_nonexistent(self, client, mock_evaluations):
        response = client.get("/api/models/nonexistent/evaluation")
        assert response.status_code == 200
        data = response.get_json()
        assert data == {}

    def test_get_model_evaluation_response_structure(self, client, mock_evaluations):
        response = client.get("/api/models/model_002/evaluation")
        data = response.get_json()
        assert "model_id" in data
        assert "model_name" in data
        assert "version" in data
        assert "metrics" in data


class TestApiDefaultEvaluationEndpoint:
    """Test GET /api/evaluation endpoint."""

    def test_default_evaluation_with_model_id(self, client, mock_evaluations):
        response = client.get("/api/evaluation?model_id=model_001")
        assert response.status_code == 200
        data = response.get_json()
        assert data["model_id"] == "model_001"

    def test_default_evaluation_without_model_id(self, client, mock_evaluations):
        response = client.get("/api/evaluation")
        assert response.status_code == 200
        data = response.get_json()
        assert data["model_id"] == "model_001"

    def test_default_evaluation_nonexistent_model(self, client, mock_evaluations):
        response = client.get("/api/evaluation?model_id=fake_model")
        assert response.status_code == 200
        data = response.get_json()
        assert data == {}
