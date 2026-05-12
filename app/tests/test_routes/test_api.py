"""Tests for API routes."""
import pytest
import json


class TestApiModelsEndpoint:
    """Test GET /api/models endpoint."""

    def test_list_models_success(self, client, monkeypatch_data_path, sample_evaluations):
        """Test successful models list endpoint."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.get_json()
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["model_id"] == "model_001"
        assert data["models"][0]["model_name"] == "Logistic Regression"

    def test_list_models_response_structure(self, client, monkeypatch_data_path):
        """Test that models list has correct structure."""
        response = client.get("/api/models")
        data = response.get_json()
        for model in data["models"]:
            assert "model_id" in model
            assert "model_name" in model
            assert "version" in model
            assert "metrics" in model

    def test_list_models_empty(self, client, monkeypatch, tmp_path):
        """Test models list when no models exist."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({"models": []}))
        monkeypatch.setattr(
            "app.db.queries.DATA_PATH",
            empty_file
        )
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.get_json()
        assert data["models"] == []


class TestApiModelEvaluationEndpoint:
    """Test GET /api/models/<model_id>/evaluation endpoint."""

    def test_get_model_evaluation_by_id(self, client, monkeypatch_data_path):
        """Test retrieving evaluation for a specific model."""
        response = client.get("/api/models/model_001/evaluation")
        assert response.status_code == 200
        data = response.get_json()
        assert data["model_id"] == "model_001"
        assert data["metrics"]["accuracy"] == 0.85

    def test_get_model_evaluation_nonexistent(self, client, monkeypatch_data_path):
        """Test retrieving evaluation for non-existent model."""
        response = client.get("/api/models/nonexistent/evaluation")
        assert response.status_code == 200
        data = response.get_json()
        assert data == {}

    def test_get_model_evaluation_response_structure(self, client, monkeypatch_data_path):
        """Test evaluation response structure."""
        response = client.get("/api/models/model_002/evaluation")
        data = response.get_json()
        assert "model_id" in data
        assert "model_name" in data
        assert "version" in data
        assert "metrics" in data


class TestApiDefaultEvaluationEndpoint:
    """Test GET /api/evaluation endpoint."""

    def test_default_evaluation_with_model_id(self, client, monkeypatch_data_path):
        """Test default evaluation endpoint with model_id parameter."""
        response = client.get("/api/evaluation?model_id=model_001")
        assert response.status_code == 200
        data = response.get_json()
        assert data["model_id"] == "model_001"

    def test_default_evaluation_without_model_id(self, client, monkeypatch_data_path):
        """Test default evaluation endpoint without model_id parameter."""
        response = client.get("/api/evaluation")
        assert response.status_code == 200
        data = response.get_json()
        # When no model_id is provided, returns None which defaults to first model
        assert data["model_id"] == "model_001"

    def test_default_evaluation_nonexistent_model(self, client, monkeypatch_data_path):
        """Test default evaluation with non-existent model_id."""
        response = client.get("/api/evaluation?model_id=fake_model")
        assert response.status_code == 200
        data = response.get_json()
        assert data == {}

    def test_api_returns_json(self, client, monkeypatch_data_path):
        """Test that API endpoints return JSON content type."""
        response = client.get("/api/models")
        assert response.content_type == "application/json"
