"""Tests for database queries."""
import pytest
import json
import tempfile
from pathlib import Path
from app.db.queries import load_all_evaluations, get_model_evaluation


class TestLoadAllEvaluations:
    """Test load_all_evaluations function."""

    def test_load_evaluations_success(self, monkeypatch_data_path, sample_evaluations):
        """Test loading evaluations from file."""
        data = load_all_evaluations()
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["model_id"] == "model_001"

    def test_load_evaluations_missing_file(self, monkeypatch, tmp_path):
        """Test loading when file doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr(
            "app.db.queries.DATA_PATH",
            nonexistent_path
        )
        data = load_all_evaluations()
        assert data == {"models": []}

    def test_load_evaluations_empty_file(self, monkeypatch, tmp_path):
        """Test loading from empty JSON file."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")
        monkeypatch.setattr(
            "app.db.queries.DATA_PATH",
            empty_file
        )
        data = load_all_evaluations()
        assert data == {}

    def test_load_evaluations_malformed_json(self, monkeypatch, tmp_path):
        """Test handling of malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json }")
        monkeypatch.setattr(
            "app.db.queries.DATA_PATH",
            bad_file
        )
        with pytest.raises(json.JSONDecodeError):
            load_all_evaluations()


class TestGetModelEvaluation:
    """Test get_model_evaluation function."""

    def test_get_model_by_id(self, monkeypatch_data_path, sample_evaluations):
        """Test retrieving a specific model by ID."""
        result = get_model_evaluation("model_001")
        assert result["model_id"] == "model_001"
        assert result["model_name"] == "Logistic Regression"
        assert result["metrics"]["accuracy"] == 0.85

    def test_get_model_not_found(self, monkeypatch_data_path):
        """Test retrieving a non-existent model returns empty dict."""
        result = get_model_evaluation("nonexistent_model")
        assert result == {}

    def test_get_first_model_when_no_id(self, monkeypatch_data_path, sample_evaluations):
        """Test getting first model when no ID provided."""
        result = get_model_evaluation()
        assert result["model_id"] == "model_001"

    def test_get_first_model_when_id_none(self, monkeypatch_data_path, sample_evaluations):
        """Test getting first model when ID is explicitly None."""
        result = get_model_evaluation(model_id=None)
        assert result["model_id"] == "model_001"

    def test_get_first_model_when_empty(self, monkeypatch, tmp_path):
        """Test getting first model from empty evaluations."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({"models": []}))
        monkeypatch.setattr(
            "app.db.queries.DATA_PATH",
            empty_file
        )
        result = get_model_evaluation()
        assert result == {}

    def test_get_model_metrics_structure(self, monkeypatch_data_path):
        """Test that returned model has correct metrics structure."""
        result = get_model_evaluation("model_002")
        assert "metrics" in result
        assert all(key in result["metrics"] for key in ["accuracy", "precision", "recall", "f1_score"])
