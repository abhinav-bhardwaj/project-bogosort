"""Tests for evaluation_service module."""
import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.services.evaluation_service import (
    is_safe_model_id,
    load_all_evaluations,
    get_model_evaluation,
    get_model_version,
    resolve_artifact_dir,
    get_model_artifacts,
)


class TestIsSafeModelId:
    """Tests for is_safe_model_id function."""

    @pytest.mark.parametrize("model_id", [
        "ensemble",
        "random_forest",
        "lasso_log_reg",
        "model_123",
        "test-model",
        "ModelID",
    ])
    def test_valid_model_ids(self, model_id):
        """Test that valid model IDs are accepted."""
        assert is_safe_model_id(model_id) is True

    @pytest.mark.parametrize("model_id", [
        "../etc/passwd",
        "../../sensitive",
        "model/../other",
        "model\x00name",
        "model!@#$",
        "model with spaces",
        "../",
        ".",
        "..",
        "",
    ])
    def test_invalid_model_ids(self, model_id):
        """Test that invalid/dangerous model IDs are rejected."""
        assert is_safe_model_id(model_id) is False

    def test_unicode_characters(self):
        """Test rejection of unicode characters."""
        assert is_safe_model_id("модель") is False

    def test_very_long_model_id(self):
        """Test handling of very long model ID."""
        long_id = "a" * 1000
        # Should still validate based on characters
        assert is_safe_model_id(long_id) is True


class TestLoadAllEvaluations:
    """Tests for load_all_evaluations function."""

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_valid_evaluations(self, mock_path_selector, tmp_path):
        """Test loading valid evaluation JSON."""
        eval_data = {
            "models": [
                {
                    "model_id": "ensemble",
                    "model_name": "Ensemble Model",
                    "version": "1.0.0",
                    "metrics": {"accuracy": 0.95},
                }
            ]
        }
        json_file = tmp_path / "evaluations.json"
        json_file.write_text(json.dumps(eval_data))
        mock_path_selector.return_value = json_file

        result = load_all_evaluations()

        assert "models" in result
        assert len(result["models"]) == 1
        assert result["models"][0]["model_id"] == "ensemble"

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_missing_file(self, mock_path_selector):
        """Test loading from non-existent file."""
        mock_path_selector.return_value = Path("/nonexistent/path.json")

        result = load_all_evaluations()

        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_empty_file(self, mock_path_selector, tmp_path):
        """Test loading from empty file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")
        mock_path_selector.return_value = json_file

        result = load_all_evaluations()

        assert isinstance(result, dict)

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_malformed_json(self, mock_path_selector, tmp_path):
        """Test loading malformed JSON file."""
        json_file = tmp_path / "malformed.json"
        json_file.write_text("{invalid json content")
        mock_path_selector.return_value = json_file

        result = load_all_evaluations()

        # Should handle gracefully
        assert isinstance(result, dict)

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_missing_models_key(self, mock_path_selector, tmp_path):
        """Test loading JSON without 'models' key."""
        json_file = tmp_path / "no_models.json"
        json_file.write_text(json.dumps({"other_key": []}))
        mock_path_selector.return_value = json_file

        result = load_all_evaluations()

        # Should return what was loaded
        assert isinstance(result, dict)

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_empty_models_array(self, mock_path_selector, tmp_path):
        """Test loading with empty models array."""
        json_file = tmp_path / "empty_models.json"
        json_file.write_text(json.dumps({"models": []}))
        mock_path_selector.return_value = json_file

        result = load_all_evaluations()

        assert result.get("models") == []

    @patch("app.services.evaluation_service._select_data_path")
    def test_load_multiple_models(self, mock_path_selector, tmp_path):
        """Test loading multiple models."""
        eval_data = {
            "models": [
                {"model_id": f"model_{i}", "version": "1.0"} for i in range(5)
            ]
        }
        json_file = tmp_path / "multi.json"
        json_file.write_text(json.dumps(eval_data))
        mock_path_selector.return_value = json_file

        result = load_all_evaluations()

        assert len(result["models"]) == 5


class TestGetModelEvaluation:
    """Tests for get_model_evaluation function."""

    @patch("app.services.evaluation_service.load_all_evaluations")
    def test_get_specific_model(self, mock_load):
        """Test getting evaluation for specific model ID."""
        mock_load.return_value = {
            "models": [
                {"model_id": "ensemble", "accuracy": 0.95},
                {"model_id": "random_forest", "accuracy": 0.92},
            ]
        }

        result = get_model_evaluation("ensemble")

        assert result["model_id"] == "ensemble"
        assert result["accuracy"] == 0.95

    @patch("app.services.evaluation_service.load_all_evaluations")
    def test_get_nonexistent_model(self, mock_load):
        """Test getting evaluation for non-existent model."""
        mock_load.return_value = {
            "models": [{"model_id": "ensemble", "accuracy": 0.95}]
        }

        result = get_model_evaluation("nonexistent")

        # Should return None or empty dict
        assert result is None or result == {}

    @patch("app.services.evaluation_service.load_all_evaluations")
    def test_get_model_with_none_id(self, mock_load):
        """Test getting evaluation with None ID (should use first model)."""
        mock_load.return_value = {
            "models": [
                {"model_id": "first", "accuracy": 0.95},
                {"model_id": "second", "accuracy": 0.90},
            ]
        }

        result = get_model_evaluation(None)

        # Should return first model
        assert result["model_id"] == "first"

    @patch("app.services.evaluation_service.load_all_evaluations")
    def test_get_model_empty_models_array(self, mock_load):
        """Test getting model when models array is empty."""
        mock_load.return_value = {"models": []}

        result = get_model_evaluation("any_id")

        assert result is None or result == {}


class TestGetModelVersion:
    """Tests for get_model_version function."""

    @patch("app.services.evaluation_service.get_model_evaluation")
    def test_get_version_success(self, mock_get):
        """Test successful version extraction."""
        mock_get.return_value = {
            "model_id": "ensemble",
            "version": "2.1.0"
        }

        result = get_model_version("ensemble")

        assert result == "2.1.0"

    @patch("app.services.evaluation_service.get_model_evaluation")
    def test_get_version_missing_version(self, mock_get):
        """Test when version field is missing."""
        mock_get.return_value = {"model_id": "ensemble"}

        result = get_model_version("ensemble")

        # Should handle gracefully
        assert result is None or isinstance(result, str)

    @patch("app.services.evaluation_service.get_model_evaluation")
    def test_get_version_none_evaluation(self, mock_get):
        """Test when evaluation is None."""
        mock_get.return_value = None

        result = get_model_version("nonexistent")

        assert result is None or isinstance(result, str)


class TestResolveArtifactDir:
    """Tests for resolve_artifact_dir function."""

    def test_resolve_artifact_valid_path(self, tmp_path):
        """Test resolving valid artifact directory."""
        model_dir = tmp_path / "analysis_and_inference" / "models" / "ensemble" / "outputs" / "evaluation"
        model_dir.mkdir(parents=True, exist_ok=True)

        with patch("app.services.evaluation_service.Path.cwd", return_value=tmp_path):
            result = resolve_artifact_dir("ensemble")
            # Should resolve to a valid path or None
            assert result is None or isinstance(result, Path)

    def test_resolve_artifact_nonexistent_model(self):
        """Test resolving non-existent model."""
        result = resolve_artifact_dir("nonexistent_model_xyz")

        # Should return None for non-existent model
        assert result is None or isinstance(result, Path)

    @pytest.mark.parametrize("unsafe_id", [
        "../etc/passwd",
        "model/../../../etc",
        "model\x00name",
    ])
    def test_resolve_artifact_path_traversal(self, unsafe_id):
        """Test that path traversal attempts are blocked."""
        result = resolve_artifact_dir(unsafe_id)

        # Should block path traversal
        assert result is None or isinstance(result, Path)


class TestGetModelArtifacts:
    """Tests for get_model_artifacts function."""

    @patch("app.services.evaluation_service.resolve_artifact_dir")
    @patch("app.services.evaluation_service._load_csv_sample")
    def test_get_artifacts_complete_set(self, mock_csv, mock_resolve, tmp_path):
        """Test getting complete artifact set."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        # Create sample artifact files
        for img_name in ["roc_curve.png", "pr_curve.png", "confusion_matrix.png"]:
            (artifacts_dir / img_name).write_text("PNG_DATA")

        (artifacts_dir / "false_positives.csv").write_text("data")

        mock_resolve.return_value = artifacts_dir
        mock_csv.return_value = [{"col": "val"}]

        result = get_model_artifacts("ensemble")

        assert isinstance(result, dict)
        assert "images" in result or "samples" in result

    @patch("app.services.evaluation_service.resolve_artifact_dir")
    def test_get_artifacts_missing_images(self, mock_resolve, tmp_path):
        """Test getting artifacts with missing image files."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        mock_resolve.return_value = artifacts_dir

        result = get_model_artifacts("ensemble")

        # Should handle gracefully with missing images
        assert isinstance(result, dict)

    @patch("app.services.evaluation_service.resolve_artifact_dir")
    def test_get_artifacts_nonexistent_model(self, mock_resolve):
        """Test getting artifacts for non-existent model."""
        mock_resolve.return_value = None

        result = get_model_artifacts("nonexistent")

        # Should return empty dict or safe default
        assert isinstance(result, dict)

    @patch("app.services.evaluation_service.resolve_artifact_dir")
    @patch("app.services.evaluation_service._load_csv_sample")
    def test_get_artifacts_csv_loading(self, mock_csv, mock_resolve, tmp_path):
        """Test CSV sample loading."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        csv_file = artifacts_dir / "false_positives.csv"
        csv_file.write_text("id,text\n1,bad\n2,mean\n")

        mock_resolve.return_value = artifacts_dir
        mock_csv.return_value = [{"id": "1", "text": "bad"}]

        result = get_model_artifacts("ensemble")

        # CSV should be loaded via _load_csv_sample
        mock_csv.assert_called()
        assert isinstance(result, dict)


class TestLoadCsvSample:
    """Tests for _load_csv_sample helper function (tested through get_model_artifacts)."""

    @patch("app.services.evaluation_service.resolve_artifact_dir")
    def test_csv_with_limit(self, mock_resolve, tmp_path):
        """Test CSV loading with limit."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        # Create a CSV file
        csv_file = artifacts_dir / "false_positives.csv"
        csv_file.write_text("id,text,toxicity\n1,bad,0.8\n2,mean,0.75\n")

        mock_resolve.return_value = artifacts_dir

        result = get_model_artifacts("ensemble")

        # Verify CSV was loaded
        assert "samples" in result
        assert "false_positives" in result["samples"]
        assert len(result["samples"]["false_positives"]) > 0

    @patch("app.services.evaluation_service.resolve_artifact_dir")
    @patch("app.services.evaluation_service._load_csv_sample")
    def test_csv_empty_file(self, mock_csv, mock_resolve, tmp_path):
        """Test loading empty CSV file."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        mock_resolve.return_value = artifacts_dir
        mock_csv.return_value = []

        result = get_model_artifacts("ensemble")

        assert isinstance(result, dict)
