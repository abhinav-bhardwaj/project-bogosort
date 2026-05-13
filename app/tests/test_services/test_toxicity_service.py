"""Tests for toxicity_service module."""
import pytest
from unittest.mock import patch, MagicMock
from app.services.toxicity_service import score_comment, DEFAULT_MODEL


class TestScoreComment:
    """Tests for score_comment function."""

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_success_with_explain(self, mock_predict):
        """Test successful scoring with explanation."""
        mock_predict.return_value = {
            "label": 1,
            "probability": 0.85,
            "top_features": [("word1", 0.5), ("word2", 0.3)],
            "explain_version": "v1",
        }

        result = score_comment("This is a toxic comment", model_name="ensemble", explain=True)

        assert result["label"] == 1
        assert result["probability"] == 0.85
        assert result["top_features"] == [("word1", 0.5), ("word2", 0.3)]
        assert result["explain_version"] == "v1"
        mock_predict.assert_called_once()

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_success_without_explain(self, mock_predict):
        """Test successful scoring without explanation."""
        mock_predict.return_value = {
            "label": 0,
            "probability": 0.2,
            "top_features": [],
            "explain_version": "v1",
        }

        result = score_comment("This is fine", model_name="ensemble", explain=False)

        assert result["label"] == 0
        assert result["probability"] == 0.2
        mock_predict.assert_called_once_with("This is fine", model_name="ensemble", explain=False)

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_uses_default_model(self, mock_predict):
        """Test that DEFAULT_MODEL is used when not specified."""
        mock_predict.return_value = {
            "label": 0,
            "probability": 0.1,
            "top_features": [],
            "explain_version": "v1",
        }

        score_comment("Text", model_name=None)

        # Should use DEFAULT_MODEL
        call_args = mock_predict.call_args
        assert call_args[1].get("model_name") == DEFAULT_MODEL

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_different_models(self, mock_predict):
        """Test scoring with different model names."""
        mock_predict.return_value = {
            "label": 1,
            "probability": 0.8,
            "top_features": [],
            "explain_version": "v1",
        }

        models = ["ensemble", "random_forest", "lasso_log_reg", "svm", "ridge_log_reg"]
        for model in models:
            score_comment("Text", model_name=model)
            # Verify model was passed correctly
            call_args = mock_predict.call_args[1]
            assert call_args.get("model_name") == model

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_with_empty_text(self, mock_predict):
        """Test scoring with empty text."""
        mock_predict.return_value = {
            "label": 0,
            "probability": 0.0,
            "top_features": [],
            "explain_version": "v1",
        }

        result = score_comment("", model_name="ensemble")

        assert result["probability"] == 0.0
        mock_predict.assert_called_once_with("", model_name="ensemble", explain=False)

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_with_very_long_text(self, mock_predict):
        """Test scoring with very long text."""
        long_text = "a" * 10000
        mock_predict.return_value = {
            "label": 1,
            "probability": 0.5,
            "top_features": [],
            "explain_version": "v1",
        }

        result = score_comment(long_text, model_name="ensemble")

        assert result["probability"] == 0.5
        mock_predict.assert_called_once()

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_with_special_characters(self, mock_predict):
        """Test scoring with special characters."""
        text = "This has !@#$%^&*() special chars"
        mock_predict.return_value = {
            "label": 0,
            "probability": 0.3,
            "top_features": [],
            "explain_version": "v1",
        }

        result = score_comment(text, model_name="ensemble")

        assert result["probability"] == 0.3
        mock_predict.assert_called_once_with(text, model_name="ensemble", explain=False)

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_probability_boundaries(self, mock_predict):
        """Test with probability at boundaries (0.0, 1.0, 0.5)."""
        for prob in [0.0, 0.5, 1.0]:
            mock_predict.return_value = {
                "label": 1 if prob > 0.5 else 0,
                "probability": prob,
                "top_features": [],
                "explain_version": "v1",
            }

            result = score_comment("Text", model_name="ensemble")

            assert result["probability"] == prob

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_different_labels(self, mock_predict):
        """Test with different label values."""
        for label in [0, 1, 2]:
            mock_predict.return_value = {
                "label": label,
                "probability": 0.5,
                "top_features": [],
                "explain_version": "v1",
            }

            result = score_comment("Text", model_name="ensemble")

            assert result["label"] == label

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_exception_handling(self, mock_predict):
        """Test that exceptions return safe defaults."""
        mock_predict.side_effect = Exception("Model error")

        result = score_comment("Text", model_name="ensemble")

        # Should return safe defaults instead of raising
        assert "probability" in result
        assert "label" in result

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_with_unicode(self, mock_predict):
        """Test scoring with unicode characters."""
        text = "こんにちは世界 🌍 مرحبا"
        mock_predict.return_value = {
            "label": 0,
            "probability": 0.1,
            "top_features": [],
            "explain_version": "v1",
        }

        result = score_comment(text, model_name="ensemble")

        assert result["probability"] == 0.1
        mock_predict.assert_called_once_with(text, model_name="ensemble", explain=False)

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_empty_features_list(self, mock_predict):
        """Test with empty features list."""
        mock_predict.return_value = {
            "label": 0,
            "probability": 0.2,
            "top_features": [],
            "explain_version": "v1",
        }

        result = score_comment("Text", model_name="ensemble", explain=True)

        assert result["top_features"] == []

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_malformed_features_missing_key(self, mock_predict):
        """Test handling of malformed response (missing explain_version)."""
        mock_predict.return_value = {
            "label": 1,
            "probability": 0.8,
            "top_features": [("word", 0.5)],
            # Missing explain_version - service should handle gracefully
        }

        result = score_comment("Text", model_name="ensemble")

        # Should still return valid response even if explain_version missing
        assert result["label"] == 1
        assert result["probability"] == 0.8

    @patch("app.services.toxicity_service.predict_comment")
    def test_score_comment_respects_explain_flag(self, mock_predict):
        """Test that explain flag controls feature extraction."""
        mock_predict.return_value = {
            "label": 1,
            "probability": 0.85,
            "top_features": [("word1", 0.5)],
            "explain_version": "v1",
        }

        # With explain=True
        result_with_explain = score_comment("Text", explain=True)
        assert mock_predict.call_args[1].get("explain") is True

        # With explain=False
        result_without_explain = score_comment("Text", explain=False)
        assert mock_predict.call_args[1].get("explain") is False
