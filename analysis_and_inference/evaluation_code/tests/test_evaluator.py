import importlib

import numpy as np
import pytest

from analysis_and_inference.evaluation_code.evaluator import evaluate_classification


# ---------------------------------------------------------------------------
# evaluate_classification
# ---------------------------------------------------------------------------

class TestEvaluateClassification:
    def test_perfect_predictions_give_all_metrics_one(self):
        # every metric collapses to 1.0 when there are no errors at all
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        result = evaluate_classification(y_true, y_pred, name="test")
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_return_dict_contains_expected_keys(self):
        # downstream code indexes into these keys — missing one would break silently
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = evaluate_classification(y_true, y_pred, name="test")
        assert {"accuracy", "f1", "precision", "recall", "classification_report"}.issubset(result.keys())

    def test_roc_pr_keys_absent_without_y_score(self):
        # curve metrics require probability scores — they must not appear when only hard labels are given
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_classification(y_true, y_pred, name="test")
        assert "roc_auc" not in result
        assert "pr_auc" not in result

    def test_roc_pr_keys_present_with_y_score(self):
        # passing probability scores should unlock the curve metrics in the returned dict
        y_true  = np.array([0, 1, 0, 1])
        y_pred  = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        # plot_curves=False: new evaluator only writes files when save_dir is also set,
        # but being explicit avoids any future change triggering file I/O in tests
        result = evaluate_classification(y_true, y_pred, y_score, name="test", plot_curves=False)
        assert "roc_auc" in result
        assert "pr_auc" in result

    def test_imbalanced_labels_do_not_crash(self):
        # zero_division=0 should absorb the 0/0 precision when no positives are predicted
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        evaluate_classification(y_true, y_pred, name="test")


# ---------------------------------------------------------------------------
# Model module resolution
# (previously tested model_run registry; run_all.py replaced that pattern —
#  models are now standalone modules, so an invalid name raises ModuleNotFoundError)
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_invalid_model_name_raises_module_not_found(self):
        # a misspelled model slug should fail loudly at import rather than silently do nothing
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("analysis_and_inference.models.nonexistent_model.nonexistent_model")
