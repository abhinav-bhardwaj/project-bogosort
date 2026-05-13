""" 
test_svm.py - smoke tests for the LinearSVC pipeline

Verifies that the shared SVM pipeline can:
- fit successfully on a minimal dataset,
- generate valid binary predictions,
- expose a decision_function score array for ranking-based evaluation.

The test checks decision_function instead of predict_proba because
LinearSVC does not provide calibrated class probabilities.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from analysis_and_inference.models._common import make_pipeline


def test_svm_pipeline_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = make_pipeline(LinearSVC(
        class_weight="balanced", random_state=42, max_iter=500,
    ))
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    # LinearSVC has no predict_proba, but decision_function should produce a 1D score array
    scores = pipe.decision_function(X)
    assert scores.shape == (len(y),)


def test_svm_reproducibility():
    # same random_state must produce identical predictions across two fits on the same data
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)

    def build_and_fit():
        pipe = make_pipeline(LinearSVC(class_weight="balanced", random_state=42, max_iter=500))
        pipe.fit(X, y)
        return pipe

    assert np.array_equal(build_and_fit().predict(X), build_and_fit().predict(X))


def test_svm_all_zero_labels():
    # LinearSVC requires at least 2 classes — single-class input must raise ValueError
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = np.zeros(20, dtype=int)

    pipe = make_pipeline(LinearSVC(class_weight="balanced", random_state=42, max_iter=500))
    with pytest.raises(ValueError):
        pipe.fit(X, y)
