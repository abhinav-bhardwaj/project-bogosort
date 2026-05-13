"""
test_ridge.py - smoke test for the ridge logistic regression pipeline

Checks that the model integrates correctly with the shared pipeline utilities,
fits successfully, and produces valid predictions and probability outputs.

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from analysis_and_inference.models._common import make_pipeline


def test_ridge_pipeline_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = make_pipeline(LogisticRegression(
        penalty="l2", solver="lbfgs", class_weight="balanced",
        max_iter=200, random_state=42,
    ))
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_ridge_reproducibility():
    # same random_state must produce identical predictions across two fits on the same data
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)

    def build_and_fit():
        pipe = make_pipeline(LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=200, random_state=42,
        ))
        pipe.fit(X, y)
        return pipe

    assert np.array_equal(build_and_fit().predict(X), build_and_fit().predict(X))


def test_ridge_all_zero_labels():
    # LogisticRegression requires at least 2 classes — single-class input must raise ValueError
    import pytest
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = np.zeros(20, dtype=int)

    pipe = make_pipeline(LogisticRegression(
        penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=200, random_state=42,
    ))
    with pytest.raises(ValueError, match="at least 2 classes"):
        pipe.fit(X, y)
