"""
test_baseline.py - smoke tests for the baseline DummyClassifier model

This script validates that the project's baseline classifier can successfully train,
generate binary toxicity predictions, and output valid probability estimates on a minimal
test dataset.

Note: These are smoke tests rather than performance tests. The goal is simply
to confirm that the baseline model pipeline executes correctly and returns
properly formatted outputs.

Run with: uv run pytest test/test_baseline.py -v
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier


def test_dummy_classifier_fits_and_predicts(tiny_data):
    X, y = tiny_data
    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_baseline_reproducibility_with_random_state():
    # same random_state must produce identical predictions across two fits
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)

    m1 = DummyClassifier(strategy="stratified", random_state=42)
    m1.fit(X, y)

    m2 = DummyClassifier(strategy="stratified", random_state=42)
    m2.fit(X, y)

    assert np.array_equal(m1.predict(X), m2.predict(X))


def test_baseline_all_zero_labels():
    # covers the edge case where the training set has no positive class at all
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = np.zeros(20, dtype=int)

    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 20
