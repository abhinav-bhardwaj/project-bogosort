"""Smoke test for the baseline DummyClassifier model."""

import numpy as np
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
