"""Smoke test for the random forest pipeline."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from analysis_and_inference.models._common import make_pipeline


def test_random_forest_pipeline_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = make_pipeline(RandomForestClassifier(
        n_estimators=10, random_state=42, n_jobs=1, class_weight="balanced",
    ))
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
