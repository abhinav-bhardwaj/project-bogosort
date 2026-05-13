"""
test_ridge.py — smoke test for the ridge logistic regression pipeline

Checks that the model integrates correctly with the shared pipeline utilities,
fits successfully, and produces valid predictions and probability outputs.

"""

import numpy as np
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
