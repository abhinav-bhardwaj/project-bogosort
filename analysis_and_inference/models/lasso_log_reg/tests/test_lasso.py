"""Smoke test for the custom Lasso pipeline."""

import numpy as np

from analysis_and_inference.models._common import make_pipeline
from analysis_and_inference.models.lasso_log_reg.core_logistic_regression_lasso import LassoLogisticRegression


def test_lasso_pipeline_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = make_pipeline(LassoLogisticRegression(alpha=0.01, learning_rate=0.1, max_iter=200))
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
