""" 
test_svm.py — smoke tests for the LinearSVC pipeline

Verifies that the shared SVM pipeline can:
- fit successfully on a minimal dataset,
- generate valid binary predictions,
- expose a decision_function score array for ranking-based evaluation.

The test checks decision_function instead of predict_proba because
LinearSVC does not provide calibrated class probabilities.
"""

import numpy as np
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
