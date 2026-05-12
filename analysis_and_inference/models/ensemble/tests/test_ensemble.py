"""Smoke test for the VotingClassifier prefit hack used by the ensemble."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder


def test_voting_classifier_prefit_hack(tiny_data):
    """Verify that we can stitch already-fitted estimators into a VotingClassifier
    by setting the post-fit attributes directly, without calling .fit() again."""
    X, y = tiny_data

    # Three trivially fitted models — majority vote should be 1
    m1 = DummyClassifier(strategy="constant", constant=1).fit(X, y)
    m2 = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    m3 = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    members = [("a", m1), ("b", m2), ("c", m3)]
    voter   = VotingClassifier(estimators=members, voting="hard")

    # The hack
    voter.estimators_       = [m for _, m in members]
    voter.named_estimators_ = dict(members)
    voter.le_               = LabelEncoder().fit([0, 1])
    voter.classes_          = voter.le_.classes_

    preds = voter.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})
    # Majority of (1, 0, 1) → 1 for every row
    assert (preds == 1).all()
