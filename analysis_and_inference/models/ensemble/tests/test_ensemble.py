"""
test_ensemble.py - smoke test for the prefit VotingClassifier ensemble

The production ensemble uses a non-standard sklearn workaround where already-
trained estimators are manually injected into a VotingClassifier without
calling fit() again. This avoids redundant retraining, overwriting tuned pipelines, unnecessary computation.

The test uses three deterministic DummyClassifier models so the expected
majority-vote output is known in advance. This isolates and validates the
ensemble stitching logic independently from model complexity.

This test confirms that:
- prefit estimators can be combined successfully,
- VotingClassifier predictions run without fit(),
- outputs remain valid binary labels,
- majority voting behaves correctly.

Run with: uv run pytest test/test_ensemble.py -v
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder


def test_voting_classifier_prefit_hack(tiny_data):
    """Verify that we can stitch already-fitted estimators into a VotingClassifier
    by setting the post-fit attributes directly, without calling .fit() again."""
    X, y = tiny_data

    # Three trivially fitted models - majority vote should be 1
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


def test_ensemble_soft_voting_predict_proba(tiny_data):
    # soft voting must expose predict_proba with valid probability values
    X, y = tiny_data

    m1 = DummyClassifier(strategy="constant", constant=1).fit(X, y)
    m2 = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    m3 = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    members = [("a", m1), ("b", m2), ("c", m3)]
    voter   = VotingClassifier(estimators=members, voting="soft")
    voter.estimators_       = [m for _, m in members]
    voter.named_estimators_ = dict(members)
    voter.le_               = LabelEncoder().fit([0, 1])
    voter.classes_          = voter.le_.classes_

    proba = voter.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_ensemble_reproducibility(tiny_data):
    # identical prefit members must produce identical predictions every time
    X, y = tiny_data

    def build_voter():
        members = [
            ("a", DummyClassifier(strategy="constant", constant=1).fit(X, y)),
            ("b", DummyClassifier(strategy="constant", constant=1).fit(X, y)),
            ("c", DummyClassifier(strategy="constant", constant=1).fit(X, y)),
        ]
        voter = VotingClassifier(estimators=members, voting="hard")
        voter.estimators_       = [m for _, m in members]
        voter.named_estimators_ = dict(members)
        voter.le_               = LabelEncoder().fit([0, 1])
        voter.classes_          = voter.le_.classes_
        return voter

    assert np.array_equal(build_voter().predict(X), build_voter().predict(X))


def test_ensemble_all_zero_labels():
    # covers the edge case where the training set has no positive class at all
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = np.zeros(20, dtype=int)

    members = [
        ("a", DummyClassifier(strategy="constant", constant=0).fit(X, y)),
        ("b", DummyClassifier(strategy="constant", constant=0).fit(X, y)),
        ("c", DummyClassifier(strategy="constant", constant=0).fit(X, y)),
    ]
    voter = VotingClassifier(estimators=members, voting="hard")
    voter.estimators_       = [m for _, m in members]
    voter.named_estimators_ = dict(members)
    voter.le_               = LabelEncoder().fit([0, 1])
    voter.classes_          = voter.le_.classes_

    preds = voter.predict(X)
    assert len(preds) == 20
