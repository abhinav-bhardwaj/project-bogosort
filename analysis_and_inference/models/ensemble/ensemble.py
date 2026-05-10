"""
ensemble.py — soft-voting ensemble classifier for toxicity detection

The script combines predictions from multiple independently trained classification
models into a single ensemble using probability-based soft voting.

The ensemble aggregates model confidence scores rather than relying on
hard majority voting, allowing stronger individual predictions to have
greater influence on the final toxicity classification.


The ensemble intentionally mixes different model families:
- L1 logistic regression for sparse feature selection,
- ridge logistic regression for stable linear boundaries,
- random forest for nonlinear feature interactions.

Using heterogeneous learners improves robustness because each model captures
different toxicity patterns in the engineered feature space.

Reasoning behind soft voting: 
Soft voting averages predicted probabilities rather than hard class labels.
This preserves confidence information and produces smoother decisions on
ambiguous or borderline toxic comments.

The approach is especially useful for imbalanced toxicity data, where model
confidence is often more informative than majority agreement alone.

Architectural decisions:
Models are trained and serialized independently, then loaded dynamically into
the ensemble. This modular design allows individual models to be:
- tested separately,
- replaced easily,
- added or removed without changing ensemble logic.

A dictionary maps model names to saved pipeline paths because it provides:
- explicit configuration,
- constant-time lookup,
- simple extensibility.

Note: VotingClassifier does not natively support prefit estimators, so fitted
pipelines are manually injected into the ensemble object. This avoids
retraining and preserves the exact tuned models used during evaluation.

The ensemble is evaluated with the shared evaluator to ensure metrics remain
directly comparable with standalone models. Feature importance plots are
intentionally skipped because a voting ensemble has no single interpretable
coefficient structure.

Generated artifacts are stored in:

    analysis_and_inference/models/ensemble/outputs/

Run with: uv run python analysis_and_inference/models/ensemble.py

"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from analysis_and_inference.models._common import load_split, precompute_features, save_outputs
from analysis_and_inference.evaluation_code.evaluator import evaluate_classification


NAME       = "ensemble_soft_vote"
OUTPUT_DIR = "analysis_and_inference/models/ensemble/outputs"

# Each entry: name → path of the pipeline saved by that model's individual run
# Note: SVM excluded (LinearSVC has no predict_proba, incompatible with soft voting)
MEMBER_PATHS = {
    "lasso_log_reg": "analysis_and_inference/models/lasso_log_reg/outputs/lasso_log_reg_tuned.pkl",
    "random_forest": "analysis_and_inference/models/random_forest/outputs/random_forest_tuned.pkl",
    "ridge_log_reg": "analysis_and_inference/models/ridge_log_reg/outputs/ridge_log_reg_tuned.pkl",
}


def run():
    X_train, X_test, _, y_test = load_split()
    _, X_test_feat = precompute_features(X_train, X_test)

    # Load already-fitted pipelines
    members = []
    for name, path in MEMBER_PATHS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: no artifact at {path}")
            continue
        with open(path, "rb") as f:
            members.append((name, pickle.load(f)))
        print(f"[loaded] {name}")

    if not members:
        raise RuntimeError("No member pipelines found — run the individual model files first.")

    # sklearn's VotingClassifier doesn't natively accept prefit estimators,
    # so bypass .fit() by setting the post-fit attributes directly.
    voter = VotingClassifier(estimators=members, voting="soft")
    voter.estimators_       = [m for _, m in members]
    voter.named_estimators_ = dict(members)
    voter.le_               = LabelEncoder().fit([0, 1])
    voter.classes_          = voter.le_.classes_

    y_pred = voter.predict(X_test_feat)
    y_score = voter.predict_proba(X_test_feat)[:, 1]

    print(f"\nSoft vote across {len(members)} models: {[n for n, _ in members]}")

    evaluate_classification(
        y_test.values, y_pred, y_score,
        name="Ensemble (soft vote)",
        plot_curves=True,
        save_dir=os.path.join(OUTPUT_DIR, "evaluation"),
        # VotingClassifier exposes no single coef_/feature_importances_, so the
        # importance plot will be skipped automatically.
    )

    save_outputs(NAME, OUTPUT_DIR, voter)


if __name__ == "__main__":
    run()
