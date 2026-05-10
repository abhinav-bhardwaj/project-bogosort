"""
_common.py — shared infrastructure for model training and evaluation

This module centralizes all reusable training logic so individual model files
remain minimal and focused only on model-specific behavior.

Provides:
- `find_project_root()`        : depth-independent walk-up to the project root
- `load_split()`               : the cached train/test split as four objects
- `precompute_features()`      : dense features computed once, cached to disk
- `load_scaler()`              : retrieve the fitted StandardScaler from cache
- `load_feature_names()`       : retrieve the feature column names from cache
- `make_pipeline(clf)`         : simple pipeline wrapping just the classifier
- `save_outputs(...)`          : persist the fitted pipeline and GridSearch results CSV
- `run_grid_search(...)`       : end-to-end GridSearchCV runner with optional threshold tuning

Every model file in models/ uses these so the per-model code stays minimal.
Features are pre-computed once and reused across all models, avoiding redundant
DenseFeatureTransformer calls during GridSearchCV folds.

Imported internally by all model training scripts.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, FixedThresholdClassifier, cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.frozen import FrozenEstimator

from analysis_and_inference.features.build_features import DenseFeatureTransformer


def find_project_root(start=None):
    """Walk up from `start` (or this file) until a folder contains pyproject.toml."""
    p = Path(start or __file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    raise RuntimeError("Could not locate project root (no pyproject.toml found)")


PROJECT_ROOT  = find_project_root()
SPLIT_PATH    = "analysis_and_inference/models/split_and_features/split.pkl"
FEATURES_PATH = "analysis_and_inference/models/split_and_features/features.pkl"


def load_split():
    with open(SPLIT_PATH, "rb") as f:
        d = pickle.load(f)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]


def precompute_features(X_train, X_test):
    """Compute dense features + scaling once; save results.

    On first call: runs DenseFeatureTransformer on train and test, fits a
    StandardScaler on train, saves both arrays + feature names to FEATURES_PATH.
    Any subsequent calls: Just loads results.

    Returns: (X_train_feat, X_test_feat) — numpy arrays.
    Use load_feature_names() to get the feature names list.
    """
    if os.path.exists(FEATURES_PATH):
        print(f"[skip] {FEATURES_PATH} already exists — loading saved features")
        with open(FEATURES_PATH, "rb") as f:
            d = pickle.load(f)
        return d["X_train"], d["X_test"]

    print("Pre-computing dense features (runs once, then cached)...")
    dense  = DenseFeatureTransformer()
    scaler = StandardScaler()

    X_train_dense = dense.transform(X_train)
    X_test_dense  = dense.transform(X_test)
    feature_names = list(X_train_dense.columns)

    X_train_feat = scaler.fit_transform(X_train_dense)
    X_test_feat  = scaler.transform(X_test_dense)

    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump({
            "X_train":       X_train_feat,
            "X_test":        X_test_feat,
            "feature_names": feature_names,
            "scaler":        scaler,
        }, f)
    print(f"Saved features to {FEATURES_PATH}")

    return X_train_feat, X_test_feat


def load_scaler():
    """Return the fitted StandardScaler from the cached features.pkl."""
    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)["scaler"]


def load_feature_names():
    """Return the list of feature names from the cached features.pkl.

    Falls back to computing from a single-row dense transform if the cache
    predates the feature_names addition.
    """
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "rb") as f:
            d = pickle.load(f)
        if "feature_names" in d:
            return d["feature_names"]
    # Fallback: compute from a fresh DenseFeatureTransformer
    X_train, _, _, _ = load_split()
    return list(DenseFeatureTransformer().transform(X_train.head(1)).columns)


def make_pipeline(clf):
    """Minimal pipeline: just the classifier.
    Features are pre-computed and scaled externally via precompute_features()."""
    return Pipeline([("clf", clf)])


def save_outputs(name, output_dir, fitted_pipeline, cv_results=None):
    """Save the fitted pipeline + GridSearch results CSV (if provided)."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{name}_tuned.pkl"), "wb") as f:
        pickle.dump(fitted_pipeline, f)
    print(f"Saved {name}_tuned.pkl to {output_dir}/")

    if cv_results is not None:
        pd.DataFrame(cv_results).to_csv(
            os.path.join(output_dir, f"{name}_tuning_results.csv"), index=False,
        )
        print(f"Saved {name}_tuning_results.csv to {output_dir}/")


def _wrap_with_tuned_threshold(estimator, X_train, y_train, cv=3):
    """Find F1-optimal threshold via out-of-fold CV, wrap in FixedThresholdClassifier.

    The wrapped estimator delegates predict_proba unchanged (so ensemble soft-voting and
    SHAP still work) but applies the tuned threshold inside predict().
    """
    proba_oof = cross_val_predict(
        estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, proba_oof)
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1)])
    print(f"Tuned threshold: {best_threshold:.4f}  (F1={f1.max():.4f} on OOF predictions)")
    return FixedThresholdClassifier(
        estimator=FrozenEstimator(estimator),
        threshold=best_threshold,
        response_method="predict_proba",
    )


def run_grid_search(name, output_dir, classifier, param_grid,
                    friendly_name=None, scoring="average_precision",
                    cv=3, tune_threshold=True):
    """End-to-end GridSearchCV runner: load data, search, tune threshold, evaluate, save.

    Parameters
    ----------
    name : str — short slug used for filenames (e.g., "lasso_log_reg")
    output_dir : str — where to write tuned pkl, tuning CSV, and evaluation/ subfolder
    classifier : sklearn estimator (or pipeline) to be searched
    param_grid : dict for GridSearchCV
    friendly_name : str — pretty name used in plot titles. Defaults to `name`.
    scoring : str — sklearn scoring metric for the search. Defaults to average_precision.
    cv : int — number of CV folds.
    tune_threshold : bool — if True (and the estimator supports predict_proba), wrap the
        best estimator in a FixedThresholdClassifier with an F1-optimal threshold.
    """
    # Local imports to avoid forcing every consumer of _common.py to pull in
    # the evaluator (and matplotlib) at import time.
    from analysis_and_inference.evaluation_code.evaluator import evaluate_classification

    X_train, X_test, y_train, y_test = load_split()
    X_train_feat, X_test_feat = precompute_features(X_train, X_test)
    feature_names = load_feature_names()

    search = GridSearchCV(classifier, param_grid, scoring=scoring,
                          cv=cv, n_jobs=-1, verbose=1)
    search.fit(X_train_feat, y_train)

    print(f"Best params  : {search.best_params_}")
    print(f"Best CV {scoring}: {search.best_score_:.4f}")

    best = search.best_estimator_
    if tune_threshold and hasattr(best, "predict_proba"):
        best = _wrap_with_tuned_threshold(best, X_train_feat, y_train, cv=cv)

    y_pred  = best.predict(X_test_feat)
    if hasattr(best, "predict_proba"):
        y_score = best.predict_proba(X_test_feat)[:, 1]
    else:
        y_score = best.decision_function(X_test_feat)

    evaluate_classification(
        y_test.values, y_pred, y_score,
        name=friendly_name or name,
        plot_curves=True,
        save_dir=os.path.join(output_dir, "evaluation"),
        model=best,
        feature_names=feature_names,
    )
    save_outputs(name, output_dir, best, search.cv_results_)
    return search
