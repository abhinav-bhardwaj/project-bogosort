"""
generate_evaluations_json.py — post-training step that writes app/data/model_evaluations.json.

Reads already-trained pkl files and tuning CSVs (produced by run_all.py) and
computes fresh metrics on the held-out test set. Does not retrain anything.
Called automatically at the end of run_all.py, or run standalone:

    uv run python analysis_and_inference/models/generate_evaluations_json.py
"""

import json
import os
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = next(
    str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path.insert(0, PROJECT_ROOT)

from analysis_and_inference.models._common import (
    load_feature_names,
    load_split,
    precompute_features,
)

OUTPUT_JSON = "app/data/model_evaluations.json"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
VERSION = "1.0.0"
DATASET = "jigsaw-toxicity"

MODEL_CONFIGS = [
    {
        "model_id": "lasso_log_reg",
        "model_name": "Lasso Logistic Regression",
        "pkl": "analysis_and_inference/models/lasso_log_reg/outputs/lasso_log_reg_tuned.pkl",
        "tuning_csv": "analysis_and_inference/models/lasso_log_reg/outputs/lasso_log_reg_tuning_results.csv",
    },
    {
        "model_id": "random_forest",
        "model_name": "Random Forest",
        "pkl": "analysis_and_inference/models/random_forest/outputs/random_forest_tuned.pkl",
        "tuning_csv": "analysis_and_inference/models/random_forest/outputs/random_forest_tuning_results.csv",
    },
    {
        "model_id": "ridge_log_reg",
        "model_name": "Ridge Logistic Regression",
        "pkl": "analysis_and_inference/models/ridge_log_reg/outputs/ridge_log_reg_tuned.pkl",
        "tuning_csv": "analysis_and_inference/models/ridge_log_reg/outputs/ridge_log_reg_tuning_results.csv",
    },
    {
        "model_id": "svm",
        "model_name": "Support Vector Machine",
        "pkl": "analysis_and_inference/models/svm/outputs/svm_tuned.pkl",
        "tuning_csv": "analysis_and_inference/models/svm/outputs/svm_tuning_results.csv",
    },
    {
        "model_id": "ensemble",
        "model_name": "Ensemble (Lasso + RF + Ridge)",
        "pkl": "analysis_and_inference/models/ensemble/outputs/ensemble_soft_vote_tuned.pkl",
        "tuning_csv": None,  # ensemble skips run_grid_search, so no tuning CSV
    },
]


def _unwrap_to_base(model):
    """Peel FixedThresholdClassifier → FrozenEstimator → Pipeline to reach the base estimator."""
    inner = model
    for _ in range(3):
        estimator = getattr(inner, "estimator", None)
        if estimator is not None:
            inner = estimator
        else:
            break
    return inner


def _get_importances(model):
    """Return (values_array, kind) or (None, None). Handles wrapped estimators."""
    base = _unwrap_to_base(model)
    for cand in [model, base]:
        named = getattr(cand, "named_steps", None)
        clf = named.get("clf") if named else None
        for obj in ([clf] if clf else []) + [cand]:
            if obj is None:
                continue
            if hasattr(obj, "feature_importances_"):
                return np.asarray(obj.feature_importances_), "importance"
            if hasattr(obj, "coef_"):
                return np.asarray(obj.coef_).ravel(), "coef"
    return None, None


def _top_perturbation(model, feature_names, top_n=3):
    values, _ = _get_importances(model)
    if values is None or not feature_names:
        return []
    order = np.argsort(np.abs(values))[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": round(float(abs(values[i])), 4)}
        for i in order
    ]


def _best_hyperparams(tuning_csv_path):
    if not tuning_csv_path or not os.path.exists(tuning_csv_path):
        return {}
    df = pd.read_csv(tuning_csv_path)
    best = df[df["rank_test_score"] == 1].iloc[0]

    params = {}
    for col in df.columns:
        if not col.startswith("param_"):
            continue
        key = col.removeprefix("param_").replace("clf__", "")
        val = best[col]
        if pd.isna(val):
            params[key] = None
        elif isinstance(val, float) and val.is_integer():
            params[key] = int(val)
        else:
            params[key] = val

    params["val_pr_auc_mean"] = round(float(best["mean_test_score"]), 4)
    params["val_pr_auc_std"] = round(float(best["std_test_score"]), 6)
    return params


def _compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }
    if y_score is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_score)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_test, y_score)), 4)

    return metrics, cm.tolist()


def main():
    os.chdir(PROJECT_ROOT)

    X_train, X_test, _, y_test = load_split()
    _, X_test_feat = precompute_features(X_train, X_test)
    feature_names = load_feature_names()

    entries = []
    today = date.today().isoformat()

    for cfg in MODEL_CONFIGS:
        if not os.path.exists(cfg["pkl"]):
            print(f"[skip] {cfg['model_id']}: pkl not found at {cfg['pkl']}")
            continue

        print(f"Evaluating {cfg['model_id']}...")
        with open(cfg["pkl"], "rb") as f:
            model = pickle.load(f)

        metrics, cm = _compute_metrics(model, X_test_feat, y_test.values)
        hyperparams = _best_hyperparams(cfg.get("tuning_csv"))
        perturbation = _top_perturbation(model, feature_names)

        entries.append({
            "model_id": cfg["model_id"],
            "model_name": cfg["model_name"],
            "version": VERSION,
            "dataset": DATASET,
            "updated_at": today,
            "metrics": metrics,
            "confusion_matrix": cm,
            "roc_curve": {"fpr": [], "tpr": [], "image_url": ""},
            "pr_curve": {"recall": [], "precision": [], "image_url": ""},
            "hyperparams": hyperparams,
            "perturbation": perturbation,
        })
        print(f"  accuracy={metrics['accuracy']}  f1={metrics['f1']}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"models": entries}, f, indent=2, cls=_NumpyEncoder)
    print(f"Wrote {len(entries)} models → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
