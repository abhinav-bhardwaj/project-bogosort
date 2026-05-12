"""
inference.py — single-comment inference utility for the Flask app

This module provides a lightweight inference layer for the Flask application,
allowing trained toxicity models to classify individual comments in real time. 
Usage:
    from analysis_and_inference.models.inference import predict_comment
    result = predict_comment("you are a complete idiot")
    # {
    #   "label":        1,
    #   "probability":  0.949,
    #   "top_features": [
    #       {"feature": "profanity_count", "value": 1.5, "shap": 0.21},
    #       ...
    #   ]
    # }

The model, scaler, and SHAP explainer are loaded once at first call and
cached for all subsequent requests, so Flask pays the one-time setup cost
only at startup.

The module returns both predictions and feature-level SHAP contributions so
users can understand which engineered signals influenced toxicity decisions.

Top features are ranked by absolute SHAP magnitude, making explanations easier
to interpret for borderline or ambiguous comments.

Implementation note: SHAP outputs vary across versions and model types, so _shap_for_class_1()
normalizes the output structure into a consistent 1D feature contribution array.

"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, PROJECT_ROOT)

from analysis_and_inference.features.build_features import DenseFeatureTransformer
from analysis_and_inference.models._common import (
    load_scaler, load_feature_names, FEATURES_PATH,
)

DEFAULT_MODEL    = "ensemble"
SHAP_BG_SIZE     = 30        # background sample size for KernelExplainer
SHAP_NSAMPLES    = 100       # samples used to estimate SHAP per row
DEFAULT_TOP_K    = 10        # how many top features to return

# Module-level cache — populated lazily, reused for every subsequent request
_cache: dict = {}


def _load(model_name: str, with_explainer: bool = False):
    """Load and cache model + scaler (+ SHAP explainer if requested)."""
    if model_name not in _cache:
        folder = os.path.join(PROJECT_ROOT, f"analysis_and_inference/models/{model_name}/outputs")
        pkls   = [f for f in os.listdir(folder) if f.endswith("_tuned.pkl")]
        if not pkls:
            raise FileNotFoundError(f"No *_tuned.pkl found in {folder}")
        with open(os.path.join(folder, pkls[0]), "rb") as f:
            _cache[model_name] = pickle.load(f)

    if "scaler" not in _cache:
        _cache["scaler"] = load_scaler()
    if "dense" not in _cache:
        _cache["dense"] = DenseFeatureTransformer()
    if "feature_names" not in _cache:
        _cache["feature_names"] = load_feature_names()

    if with_explainer and f"explainer:{model_name}" not in _cache:
        import shap
        # Background = small random sample of training features (fast & representative)
        with open(FEATURES_PATH, "rb") as f:
            X_train = pickle.load(f)["X_train"]
        background = shap.sample(X_train, SHAP_BG_SIZE, random_state=42)

        model = _cache[model_name]
        score_fn = model.predict_proba if hasattr(model, "predict_proba") else model.decision_function
        _cache[f"explainer:{model_name}"] = shap.KernelExplainer(score_fn, background)


def _shap_for_class_1(shap_values):
    """Normalise SHAP output across versions → (n_features,) array for the toxic class."""
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[..., 0]
    if sv.ndim == 2:
        sv = sv[0]
    return sv


def predict_comment(text: str,
                    model_name: str = DEFAULT_MODEL,
                    explain: bool = True,
                    top_k: int = DEFAULT_TOP_K) -> dict:
    """Classify a single comment and (optionally) return SHAP explanations.

    Parameters
    ----------
    text : str
        The raw comment to classify.
    model_name : str
        Subfolder name under analysis_and_inference/models/ (e.g., "ensemble",
        "lasso_log_reg"). Defaults to the soft-vote ensemble.
    explain : bool
        If True, compute SHAP values and return the top contributing features.
        Adds ~1-3 seconds per call (KernelExplainer). Defaults to True.
    top_k : int
        Number of features to return in `top_features`, sorted by |SHAP|.

    Returns
    -------
    dict with keys:
        label : int            (0 = non-toxic, 1 = toxic)
        probability : float | None
        top_features : list[dict]   (only present if explain=True)
            Each dict: {"feature": str, "value": float, "shap": float}
            Sorted by |shap| descending. Positive shap pushes toward toxic.
    """
    _load(model_name, with_explainer=explain)

    model    = _cache[model_name]
    scaler   = _cache["scaler"]
    dense    = _cache["dense"]
    feat_nms = _cache["feature_names"]

    X_raw    = pd.DataFrame({"comment_text": [text]})
    X_dense  = dense.transform(X_raw)
    X_scaled = scaler.transform(X_dense)

    label = int(model.predict(X_scaled)[0])
    prob  = float(model.predict_proba(X_scaled)[0, 1]) if hasattr(model, "predict_proba") else None

    result = {"label": label, "probability": prob}

    if explain:
        explainer   = _cache[f"explainer:{model_name}"]
        shap_values = explainer.shap_values(X_scaled, nsamples=SHAP_NSAMPLES, silent=True)
        sv          = _shap_for_class_1(shap_values)

        # Pair raw (unscaled) feature values with SHAP contributions
        raw_values = X_dense.iloc[0].values
        order      = np.argsort(np.abs(sv))[::-1][:top_k]
        result["top_features"] = [
            {"feature": feat_nms[i], "value": float(raw_values[i]), "shap": float(sv[i])}
            for i in order
        ]

    return result


if __name__ == "__main__":
    test_cases = [
        "you are a complete idiot",
        "thanks for the edit, looks great",
        "kys you worthless piece of trash",
        "I really enjoyed reading this article",
    ]
    for text in test_cases:
        result = predict_comment(text, top_k=5)
        tag      = "TOXIC" if result["label"] == 1 else "clean"
        prob_str = f"{result['probability']:.3f}" if result["probability"] is not None else "n/a"
        print(f"\n[{tag}  p={prob_str}]  {text}")
        print("  Top contributors (positive = pushes toward toxic):")
        for feat in result["top_features"]:
            arrow = "+" if feat["shap"] > 0 else "-"
            print(f"    {arrow} {feat['feature']:<30} value={feat['value']:>7.2f}  shap={feat['shap']:+.3f}")
