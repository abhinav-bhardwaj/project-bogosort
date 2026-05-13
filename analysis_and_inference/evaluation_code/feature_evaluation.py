"""Heavy feature-importance analyses (permutation + SHAP) and a coef CSV.

The basic feature-importance plot (model.coef_ or model.feature_importances_) is
already produced by evaluator.py during run_all.py and saved as
`evaluation/feature_importance.png` — no need to re-render it here. This script
adds the deeper, expensive analyses on top of that.

Run manually for a given model after run_all.py has saved its tuned pipeline:

    python analysis_and_inference/evaluation_code/feature_evaluation.py lasso_log_reg
    python analysis_and_inference/evaluation_code/feature_evaluation.py random_forest
    python analysis_and_inference/evaluation_code/feature_evaluation.py ridge_log_reg
    python analysis_and_inference/evaluation_code/feature_evaluation.py svm

Saves to analysis_and_inference/models/<model>/outputs/evaluation/:
    coef_importance.csv        (only for models with coef_; ranked CSV)
    perm_importance.png + .csv (always)
    shap_summary.png + .csv    (always; TreeExplainer for RF, KernelExplainer otherwise)
"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from analysis_and_inference.models._common import load_split, precompute_features, load_feature_names


TOP_N    = 20
N_REPEATS = 10
SHAP_N    = 500
SEED      = 42


def _unwrap(estimator):
    """Return the underlying classifier (Pipeline -> its 'clf' step, else estimator itself)."""
    named = getattr(estimator, "named_steps", None)
    if named is not None and "clf" in named:
        return named["clf"]
    return estimator


def coef_importance_csv(estimator, feature_names, save_dir):
    """Save a ranked CSV of model coefficients. Plot is produced by evaluator.py."""
    inner = _unwrap(estimator)
    if not hasattr(inner, "coef_"):
        print("Coefficient CSV: skipped (model has no coef_)")
        return None
    coefs = np.asarray(inner.coef_).ravel()
    df = pd.DataFrame({
        "feature":  feature_names,
        "coef":     coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(save_dir, "coef_importance.csv")
    df.to_csv(csv_path, index=False)
    n_zero = int((df["coef"] == 0).sum())
    print(f"Saved {csv_path}  (zeroed: {n_zero}/{len(df)})")
    return df


def perm_importance(estimator, X_test, y_test, feature_names, save_dir):
    print(f"Permutation importance ({N_REPEATS} repeats, {X_test.shape[1]} features)...")
    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=N_REPEATS, random_state=SEED, scoring="roc_auc", n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top = df.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, TOP_N * 0.35)))
    ax.barh(top["feature"], top["importance"], xerr=top["std"],
            color="steelblue", alpha=0.85, capsize=3)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean decrease in ROC-AUC when feature is shuffled")
    ax.set_title(f"Top {TOP_N} Features — Permutation Importance (test set)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    png = os.path.join(save_dir, "perm_importance.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(os.path.join(save_dir, "perm_importance.csv"), index=False)
    print(f"Saved {png}")
    return df


def shap_importance(estimator, X_test, feature_names, save_dir):
    try:
        import shap
    except ImportError:
        print("SHAP not installed (`pip install shap`); skipping SHAP analysis")
        return None

    inner = _unwrap(estimator)
    rng = np.random.default_rng(SEED)
    sample_size = min(SHAP_N, X_test.shape[0])
    idx = rng.choice(X_test.shape[0], size=sample_size, replace=False)
    X_sample = X_test[idx]

    print(f"SHAP on {sample_size} samples...")
    if hasattr(inner, "feature_importances_"):
        explainer = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(X_sample)
    else:
        # KernelExplainer requires a function; predict_proba on the pipeline level
        background = shap.sample(X_sample, 50, random_state=SEED)
        explainer  = shap.KernelExplainer(estimator.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    # Normalise to (n_samples, n_features) for the positive class
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[..., 0]

    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      max_display=TOP_N, show=False)
    plt.tight_layout()
    png = os.path.join(save_dir, "shap_summary.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    df.to_csv(os.path.join(save_dir, "shap_importance.csv"), index=False)
    print(f"Saved {png}")
    return df


def main(model_name):
    model_path = f"analysis_and_inference/models/{model_name}/outputs/{model_name}_tuned.pkl"
    save_dir   = f"analysis_and_inference/models/{model_name}/outputs/evaluation"
    os.makedirs(save_dir, exist_ok=True)

    with open(model_path, "rb") as f:
        estimator = pickle.load(f)
    print(f"Loaded {model_path}")

    X_train, X_test, y_train, y_test = load_split()
    _, X_test_feat = precompute_features(X_train, X_test)
    feature_names = load_feature_names()
    y_test = y_test.values.ravel()

    print("\n-- Coefficient CSV (plot already produced by evaluator.py) --")
    coef_importance_csv(estimator, feature_names, save_dir)

    print("\n-- Permutation importance --")
    perm_importance(estimator, X_test_feat, y_test, feature_names, save_dir)

    print("\n-- SHAP --")
    shap_importance(estimator, X_test_feat, feature_names, save_dir)

    print(f"\nAll outputs saved to {save_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python feature_evaluation.py <model_name>")
        print("Example: python feature_evaluation.py lasso_log_reg")
        sys.exit(1)
    main(sys.argv[1])
