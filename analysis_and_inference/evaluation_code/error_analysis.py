"""Error-pattern analysis: FP/FN samples, error patterns by feature, confidence distribution.

Run manually for a given model after run_all.py has saved its tuned pipeline:

    python analysis_and_inference/evaluation_code/error_analysis.py lasso_log_reg

Saves to analysis_and_inference/models/<model>/outputs/evaluation/:
    false_positives.csv
    false_negatives.csv
    error_patterns_by_feature.png + .csv
    error_confidence_distribution.png   (only if model exposes predict_proba)
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

from analysis_and_inference.models._common import load_split, precompute_features, load_feature_names


TOP_N  = 20
N_SHOW = 20


def _unwrap(estimator):
    while hasattr(estimator, "estimator"):
        estimator = estimator.estimator
    named = getattr(estimator, "named_steps", None)
    if named is not None and "clf" in named:
        return named["clf"]
    return estimator


def _get_scores(estimator, X):
    """Return positive-class scores in [0,1] if possible, else raw decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1], True
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X), False
    return None, False


def inspect_errors(X_test_text, y_test, y_pred, scores, save_dir, verbose=True):
    fp_mask = (y_pred == 1) & (y_test == 0)
    fn_mask = (y_pred == 0) & (y_test == 1)

    fp_df = pd.DataFrame({"text": X_test_text[fp_mask].values, "score": scores[fp_mask]})
    fn_df = pd.DataFrame({"text": X_test_text[fn_mask].values, "score": scores[fn_mask]})

    fp_df = fp_df.sort_values("score", ascending=False).head(N_SHOW)
    fn_df = fn_df.sort_values("score", ascending=True).head(N_SHOW)

    fp_path = os.path.join(save_dir, "false_positives.csv")
    fn_path = os.path.join(save_dir, "false_negatives.csv")
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)

    if verbose:
        print(f"\nTop {N_SHOW} False Positives (highest-confidence non-toxic flagged toxic):")
        for _, row in fp_df.iterrows():
            print(f"  [{row['score']:.3f}] {row['text'][:120]}")
        print(f"\nTop {N_SHOW} False Negatives (lowest-confidence toxic missed):")
        for _, row in fn_df.iterrows():
            print(f"  [{row['score']:.3f}] {row['text'][:120]}")
    print(f"Saved {fp_path} and {fn_path}")


def error_patterns_by_feature(estimator, X_test_feat, y_test, y_pred, feature_names, save_dir):
    inner = _unwrap(estimator)
    if hasattr(inner, "coef_"):
        ranking = np.abs(np.asarray(inner.coef_).ravel())
    elif hasattr(inner, "feature_importances_"):
        ranking = np.asarray(inner.feature_importances_)
    else:
        print("Error patterns: skipped (no coef_ or feature_importances_ to rank features)")
        return

    top_idx = np.argsort(ranking)[::-1][:TOP_N]
    X_top   = X_test_feat[:, top_idx]
    names   = [feature_names[i] for i in top_idx]

    fp_mask      = (y_pred == 1) & (y_test == 0)
    fn_mask      = (y_pred == 0) & (y_test == 1)
    correct_mask = y_pred == y_test

    df = pd.DataFrame({
        "correct":        X_top[correct_mask].mean(axis=0),
        "false_positive": X_top[fp_mask].mean(axis=0),
        "false_negative": X_top[fn_mask].mean(axis=0),
    }, index=names)
    df["fp_vs_correct"] = df["false_positive"] - df["correct"]
    df["fn_vs_correct"] = df["false_negative"] - df["correct"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, TOP_N * 0.35)))
    for ax, col, title, colour in [
        (axes[0], "fp_vs_correct", "False Positive − Correct\n(features over-triggering)", "tomato"),
        (axes[1], "fn_vs_correct", "False Negative − Correct\n(features under-triggering)", "steelblue"),
    ]:
        vals = df[col].sort_values()
        ax.barh(vals.index, vals.values, color=colour, alpha=0.8)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Mean feature value difference")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Error Patterns by Feature Value", fontsize=13, y=1.01)
    plt.tight_layout()
    png = os.path.join(save_dir, "error_patterns_by_feature.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(os.path.join(save_dir, "error_patterns_by_feature.csv"))
    print(f"Saved {png}")


def confidence_distribution(y_test, y_pred, scores, threshold, save_dir):
    fp_mask = (y_pred == 1) & (y_test == 0)
    fn_mask = (y_pred == 0) & (y_test == 1)
    tp_mask = (y_pred == 1) & (y_test == 1)
    tn_mask = (y_pred == 0) & (y_test == 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(scores.min(), scores.max(), 30)

    axes[0].hist(scores[fp_mask], bins=bins, color="tomato",    alpha=0.7, label=f"FP (n={fp_mask.sum()})")
    axes[0].hist(scores[tn_mask], bins=bins, color="steelblue", alpha=0.7, label=f"TN (n={tn_mask.sum()})")
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.2f}")
    axes[0].set_title("Non-toxic samples — score distribution")
    axes[0].set_xlabel("Score")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(scores[fn_mask], bins=bins, color="tomato",    alpha=0.7, label=f"FN (n={fn_mask.sum()})")
    axes[1].hist(scores[tp_mask], bins=bins, color="steelblue", alpha=0.7, label=f"TP (n={tp_mask.sum()})")
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.2f}")
    axes[1].set_title("Toxic samples — score distribution")
    axes[1].set_xlabel("Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    png = os.path.join(save_dir, "error_confidence_distribution.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")


def main(model_name, verbose=True):
    folder   = f"analysis_and_inference/models/{model_name}/outputs"
    save_dir = os.path.join(folder, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    pkls = [f for f in os.listdir(folder) if f.endswith("_tuned.pkl")]
    if not pkls:
        raise FileNotFoundError(f"No *_tuned.pkl found in {folder}")
    model_path = os.path.join(folder, pkls[0])
    with open(model_path, "rb") as f:
        estimator = pickle.load(f)
    print(f"Loaded {model_path}")

    X_train, X_test, y_train, y_test = load_split()
    _, X_test_feat = precompute_features(X_train, X_test)
    feature_names = load_feature_names()

    X_test_text = X_test["comment_text"]
    y_test = y_test.values.ravel()
    y_pred = estimator.predict(X_test_feat)

    scores, is_proba = _get_scores(estimator, X_test_feat)
    if scores is None:
        print("Model has neither predict_proba nor decision_function; using y_pred as score")
        scores = y_pred.astype(float)
        is_proba = True

    threshold = 0.5 if is_proba else 0.0

    print("\n-- FP / FN samples --")
    inspect_errors(X_test_text, y_test, y_pred, scores, save_dir, verbose=verbose)

    print("\n-- Error patterns by feature --")
    error_patterns_by_feature(estimator, X_test_feat, y_test, y_pred, feature_names, save_dir)

    print("\n-- Confidence distribution --")
    confidence_distribution(y_test, y_pred, scores, threshold, save_dir)

    print(f"All outputs saved to {save_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python error_analysis.py <model_name>")
        print("Example: python error_analysis.py lasso_log_reg")
        sys.exit(1)
    main(sys.argv[1])
