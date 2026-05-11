"""
feature_selection.py — post-hoc feature-selection analysis for the tuned Random Forest

This module evaluates how Random Forest performance changes when training on
only the most important engineered features. Feature importance rankings are
used to retrain the model on top-5 and top-10 subsets, allowing comparison
against the full feature space (using PR-AUC, ROC-AUC, and Macro-F1).
It saves a comparison plot and CSV to random_forest/outputs/evaluation/.

Note: The analysis reuses the tuned Random Forest configuration to isolate the effect
of feature selection rather than hyperparameter changes. Feature subsets are represented as ranked index arrays because they allow fast,
memory-efficient slicing of sparse feature matrices.

Run manually after run_all.py has produced random_forest_tuned.pkl:
    python analysis_and_inference/models/random_forest/feature_selection.py
"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis_and_inference.models._common import load_split, precompute_features
from analysis_and_inference.features.build_features import DenseFeatureTransformer


MODEL_PATH = "analysis_and_inference/models/random_forest/outputs/random_forest_tuned.pkl"
OUTPUT_DIR = "analysis_and_inference/models/random_forest/outputs/evaluation"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        tuned_pipeline = pickle.load(f)
    tuned_rf = tuned_pipeline.named_steps["clf"]

    X_train, X_test, y_train, y_test = load_split()
    X_train_feat, X_test_feat = precompute_features(X_train, X_test)
    y_train = y_train.values.ravel()
    y_test  = y_test.values.ravel()

    # Feature names: re-derive from a single-row dense transform.
    feat_names = DenseFeatureTransformer().transform(X_train.head(1)).columns.tolist()
    importances = tuned_rf.feature_importances_

    ranked_idx = np.argsort(importances)[::-1]
    print("\nAll features ranked by importance:")
    print(f"{'#':<4} {'feature':<35} {'importance':>10}")
    print(f"{'-'*4} {'-'*35} {'-'*10}")
    for rank, idx in enumerate(ranked_idx, 1):
        print(f"{rank:<4} {feat_names[idx]:<35} {importances[idx]:>10.4f}")

    top_5_idx  = ranked_idx[:5]
    top_10_idx = ranked_idx[:10]

    sample_weights = compute_sample_weight("balanced", y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for label, sel_idx in [("top_5", top_5_idx), ("top_10", top_10_idx)]:
        print(f"\n-- Training on {label} features ---------------------------------")
        X_train_sel = X_train_feat[:, sel_idx]
        X_test_sel  = X_test_feat[:, sel_idx]

        model = RandomForestClassifier(
            n_estimators=tuned_rf.n_estimators,
            max_depth=tuned_rf.max_depth,
            min_samples_split=tuned_rf.min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_sel, y_train, sample_weight=sample_weights)

        # Threshold tuning on last CV fold
        last_train_idx, last_val_idx = list(cv.split(X_train_sel, y_train))[-1]
        rf_thresh = RandomForestClassifier(
            n_estimators=tuned_rf.n_estimators,
            max_depth=tuned_rf.max_depth,
            min_samples_split=tuned_rf.min_samples_split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_thresh.fit(
            X_train_sel[last_train_idx],
            y_train[last_train_idx],
            sample_weight=sample_weights[last_train_idx],
        )
        y_proba_val = rf_thresh.predict_proba(X_train_sel[last_val_idx])[:, 1]
        y_val       = y_train[last_val_idx]
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        threshold = float(thresholds[np.argmax(f1_scores)])

        y_proba_test = model.predict_proba(X_test_sel)[:, 1]
        y_pred_test  = (y_proba_test >= threshold).astype(int)
        pr_auc   = average_precision_score(y_test, y_proba_test)
        roc_auc  = roc_auc_score(y_test, y_proba_test)
        macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

        print(f"  PR-AUC:   {pr_auc:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Threshold: {threshold:.4f}")

        results[label] = {
            "n_features": len(sel_idx),
            "features":   [feat_names[i] for i in sel_idx],
            "pr_auc":     pr_auc,
            "roc_auc":    roc_auc,
            "macro_f1":   macro_f1,
            "threshold":  threshold,
        }

    # Full feature set baseline (using already-tuned pipeline)
    y_proba_full = tuned_pipeline.predict_proba(X_test_feat)[:, 1]
    y_pred_full  = tuned_pipeline.predict(X_test_feat)
    full_pr_auc  = average_precision_score(y_test, y_proba_full)
    full_roc_auc = roc_auc_score(y_test, y_proba_full)
    full_f1      = f1_score(y_test, y_pred_full, average="macro", zero_division=0)

    print("\n-- Performance comparison -------------------------------------------")
    print(f"{'model':<25} {'PR-AUC':>8} {'ROC-AUC':>9} {'Macro-F1':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*9} {'-'*10}")
    print(f"{'all features ('+str(len(feat_names))+')':<25} {full_pr_auc:>8.4f} {full_roc_auc:>9.4f} {full_f1:>10.4f}")
    for label, res in results.items():
        n = res["n_features"]
        print(f"{label+' ('+str(n)+')':<25} {res['pr_auc']:>8.4f} {res['roc_auc']:>9.4f} {res['macro_f1']:>10.4f}")

    # Save comparison CSV + plot
    rows = [
        {"model": f"all features ({len(feat_names)})", "pr_auc": full_pr_auc, "roc_auc": full_roc_auc, "macro_f1": full_f1},
    ]
    for label, res in results.items():
        rows.append({
            "model":    f"{label} ({res['n_features']})",
            "pr_auc":   res["pr_auc"],
            "roc_auc":  res["roc_auc"],
            "macro_f1": res["macro_f1"],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "feature_selection.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.27
    ax.bar(x - width, df["pr_auc"],   width, label="PR-AUC")
    ax.bar(x,         df["roc_auc"],  width, label="ROC-AUC")
    ax.bar(x + width, df["macro_f1"], width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest — feature-selection comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "feature_selection.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    run()
