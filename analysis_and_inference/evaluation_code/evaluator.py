"""Standard classification evaluation: prints metrics and saves individual PNGs.

Plots produced (each as a separate file inside `save_dir`):
    confusion_matrix.png
    roc_curve.png            (only if y_score given)
    pr_curve.png             (only if y_score given)
    calibration.png          (only if y_score is in [0, 1] - skipped for raw SVM scores)
    feature_importance.png   (only if `model` is provided and exposes coef_ or
                              feature_importances_; skipped silently otherwise)

Heavy analyses (SHAP, permutation importance, error analysis) live in
feature_evaluation.py and error_analysis.py - run those separately.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    auc,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def _get_importances(model):
    """Return (values, kind) where kind is 'coef' or 'importance', or (None, None)."""
    if model is None:
        return None, None
    candidates = [model]
    named = getattr(model, "named_steps", None)
    if named is not None and "clf" in named:
        candidates.append(named["clf"])
    for cand in candidates:
        if hasattr(cand, "feature_importances_"):
            return np.asarray(cand.feature_importances_), "importance"
        if hasattr(cand, "coef_"):
            return np.asarray(cand.coef_).ravel(), "coef"
    return None, None


def _save_confusion_matrix(cm, name, save_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{name} - Confusion Matrix")
    labels = ["Non-toxic", "Toxic"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_roc_curve(fpr, tpr, roc_auc, name, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title(f"{name} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_pr_curve(pr_recall, pr_precision, pr_auc_trapz, ap_score, name, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pr_recall, pr_precision, label=f"PR AUC = {pr_auc_trapz:.4f}\nAP = {ap_score:.4f}")
    ax.set_title(f"{name} - Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_calibration_curve(y_true, y_score, name, save_dir):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.set_title(f"{name} - Calibration Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_feature_importance(values, kind, feature_names, name, save_dir, top_n=20):
    n_feat = len(values)
    if feature_names is None or len(feature_names) != n_feat:
        feature_names = [f"feature_{i}" for i in range(n_feat)]
    order = np.argsort(np.abs(values))[::-1][:top_n]
    sel_vals  = values[order]
    sel_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.32)))
    if kind == "coef":
        colours = ["steelblue" if v >= 0 else "tomato" for v in sel_vals]
        ax.barh(sel_names[::-1], sel_vals[::-1], color=colours[::-1], alpha=0.85)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Coefficient (blue = positive, red = negative)")
        ax.set_title(f"{name} - Top {top_n} Feature Coefficients")
    else:
        ax.barh(sel_names[::-1], sel_vals[::-1], color="steelblue", alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(f"{name} - Top {top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_classification(y_true, y_pred, y_score=None, name="Model",
                            plot_curves=True, save_dir=None,
                            model=None, feature_names=None):
    """Print metrics + (optionally) save individual evaluation PNGs to save_dir.

    Parameters
    ----------
    y_true, y_pred : array-like (binary 0/1)
    y_score : array-like (probability or decision score). Optional.
    name : str - used in plot titles
    plot_curves : bool - if False, skip all plotting
    save_dir : str or None - directory for PNGs (created if missing). If None, no plots saved.
    model : fitted estimator - optional; used to extract coef_ or feature_importances_
    feature_names : list[str] - optional; pairs with `model` for the importance plot
    """
    print(f"\n===== {name} Evaluation =====")

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\nClassification Report:\n")
    clf_report = classification_report(y_true, y_pred)
    print(clf_report)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  TP: {tp}  FP: {fp}")
    print(f"  FN: {fn}  TN: {tn}")

    metrics = {
        "accuracy":              acc,
        "f1":                    f1,
        "precision":             precision,
        "recall":                recall,
        "classification_report": clf_report,
        "confusion_matrix":      cm,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if plot_curves and save_dir is not None:
        _save_confusion_matrix(cm, name, save_dir)

    if y_score is not None:
        y_score = np.asarray(y_score).ravel()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc_trapz = auc(pr_recall, pr_precision)
        ap_score = average_precision_score(y_true, y_score)

        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc_trapz:.4f}")
        print(f"Avg Prec.: {ap_score:.4f}")

        metrics.update({
            "roc_auc":           roc_auc,
            "pr_auc":            pr_auc_trapz,
            "average_precision": ap_score,
        })

        if plot_curves and save_dir is not None:
            _save_roc_curve(fpr, tpr, roc_auc, name, save_dir)
            _save_pr_curve(pr_recall, pr_precision, pr_auc_trapz, ap_score, name, save_dir)

            # Calibration curve only makes sense for [0, 1] probability scores.
            score_min, score_max = float(y_score.min()), float(y_score.max())
            if score_min >= 0.0 and score_max <= 1.0:
                _save_calibration_curve(y_true, y_score, name, save_dir)
            else:
                print(f"Calibration plot skipped (y_score range [{score_min:.2f}, {score_max:.2f}] is not [0, 1])")
    else:
        print("y_score not provided -> skipping ROC/PR/AUC computation.")

    if plot_curves and save_dir is not None and model is not None:
        values, kind = _get_importances(model)
        if values is not None:
            _save_feature_importance(values, kind, feature_names, name, save_dir)
        else:
            print(f"Feature importance plot skipped (model has no coef_ or feature_importances_)")

    if save_dir is not None and plot_curves:
        print(f"Saved evaluation plots to: {save_dir}/")

    return metrics
