import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis.models.data_pipeline import DataPipeline
from analysis.features.build_features import FeatureBuilder, FeaturePreprocessor


# ── Model ─────────────────────────────────────────────────────────────────────

class LassoLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01, learning_rate=0.1, max_iter=1000, tol=1e-4, fit_intercept=True, decision_threshold=0.5, verbose=False):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.decision_threshold = decision_threshold
        self.verbose = verbose

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _soft_threshold(self, beta, threshold):
        return np.sign(beta) * np.maximum(np.abs(beta) - threshold, 0)

    def fit(self, X, y, sample_weight=None):
        X = check_array(X, accept_sparse=True)
        y = np.asarray(y)
        if not set(np.unique(y).tolist()).issubset({0, 1}):
            raise ValueError(f"y must contain only 0/1 labels, got {np.unique(y)}")
        n, n_features = X.shape
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Need both 0 and 1 classes in y, got {self.classes_}")
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sw_sum = sw.sum()
        if sw_sum <= 0:
            raise ValueError(f"sample_weight must have a positive total, got sw_sum={sw_sum}")

        converged = False
        iteration = -1
        for iteration in range(self.max_iter):
            p_hat = self._sigmoid(X @ self.coef_ + self.intercept_)
            residual = p_hat - y
            weighted_residual = sw * residual
            grad_coef = (X.T @ weighted_residual) / sw_sum

            coef_new = self.coef_ - self.learning_rate * grad_coef
            coef_new = self._soft_threshold(coef_new, self.alpha * self.learning_rate)

            intercept_new = self.intercept_
            if self.fit_intercept:
                intercept_new = self.intercept_ - self.learning_rate * (weighted_residual.sum() / sw_sum)

            delta = max(
                np.max(np.abs(coef_new - self.coef_)),
                abs(intercept_new - self.intercept_),
            )
            self.coef_ = coef_new
            self.intercept_ = intercept_new

            if delta < self.tol:
                converged = True
                if self.verbose:
                    print(f"Model converged at iteration {iteration}")
                break

        self.n_iter_ = iteration + 1
        if not converged:
            warnings.warn(
                f"LassoLogisticRegression did not converge after {self.max_iter} iterations "
                f"(final delta={delta:.2e}, tol={self.tol}). Consider increasing max_iter or learning_rate.",
                ConvergenceWarning,
            )
        return self

    def predict_proba(self, X):
        p = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return (self._sigmoid(X @ self.coef_ + self.intercept_) >= self.decision_threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR     = "./data/processed"
X_TRAIN_PROC_PATH = os.path.join(PROCESSED_DIR, "X_train_proc.npz")
X_TEST_PROC_PATH  = os.path.join(PROCESSED_DIR, "X_test_proc.npz")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.pkl")


# ── Load data ─────────────────────────────────────────────────────────────────

dp = DataPipeline("./data/processed/test_train_data.pkl", label_columns=["toxic"])
X_train, X_test, y_train, y_test = dp.get_data()

y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")
print(f"Toxic rate (train): {y_train.mean():.1%}")


# ── Features + preprocessing (cached) ────────────────────────────────────────

if (os.path.exists(X_TRAIN_PROC_PATH) and
    os.path.exists(X_TEST_PROC_PATH)  and
    os.path.exists(PREPROCESSOR_PATH)):

    print("Loading cached processed matrices...")
    X_train_proc = sp.load_npz(X_TRAIN_PROC_PATH).tocsr()
    X_test_proc  = sp.load_npz(X_TEST_PROC_PATH).tocsr()
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

else:
    fb = FeatureBuilder()

    if os.path.exists(fb.tfidf_path):
        fb.load()
    else:
        print("Fitting TF-IDF vectorizers...")
        fb.fit(X_train)

    print("Transforming train features...")
    X_train_feat = fb.transform(X_train, split="train")
    print("Transforming test features...")
    X_test_feat  = fb.transform(X_test,  split="test")

    preprocessor = FeaturePreprocessor()
    print("Preprocessing features...")
    X_train_proc = preprocessor.fit_transform(X_train_feat).tocsr()
    X_test_proc  = preprocessor.transform(X_test_feat).tocsr()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    sp.save_npz(X_TRAIN_PROC_PATH, X_train_proc)
    sp.save_npz(X_TEST_PROC_PATH,  X_test_proc)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    print("Processed matrices cached to disk.")


# ── Grid search ───────────────────────────────────────────────────────────────

alpha_values         = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
learning_rate_values = [0.001, 0.01, 0.05, 0.1]

# PR-AUC is used as the primary metric rather than ROC-AUC because the dataset
# is heavily imbalanced (~9:1 non-toxic:toxic). ROC-AUC can be misleadingly
# high in imbalanced settings as it accounts for true negatives, which are
# abundant. PR-AUC focuses only on the positive (toxic) class and is a more
# honest measure of how well the model identifies toxic comments.
cv             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sample_weights = compute_sample_weight("balanced", y_train)

records = []
for alpha in alpha_values:
    for lr in learning_rate_values:
        fold_pr_aucs  = []
        fold_roc_aucs = []
        fold_f1s      = []

        for train_idx, val_idx in cv.split(X_train_proc, y_train):
            X_fold_train = X_train_proc[train_idx]
            X_fold_val   = X_train_proc[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val   = y_train[val_idx]
            w_fold       = sample_weights[train_idx]

            lasso = LassoLogisticRegression(
                alpha=alpha,
                learning_rate=lr,
                max_iter=1000,
                verbose=False,
            )
            lasso.fit(X_fold_train, y_fold_train, sample_weight=w_fold)

            y_proba = lasso.predict_proba(X_fold_val)[:, 1]
            y_pred  = lasso.predict(X_fold_val)

            fold_pr_aucs.append(average_precision_score(y_fold_val, y_proba))
            fold_roc_aucs.append(roc_auc_score(y_fold_val, y_proba))
            fold_f1s.append(f1_score(y_fold_val, y_pred, average="macro", zero_division=0))

        records.append({
            "alpha":             alpha,
            "learning_rate":     lr,
            "val_pr_auc_mean":   np.mean(fold_pr_aucs),
            "val_pr_auc_std":    np.std(fold_pr_aucs),
            "val_roc_auc_mean":  np.mean(fold_roc_aucs),
            "val_macro_f1_mean": np.mean(fold_f1s),
        })
        print(f"alpha={alpha:<6}  lr={lr:<6}  PR-AUC: {np.mean(fold_pr_aucs):.4f} +/- {np.std(fold_pr_aucs):.4f}")

results_df = pd.DataFrame(records).sort_values("val_pr_auc_mean", ascending=False)

print()
print("Best CV PR-AUC:", round(results_df.iloc[0]["val_pr_auc_mean"], 4))
print("Best alpha:", results_df.iloc[0]["alpha"])
print("Best learning_rate:", results_df.iloc[0]["learning_rate"])
print()
print(results_df[["alpha", "learning_rate", "val_pr_auc_mean", "val_pr_auc_std", "val_roc_auc_mean", "val_macro_f1_mean"]].to_string(index=False))


# ── Threshold tuning ──────────────────────────────────────────────────────────

# The default threshold of 0.5 assumes balanced classes. With a 9:1 imbalance
# the optimal decision boundary is lower — we sweep the precision-recall curve
# and pick the threshold that maximises F1.
best_alpha = results_df.iloc[0]["alpha"]
best_lr    = results_df.iloc[0]["learning_rate"]

final_lasso = LassoLogisticRegression(
    alpha=best_alpha,
    learning_rate=best_lr,
    max_iter=1000,
    verbose=False,
)
final_lasso.fit(X_train_proc, y_train, sample_weight=sample_weights)

y_proba_train  = final_lasso.predict_proba(X_train_proc)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_train, y_proba_train)
f1_scores      = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
best_threshold = float(thresholds[np.argmax(f1_scores)])

print(f"\nOptimal classification threshold: {best_threshold:.4f}  (default 0.5)")


# ── Test set evaluation ───────────────────────────────────────────────────────

final_lasso.decision_threshold = best_threshold

y_proba_test = final_lasso.predict_proba(X_test_proc)[:, 1]
y_pred_test  = final_lasso.predict(X_test_proc)

print()
print("Test set performance:")
print(f"  PR-AUC:   {average_precision_score(y_test, y_proba_test):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"  Macro-F1: {f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs("./analysis/models/artifacts", exist_ok=True)

results_df.to_csv("./analysis/models/artifacts/tuning_results.csv", index=False)

with open("./analysis/models/artifacts/best_model.pkl", "wb") as f:
    pickle.dump({
        "model":        final_lasso,
        "preprocessor": preprocessor,
        "threshold":    best_threshold,
    }, f)

print("\nSaved tuning_results.csv and best_model.pkl to artifacts/")