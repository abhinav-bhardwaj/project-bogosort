import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from analysis_and_inference.models._common import make_pipeline
from analysis_and_inference.evaluation_code.evaluator import evaluate_classification


def make_synthetic_data():
    # synthetic numeric data mimicking precompute_features output
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X[:80], X[80:], y[:80], y[80:]


def test_random_forest_pipeline_to_evaluator():
    # full chain from pipeline to evaluator with probability scores
    X_train, X_test, y_train, y_test = make_synthetic_data()

    pipe = make_pipeline(RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1, class_weight="balanced"))
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]

    result = evaluate_classification(y_test, y_pred, y_score=y_score, name="RF Integration", plot_curves=False, save_dir=None)

    assert isinstance(result, dict)
    assert {"accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"}.issubset(result.keys())
    assert all(0.0 <= result[k] <= 1.0 for k in ("accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"))


def test_svm_pipeline_to_evaluator():
    # integration with decision_function instead of predict_proba
    X_train, X_test, y_train, y_test = make_synthetic_data()

    pipe = make_pipeline(LinearSVC(class_weight="balanced", random_state=42, max_iter=500))
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_score = pipe.decision_function(X_test)

    result = evaluate_classification(y_test, y_pred, y_score=y_score, name="SVM Integration", plot_curves=False, save_dir=None)

    assert isinstance(result, dict)
    assert {"accuracy", "f1", "precision", "recall"}.issubset(result.keys())


def test_ridge_pipeline_to_evaluator_no_scores():
    # evaluator without probability scores should still return base metrics
    X_train, X_test, y_train, y_test = make_synthetic_data()

    pipe = make_pipeline(LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=200, random_state=42))
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    result = evaluate_classification(y_test, y_pred, y_score=None, name="Ridge Integration", plot_curves=False, save_dir=None)

    assert isinstance(result, dict)
    assert {"accuracy", "f1", "precision", "recall"}.issubset(result.keys())
    assert "roc_auc" not in result
