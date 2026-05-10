"""
baseline.py — random baseline classifier for toxicity detection

This model implements a simple benchmark model using sklearn's DummyClassifier.
This establishes a minimum performance baseline that all real machine
learning models in the project should outperform.

This script trains a DummyClassifier using the "stratified" strategy 
(predictions are generated randomly and class probabilities follow the training 
label distribution). It produces predictions and probability scores on the test set 
and evaluates classification performance using the shared evaluator: classification 
metrics, ROC/PR curves, optional feature importance handling.
It saves the trained baseline model, the evaluation artifacts, and the output files 
to the model output directory.

The baseline serves as a reference point for measuring whether more
advanced models (logistic regression, random forest, XGBoost, etc.)
actually learn meaningful toxicity patterns beyond random guessing.

Generated artifacts are stored in:

    analysis_and_inference/models/baseline/outputs/

Run with: uv run python analysis_and_inference/models/baseline.py
"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from sklearn.dummy import DummyClassifier

from analysis_and_inference.models._common import load_split, precompute_features, load_feature_names, save_outputs
from analysis_and_inference.evaluation_code.evaluator import evaluate_classification


NAME       = "baseline"
OUTPUT_DIR = "analysis_and_inference/models/baseline/outputs"


def run():
    X_train, X_test, y_train, y_test = load_split()
    X_train_feat, X_test_feat = precompute_features(X_train, X_test)
    feature_names = load_feature_names()

    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(X_train_feat, y_train)

    y_pred  = model.predict(X_test_feat)
    y_score = model.predict_proba(X_test_feat)[:, 1]

    evaluate_classification(
        y_test.values, y_pred, y_score,
        name="Dummy Baseline",
        plot_curves=True,
        save_dir=os.path.join(OUTPUT_DIR, "evaluation"),
        model=model,
        feature_names=feature_names,
    )

    save_outputs(NAME, OUTPUT_DIR, model)



if __name__ == "__main__":
    run()
