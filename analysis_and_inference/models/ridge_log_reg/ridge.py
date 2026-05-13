"""
ridge.py - L2-regularized logistic regression for toxicity classification

This module trains a ridge logistic regression model on the engineered feature
space. We chose L2 regularization because it stabilizes coefficient estimates
while retaining information from correlated features.

The model is wrapped in the shared pipeline and grid search utilities so
training, tuning, evaluation, and serialization remain consistent across all
models in the project.

Class balancing is enabled to better handle the toxicity label imbalance.

Run with: uv run python analysis_and_inference/models/ridge_log_reg/ridge.py
"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression

from analysis_and_inference.models._common import make_pipeline, run_grid_search


NAME       = "ridge_log_reg"
OUTPUT_DIR = "analysis_and_inference/models/ridge_log_reg/outputs"


def run():
    pipe = make_pipeline(LogisticRegression(
        solver="lbfgs", class_weight="balanced",
        max_iter=1000, random_state=42,
    ))
    grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    run_grid_search(NAME, OUTPUT_DIR, pipe, grid,
                    friendly_name="Ridge Logistic Regression (sklearn L2)")


if __name__ == "__main__":
    run()
