"""
lasso.py — training entry point for custom L1 logistic regression

This module trains the project's from-scratch L1-regularized logistic
regression model using shared preprocessing and evaluation utilities.
L1 regularization was chosen to encourage sparse feature weights, making the
model more interpretable and reducing dependence on weak engineered features.

Training logic is delegated to the shared run_grid_search utility such that tuning,
evaluation, serialization, and reporting remain consistent across all models.
Hyperparameters are defined in a small search grid to balance optimisation
quality with computational cost during experimentation.

Run with: uv run python analysis_and_inference/models/lasso_log_reg/lasso.py
"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from analysis_and_inference.models._common import run_grid_search
from analysis_and_inference.models.lasso_log_reg.core_logistic_regression_lasso import LassoLogisticRegression


NAME       = "lasso_log_reg"
OUTPUT_DIR = "analysis_and_inference/models/lasso_log_reg/outputs"


def run():
    clf  = LassoLogisticRegression(max_iter=2000)
    grid = {
        "alpha":         [0.001, 0.01, 0.1],
        "learning_rate": [0.01, 0.1],
    }
    run_grid_search(NAME, OUTPUT_DIR, clf, grid,
                    friendly_name="Lasso (custom from-scratch)")


if __name__ == "__main__":
    run()
