"""
random_forest.py - random forest model for toxicity classification


This module trains a RandomForestClassifier on the engineered feature space.
Random forests were chosen because they capture nonlinear feature interactions
and are robust to noisy or correlated predictors.

The model is wrapped in the shared pipeline and grid search utilities so
training, tuning, evaluation, and serialization remain consistent across all
models in the project.

Class balancing is enabled to better handle the toxicity label imbalance.

Run with: uv run python analysis_and_inference/models/random_forest/random_forest.py

"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier

from analysis_and_inference.models._common import make_pipeline, run_grid_search


NAME       = "random_forest"
OUTPUT_DIR = "analysis_and_inference/models/random_forest/outputs"


def run():
    pipe = make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    grid = {
        "clf__n_estimators":      [100, 200],
        "clf__max_depth":         [10, 20, None],
        "clf__min_samples_split": [2, 5],
    }
    run_grid_search(NAME, OUTPUT_DIR, pipe, grid, friendly_name="Random Forest")


if __name__ == "__main__":
    run()
