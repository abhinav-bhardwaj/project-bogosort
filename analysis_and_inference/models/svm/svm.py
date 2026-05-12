import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from sklearn.svm import LinearSVC

from analysis_and_inference.models._common import make_pipeline, run_grid_search


NAME       = "svm"
OUTPUT_DIR = "analysis_and_inference/models/svm/outputs"


def run():
    pipe = make_pipeline(LinearSVC(
        class_weight="balanced", random_state=42, max_iter=2000,
    ))
    grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    # LinearSVC has no predict_proba; helper falls back to decision_function for y_score
    # and skips threshold tuning automatically.
    run_grid_search(NAME, OUTPUT_DIR, pipe, grid, friendly_name="LinearSVM")


if __name__ == "__main__":
    run()
