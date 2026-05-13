"""End-to-end orchestrator: prepare data → train every model → train ensemble.

After this finishes, the following bundles are on disk and ready to use:
    baseline/outputs/baseline_tuned.pkl
    lasso_log_reg/outputs/lasso_log_reg_tuned.pkl
    random_forest/outputs/random_forest_tuned.pkl
    ridge_log_reg/outputs/ridge_log_reg_tuned.pkl
    svm/outputs/svm_tuned.pkl
    ensemble/outputs/ensemble_hard_vote_tuned.pkl   ← final classifier

Run with:
    uv run python analysis_and_inference/models/run_all.py
"""

import logging
import os
import sys
import time

from pathlib import Path

PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Headless backend so plt.show() doesn't block when running as a script
import matplotlib
matplotlib.use("Agg")

# initialize logging to get those exceptions written with timestamps and saved
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # module-level logger for structured output


def banner(text):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def main():
    overall_start = time.time()

    # 1. Prepare data (skip if split already exists)
    split_path = "analysis_and_inference/models/split_and_features/split.pkl"
    if os.path.exists(split_path):
        log.info("[skip] %s already exists - using existing split", split_path)
    else:
        banner("STEP 1/8  Preparing data")
        try:
            from analysis_and_inference.models.split_and_features.prepare_split import main as prepare_split_main
            prepare_split_main()
        except Exception as e:
            raise RuntimeError(f"STEP 1 failed - could not prepare data split: {e}") from e  # abort if raw data missing/ malformed

    # 1b. Pre-compute dense features (skip if cache already exists)
    features_path = "analysis_and_inference/models/split_and_features/features.pkl"
    if os.path.exists(features_path):
        log.info("[skip] %s already exists - using cached features", features_path)
    else:
        banner("STEP 1b   Pre-computing dense features (one-time)")
        try:
            from analysis_and_inference.models._common import load_split, precompute_features
            X_train, X_test, _, _ = load_split()
            precompute_features(X_train, X_test)
        except Exception as e:
            raise RuntimeError(f"STEP 1b failed - could not pre-compute features: {e}") from e  #if split.pkl missing or feature extraction fails

    # 2-6. Individual models
    try:
        from analysis_and_inference.models.baseline.baseline           import run as run_baseline
        from analysis_and_inference.models.lasso_log_reg.lasso         import run as run_lasso
        from analysis_and_inference.models.random_forest.random_forest import run as run_rf
        from analysis_and_inference.models.ridge_log_reg.ridge         import run as run_ridge
        from analysis_and_inference.models.svm.svm                     import run as run_svm
    except ImportError as e:
        raise ImportError(f"Could not import one or more model modules: {e}") from e #raise Error and log if model could not be loaded using try except

    steps = [
        ("STEP 2/8  Baseline (Dummy)", run_baseline),
        ("STEP 3/8  Custom Lasso",     run_lasso),
        ("STEP 4/8  Random Forest",    run_rf),
        ("STEP 5/8  Ridge LogReg",     run_ridge),
        ("STEP 6/8  Linear SVM",       run_svm),
    ]
    for label, fn in steps:
        banner(label)
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            raise RuntimeError(f"{label} failed: {e}") from e  # abort pipeline and name the failing step
        log.info("%s finished in %.1fs", label, time.time() - t0)

    # 7. Ensemble (depends on the saved member pkls)
    banner("STEP 7/8  Ensemble (soft vote)")
    try:
        from analysis_and_inference.models.ensemble.ensemble import run as run_ensemble
        t0 = time.time()
        run_ensemble()
        log.info("Ensemble finished in %.1fs", time.time() - t0)
    except Exception as e:
        raise RuntimeError(f"STEP 7 failed - ensemble could not be trained: {e}") from e  #one or more member pkls missing

    # 8. Error analysis for every saved model
    banner("STEP 8/8  Error analysis (FP/FN, error patterns, confidence dist)")
    try:
        from analysis_and_inference.evaluation_code.error_analysis import main as run_error_analysis
        t0 = time.time()
        for model_name in ["baseline", "lasso_log_reg", "random_forest",
                           "ridge_log_reg", "svm", "ensemble"]:
            print(f"\n--- {model_name} ---")
            run_error_analysis(model_name, verbose=False)
        log.info("Error analysis finished in %.1fs", time.time() - t0)
    except Exception as e:
        raise RuntimeError(f"STEP 8 failed - error analysis could not be run: {e}") from e  # most likely saved model pkl missing or predictions empty

    banner(f"DONE in {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Pipeline aborted: %s", e)  # log final error and exit 
