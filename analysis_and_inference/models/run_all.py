"""End-to-end orchestrator: prepare data → train every model → train ensemble.

After this finishes, the following bundles are on disk and ready to use:
    baseline/outputs/baseline_tuned.pkl
    lasso_log_reg/outputs/lasso_log_reg_tuned.pkl
    random_forest/outputs/random_forest_tuned.pkl
    ridge_log_reg/outputs/ridge_log_reg_tuned.pkl
    svm/outputs/svm_tuned.pkl
    ensemble/outputs/ensemble_hard_vote_tuned.pkl   ← final classifier

Run with: uv run python analysis_and_inference/models/run_all.py
"""

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


def banner(text):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def main():
    overall_start = time.time()

    # 1. Prepare data (skip if split already exists)
    split_path = "analysis_and_inference/models/split_and_features/split.pkl"
    if os.path.exists(split_path):
        print(f"[skip] {split_path} already exists - using existing split")
    else:
        banner("STEP 1/7  Preparing data")
        from analysis_and_inference.models.split_and_features.prepare_split import main as prepare_split_main
        prepare_split_main()

    # 1b. Pre-compute dense features (skip if cache already exists)
    features_path = "analysis_and_inference/models/split_and_features/features.pkl"
    if os.path.exists(features_path):
        print(f"[skip] {features_path} already exists - using cached features")
    else:
        banner("STEP 1b   Pre-computing dense features (one-time)")
        import pickle
        from analysis_and_inference.models._common import load_split, precompute_features
        X_train, X_test, _, _ = load_split()
        precompute_features(X_train, X_test)

    # 2-6. Individual models
    from analysis_and_inference.models.baseline.baseline             import run as run_baseline
    from analysis_and_inference.models.lasso_log_reg.lasso           import run as run_lasso
    from analysis_and_inference.models.random_forest.random_forest   import run as run_rf
    from analysis_and_inference.models.ridge_log_reg.ridge           import run as run_ridge
    from analysis_and_inference.models.svm.svm                       import run as run_svm

    steps = [
        ("STEP 2/8  Baseline (Dummy)",      run_baseline),
        ("STEP 3/8  Custom Lasso",          run_lasso),
        ("STEP 4/8  Random Forest",         run_rf),
        ("STEP 5/8  Ridge LogReg",          run_ridge),
        ("STEP 6/8  Linear SVM",            run_svm),
    ]
    for label, fn in steps:
        banner(label)
        t0 = time.time()
        fn()
        print(f"\n[{label}] finished in {time.time() - t0:.1f}s")

    # 7. Ensemble (depends on the saved member pkls)
    banner("STEP 7/8  Ensemble (soft vote)")
    from analysis_and_inference.models.ensemble.ensemble import run as run_ensemble
    t0 = time.time()
    run_ensemble()
    print(f"\n[ensemble] finished in {time.time() - t0:.1f}s")

    # 8. Error analysis for every saved model
    banner("STEP 8/8  Error analysis (FP/FN, error patterns, confidence dist)")
    from analysis_and_inference.evaluation_code.error_analysis import main as run_error_analysis
    t0 = time.time()
    for model_name in ["baseline", "lasso_log_reg", "random_forest",
                       "ridge_log_reg", "svm", "ensemble"]:
        print(f"\n--- {model_name} ---")
        run_error_analysis(model_name, verbose=False)
    print(f"\n[error analysis] finished in {time.time() - t0:.1f}s")

    banner(f"DONE in {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
