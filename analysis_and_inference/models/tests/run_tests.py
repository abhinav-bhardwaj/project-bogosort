"""Master test runner: unit tests, integration tests, and end-to-end discovery across all model folders."""

import sys
import pytest


def main():
    # per-model unit test folders
    model_test_dirs = [
        "analysis_and_inference/models/baseline/tests",
        "analysis_and_inference/models/ensemble/tests",
        "analysis_and_inference/models/lasso_log_reg/tests",
        "analysis_and_inference/models/random_forest/tests",
        "analysis_and_inference/models/ridge_log_reg/tests",
        "analysis_and_inference/models/svm/tests",
    ]

    # integration and evaluator tests
    integration_dir = "analysis_and_inference/models/tests"
    evaluator_dir   = "analysis_and_inference/evaluation_code/tests"

    # run everything in one pytest invocation
    args = model_test_dirs + [integration_dir, evaluator_dir, "-v"]
    sys.exit(pytest.main(args))


if __name__ == "__main__":
    main()
