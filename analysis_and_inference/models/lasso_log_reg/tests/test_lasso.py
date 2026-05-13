"""
test_lasso.py - smoke test for the custom L1 logistic regression pipeline

Checks that the model integrates correctly with the shared pipeline utilities,
fits without errors, and produces valid predictions and probability scores.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from analysis_and_inference.models._common import make_pipeline
from analysis_and_inference.models.lasso_log_reg.core_logistic_regression_lasso import LassoLogisticRegression


def test_lasso_pipeline_fits_and_predicts(tiny_data):
    X, y = tiny_data
    pipe = make_pipeline(LassoLogisticRegression(alpha=0.01, learning_rate=0.1, max_iter=200))
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# Fixtures for unit tests below (raw numpy arrays, independent of tiny_data)
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return LassoLogisticRegression()


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 4))
    y = (X[:, 0] > 0).astype(int)
    m = LassoLogisticRegression(alpha=0.01, max_iter=500)
    m.fit(X, y)
    return m, X


@pytest.fixture
def sparse_data():
    # 200 samples, 20 features — only the first two actually predict y
    rng = np.random.default_rng(42)
    n, p = 200, 20
    X = rng.standard_normal((n, p))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Unit tests — _sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero_input_returns_half(self, model):
        # zero is the symmetry point — neither class is favoured
        assert model._sigmoid(0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self, model):
        # strong positive signal should mean near-certain class 1
        assert model._sigmoid(100) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_zero(self, model):
        # strong negative signal should mean near-certain class 0
        assert model._sigmoid(-100) == pytest.approx(0.0, abs=1e-6)

    def test_clip_prevents_overflow(self, model):
        # values beyond ±500 are clipped — result should still be a valid float, not nan
        result = model._sigmoid(1e9)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Unit tests — _soft_threshold
# ---------------------------------------------------------------------------

# covers inside-threshold zeroing, outside-threshold shrinkage, sign preservation, and symmetry
@pytest.mark.parametrize("value, threshold, expected", [
    ( 0.03,  0.05,  0.0),
    (-0.03,  0.05,  0.0),
    ( 0.8,   0.05,  0.75),
    (-0.8,   0.05, -0.75),
    (-0.9,   0.05, -0.85),
])
def test_soft_threshold_parametrized(model, value, threshold, expected):
    assert model._soft_threshold(value, threshold) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Unit tests — sparsity
# ---------------------------------------------------------------------------

class TestSparsity:
    def test_high_alpha_zeros_most_coefficients(self, sparse_data):
        # strong regularization should drive noise features to exactly zero
        X, y = sparse_data
        model = LassoLogisticRegression(alpha=1.0, max_iter=1000)
        model.fit(X, y)
        assert np.sum(model.coef_ == 0) > len(model.coef_) // 2

    def test_zero_alpha_produces_fewer_zeros_than_high_alpha(self, sparse_data):
        # without a penalty there is no pressure to zero anything out
        X, y = sparse_data
        model_low  = LassoLogisticRegression(alpha=0.0, max_iter=1000)
        model_high = LassoLogisticRegression(alpha=1.0, max_iter=1000)
        model_low.fit(X, y)
        model_high.fit(X, y)
        assert np.sum(model_low.coef_ == 0) < np.sum(model_high.coef_ == 0)


# ---------------------------------------------------------------------------
# Unit tests — predictions
# ---------------------------------------------------------------------------

class TestPredictions:
    def test_predict_proba_shape(self, fitted_model):
        # one probability pair per sample — (n, 2) is the sklearn contract
        m, X = fitted_model
        assert m.predict_proba(X).shape == (len(X), 2)

    def test_predict_proba_values_in_range(self, fitted_model):
        # every value must be a valid probability — nothing below 0 or above 1
        m, X = fitted_model
        proba = m.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_rows_sum_to_one(self, fitted_model):
        # the two columns are complements — P(class 0) + P(class 1) must equal 1
        m, X = fitted_model
        row_sums = m.predict_proba(X).sum(axis=1)
        assert row_sums == pytest.approx(np.ones(len(X)))

    def test_predict_returns_binary(self, fitted_model):
        # a binary classifier must only ever output 0 or 1, nothing in between
        m, X = fitted_model
        assert set(m.predict(X)).issubset({0, 1})

    def test_threshold_zero_predicts_all_ones(self):
        # sigmoid is always > 0, so decision_threshold=0.0 forces every prediction to 1
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4))
        y = (X[:, 0] > 0).astype(int)
        m = LassoLogisticRegression(alpha=0.01, max_iter=500, decision_threshold=0.0)
        m.fit(X, y)
        assert np.all(m.predict(X) == 1)


# ---------------------------------------------------------------------------
# Unit tests — score
# ---------------------------------------------------------------------------

class TestScore:
    def test_perfect_accuracy_on_separable_data(self):
        # a wide decision boundary leaves no room for misclassification
        rng = np.random.default_rng(7)
        n = 100
        X = rng.standard_normal((n, 4))
        X[:n // 2, 0] += 10
        X[n // 2:, 0] -= 10
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        model = LassoLogisticRegression(alpha=0.01, max_iter=1000)
        model.fit(X, y)
        assert model.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Unit tests — intercept
# ---------------------------------------------------------------------------

class TestIntercept:
    def test_fit_intercept_false_keeps_intercept_at_zero(self, sparse_data):
        # the flag must suppress intercept updates, not just initialise to zero
        X, y = sparse_data
        model = LassoLogisticRegression(fit_intercept=False, max_iter=500)
        model.fit(X, y)
        assert model.intercept_ == 0.0


# ---------------------------------------------------------------------------
# Unit tests — convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_converges_before_max_iter_on_separable_data(self):
        # the model stores n_iter_ after fitting — on separable data it should be well under max_iter
        rng = np.random.default_rng(0)
        n = 100
        X = rng.standard_normal((n, 5))
        X[:n // 2, 0] += 10   # class 1 is far positive on feature 0
        X[n // 2:, 0] -= 10   # class 0 is far negative on feature 0
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        model = LassoLogisticRegression(alpha=0.01, max_iter=1000, tol=1e-4)
        model.fit(X, y)

        assert model.n_iter_ < model.max_iter


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_lasso_reproducibility_with_random_state():
    # LassoLogisticRegression has no random_state — it is fully deterministic (gradient descent
    # from fixed zero initialization), so two fits on identical data always produce equal predictions
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)

    m1 = LassoLogisticRegression(alpha=0.01, max_iter=1000)
    m1.fit(X, y)

    m2 = LassoLogisticRegression(alpha=0.01, max_iter=1000)
    m2.fit(X, y)

    assert np.array_equal(m1.predict(X), m2.predict(X))
