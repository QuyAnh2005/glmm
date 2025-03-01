"""Tests for OrderedBinomialGLMM model."""

import numpy as np
import pytest
from scipy.stats import norm

from glmm.models import OrderedBinomialGLMM


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    T, K = 10, 3  # 10 time periods, 3 rating categories

    # Generate true parameters
    mu_true = np.array([-2, -4, -6])  # Ordered intercepts
    beta_true = np.array([0.5, 1.0, 1.5])  # Rating-specific slopes
    phi_true = np.array([0.8, 1.0, 1.2])  # Rating-specific loadings
    alpha_true = 0.7  # AR(1) coefficient

    # Generate AR(1) random effects
    b_true = np.zeros(T)
    b_true[0] = np.random.normal(0, np.sqrt(1 / (1 - alpha_true**2)))
    for t in range(1, T):
        b_true[t] = alpha_true * b_true[t - 1] + np.random.normal(0, 1)

    # Generate covariates
    x_t = np.random.normal(0, 1, T)

    # Generate default counts
    N_tk = np.zeros((T, K), dtype=int)
    n_tk = np.full((T, K), 100)  # 100 firms per rating category

    for t in range(T):
        for k in range(K):
            p = 1 / (1 + np.exp(-(mu_true[k] - x_t[t] * beta_true[k] - phi_true[k] * b_true[t])))
            N_tk[t, k] = np.random.binomial(n_tk[t, k], p)

    return {
        "N_tk": N_tk,
        "n_tk": n_tk,
        "x_t": x_t,
        "true_params": {
            "mu": mu_true,
            "beta": beta_true,
            "phi": phi_true,
            "alpha": alpha_true,
            "b": b_true,
        },
    }


def test_model_initialization():
    """Test model initialization with default and custom parameters."""
    # Test default initialization
    model = OrderedBinomialGLMM()
    assert model.n_iter == 15000
    assert model.burn_in == 5000
    assert model.tau == 100.0
    assert model.eta == 2.0
    assert model.v == 1.0

    # Test custom initialization
    model = OrderedBinomialGLMM(n_iter=10000, burn_in=2000, tau=50.0, eta=3.0, v=2.0)
    assert model.n_iter == 10000
    assert model.burn_in == 2000
    assert model.tau == 50.0
    assert model.eta == 3.0
    assert model.v == 2.0


def test_parameter_initialization(sample_data):
    """Test parameter initialization method."""
    model = OrderedBinomialGLMM(random_seed=42)
    T, K = sample_data["N_tk"].shape

    mu, beta, phi, alpha, b = model._initialize_parameters(T, K)

    # Check shapes
    assert mu.shape == (K,)
    assert beta.shape == (K,)
    assert phi.shape == (K,)
    assert isinstance(alpha, float)
    assert b.shape == (T,)

    # Check ordering constraint on mu
    assert np.all(np.diff(mu) < 0)

    # Check phi positivity
    assert np.all(phi > 0)

    # Check alpha bounds
    assert -1 < alpha < 1


def test_input_validation(sample_data):
    """Test input validation."""
    model = OrderedBinomialGLMM()
    N_tk = sample_data["N_tk"]
    n_tk = sample_data["n_tk"]
    x_t = sample_data["x_t"]

    # Test valid input
    model._validate_input(N_tk, n_tk, x_t)

    # Test mismatched shapes
    with pytest.raises(ValueError):
        model._validate_input(N_tk[:, :-1], n_tk, x_t)

    # Test invalid x_t length
    with pytest.raises(ValueError):
        model._validate_input(N_tk, n_tk, x_t[:-1])

    # Test invalid default counts
    invalid_N_tk = N_tk.copy()
    invalid_N_tk[0, 0] = n_tk[0, 0] + 1
    with pytest.raises(ValueError):
        model._validate_input(invalid_N_tk, n_tk, x_t)


def test_full_conditionals(sample_data):
    """Test full conditional distributions."""
    model = OrderedBinomialGLMM(random_seed=42)
    N_tk = sample_data["N_tk"]
    n_tk = sample_data["n_tk"]
    x_t = sample_data["x_t"]
    true_params = sample_data["true_params"]
    T, K = N_tk.shape

    # Initialize parameters near true values
    mu = true_params["mu"] + np.random.normal(0, 0.1, K)
    beta = true_params["beta"] + np.random.normal(0, 0.1, K)
    phi = true_params["phi"] + np.random.normal(0, 0.1, K)
    alpha = true_params["alpha"] + np.random.normal(0, 0.1)
    b = true_params["b"] + np.random.normal(0, 0.1, T)

    # Test mu_k full conditional
    for k in range(K):
        log_prob = model._log_full_conditional_mu_k(
            k, mu[k], mu, beta[k], phi[k], b, N_tk, n_tk, x_t
        )
        assert np.isfinite(log_prob)

    # Test beta_k full conditional
    for k in range(K):
        log_prob = model._log_full_conditional_beta_k(k, mu[k], beta[k], phi[k], b, N_tk, n_tk, x_t)
        assert np.isfinite(log_prob)

    # Test phi_k full conditional
    for k in range(K):
        log_prob = model._log_full_conditional_phi_k(k, mu[k], beta[k], phi[k], b, N_tk, n_tk, x_t)
        assert np.isfinite(log_prob)

    # Test b_t full conditional
    for t in range(T):
        log_prob = model._log_full_conditional_b_t(
            t, b[t], b, mu, beta, phi, alpha, N_tk, n_tk, x_t
        )
        assert np.isfinite(log_prob)

    # Test alpha full conditional
    log_prob = model._log_full_conditional_alpha(alpha, b)
    assert np.isfinite(log_prob)


def test_model_fitting(sample_data):
    """Test model fitting with small number of iterations."""
    model = OrderedBinomialGLMM(n_iter=100, burn_in=50, random_seed=42)
    N_tk = sample_data["N_tk"]
    n_tk = sample_data["n_tk"]
    x_t = sample_data["x_t"]

    # Fit model
    traces = model.fit(N_tk, n_tk, x_t)

    # Check trace shapes
    T, K = N_tk.shape
    assert traces["mu"].shape == (50, K)  # After burn-in
    assert traces["beta"].shape == (50, K)
    assert traces["phi"].shape == (50, K)
    assert traces["alpha"].shape == (50,)
    assert traces["b"].shape == (50, T)

    # Check CPO statistics
    assert traces["log_cpo_t"].shape == (T,)
    assert isinstance(traces["avg_log_cpo"], float)
    assert isinstance(traces["sum_log_cpo"], float)

    # Get parameter estimates
    estimates = model.get_estimates()
    for param in ["mu", "beta", "phi", "alpha", "b"]:
        assert "mean" in estimates[param]
        assert "variance" in estimates[param]


def test_trace_retrieval(sample_data):
    """Test trace retrieval with and without burn-in."""
    model = OrderedBinomialGLMM(n_iter=100, burn_in=50, random_seed=42)
    N_tk = sample_data["N_tk"]
    n_tk = sample_data["n_tk"]
    x_t = sample_data["x_t"]

    # Fit model
    model.fit(N_tk, n_tk, x_t)

    # Get traces with burn-in
    traces_with_burnin = model.get_traces(discard_burnin=False)
    assert all(
        v.shape[0] == 100
        for v in traces_with_burnin.values()
        if isinstance(v, np.ndarray) and v.ndim > 1
    )

    # Get traces without burn-in
    traces_without_burnin = model.get_traces(discard_burnin=True)
    assert all(
        v.shape[0] == 50
        for v in traces_without_burnin.values()
        if isinstance(v, np.ndarray) and v.ndim > 1
    )
