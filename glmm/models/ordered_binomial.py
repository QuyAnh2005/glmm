"""Ordered Binomial GLMM with AR(1) Random Effects."""

from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import norm
from scipy.special import binom

from .base import BaseGLMM
from ..utils.math import logit


class OrderedBinomialGLMM(BaseGLMM):
    """Ordered Binomial GLMM with AR(1) Random Effects.

    This model implements a binomial GLMM with:
    - Ordered fixed effects (mu_k) across rating categories
    - Rating-specific covariate effects (beta_k)
    - Rating-specific random effect loadings (phi_k)
    - AR(1) random effects (b_t)
    """

    def __init__(
        self,
        n_iter: int = 15000,
        burn_in: int = 5000,
        tau: float = 100.0,
        eta: float = 2.0,
        v: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize model.

        Args:
            n_iter: Number of MCMC iterations
            burn_in: Number of burn-in iterations
            tau: Prior variance for mu and beta
            eta: Shape parameter for phi inverse-gamma prior
            v: Scale parameter for phi inverse-gamma prior
            random_seed: Random seed for reproducibility
        """
        super().__init__(n_iter=n_iter, burn_in=burn_in, tau=tau, random_seed=random_seed)
        self.eta = eta
        self.v = v

    def _initialize_parameters(self, T: int, K: int) -> tuple:
        """Initialize model parameters.

        Args:
            T: Number of time periods
            K: Number of rating categories

        Returns:
            Tuple of initialized parameters (mu, beta, phi, alpha, b)
        """
        mu = np.linspace(-2, -10, K)  # Ordered initial values
        beta = np.random.normal(0, 1, K)
        phi = np.ones(K) * 0.5
        alpha = 0.5
        b = np.random.normal(0, 1, T)
        return mu, beta, phi, alpha, b

    def _log_full_conditional_mu_k(
        self,
        k: int,
        mu_k: float,
        mu: np.ndarray,
        beta_k: float,
        phi_k: float,
        b: np.ndarray,
        N_tk: np.ndarray,
        n_tk: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """Full conditional for mu_k: [mu_k | ·] ∝ Π_t [N_tk | n_tk, b_t, mu_k, beta_k, phi_k] [mu_k]
        - Likelihood: Product of binomial probabilities over time for rating k
        - Prior: N(0, tau^2) with ordering constraint mu_1 > mu_2 > ... > mu_K
        """
        if k > 0 and mu_k >= mu[k - 1]:  # Check ordering constraint
            return -np.inf
        if k < len(mu) - 1 and mu_k <= mu[k + 1]:
            return -np.inf

        ll = 0
        for t in range(len(b)):
            p = logit(mu_k - x_t[t] * beta_k - phi_k * b[t])
            ll += N_tk[t, k] * np.log(p + 1e-10) + (n_tk[t, k] - N_tk[t, k]) * np.log(1 - p + 1e-10)
        prior = norm.logpdf(mu_k, 0, self.tau)
        return ll + prior

    def _log_full_conditional_beta_k(
        self,
        k: int,
        mu_k: float,
        beta_k: float,
        phi_k: float,
        b: np.ndarray,
        N_tk: np.ndarray,
        n_tk: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """Full conditional for beta_k: [beta_k | ·] ∝ Π_t [N_tk | n_tk, b_t, mu_k, beta_k, phi_k] [beta_k]
        - Likelihood: Product of binomial probabilities over time for rating k
        - Prior: N(0, tau^2)
        """
        ll = 0
        for t in range(len(b)):
            p = logit(mu_k - x_t[t] * beta_k - phi_k * b[t])
            ll += N_tk[t, k] * np.log(p + 1e-10) + (n_tk[t, k] - N_tk[t, k]) * np.log(1 - p + 1e-10)
        prior = norm.logpdf(beta_k, 0, self.tau)
        return ll + prior

    def _log_full_conditional_phi_k(
        self,
        k: int,
        mu_k: float,
        beta_k: float,
        phi_k: float,
        b: np.ndarray,
        N_tk: np.ndarray,
        n_tk: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """Full conditional for phi_k: [phi_k | ·] ∝ Π_t [N_tk | n_tk, b_t, mu_k, beta_k, phi_k] [phi_k]
        - Likelihood: Product of binomial probabilities over time for rating k
        - Prior: phi_k^2 ~ Inverse-Gamma(eta, v), phi_k > 0, so [phi_k] ∝ phi_k^(-2(eta+1)) exp(-v/phi_k^2)
        """
        if phi_k <= 0:  # Enforce positivity
            return -np.inf

        ll = 0
        for t in range(len(b)):
            p = logit(mu_k - x_t[t] * beta_k - phi_k * b[t])
            ll += N_tk[t, k] * np.log(p + 1e-10) + (n_tk[t, k] - N_tk[t, k]) * np.log(1 - p + 1e-10)
        prior = -(self.eta + 1) * np.log(phi_k**2) - self.v / phi_k**2
        return ll + prior

    def _log_full_conditional_b_t(
        self,
        t: int,
        b_t: float,
        b: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        phi: np.ndarray,
        alpha: float,
        N_tk: np.ndarray,
        n_tk: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """Full conditional for b_t: [b_t | ·] ∝ Π_k [N_tk | n_tk, b_t, mu_k, beta_k, phi_k] [b_t | b_{t-1}, b_{t+1}, alpha]
        - Likelihood: Product of binomial probabilities over all ratings k at time t
        - Prior: AR(1) process with innovation variance 1
          - For t = 0: b_0 ~ N(0, 1/(1-alpha^2)), contributes to b_1 ~ N(alpha b_0, 1)
          - For t = 1, ..., T-2: [b_t | b_{t-1}, b_{t+1}, alpha] ~ N(α (b_{t-1} + b_{t+1}) / (1 + α^2), 1 / (1 + α^2))
            - Derived from b_t = α b_{t-1} + ε_t and b_{t+1} = α b_t + ε_{t+1}, ε_t, ε_{t+1} ~ N(0, 1)
            - The mean α (b_{t-1} + b_{t+1}) / (1 + α^2) balances past and future, variance 1 / (1 + α^2) reflects correlation
          - For t = T-1: b_{T-1} ~ N(α b_{T-2}, 1), no future term
        """
        T = len(b)

        # Likelihood
        ll = 0
        for k in range(len(mu)):
            p = logit(mu[k] - x_t[t] * beta[k] - phi[k] * b_t)
            ll += N_tk[t, k] * np.log(p + 1e-10) + (n_tk[t, k] - N_tk[t, k]) * np.log(1 - p + 1e-10)

        # AR(1) prior
        if t == 0:
            prior = norm.logpdf(b_t, 0, 1 / np.sqrt(1 - alpha**2 + 1e-10))
            prior += norm.logpdf(b[1], alpha * b_t, 1)
        elif t == T - 1:
            prior = norm.logpdf(b_t, alpha * b[t - 1], 1)
        else:
            mean_prior = alpha * (b[t - 1] + b[t + 1]) / (1 + alpha**2)
            var_prior = 1 / (1 + alpha**2)
            prior = norm.logpdf(b_t, mean_prior, np.sqrt(var_prior))

        return ll + prior

    def _log_full_conditional_alpha(self, alpha: float, b: np.ndarray) -> float:
        """Full conditional for alpha: [alpha | ·] ∝ [b | alpha] [alpha]
        - Likelihood: Multivariate normal for b with AR(1) covariance
        - Prior: Uniform(-1, 1)
        """
        if not (-1 < alpha < 1):
            return -np.inf

        T = len(b)
        Sigma_inv = np.zeros((T, T))
        for i in range(T):
            Sigma_inv[i, i] = 1 + alpha**2 if 0 < i < T - 1 else 1
            if i < T - 1:
                Sigma_inv[i, i + 1] = Sigma_inv[i + 1, i] = -alpha

        det_Sigma_inv = 1 - alpha**2  # Simplified determinant for AR(1)
        ll = 0.5 * np.log(det_Sigma_inv + 1e-10) - 0.5 * b @ Sigma_inv @ b
        return ll  # Uniform prior on (-1, 1) contributes constant

    def _log_likelihood_t(
        self,
        t: int,
        mu: np.ndarray,
        beta: np.ndarray,
        phi: np.ndarray,
        b_t: float,
        N_tk: np.ndarray,
        n_tk: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """Compute log likelihood for time period t."""
        ll = 0
        for k in range(len(mu)):
            p = logit(mu[k] - x_t[t] * beta[k] - phi[k] * b_t)
            ll += np.log(binom(n_tk[t, k], N_tk[t, k]) + 1e-10)
            ll += N_tk[t, k] * np.log(p + 1e-10)
            ll += (n_tk[t, k] - N_tk[t, k]) * np.log(1 - p + 1e-10)
        return ll

    def fit(self, N_tk: np.ndarray, n_tk: np.ndarray, x_t: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit the model using MCMC.

        Args:
            N_tk: Array of default counts (T x K)
            n_tk: Array of total counts (T x K)
            x_t: Array of covariates (T,)

        Returns:
            Dictionary containing:
            - Parameter traces (mu, beta, phi, alpha, b)
            - Log CPO for each time period
            - Average and sum of log CPO
        """
        self._validate_input(N_tk, n_tk, x_t)
        T, K = N_tk.shape

        # Initialize parameters
        mu, beta, phi, alpha, b = self._initialize_parameters(T, K)

        # Initialize storage
        self.traces = {
            "mu": np.zeros((self.n_iter, K)),
            "beta": np.zeros((self.n_iter, K)),
            "phi": np.zeros((self.n_iter, K)),
            "alpha": np.zeros(self.n_iter),
            "b": np.zeros((self.n_iter, T)),
        }
        likelihood_trace = np.zeros((self.n_iter, T))

        # MCMC iterations
        for it in range(self.n_iter):
            # Update mu_k
            for k in range(K):
                mu_prop = mu[k] + np.random.normal(0, 0.5)
                mu_temp = mu.copy()
                mu_temp[k] = mu_prop
                log_ratio = self._log_full_conditional_mu_k(
                    k, mu_prop, mu_temp, beta[k], phi[k], b, N_tk, n_tk, x_t
                ) - self._log_full_conditional_mu_k(
                    k, mu[k], mu, beta[k], phi[k], b, N_tk, n_tk, x_t
                )
                if np.isfinite(log_ratio) and np.log(np.random.rand()) < log_ratio:
                    mu[k] = mu_prop

            # Update beta_k
            for k in range(K):
                beta_prop = beta[k] + np.random.normal(0, 0.5)
                log_ratio = self._log_full_conditional_beta_k(
                    k, mu[k], beta_prop, phi[k], b, N_tk, n_tk, x_t
                ) - self._log_full_conditional_beta_k(k, mu[k], beta[k], phi[k], b, N_tk, n_tk, x_t)
                if np.isfinite(log_ratio) and np.log(np.random.rand()) < log_ratio:
                    beta[k] = beta_prop

            # Update phi_k
            for k in range(K):
                phi_prop = phi[k] + np.random.normal(0, 0.5)
                log_ratio = self._log_full_conditional_phi_k(
                    k, mu[k], beta[k], phi_prop, b, N_tk, n_tk, x_t
                ) - self._log_full_conditional_phi_k(k, mu[k], beta[k], phi[k], b, N_tk, n_tk, x_t)
                if np.isfinite(log_ratio) and np.log(np.random.rand()) < log_ratio:
                    phi[k] = phi_prop

            # Update b_t
            for t in range(T):
                b_prop = b[t] + np.random.normal(0, 0.5)
                log_ratio = self._log_full_conditional_b_t(
                    t, b_prop, b, mu, beta, phi, alpha, N_tk, n_tk, x_t
                ) - self._log_full_conditional_b_t(
                    t, b[t], b, mu, beta, phi, alpha, N_tk, n_tk, x_t
                )
                if np.isfinite(log_ratio) and np.log(np.random.rand()) < log_ratio:
                    b[t] = b_prop

            # Update alpha
            alpha_prop = alpha + np.random.normal(0, 0.5)
            log_ratio = self._log_full_conditional_alpha(
                alpha_prop, b
            ) - self._log_full_conditional_alpha(alpha, b)
            if np.isfinite(log_ratio) and np.log(np.random.rand()) < log_ratio:
                alpha = alpha_prop

            # Store results
            self.traces["mu"][it] = mu
            self.traces["beta"][it] = beta
            self.traces["phi"][it] = phi
            self.traces["alpha"][it] = alpha
            self.traces["b"][it] = b

            # Store log-likelihood for CPO
            for t in range(T):
                likelihood_trace[it, t] = self._log_likelihood_t(
                    t, mu, beta, phi, b[t], N_tk, n_tk, x_t
                )

        # Compute CPO statistics
        log_cpo_t = np.zeros(T)
        for t in range(T):
            inv_likelihood = 1 / (np.exp(likelihood_trace[self.burn_in :, t]) + 1e-10)
            mean_inv_likelihood = np.mean(inv_likelihood)
            log_cpo_t[t] = -np.log(mean_inv_likelihood + 1e-10)

        # Get MCMC traces
        traces = self.get_traces()

        # Add CPO statistics
        traces["log_cpo_t"] = log_cpo_t
        traces["avg_log_cpo"] = np.mean(log_cpo_t)
        traces["sum_log_cpo"] = np.sum(log_cpo_t)

        return traces
