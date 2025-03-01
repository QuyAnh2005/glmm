"""Base class for Generalized Linear Mixed Models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import norm

from ..utils.math import logit, log_likelihood_binomial


class BaseGLMM(ABC):
    """Abstract base class for GLMM models."""

    def __init__(
        self,
        n_iter: int = 10000,
        burn_in: int = 1000,
        tau: float = 100.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize GLMM model.

        Args:
            n_iter: Number of MCMC iterations
            burn_in: Number of burn-in iterations to discard
            tau: Prior standard deviation for normal priors
            random_seed: Random seed for reproducibility
        """
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.tau = tau

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize storage for MCMC traces
        self.traces = {}

    @abstractmethod
    def fit(self, N_tk: np.ndarray, n_tk: np.ndarray, x_t: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit the model using MCMC.

        Args:
            N_tk: Array of default counts (T x K)
            n_tk: Array of total counts (T x K)
            x_t: Array of covariates (T,)

        Returns:
            Dictionary containing parameter traces
        """
        pass

    def _validate_input(self, N_tk: np.ndarray, n_tk: np.ndarray, x_t: np.ndarray) -> None:
        """Validate input data dimensions and values.

        Args:
            N_tk: Array of default counts
            n_tk: Array of total counts
            x_t: Array of covariates

        Raises:
            ValueError: If inputs are invalid
        """
        if N_tk.shape != n_tk.shape:
            raise ValueError("N_tk and n_tk must have same shape")
        if x_t.shape[0] != N_tk.shape[0]:
            raise ValueError("x_t length must match first dimension of N_tk")
        if np.any(N_tk > n_tk):
            raise ValueError("Default counts cannot exceed total counts")

    def get_traces(self, discard_burnin: bool = True) -> Dict[str, np.ndarray]:
        """Get MCMC traces.

        Args:
            discard_burnin: Whether to discard burn-in iterations

        Returns:
            Dictionary of parameter traces
        """
        if discard_burnin:
            return {k: v[self.burn_in :] for k, v in self.traces.items()}
        return self.traces

    def get_estimates(self, include_variance: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """Get parameter estimates (posterior means and variances).

        Args:
            include_variance: Whether to include variance estimates

        Returns:
            Dictionary containing parameter estimates with structure:
            {
                'parameter_name': {
                    'mean': array_like,
                    'variance': array_like  # Only if include_variance=True
                }
            }
        """
        traces = self.get_traces()
        estimates = {}

        for param_name, trace in traces.items():
            param_dict = {"mean": np.mean(trace, axis=0)}
            if include_variance:
                param_dict["variance"] = np.var(trace, axis=0)
                # Add 95% credible intervals
                param_dict["ci_lower"] = np.percentile(trace, 2.5, axis=0)
                param_dict["ci_upper"] = np.percentile(trace, 97.5, axis=0)
            estimates[param_name] = param_dict

        return estimates
