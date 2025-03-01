"""Math utility functions for GLMM models."""

import numpy as np


def logit(x: float) -> float:
    """Logistic function (sigmoid).

    Args:
        x: Input value

    Returns:
        Transformed value between 0 and 1
    """
    return 1 / (1 + np.exp(-x))


def log_likelihood_binomial(n: int, k: int, p: float, eps: float = 1e-10) -> float:
    """Calculate log likelihood for binomial distribution.

    Args:
        n: Number of trials
        k: Number of successes
        p: Probability of success
        eps: Small value to avoid log(0)

    Returns:
        Log likelihood value

    Raises:
        ValueError: If inputs are invalid (n < 0, k < 0, k > n)
    """
    if n < 0:
        raise ValueError("Number of trials must be non-negative")
    if k < 0:
        raise ValueError("Number of successes must be non-negative")
    if k > n:
        raise ValueError("Number of successes cannot exceed number of trials")

    return k * np.log(p + eps) + (n - k) * np.log(1 - p + eps)
