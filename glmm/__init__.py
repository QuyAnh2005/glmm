"""GLMM package for Bayesian portfolio credit risk modeling."""

from .models import OrderedBinomialGLMM
from .utils.math import logit, log_likelihood_binomial

__version__ = "0.1.0"

__all__ = [
    "OrderedBinomialGLMM",
    "logit",
    "log_likelihood_binomial",
]
