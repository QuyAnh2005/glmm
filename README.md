# GLMM: Generalized Linear Mixed Models for Portfolio Credit Risk

This package implements a Bayesian Generalized Linear Mixed Model (GLMM) for analyzing portfolio credit risk, based on the methodology described in McNeil and Wendin's paper. The implementation focuses on an ordered binomial model with rating-specific parameters and temporal random effects.

## Features

- **Ordered Effects**:
  - Ordered intercepts (mu) across rating categories with strict ordering constraints
  - Rating-specific covariate effects (beta)
  - Rating-specific random effect loadings (phi)

- **Random Effects**:
  - AR(1) process for temporal random effects
  - Captures unobserved time-varying risk factors

- **Model Evaluation**:
  - Conditional Predictive Ordinate (CPO) computation
  - Full parameter uncertainty quantification
  - MCMC diagnostics and trace analysis

- **Implementation**:
  - Full Bayesian inference using MCMC sampling
  - Efficient parameter estimation with uncertainty quantification
  - Comprehensive test suite with unit and integration tests
  - Flexible model specification and fitting

## Installation

To install the package in development mode:

```bash
# Install main package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

### Basic Example

```python
from glmm.models import OrderedBinomialGLMM
import numpy as np

# Prepare your data
N_tk = np.array(...)  # Default counts (T x K)
n_tk = np.array(...)  # Total counts (T x K)
x_t = np.array(...)   # Covariates (T,)
rating_labels = ['CCC', 'B', 'BB', 'BBB', 'A']  # Rating categories

# Initialize and fit OrderedBinomialGLMM
model = OrderedBinomialGLMM(
    n_iter=15000,    # Number of MCMC iterations
    burn_in=5000,    # Number of burn-in iterations
    tau=100.0,       # Prior variance for mu and beta
    eta=2.0,         # Shape parameter for phi inverse-gamma prior
    v=1.0            # Scale parameter for phi inverse-gamma prior
)

# Fit model and get traces
traces = model.fit(N_tk, n_tk, x_t)

# Get parameter estimates with uncertainty
estimates = model.get_estimates()

# Print parameter estimates
print("\nOrdered Intercepts (mu):")
for k, label in enumerate(rating_labels):
    mean = estimates['mu']['mean'][k]
    std = np.sqrt(estimates['mu']['variance'][k])
    print(f"{label:4s}: {mean:6.3f} ± {std:5.3f}")

print("\nRating-specific Slopes (beta):")
for k, label in enumerate(rating_labels):
    mean = estimates['beta']['mean'][k]
    std = np.sqrt(estimates['beta']['variance'][k])
    print(f"{label:4s}: {mean:6.3f} ± {std:5.3f}")

# Model fit statistics
print(f"\nAverage log(CPO): {traces['avg_log_cpo']:.3f}")
print(f"Sum of log(CPO): {traces['sum_log_cpo']:.3f}")
```

or you can refer the [Notebook Example](./notebooks/implementation.ipynb) for detail implementation and visualization.

## Project Structure

```
glmm/
├── glmm/
│   ├── models/                    # GLMM model implementation
│   │   ├── base.py                # Abstract base class
│   │   └── ordered_binomial.py    # OrderedBinomialGLMM implementation
│   └── utils/                     # Utility functions
│       └── math.py                # Mathematical helper functions
├── examples/                      # Example scripts
│   └── run_ordered_binomial.py    # Example script for OrderedBinomialGLMM
├── tests/                         # Test suite
│   ├── test_ordered_binomial.py   # Tests for OrderedBinomialGLMM
├── setup.py                       # Package configuration
└── pyproject.toml                 # Build system requirements
```

## Development

### Running Tests

```bash
# Run quick tests
pytest tests/

# Run all tests including slow ones
pytest tests/ --runslow

# Run with coverage
pytest tests/ --cov=glmm
```

### Code Style

The codebase follows PEP 8 guidelines. To format your code:

```bash
black .
isort .
flake8
```

## License

MIT License

## References

McNeil, A. J., & Wendin, J. P. (2007). Bayesian inference for generalized linear mixed models of portfolio credit risk. Journal of Empirical Finance, 14(2), 131-149.
