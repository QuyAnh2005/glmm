"""Example script for running OrderedBinomialGLMM model."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from glmm import OrderedBinomialGLMM

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data_dir = Path("../data/processed")
ratings = pd.read_csv(data_dir / "ratings.csv", header=None)
defaults = pd.read_csv(data_dir / "defaults.csv", header=None)

# Convert to numpy arrays
n_tk = ratings.values
N_tk = defaults.values

# Define rating labels and covariates
rating_labels = ['CCC', 'B', 'BB', 'BBB', 'A']  # Original order in data
x_t = np.array([2.7, 3.9, 3.1, 2.8, 3.3, 2.7, -2.9, 6.4, 3.2, 2.8])

T, K = N_tk.shape
print(f"Data dimensions: {T} time periods, {K} rating categories")
print("\nRating categories:", ", ".join(rating_labels))

# Initialize and fit model
model = OrderedBinomialGLMM(
    n_iter=15000,
    burn_in=5000,
    tau=100.0,
    eta=2.0,
    v=1.0,
    random_seed=42
)

print("\nFitting model...")
traces = model.fit(N_tk, n_tk, x_t)
estimates = model.get_estimates()

# Print parameter estimates
print("\nParameter Estimates:")
print("-" * 50)

# Mu (ordered intercepts)
print("\nOrdered Intercepts (mu):")
for k, label in enumerate(rating_labels):
    mean = estimates['mu']['mean'][k]
    std = np.sqrt(estimates['mu']['variance'][k])
    print(f"{label:4s}: {mean:6.3f} ± {std:5.3f}")

# Beta (rating-specific slopes)
print("\nRating-specific Slopes (beta):")
for k, label in enumerate(rating_labels):
    mean = estimates['beta']['mean'][k]
    std = np.sqrt(estimates['beta']['variance'][k])
    print(f"{label:4s}: {mean:6.3f} ± {std:5.3f}")

# Phi (rating-specific loadings)
print("\nRating-specific Loadings (phi):")
for k, label in enumerate(rating_labels):
    mean = estimates['phi']['mean'][k]
    std = np.sqrt(estimates['phi']['variance'][k])
    print(f"{label:4s}: {mean:6.3f} ± {std:5.3f}")

# Alpha (AR(1) coefficient)
alpha_mean = estimates['alpha']['mean']
alpha_std = np.sqrt(estimates['alpha']['variance'])
print(f"\nAR(1) coefficient (alpha): {alpha_mean:.3f} ± {alpha_std:.3f}")

# Plot traces
plt.figure(figsize=(15, 10))

# Plot mu traces
plt.subplot(2, 2, 1)
for k, label in enumerate(rating_labels):
    plt.plot(traces['mu'][:, k], label=label, alpha=0.7)
plt.title('Ordered Intercepts (mu)')
plt.xlabel('Iteration')
plt.legend()

# Plot beta traces
plt.subplot(2, 2, 2)
for k, label in enumerate(rating_labels):
    plt.plot(traces['beta'][:, k], label=label, alpha=0.7)
plt.title('Rating-specific Slopes (beta)')
plt.xlabel('Iteration')
plt.legend()

# Plot phi traces
plt.subplot(2, 2, 3)
for k, label in enumerate(rating_labels):
    plt.plot(traces['phi'][:, k], label=label, alpha=0.7)
plt.title('Rating-specific Loadings (phi)')
plt.xlabel('Iteration')
plt.legend()

# Plot alpha trace
plt.subplot(2, 2, 4)
plt.plot(traces['alpha'], label='alpha', alpha=0.7)
plt.title('AR(1) coefficient (alpha)')
plt.xlabel('Iteration')
plt.legend()

plt.tight_layout()
plt.savefig('parameter_traces.png')
plt.close()

# Plot random effects
plt.figure(figsize=(12, 6))
b_mean = estimates['b']['mean']
b_std = np.sqrt(estimates['b']['variance'])

plt.plot(range(T), b_mean, 'b-', label='Posterior Mean')
plt.fill_between(range(T), 
                b_mean - 2*b_std,
                b_mean + 2*b_std,
                alpha=0.2,
                label='95% Credible Interval')
plt.title('Random Effects (b)')
plt.xlabel('Time')
plt.ylabel('Effect Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('random_effects.png')
plt.close()

# Print model fit statistics
print("\nModel Fit Statistics:")
print("-" * 50)
print(f"Average log(CPO): {traces['avg_log_cpo']:.3f}")
print(f"Sum of log(CPO): {traces['sum_log_cpo']:.3f}")

# Save results
results = {
    'estimates': estimates,
    'traces': traces,
    'model_info': {
        'n_iter': model.n_iter,
        'burn_in': model.burn_in,
        'tau': model.tau,
        'eta': model.eta,
        'v': model.v
    }
}

np.save('model_results.npy', results)
