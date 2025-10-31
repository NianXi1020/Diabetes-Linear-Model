"""Bayesian model selection via Gibbs sampling for the g-prior."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .gprior import analytic_posterior_mean, analytic_ssr_g
from .linalg import Matrix, Vector, cholesky, matmul, quantiles, transpose, vector_mean, vector_quantiles


@dataclass
class ModelSelectionResult:
    """Posterior summaries returned by :class:`GPriorModelSelector`."""

    inclusion_probabilities: List[float]
    beta_mean: Vector
    beta_ci: List[Vector]
    sigma2_mean: float
    sigma2_ci: tuple[float, float]
    beta_samples: List[Vector]
    sigma2_samples: List[float]
    z_samples: List[List[int]]


def _subset_design_matrix(X: Matrix, z: List[int]) -> Matrix:
    columns = [0] + [idx + 1 for idx, value in enumerate(z) if value == 1]
    subset = []
    for row in X:
        subset.append([row[idx] for idx in columns])
    return subset


def _log_marginal_likelihood(X: Matrix, y: Vector, g: float, nu0: float, sigma0_sq: float) -> float:
    n = len(y)
    p = len(X[0])
    ssr_g = analytic_ssr_g(X, y, g)
    return (
        -0.5 * n * math.log(math.pi)
        + math.lgamma(0.5 * (nu0 + n))
        - math.lgamma(0.5 * nu0)
        - 0.5 * p * math.log(1.0 + g)
        + 0.5 * nu0 * math.log(nu0 * sigma0_sq)
        - 0.5 * (nu0 + n) * math.log(nu0 * sigma0_sq + ssr_g)
    )


class GPriorModelSelector:
    """Gibbs sampler for Bayesian variable selection with a g-prior."""

    def __init__(
        self,
        X: Matrix,
        y: Vector,
        g: float,
        nu0: float,
        sigma0_sq: float,
        inclusion_prob: float = 0.5,
        rng: random.Random | None = None,
    ) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of observations.")
        if not 0.0 < inclusion_prob < 1.0:
            raise ValueError("The prior inclusion probability must lie in (0, 1).")
        self.X = X
        self.y = y
        self.g = g
        self.nu0 = nu0
        self.sigma0_sq = sigma0_sq
        self.inclusion_prob = inclusion_prob
        self.rng = rng or random.Random()
        self.num_predictors = len(X[0]) - 1
        if self.num_predictors < 0:
            raise ValueError("Design matrix must contain at least an intercept column.")

    def _conditional_inclusion_probability(self, z: List[int], j: int) -> float:
        z_with = z.copy()
        z_with[j] = 1
        z_without = z.copy()
        z_without[j] = 0

        X_with = _subset_design_matrix(self.X, z_with)
        X_without = _subset_design_matrix(self.X, z_without)

        log_ml_with = _log_marginal_likelihood(X_with, self.y, self.g, self.nu0, self.sigma0_sq)
        log_ml_without = _log_marginal_likelihood(X_without, self.y, self.g, self.nu0, self.sigma0_sq)

        log_prior_odds = math.log(self.inclusion_prob) - math.log(1.0 - self.inclusion_prob)
        log_odds = log_prior_odds + (log_ml_with - log_ml_without)
        return 1.0 / (1.0 + math.exp(-log_odds))

    def run(self, num_samples: int, burn_in: int = 1000, thin: int = 1) -> ModelSelectionResult:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in cannot be negative.")
        if thin <= 0:
            raise ValueError("thin must be positive.")

        total_iterations = burn_in + num_samples * thin
        z = [1] * self.num_predictors

        beta_samples: List[Vector] = []
        sigma2_samples: List[float] = []
        z_samples: List[List[int]] = []

        lower = 0.025
        upper = 0.975

        for iteration in range(total_iterations):
            for j in range(self.num_predictors):
                prob = self._conditional_inclusion_probability(z, j)
                z[j] = 1 if self.rng.random() < prob else 0

            X_current = _subset_design_matrix(self.X, z)
            beta_mean = analytic_posterior_mean(X_current, self.y, self.g)
            ssr_g = analytic_ssr_g(X_current, self.y, self.g)

            shape = 0.5 * (self.nu0 + len(self.X))
            scale = 0.5 * (self.nu0 * self.sigma0_sq + ssr_g)
            sigma2 = 1.0 / self.rng.gammavariate(shape, 1.0 / scale)

            # Sample beta conditional on sigma2.
            Xt = transpose(X_current)
            XtX = matmul(Xt, X_current)
            chol = cholesky(XtX)
            std_normals = [self.rng.gauss(0.0, 1.0) for _ in beta_mean]
            Lt = transpose(chol)
            v = [0.0] * len(beta_mean)
            for i in range(len(beta_mean) - 1, -1, -1):
                s = sum(Lt[i][k] * v[k] for k in range(i + 1, len(beta_mean)))
                v[i] = (std_normals[i] - s) / Lt[i][i]
            scale_factor = math.sqrt(self.g / (self.g + 1.0) * sigma2)
            beta_draw = [beta_mean[i] + scale_factor * v[i] for i in range(len(beta_mean))]

            full_beta = [0.0] * (self.num_predictors + 1)
            full_beta[0] = beta_draw[0]
            included_indices = [idx for idx, value in enumerate(z) if value == 1]
            for position, idx in enumerate(included_indices, start=1):
                full_beta[idx + 1] = beta_draw[position]

            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                beta_samples.append(full_beta)
                sigma2_samples.append(sigma2)
                z_samples.append(z.copy())

        inclusion_probabilities = [sum(column) / len(z_samples) for column in zip(*z_samples)] if z_samples else [0.0] * self.num_predictors
        beta_ci = quantiles(beta_samples, [lower, upper])
        sigma2_ci = tuple(vector_quantiles(sigma2_samples, [lower, upper]))

        return ModelSelectionResult(
            inclusion_probabilities=inclusion_probabilities,
            beta_mean=vector_mean(beta_samples),
            beta_ci=beta_ci,
            sigma2_mean=sum(sigma2_samples) / len(sigma2_samples),
            sigma2_ci=sigma2_ci,
            beta_samples=beta_samples,
            sigma2_samples=sigma2_samples,
            z_samples=z_samples,
        )
