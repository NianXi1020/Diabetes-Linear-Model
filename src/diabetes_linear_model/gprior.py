"""Posterior computations for g-prior linear regression."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .linalg import (
    Matrix,
    Vector,
    add_vectors,
    cholesky,
    matvec,
    matmul,
    quantiles,
    scale_vector,
    solve_pos_def,
    transpose,
    vecdot,
    vector_mean,
    vector_quantiles,
)


@dataclass
class GPriorPosterior:
    """Summary of the posterior distribution under a g-prior."""

    beta_mean: Vector
    beta_ci: List[Vector]
    sigma2_mean: float
    sigma2_ci: Tuple[float, float]
    beta_samples: List[Vector]
    sigma2_samples: List[float]


def _posterior_hyperparameters(
    X: Matrix, y: Vector, g: float
) -> Tuple[Vector, Matrix, Matrix, float]:
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = matvec(Xt, y)
    beta_ols = solve_pos_def(XtX, Xty)
    shrinkage = g / (g + 1.0)
    beta_mean = scale_vector(beta_ols, shrinkage)
    ssr_g = vecdot(y, y) - shrinkage * vecdot(Xty, beta_ols)
    chol = cholesky(XtX)
    return beta_mean, XtX, chol, ssr_g


def _sample_beta(
    chol: Matrix,
    beta_mean: Vector,
    sigma2: float,
    g: float,
    rng: random.Random,
) -> Vector:
    shrinkage = g / (g + 1.0)
    std_normals = [rng.gauss(0.0, 1.0) for _ in beta_mean]
    Lt = transpose(chol)
    v = [0.0] * len(beta_mean)
    for i in range(len(beta_mean) - 1, -1, -1):
        s = sum(Lt[i][k] * v[k] for k in range(i + 1, len(beta_mean)))
        v[i] = (std_normals[i] - s) / Lt[i][i]
    scale = math.sqrt(shrinkage * sigma2)
    return add_vectors(beta_mean, [scale * value for value in v])


def fit_g_prior_model(
    X: Matrix,
    y: Vector,
    g: float,
    nu0: float,
    sigma0_sq: float,
    num_samples: int = 5000,
    cred_mass: float = 0.95,
    rng: random.Random | None = None,
) -> GPriorPosterior:
    """Draw posterior samples for a g-prior regression model."""

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of observations.")
    if not X or not X[0]:
        raise ValueError("Design matrix cannot be empty.")

    rng = rng or random.Random()

    beta_mean_conditional, XtX, chol, ssr_g = _posterior_hyperparameters(X, y, g)

    shape = 0.5 * (nu0 + len(X))
    scale = 0.5 * (nu0 * sigma0_sq + ssr_g)

    sigma2_samples: List[float] = []
    beta_samples: List[Vector] = []

    for _ in range(num_samples):
        sigma2 = 1.0 / rng.gammavariate(shape, 1.0 / scale)
        sigma2_samples.append(sigma2)
        beta_samples.append(_sample_beta(chol, beta_mean_conditional, sigma2, g, rng))

    lower = (1.0 - cred_mass) / 2.0
    upper = 1.0 - lower
    beta_ci = quantiles(beta_samples, [lower, upper])
    sigma2_ci = tuple(vector_quantiles(sigma2_samples, [lower, upper]))

    return GPriorPosterior(
        beta_mean=vector_mean(beta_samples),
        beta_ci=beta_ci,
        sigma2_mean=sum(sigma2_samples) / len(sigma2_samples),
        sigma2_ci=sigma2_ci,
        beta_samples=beta_samples,
        sigma2_samples=sigma2_samples,
    )


def analytic_posterior_mean(X: Matrix, y: Vector, g: float) -> Vector:
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = matvec(Xt, y)
    beta_ols = solve_pos_def(XtX, Xty)
    shrinkage = g / (g + 1.0)
    return scale_vector(beta_ols, shrinkage)


def analytic_ssr_g(X: Matrix, y: Vector, g: float) -> float:
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = matvec(Xt, y)
    beta_ols = solve_pos_def(XtX, Xty)
    shrinkage = g / (g + 1.0)
    return vecdot(y, y) - shrinkage * vecdot(Xty, beta_ols)
