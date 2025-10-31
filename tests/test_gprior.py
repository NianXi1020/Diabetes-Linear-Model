import random

from diabetes_linear_model.gprior import analytic_posterior_mean, fit_g_prior_model
from diabetes_linear_model.linalg import vecdot


def test_posterior_mean_matches_closed_form():
    rng = random.Random(123)
    n, p = 40, 3
    X = [[rng.gauss(0.0, 1.0) for _ in range(p)] for _ in range(n)]
    beta_true = [1.5, -2.0, 0.5]
    y = [vecdot(row, beta_true) + rng.gauss(0.0, 0.5) for row in X]

    g = float(n)
    nu0 = 2.0
    sigma0_sq = 1.0

    posterior = fit_g_prior_model(X, y, g=g, nu0=nu0, sigma0_sq=sigma0_sq, num_samples=2000, rng=rng)
    expected_mean = analytic_posterior_mean(X, y, g)

    for est, expected in zip(posterior.beta_mean, expected_mean):
        assert abs(est - expected) < 0.3
    assert len(posterior.beta_ci) == 2
    assert posterior.sigma2_ci[0] > 0
