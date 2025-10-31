import random

from diabetes_linear_model.model_selection import GPriorModelSelector
from diabetes_linear_model.linalg import vecdot


def test_model_selection_outputs_shapes():
    rng = random.Random(42)
    n = 60
    X_base = [[rng.gauss(0.0, 1.0) for _ in range(2)] for _ in range(n)]
    X = [[1.0] + row for row in X_base]
    beta = [0.5, 1.0, 0.0]
    y = [vecdot(row, beta) + rng.gauss(0.0, 0.3) for row in X]

    selector = GPriorModelSelector(X, y, g=float(n), nu0=2.0, sigma0_sq=1.0, rng=rng)
    result = selector.run(num_samples=200, burn_in=50, thin=2)

    assert len(result.beta_samples[0]) == len(X[0])
    assert len(result.z_samples[0]) == len(X[0]) - 1
    assert all(0.0 <= prob <= 1.0 for prob in result.inclusion_probabilities)
    assert len(result.beta_ci) == 2
    assert result.sigma2_ci[0] > 0
