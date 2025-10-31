"""Run the Hoff (2009) azdiabetes linear model analysis."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure the ``src`` directory is on the Python path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diabetes_linear_model.data import load_az_diabetes, prepare_regression_matrices
from diabetes_linear_model.gprior import fit_g_prior_model
from diabetes_linear_model.model_selection import GPriorModelSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/azdiabetes.dat.txt"),
        help="Path to the azdiabetes data file.",
    )
    parser.add_argument("--response", default="glu", help="Name of the response variable")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["diabetes"],
        help="Columns to exclude from the predictor matrix.",
    )
    parser.add_argument("--g", type=float, default=None, help="Value of Zellner's g. Defaults to the sample size.")
    parser.add_argument("--nu0", type=float, default=2.0, help="Prior degrees of freedom for sigma^2")
    parser.add_argument("--sigma0-sq", dest="sigma0_sq", type=float, default=1.0, help="Prior scale for sigma^2")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--samples", type=int, default=5000, help="Posterior samples for part (a)")
    parser.add_argument("--selection-samples", type=int, default=4000, help="Number of retained Gibbs samples for part (b)")
    parser.add_argument("--burn-in", type=int, default=1000, help="Burn-in iterations for the Gibbs sampler")
    parser.add_argument("--thin", type=int, default=5, help="Thinning interval for the Gibbs sampler")
    return parser.parse_args()


def print_coefficient_table(names: list[str], means: list[float], lowers: list[float], uppers: list[float]) -> None:
    width = max(len(name) for name in names) + 2
    header = f"{'variable'.ljust(width)}{'mean':>12}{'lower':>12}{'upper':>12}"
    print(header)
    print("-" * len(header))
    for name, mean, lower, upper in zip(names, means, lowers, uppers):
        print(f"{name.ljust(width)}{mean:12.3f}{lower:12.3f}{upper:12.3f}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data = load_az_diabetes(args.data_path)
    predictors, _, X, y = prepare_regression_matrices(data, response=args.response, exclude=args.exclude)

    g_value = args.g if args.g is not None else float(len(X))

    print("=== Part (a): Full model with g-prior ===")
    posterior = fit_g_prior_model(
        X,
        y,
        g=g_value,
        nu0=args.nu0,
        sigma0_sq=args.sigma0_sq,
        num_samples=args.samples,
        rng=rng,
    )
    print_coefficient_table(
        predictors,
        posterior.beta_mean,
        posterior.beta_ci[0],
        posterior.beta_ci[1],
    )
    print(
        "Sigma^2 mean: {:.3f}, 95% CI: ({:.3f}, {:.3f})".format(
            posterior.sigma2_mean, posterior.sigma2_ci[0], posterior.sigma2_ci[1]
        )
    )

    print("\n=== Part (b): Model selection and averaging ===")
    selector = GPriorModelSelector(
        X,
        y,
        g=g_value,
        nu0=args.nu0,
        sigma0_sq=args.sigma0_sq,
        rng=rng,
    )
    result = selector.run(
        num_samples=args.selection_samples,
        burn_in=args.burn_in,
        thin=args.thin,
    )

    print("Posterior inclusion probabilities (excluding intercept):")
    for name, prob in zip(predictors[1:], result.inclusion_probabilities):
        print(f"  {name}: {prob:.3f}")

    print("\nPosterior coefficient summaries (model averaged):")
    print_coefficient_table(
        predictors,
        result.beta_mean,
        result.beta_ci[0],
        result.beta_ci[1],
    )
    print(
        "Sigma^2 mean: {:.3f}, 95% CI: ({:.3f}, {:.3f})".format(
            result.sigma2_mean, result.sigma2_ci[0], result.sigma2_ci[1]
        )
    )


if __name__ == "__main__":
    main()
