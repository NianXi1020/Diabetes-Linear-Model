"""Utilities for Bayesian linear modeling of the azdiabetes dataset."""

from .data import TabularData, load_az_diabetes, prepare_regression_matrices
from .gprior import GPriorPosterior, fit_g_prior_model
from .model_selection import GPriorModelSelector, ModelSelectionResult

__all__ = [
    "TabularData",
    "load_az_diabetes",
    "prepare_regression_matrices",
    "GPriorPosterior",
    "fit_g_prior_model",
    "GPriorModelSelector",
    "ModelSelectionResult",
]
