"""Isolated Python 3.12 Bayesian and JAX-native evidence profile."""

from .contracts import BayesianSmokeConfig
from .numpyro_smoke import (
    jax_fem_differentiation_status,
    require_jax_fem_differentiation,
    run_numpyro_smoke,
)
from .oracle import exact_normal_posterior
from .profile import run_bayesian_jax_profile, stable_environment_checks
from .pymc_smoke import run_pymc_smoke

__all__ = [
    "BayesianSmokeConfig",
    "exact_normal_posterior",
    "jax_fem_differentiation_status",
    "require_jax_fem_differentiation",
    "run_bayesian_jax_profile",
    "run_numpyro_smoke",
    "run_pymc_smoke",
    "stable_environment_checks",
]
