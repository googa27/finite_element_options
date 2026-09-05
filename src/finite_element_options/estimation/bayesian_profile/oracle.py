"""Pure-Python posterior oracle and diagnostic decision ownership."""

from __future__ import annotations

from math import sqrt
from typing import Any

from .contracts import BayesianSmokeConfig


def exact_normal_posterior(config: BayesianSmokeConfig) -> dict[str, float]:
    """Return the conjugate posterior for the synthetic known-variance mean."""

    precision = 1.0 / config.prior_sigma**2 + len(config.observations) / config.known_sigma**2
    variance = 1.0 / precision
    mean = variance * (
        config.prior_mean / config.prior_sigma**2 + sum(config.observations) / config.known_sigma**2
    )
    return {
        "mean": mean,
        "sd": sqrt(variance),
        "posterior_predictive_mean": mean,
        "posterior_predictive_sd": sqrt(config.known_sigma**2 + variance),
    }


def build_diagnostic_summary(
    *,
    engine: str,
    sampler: str,
    version: str,
    arviz_version: str,
    posterior_mean: float,
    posterior_sd: float,
    posterior_predictive_mean: float,
    posterior_predictive_sd: float,
    rhat: float,
    ess_bulk: float,
    divergences: int,
    elapsed_seconds: float,
    finite_log_density: bool,
    finite_posterior: bool,
    finite_predictive: bool,
    config: BayesianSmokeConfig,
) -> dict[str, Any]:
    """Apply one shared diagnostic contract to either inference engine."""

    exact = exact_normal_posterior(config)
    checks = {
        "finite_log_density": finite_log_density,
        "finite_posterior": finite_posterior,
        "finite_predictive": finite_predictive,
        "posterior_mean": (abs(posterior_mean - exact["mean"]) <= config.posterior_mean_tolerance),
        "posterior_sd": abs(posterior_sd - exact["sd"]) <= config.posterior_sd_tolerance,
        "posterior_predictive_mean": (
            abs(posterior_predictive_mean - exact["posterior_predictive_mean"])
            <= config.predictive_mean_tolerance
        ),
        "posterior_predictive_sd": (
            abs(posterior_predictive_sd - exact["posterior_predictive_sd"])
            <= config.predictive_sd_tolerance
        ),
        "rhat": rhat <= config.maximum_rhat,
        "ess_bulk": ess_bulk >= config.minimum_bulk_ess,
        "zero_divergences": divergences == 0,
    }
    return {
        "engine": engine,
        "sampler": sampler,
        "version": version,
        "arviz_version": arviz_version,
        "seeded": True,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "divergences": divergences,
        "posterior_predictive_mean": posterior_predictive_mean,
        "posterior_predictive_sd": posterior_predictive_sd,
        "finite_log_density": finite_log_density,
        "elapsed_seconds": elapsed_seconds,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = ["build_diagnostic_summary", "exact_normal_posterior"]
