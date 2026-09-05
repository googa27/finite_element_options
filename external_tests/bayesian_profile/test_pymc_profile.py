"""Real PyMC/ArviZ smoke for the isolated Python 3.12 Bayesian profile."""

from __future__ import annotations

try:
    import arviz as _arviz  # noqa: F401
    import pymc as _pymc  # noqa: F401
except ImportError as exc:  # pragma: no cover - selected profile must fail, never skip
    raise RuntimeError("install the locked bayesian profile; this test never skips") from exc

from finite_element_options.estimation.bayesian_profile import (
    BayesianSmokeConfig,
    exact_normal_posterior,
    run_pymc_smoke,
)


def test_pymc_posterior_and_predictive_diagnostics() -> None:
    """Native PyMC NUTS must recover the identifiable exact posterior."""

    config = BayesianSmokeConfig(draws=300, warmup=300)
    result = run_pymc_smoke(config)
    exact = exact_normal_posterior(config)
    assert result["passed"] is True
    assert result["finite_log_density"] is True
    assert result["divergences"] == 0
    assert result["rhat"] <= config.maximum_rhat
    assert result["ess_bulk"] >= config.minimum_bulk_ess
    assert abs(result["posterior_mean"] - exact["mean"]) <= config.posterior_mean_tolerance
    assert (
        abs(result["posterior_predictive_sd"] - exact["posterior_predictive_sd"])
        <= config.predictive_sd_tolerance
    )
