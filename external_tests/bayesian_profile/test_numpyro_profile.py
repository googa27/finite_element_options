"""Real NumPyro/JAX smoke for the isolated Python 3.12 Bayesian-JAX profile."""

from __future__ import annotations

import pytest

try:
    import arviz as _arviz  # noqa: F401
    import jax as _jax  # noqa: F401
    import numpyro as _numpyro  # noqa: F401
except ImportError as exc:  # pragma: no cover - selected profile must fail, never skip
    raise RuntimeError("install the locked bayesian-jax profile; this test never skips") from exc

from finite_element_options.estimation.bayesian_profile import (
    BayesianSmokeConfig,
    exact_normal_posterior,
    jax_fem_differentiation_status,
    require_jax_fem_differentiation,
    run_numpyro_smoke,
)


def test_numpyro_posterior_predictive_and_log_density() -> None:
    """JAX-native NumPyro NUTS must recover the identifiable exact posterior."""

    config = BayesianSmokeConfig(draws=300, warmup=300)
    result = run_numpyro_smoke(config)
    exact = exact_normal_posterior(config)
    assert result["passed"] is True
    assert result["sampler"] == "jax_native_nuts"
    assert result["jax_backend"] == "cpu"
    assert result["finite_log_density"] is True
    assert result["divergences"] == 0
    assert abs(result["posterior_mean"] - exact["mean"]) <= config.posterior_mean_tolerance
    assert (
        abs(result["posterior_predictive_sd"] - exact["posterior_predictive_sd"])
        <= config.predictive_sd_tolerance
    )


def test_jax_fem_differentiation_remains_fail_closed() -> None:
    """JAX inference must not imply automatic differentiation through FEM."""

    assert jax_fem_differentiation_status()["supported"] is False
    with pytest.raises(NotImplementedError, match="not a pure JAX trace"):
        require_jax_fem_differentiation()
