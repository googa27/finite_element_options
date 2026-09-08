"""Real isolated-profile tests for DYNAMAX, NumPyro, and Diffrax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from finite_element_options.examples.regime_switching_quanto.jax_regime.contracts import (
    JaxRegimeStudyConfig,
)

from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.dynamax_adapter import (
    dynamax_marginal_log_prob,
    fit_dynamax_hmm,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.forward import (
    forecast_next_state_probs,
    gaussian_hmm_log_prob,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.generator_check import (
    check_ctmc_generator,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.numpyro_model import (
    run_numpyro_hmm,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing.diffrax_sde import (
    simulate_diffrax_terminal_states,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing.exact import (
    simulate_exact_terminal_states,
    simulate_regime_paths,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing.validation import (
    martingale_checks,
    refinement_invariance,
    strike_monotonicity,
)

jax.config.update("jax_enable_x64", True)


def _fixture() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    initial = jnp.array([0.65, 0.35])
    transition = jnp.array([[0.96, 0.04], [0.10, 0.90]])
    means = jnp.array([[0.01, -0.01], [-0.04, 0.02]])
    covariances = jnp.array(
        [
            [[0.05, 0.012], [0.012, 0.03]],
            [[0.20, -0.04], [-0.04, 0.12]],
        ]
    )
    return initial, transition, means, covariances


def test_first_forecast_interval_advances_the_filtered_hmm_law() -> None:
    transition = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    forecast = forecast_next_state_probs(jnp.array([1.0, 0.0]), transition)
    np.testing.assert_array_equal(forecast, jnp.array([0.0, 1.0]))
    paths = simulate_regime_paths(jr.key(99), forecast, transition, paths=32, steps=1)
    np.testing.assert_array_equal(paths, jnp.ones((32, 1), dtype=paths.dtype))


def test_forward_likelihood_matches_dynamax() -> None:
    initial, transition, means, covariances = _fixture()
    observations = jr.multivariate_normal(jr.key(1), means[0], covariances[0], (64,))
    ours = gaussian_hmm_log_prob(observations, initial, transition, means, covariances)
    theirs = dynamax_marginal_log_prob(observations, initial, transition, means, covariances)
    np.testing.assert_allclose(ours, theirs, atol=1.0e-9, rtol=1.0e-9)


def test_ctmc_generator_satisfies_cone_and_reconstruction_laws() -> None:
    _initial, transition, _means, _covariances = _fixture()
    result = check_ctmc_generator(transition)
    assert result["passed"] is True
    assert result["minimum_off_diagonal"] >= 0.0
    assert result["maximum_diagonal"] <= 0.0
    assert result["maximum_row_sum_error"] <= 1.0e-10


def test_dynamax_synthetic_fit_is_lawful_and_ordered() -> None:
    initial, transition, means, covariances = _fixture()
    # Deliberately concatenate calm/crisis blocks: this is a fit-law smoke, not a recovery claim.
    first = jr.multivariate_normal(jr.key(2), means[0], covariances[0], (180,))
    second = jr.multivariate_normal(jr.key(3), means[1], covariances[1], (80,))
    result = fit_dynamax_hmm(jnp.concatenate([first, second]), num_states=2, seed=7, em_iters=25)
    params = result["parameters"]
    np.testing.assert_allclose(np.sum(params["transition_matrix"], axis=1), 1.0, atol=1.0e-10)
    assert np.all(np.linalg.eigvalsh(params["covariances"]) > 0.0)
    assert np.diff(result["annualized_composite_volatility"]).min() > 0.0
    assert result["finite"]


def test_numpyro_hmm_runs_marginalized_two_chain_smoke() -> None:
    initial, transition, means, covariances = _fixture()
    observations = jr.multivariate_normal(jr.key(4), means[0], covariances[0], (80,))
    result = run_numpyro_hmm(
        observations,
        initial=initial,
        transition=transition,
        means=means,
        covariances=covariances,
        seed=17,
        warmup=40,
        samples=40,
        chains=2,
    )
    assert result["chains"] == 2
    assert result["divergences"] == 0
    assert "diverging" in result["diagnostic_fields"]
    assert result["grouped_samples"]["transition_matrix"].shape[:2] == (2, 40)
    assert result["maximum_rhat"] <= 1.2
    assert result["minimum_ess"] >= 5.0
    assert result["finite"]


def test_posterior_repricing_stratifies_draws_within_each_chain(
    monkeypatch: object,
) -> None:
    from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing import (
        study as pricing_study,
    )

    initial, transition, means, covariances = _fixture()
    grouped = {
        "initial_probs": jnp.tile(initial, (2, 4, 1)),
        "transition_matrix": jnp.tile(transition, (2, 4, 1, 1)),
        "means": jnp.tile(means, (2, 4, 1, 1)),
        "covariances": jnp.tile(covariances, (2, 4, 1, 1, 1)),
    }
    monkeypatch.setattr(pricing_study, "_POSTERIOR_INTERVAL_PATHS", 16)  # type: ignore[attr-defined]
    result = pricing_study._posterior_price_intervals(
        {"grouped_samples": grouped},
        jr.multivariate_normal(jr.key(23), means[0], covariances[0], (12,)),
        [{"name": "call", "kind": "composite_call", "strike": 80_000.0}],
        equity_spot=100.0,
        fx_spot=800.0,
        config=JaxRegimeStudyConfig(pricing_paths=16),
    )["call"]
    assert result["posterior_draws"] == 8
    assert result["posterior_draws_per_chain"] == 4
    assert result["available_draws_per_chain"] == 4
    assert "within-chain stratification" in result["selection"]


def test_statsmodels_var_baseline_scores_identical_bivariate_holdout() -> None:
    from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.statsmodels_baseline import (
        fit_statsmodels_var_baseline,
    )

    observations = 0.2 * jr.normal(jr.key(19), (160, 2))
    result = fit_statsmodels_var_baseline(observations, 120)
    assert result["engine"] == "statsmodels_VAR_1"
    assert result["finite"] is True
    assert result["lags"] == 1
    assert result["observed_lags_used_for_sequential_scoring"] is True
    assert np.isfinite(result["heldout_mean_log_score"])


def test_pricing_law_helpers_cover_zero_variance_and_monotonicity() -> None:
    config = JaxRegimeStudyConfig()
    deterministic_log_returns = jnp.array(
        [
            (config.foreign_rate - config.dividend_yield) * config.maturity_years,
            (config.domestic_rate - config.foreign_rate) * config.maturity_years,
        ]
    )
    states = jnp.tile(deterministic_log_returns, (32, 1))
    martingale = martingale_checks(
        states,
        equity_spot=100.0,
        fx_spot=900.0,
        config=config,
    )
    monotonicity = strike_monotonicity(
        states,
        equity_spot=100.0,
        fx_spot=900.0,
        config=config,
    )
    assert martingale["passed"] is True
    assert monotonicity["passed"] is True


def test_diffrax_matches_exact_aligned_log_sde_pathwise() -> None:
    regimes = jnp.array([[0, 1, 1, 0], [1, 1, 0, 0]])
    increments = jnp.array(
        [
            [[0.1, -0.2], [0.05, 0.3], [-0.2, 0.1], [0.4, -0.1]],
            [[-0.2, 0.1], [0.1, 0.05], [0.3, -0.2], [-0.1, 0.4]],
        ]
    )
    drift = jnp.array([[0.1, 0.2], [-0.1, 0.05]])
    diffusion = jnp.array(
        [
            [[0.2, 0.0], [0.05, 0.1]],
            [[0.3, 0.0], [-0.04, 0.2]],
        ]
    )
    diffrax_states = simulate_diffrax_terminal_states(regimes, increments, drift, diffusion, 1.0)
    exact_states = simulate_exact_terminal_states(regimes, increments, drift, diffusion, 1.0)
    np.testing.assert_allclose(diffrax_states, exact_states, atol=1.0e-12, rtol=1.0e-12)
    refinement = refinement_invariance(regimes, increments, drift, diffusion, 1.0)
    assert refinement["passed"] is True
