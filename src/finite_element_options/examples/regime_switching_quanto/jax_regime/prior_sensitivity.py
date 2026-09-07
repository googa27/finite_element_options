"""Weak/reference/strong prior sensitivity for the marginalized NumPyro HMM."""

from __future__ import annotations

from typing import Any

from .contracts import (
    JaxRegimeStudyConfig,
    NumPyroPriorConfig,
    PUBLICATION_MAX_RHAT,
    PUBLICATION_MIN_CHAINS,
    PUBLICATION_MIN_DRAWS_PER_CHAIN,
    PUBLICATION_MIN_ESS,
)
from .hmm.forward import gaussian_hmm_filter_probs
from .hmm.numpyro_model import run_numpyro_hmm
from .pricing.exact import draw_paths_and_increments, simulate_exact_terminal_states
from .pricing.study import _price_contracts, _risk_neutral_coefficients
from .utils import stack as _stack, to_python as _python


def _reprice_posterior_mean(
    posterior: dict[str, Any],
    observations: Any,
    contracts: list[dict[str, Any]],
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, Any]:
    _jax, _jnp, jr = _stack()
    mean = posterior["posterior_mean"]
    current = gaussian_hmm_filter_probs(
        observations,
        mean["initial_probs"],
        mean["transition_matrix"],
        mean["means"],
        mean["covariances"],
    )[-1]
    paths = max(4_096, config.pricing_paths)
    steps = round(config.maturity_years * config.steps_per_year)
    regimes, increments = draw_paths_and_increments(
        jr.key(config.seed + 1_700),
        current,
        mean["transition_matrix"],
        paths=paths,
        steps=steps,
        maturity=config.maturity_years,
    )
    drift, diffusion, equity_vol, fx_vol, correlation = _risk_neutral_coefficients(
        mean["covariances"], config
    )
    states = simulate_exact_terminal_states(
        regimes, increments, drift, diffusion, config.maturity_years
    )
    return {
        "initial_probs": _python(mean["initial_probs"]),
        "transition_matrix": _python(mean["transition_matrix"]),
        "annualized_equity_volatility": _python(equity_vol),
        "annualized_fx_volatility": _python(fx_vol),
        "correlation": _python(correlation),
        "paths": paths,
        "point_prices": _price_contracts(
            states,
            contracts,
            equity_spot=equity_spot,
            fx_spot=fx_spot,
            config=config,
        ),
    }


def run_prior_sensitivity(
    observations: Any,
    reference_parameters: dict[str, Any],
    reference_posterior: dict[str, Any],
    contracts: list[dict[str, Any]],
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, Any]:
    """Run weak/strong full-data fits and compare them with the reference posterior."""

    profiles: dict[str, Any] = {
        "reference": {
            "prior": reference_posterior["prior"],
            "diagnostics": {
                "divergences": reference_posterior["divergences"],
                "maximum_rhat": reference_posterior["maximum_rhat"],
                "minimum_ess": reference_posterior["minimum_ess"],
                "worst_rhat": reference_posterior["worst_rhat"],
                "worst_ess": reference_posterior["worst_ess"],
                "finite": reference_posterior["finite"],
            },
            "pricing": _reprice_posterior_mean(
                reference_posterior,
                observations,
                contracts,
                equity_spot=equity_spot,
                fx_spot=fx_spot,
                config=config,
            ),
        }
    }
    warmup = config.warmup
    samples = config.posterior_samples
    for offset, prior in enumerate((NumPyroPriorConfig.weak(), NumPyroPriorConfig.strong())):
        fitted = run_numpyro_hmm(
            observations,
            initial=reference_parameters["initial_probs"],
            transition=reference_parameters["transition_matrix"],
            means=reference_parameters["means"],
            covariances=reference_parameters["covariances"],
            seed=config.seed + 1_500 + offset,
            warmup=warmup,
            samples=samples,
            chains=config.chains,
            prior=prior,
        )
        profiles[prior.name] = {
            "prior": fitted["prior"],
            "diagnostics": {
                "divergences": fitted["divergences"],
                "maximum_rhat": fitted["maximum_rhat"],
                "minimum_ess": fitted["minimum_ess"],
                "worst_rhat": fitted["worst_rhat"],
                "worst_ess": fitted["worst_ess"],
                "finite": fitted["finite"],
            },
            "pricing": _reprice_posterior_mean(
                fitted,
                observations,
                contracts,
                equity_spot=equity_spot,
                fx_spot=fx_spot,
                config=config,
            ),
        }
    reference_prices = profiles["reference"]["pricing"]["point_prices"]
    maximum_relative_delta = 0.0
    for profile_name in ("weak", "strong"):
        price_deltas: dict[str, Any] = {}
        for contract_name, result in profiles[profile_name]["pricing"]["point_prices"].items():
            reference_price = reference_prices[contract_name]["price_clp"]
            difference = result["price_clp"] - reference_price
            relative = abs(difference) / max(abs(reference_price), 1.0)
            maximum_relative_delta = max(maximum_relative_delta, relative)
            price_deltas[contract_name] = {
                "difference_clp": difference,
                "absolute_relative_difference": relative,
            }
        profiles[profile_name]["price_delta_vs_reference"] = price_deltas
    sensitivity_passed = (
        config.chains >= PUBLICATION_MIN_CHAINS
        and samples >= PUBLICATION_MIN_DRAWS_PER_CHAIN
        and all(
            profiles[name]["diagnostics"]["divergences"] == 0
            and profiles[name]["diagnostics"]["maximum_rhat"] <= PUBLICATION_MAX_RHAT
            and profiles[name]["diagnostics"]["minimum_ess"] >= PUBLICATION_MIN_ESS
            and profiles[name]["diagnostics"]["finite"]
            for name in ("reference", "weak", "strong")
        )
    )
    return {
        "same_full_observation_count": len(observations),
        "sensitivity_warmup": warmup,
        "sensitivity_draws_per_chain": samples,
        "chains": config.chains,
        "publication_thresholds": {
            "minimum_chains": PUBLICATION_MIN_CHAINS,
            "minimum_draws_per_chain": PUBLICATION_MIN_DRAWS_PER_CHAIN,
            "maximum_rhat": PUBLICATION_MAX_RHAT,
            "minimum_ess": PUBLICATION_MIN_ESS,
            "zero_divergences": True,
            "finite": True,
        },
        "common_random_numbers_for_pricing": True,
        "profiles": profiles,
        "maximum_absolute_relative_price_delta_vs_reference": maximum_relative_delta,
        "passed": sensitivity_passed,
    }
