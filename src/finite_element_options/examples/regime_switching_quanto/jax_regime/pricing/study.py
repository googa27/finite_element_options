"""Pricing experiment composition and independent oracle summaries."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
import math
from typing import Any
from zipfile import ZipFile

from ..contracts import (
    MAX_POSTERIOR_PRICING_PATH_STEPS,
    MAX_PRICING_PATH_STEPS,
    JaxRegimeStudyConfig,
    PriceEstimate,
)
from ..data import _MEMBER_ROOT
from ..hmm.forward import gaussian_hmm_filter_probs
from ..utils import stack as _stack
from .analytic import one_state_price
from .exact import correlated_diffusion, draw_paths_and_increments, simulate_exact_terminal_states
from .payoffs import discounted_summary, payoff_samples

_POSTERIOR_INTERVAL_PATHS = 131_072
_POSTERIOR_INTERVAL_DRAWS_PER_CHAIN = 64


def _bounded_pricing_paths(preferred: int, requested: int, steps: int) -> int:
    """Bound one simulation allocation by the validated path-step ceiling."""

    if steps < 1:
        raise ValueError("steps must be positive")
    maximum_paths = MAX_PRICING_PATH_STEPS // steps
    if maximum_paths < 2:
        raise ValueError("path-step ceiling cannot support finite Monte Carlo diagnostics")
    return min(max(preferred, requested), maximum_paths)


def _publication_or_requested_paths(
    config: JaxRegimeStudyConfig, canonical_floor: int, steps: int
) -> int:
    """Use a publication floor only for exact canonical config; otherwise honor the request."""

    preferred = canonical_floor if config == JaxRegimeStudyConfig() else config.pricing_paths
    return _bounded_pricing_paths(preferred, config.pricing_paths, steps)


def _posterior_pricing_paths(config: JaxRegimeStudyConfig, posterior_draws: int) -> int:
    """Bound both per-draw memory and total posterior repricing work."""

    if posterior_draws < 1:
        raise ValueError("posterior_draws must be positive")
    per_draw = _publication_or_requested_paths(
        config, _POSTERIOR_INTERVAL_PATHS, config.pricing_steps
    )
    total_cap = MAX_POSTERIOR_PRICING_PATH_STEPS // (posterior_draws * config.pricing_steps)
    if total_cap < 2:
        raise ValueError("total posterior path-step ceiling cannot support finite diagnostics")
    return min(per_draw, total_cap)


def _matched_oracle_config(
    config: JaxRegimeStudyConfig,
    *,
    maturity_years: float,
    domestic_rate: float,
    foreign_rate: float,
    dividend_yield: float,
) -> JaxRegimeStudyConfig:
    """Return an archive-aligned config whose paths are capped before validation."""

    steps = int(round(maturity_years * config.steps_per_year))
    paths = _publication_or_requested_paths(config, 16_384, steps)
    return replace(
        config,
        maturity_years=maturity_years,
        pricing_paths=paths,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        dividend_yield=dividend_yield,
    )


def _oracle_z_score(error: float, standard_error: float) -> float | None:
    """Return a fail-closed standardized error, including zero-variance samples."""

    if standard_error > 0.0:
        return error / standard_error
    if error == 0.0:
        return 0.0
    return None


def _risk_neutral_coefficients(
    covariances_percent: Any, config: JaxRegimeStudyConfig
) -> tuple[Any, Any, Any, Any, Any]:
    _jax, jnp, _jr = _stack()
    annual_covariance = jnp.asarray(covariances_percent) * 252.0 / 10_000.0
    equity_vol = jnp.sqrt(annual_covariance[:, 0, 0])
    fx_vol = jnp.sqrt(annual_covariance[:, 1, 1])
    correlation = annual_covariance[:, 0, 1] / (equity_vol * fx_vol)
    correlation = jnp.clip(correlation, -0.98, 0.98)
    drift = jnp.stack(
        [
            config.foreign_rate
            - config.dividend_yield
            - correlation * equity_vol * fx_vol
            - 0.5 * equity_vol**2,
            config.domestic_rate - config.foreign_rate - 0.5 * fx_vol**2,
        ],
        axis=1,
    )
    diffusion = correlated_diffusion(equity_vol, fx_vol, correlation)
    return drift, diffusion, equity_vol, fx_vol, correlation


def _contracts(equity_spot: float, fx_spot: float) -> list[dict[str, Any]]:
    composite = equity_spot * fx_spot
    return [
        {"name": "ATM composite call", "kind": "composite_call", "strike": composite},
        {"name": "ATM composite put", "kind": "composite_put", "strike": composite},
        {
            "name": "Composite digital",
            "kind": "composite_digital",
            "strike": composite,
            "payout": 1_000_000.0,
        },
        {
            "name": "ATM fixed-FX quanto call",
            "kind": "quanto_call",
            "strike": equity_spot,
            "fixed_fx": fx_spot,
        },
        {
            "name": "Dual-trigger protection",
            "kind": "dual_trigger_protection",
            "equity_barrier": 0.90 * equity_spot,
            "fx_barrier": 1.08 * fx_spot,
            "payout": 1_000_000.0,
        },
    ]


def _price_contracts(
    states: Any,
    contracts: list[dict[str, Any]],
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, dict[str, float]]:
    prices: dict[str, dict[str, float]] = {}
    for contract in contracts:
        terms = {key: value for key, value in contract.items() if key not in {"name", "kind"}}
        samples = payoff_samples(
            contract["kind"],
            states,
            equity_spot=equity_spot,
            fx_spot=fx_spot,
            **terms,
        )
        summary = discounted_summary(
            samples, rate=config.domestic_rate, maturity=config.maturity_years
        )
        prices[contract["name"]] = PriceEstimate(
            price_clp=float(summary["price"]),
            standard_error_clp=float(summary["standard_error"]),
        ).to_dict()
    return prices


def _one_state_oracles(
    increments: Any,
    drift: Any,
    diffusion: Any,
    equity_vol: Any,
    fx_vol: Any,
    correlation: Any,
    contracts: list[dict[str, Any]],
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, dict[str, float | bool | None]]:
    _jax, jnp, _jr = _stack()
    regimes = jnp.zeros(increments.shape[:2], dtype=int)
    states = simulate_exact_terminal_states(
        regimes, increments, drift, diffusion, config.maturity_years
    )
    monte_carlo = _price_contracts(
        states,
        contracts,
        equity_spot=equity_spot,
        fx_spot=fx_spot,
        config=config,
    )
    results: dict[str, dict[str, float | bool | None]] = {}
    for contract in contracts:
        terms = {key: value for key, value in contract.items() if key not in {"name", "kind"}}
        analytic = one_state_price(
            contract["kind"],
            equity_spot=equity_spot,
            fx_spot=fx_spot,
            equity_vol=float(equity_vol[0]),
            fx_vol=float(fx_vol[0]),
            correlation=float(correlation[0]),
            domestic_rate=config.domestic_rate,
            foreign_rate=config.foreign_rate,
            dividend_yield=config.dividend_yield,
            maturity=config.maturity_years,
            **terms,
        )
        estimate = monte_carlo[contract["name"]]
        error = estimate["price_clp"] - analytic
        standard_error = estimate["standard_error_clp"]
        z_score = _oracle_z_score(error, standard_error)
        results[contract["name"]] = {
            "analytic_price_clp": analytic,
            "mc_price_clp": estimate["price_clp"],
            "mc_standard_error_clp": standard_error,
            "z_score": z_score,
            "passed_5se": z_score is not None and abs(z_score) <= 5.0,
        }
    return results


def _posterior_price_intervals(
    posterior: dict[str, Any],
    observations: Any,
    contracts: list[dict[str, Any]],
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, dict[str, Any]]:
    _jax, jnp, jr = _stack()
    grouped = posterior["grouped_samples"]
    chain_count, available_draws_per_chain = grouped["transition_matrix"].shape[:2]
    selected_draws_per_chain = min(
        _POSTERIOR_INTERVAL_DRAWS_PER_CHAIN, int(available_draws_per_chain)
    )
    draw_pairs = [
        (chain, int(draw))
        for chain in range(int(chain_count))
        for draw in jnp.linspace(0, available_draws_per_chain - 1, selected_draws_per_chain).astype(
            int
        )
    ]
    steps = config.pricing_steps
    reduced_paths = _posterior_pricing_paths(config, len(draw_pairs))
    common_key = jr.key(config.seed + 200)
    collected: dict[str, list[float]] = {contract["name"]: [] for contract in contracts}
    collected_mc_se: dict[str, list[float]] = {contract["name"]: [] for contract in contracts}
    for chain, draw in draw_pairs:
        transition = grouped["transition_matrix"][chain, draw]
        initial_draw = grouped["initial_probs"][chain, draw]
        covariances = grouped["covariances"][chain, draw]
        current = gaussian_hmm_filter_probs(
            observations,
            initial_draw,
            transition,
            grouped["means"][chain, draw],
            covariances,
        )[-1]
        regimes, increments = draw_paths_and_increments(
            common_key,
            current,
            transition,
            paths=reduced_paths,
            steps=steps,
            maturity=config.maturity_years,
        )
        drift, diffusion, _equity_vol, _fx_vol, _correlation = _risk_neutral_coefficients(
            covariances, config
        )
        states = simulate_exact_terminal_states(
            regimes, increments, drift, diffusion, config.maturity_years
        )
        priced = _price_contracts(
            states,
            contracts,
            equity_spot=equity_spot,
            fx_spot=fx_spot,
            config=config,
        )
        for name, summary in priced.items():
            collected[name].append(summary["price_clp"])
            collected_mc_se[name].append(summary["standard_error_clp"])
    result: dict[str, Any] = {}
    for name, values in collected.items():
        array = jnp.asarray(values)
        mc_errors = jnp.asarray(collected_mc_se[name])
        quantile_levels = jnp.array([0.05, 0.5, 0.95])
        q05, median, q95 = jnp.quantile(array, quantile_levels)
        posterior_half_width = (q95 - q05) / 2.0
        split_quantiles = jnp.stack(
            [jnp.quantile(array[offset::2], quantile_levels) for offset in (0, 1)]
        )
        split_delta = jnp.max(jnp.abs(split_quantiles - jnp.array([q05, median, q95])))
        split_delta_ratio = split_delta / jnp.maximum(posterior_half_width, 1.0e-12)
        median_mc_error = jnp.median(mc_errors)
        ratio = median_mc_error / jnp.maximum(posterior_half_width, 1.0e-12)
        result[name] = {
            "posterior_parameter_q05_clp": float(q05),
            "posterior_parameter_median_clp": float(median),
            "posterior_parameter_q95_clp": float(q95),
            "conditional_mc_standard_error_median_clp": float(median_mc_error),
            "conditional_mc_standard_error_maximum_clp": float(jnp.max(mc_errors)),
            "posterior_parameter_half_width_clp": float(posterior_half_width),
            "mc_se_to_posterior_half_width": float(ratio),
            "total_computational_q05_clp": float(q05 - 1.645 * median_mc_error),
            "total_computational_q95_clp": float(q95 + 1.645 * median_mc_error),
            "posterior_draws": len(values),
            "posterior_draws_per_chain": selected_draws_per_chain,
            "available_draws_per_chain": int(available_draws_per_chain),
            "split_subsample_max_quantile_delta_to_full_half_width": float(split_delta_ratio),
            "paths_per_draw": reduced_paths,
            "common_random_numbers": True,
            "selection": (
                "deterministic within-chain stratification; equal draws from every chain, with "
                "even/odd split-subsample quantile stability recorded"
            ),
            "conditioning": (
                "empirical-Bayes NumPyro posterior conditional on the volatility-ordered DYNAMAX "
                "EM reference fit"
            ),
        }
    return result


def _matched_historical_oracle(
    archive_snapshot: bytes,
    contracts: list[dict[str, Any]],
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, Any]:
    """Replay the archived three-state model through JAX exact-step Monte Carlo."""

    _jax, jnp, jr = _stack()
    with ZipFile(BytesIO(archive_snapshot)) as bundle:
        calibration = json.loads(bundle.read(f"{_MEMBER_ROOT}calibration.json"))
        prior = json.loads(bundle.read(f"{_MEMBER_ROOT}pricing_results.json"))
    prior_model = prior["model"]
    matched_config = _matched_oracle_config(
        config,
        maturity_years=float(prior["maturity_years"]),
        domestic_rate=float(prior_model["domestic_rate"]),
        foreign_rate=float(prior_model["foreign_rate"]),
        dividend_yield=float(prior_model["dividend_yield"]),
    )
    transition = jnp.asarray(calibration["transition_matrix"])
    current = jnp.asarray(calibration["current_probabilities"])
    drift = jnp.stack(
        [
            jnp.asarray(prior_model["equity_log_drift"]),
            jnp.asarray(prior_model["fx_log_drift"]),
        ],
        axis=1,
    )
    diffusion = correlated_diffusion(
        jnp.asarray(prior_model["equity_vol"]),
        jnp.asarray(prior_model["fx_vol"]),
        jnp.asarray(prior_model["correlation"]),
    )
    steps = matched_config.pricing_steps
    paths = matched_config.pricing_paths
    regimes, increments = draw_paths_and_increments(
        jr.key(config.seed + 1_200),
        current,
        transition,
        paths=paths,
        steps=steps,
        maturity=matched_config.maturity_years,
    )
    states = simulate_exact_terminal_states(
        regimes, increments, drift, diffusion, matched_config.maturity_years
    )
    jax_prices = _price_contracts(
        states,
        contracts,
        equity_spot=equity_spot,
        fx_spot=fx_spot,
        config=matched_config,
    )
    prior_prices = {row["contract"]: row for row in prior["pricing"]}
    comparisons: dict[str, Any] = {}
    for name, current_result in jax_prices.items():
        previous = prior_prices[name]
        difference = current_result["price_clp"] - float(previous["mc_clp"])
        combined_error = math.hypot(
            current_result["standard_error_clp"],
            float(previous["mc_standard_error_clp"]),
        )
        z_score = _oracle_z_score(difference, combined_error)
        comparisons[name] = {
            "jax_exact_step_clp": current_result["price_clp"],
            "jax_standard_error_clp": current_result["standard_error_clp"],
            "archived_numpy_exact_step_clp": float(previous["mc_clp"]),
            "archived_numpy_standard_error_clp": float(previous["mc_standard_error_clp"]),
            "difference_clp": difference,
            "combined_standard_error_clp": combined_error,
            "z_score": z_score,
            "passed_5se": z_score is not None and abs(z_score) <= 5.0,
        }
    assumption_match = {
        "equity_spot": math.isclose(equity_spot, float(prior["spot"]["sp500"])),
        "fx_spot": math.isclose(fx_spot, float(prior["spot"]["usdclp"])),
        "maturity": math.isclose(config.maturity_years, float(prior["maturity_years"])),
        "domestic_rate": math.isclose(config.domestic_rate, float(prior_model["domestic_rate"])),
        "foreign_rate": math.isclose(config.foreign_rate, float(prior_model["foreign_rate"])),
        "dividend_yield": math.isclose(config.dividend_yield, float(prior_model["dividend_yield"])),
    }
    return {
        "purpose": "numerical parity only; this reuses the archived three-state parameters and is distinct from the selected four-state model",
        "paths": paths,
        "seed": config.seed + 1_200,
        "assumption_match": assumption_match,
        "comparisons": comparisons,
        "passed": all(assumption_match.values())
        and all(bool(row["passed_5se"]) for row in comparisons.values()),
    }


def _historical_prices(archive_snapshot: bytes) -> list[dict[str, Any]]:
    with ZipFile(BytesIO(archive_snapshot)) as bundle:
        prior = json.loads(bundle.read(f"{_MEMBER_ROOT}pricing_results.json"))
    return [
        {
            "contract": row["contract"],
            "fine_fem_clp": row["fem_fine_clp"],
            "richardson_heuristic_clp": row["richardson_extrapolated_clp"],
            "numpy_exact_step_mc_clp": row["mc_clp"],
            "numpy_mc_standard_error_clp": row["mc_standard_error_clp"],
        }
        for row in prior["pricing"]
    ]
