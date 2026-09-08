"""No-arbitrage and grid-refinement checks for regime pricing paths."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from ..contracts import JaxRegimeStudyConfig
from ..utils import stack as _stack
from .diffrax_sde import simulate_diffrax_terminal_states
from .exact import simulate_exact_terminal_states
from .payoffs import discounted_summary, payoff_samples


def martingale_checks(
    states: Any,
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, Any]:
    """Check domestic-discounted tradable expectations within five MC errors."""

    _jax, jnp, _jr = _stack()
    discount = jnp.exp(-config.domestic_rate * config.maturity_years)
    terminal_equity = equity_spot * jnp.exp(states[:, 0])
    terminal_fx = fx_spot * jnp.exp(states[:, 1])
    samples = {
        "foreign_currency": discount * terminal_fx,
        "foreign_equity_in_domestic_currency": discount * terminal_equity * terminal_fx,
    }
    targets = {
        "foreign_currency": fx_spot * jnp.exp(-config.foreign_rate * config.maturity_years),
        "foreign_equity_in_domestic_currency": equity_spot
        * fx_spot
        * jnp.exp(-config.dividend_yield * config.maturity_years),
    }
    results: dict[str, Any] = {}
    for name, values in samples.items():
        estimate = jnp.mean(values)
        error = estimate - targets[name]
        standard_error = jnp.std(values, ddof=1) / jnp.sqrt(len(values))
        z_score = jnp.where(
            standard_error > 0.0,
            error / standard_error,
            jnp.where(jnp.abs(error) <= 1.0e-10, 0.0, jnp.inf),
        )
        results[name] = {
            "discounted_estimate": float(estimate),
            "discounted_target": float(targets[name]),
            "standard_error": float(standard_error),
            "z_score": float(z_score),
            "passed_5se": bool(jnp.abs(z_score) <= 5.0),
        }
    return {
        "checks": results,
        "passed": all(bool(row["passed_5se"]) for row in results.values()),
    }


def strike_monotonicity(
    states: Any,
    *,
    equity_spot: float,
    fx_spot: float,
    config: JaxRegimeStudyConfig,
) -> dict[str, Any]:
    """Verify pathwise monotonicity of composite calls and puts over three strikes."""

    composite_spot = equity_spot * fx_spot
    strikes = [0.90 * composite_spot, composite_spot, 1.10 * composite_spot]
    calls: list[float] = []
    puts: list[float] = []
    for strike in strikes:
        for kind, values in (
            (
                "composite_call",
                payoff_samples(
                    "composite_call",
                    states,
                    equity_spot=equity_spot,
                    fx_spot=fx_spot,
                    strike=strike,
                ),
            ),
            (
                "composite_put",
                payoff_samples(
                    "composite_put",
                    states,
                    equity_spot=equity_spot,
                    fx_spot=fx_spot,
                    strike=strike,
                ),
            ),
        ):
            price = float(
                discounted_summary(
                    values,
                    rate=config.domestic_rate,
                    maturity=config.maturity_years,
                )["price"]
            )
            (calls if kind == "composite_call" else puts).append(price)
    call_passed = all(left >= right for left, right in pairwise(calls))
    put_passed = all(left <= right for left, right in pairwise(puts))
    return {
        "strikes_clp": strikes,
        "call_prices_clp": calls,
        "put_prices_clp": puts,
        "call_nonincreasing": call_passed,
        "put_nondecreasing": put_passed,
        "passed": call_passed and put_passed,
    }


def refinement_invariance(
    regimes: Any,
    increments: Any,
    drift: Any,
    diffusion: Any,
    maturity: float,
    *,
    seed: int = 20260907,
) -> dict[str, Any]:
    """Check stochastic Brownian-bridge refinement against the exact additive endpoint."""

    _jax, jnp, jr = _stack()
    paths = min(512, int(regimes.shape[0]))
    coarse_regimes = regimes[:paths]
    coarse_increments = increments[:paths]
    coarse_exact = simulate_exact_terminal_states(
        coarse_regimes, coarse_increments, drift, diffusion, maturity
    )
    refined_regimes = jnp.repeat(coarse_regimes, 2, axis=1)
    dt = maturity / coarse_regimes.shape[1]
    bridge = jr.normal(jr.key(seed), coarse_increments.shape) * jnp.sqrt(dt / 4.0)
    first_half = 0.5 * coarse_increments + bridge
    second_half = 0.5 * coarse_increments - bridge
    refined_increments = jnp.stack((first_half, second_half), axis=2).reshape(
        paths, refined_regimes.shape[1], coarse_increments.shape[2]
    )
    refined_exact = simulate_exact_terminal_states(
        refined_regimes, refined_increments, drift, diffusion, maturity
    )
    refined_diffrax = simulate_diffrax_terminal_states(
        refined_regimes, refined_increments, drift, diffusion, maturity
    )
    exact_error = float(jnp.max(jnp.abs(refined_exact - coarse_exact)))
    diffrax_error = float(jnp.max(jnp.abs(refined_diffrax - coarse_exact)))
    return {
        "paths": paths,
        "coarse_steps": int(coarse_regimes.shape[1]),
        "refined_steps": int(refined_regimes.shape[1]),
        "exact_brownian_bridge_maximum_error": exact_error,
        "diffrax_brownian_bridge_maximum_error": diffrax_error,
        "construction": (
            "conditional Brownian bridge: each daily Wiener increment is split into "
            "two stochastic half-day increments with unchanged regime and exact sum"
        ),
        "passed": exact_error <= 1.0e-10 and diffrax_error <= 1.0e-10,
    }
