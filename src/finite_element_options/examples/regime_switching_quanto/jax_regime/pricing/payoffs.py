"""Vectorized European payoff and Monte Carlo summary helpers."""

from __future__ import annotations

from typing import Any


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError(
            "JAX payoff evaluation requires finite-element-options[jax-regime]."
        ) from exc
    return jnp


def terminal_levels(states: Any, *, equity_spot: float, fx_spot: float) -> tuple[Any, Any]:
    """Map terminal log returns to equity and FX levels."""

    jnp = _jnp()
    states = jnp.asarray(states)
    return equity_spot * jnp.exp(states[:, 0]), fx_spot * jnp.exp(states[:, 1])


def _required(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required for this contract")
    return value


def payoff_samples(
    kind: str,
    states: Any,
    *,
    equity_spot: float,
    fx_spot: float,
    strike: float | None = None,
    payout: float = 1.0,
    fixed_fx: float | None = None,
    equity_barrier: float | None = None,
    fx_barrier: float | None = None,
) -> Any:
    """Evaluate one of the five accepted European terminal payoff contracts."""

    jnp = _jnp()
    equity, fx = terminal_levels(states, equity_spot=equity_spot, fx_spot=fx_spot)
    composite = equity * fx
    if kind == "composite_call":
        return jnp.maximum(composite - _required(strike, "strike"), 0.0)
    if kind == "composite_put":
        return jnp.maximum(_required(strike, "strike") - composite, 0.0)
    if kind == "composite_digital":
        return payout * (composite >= _required(strike, "strike"))
    if kind == "quanto_call":
        return _required(fixed_fx, "fixed_fx") * jnp.maximum(
            equity - _required(strike, "strike"), 0.0
        )
    if kind == "dual_trigger_protection":
        return payout * (
            (equity <= _required(equity_barrier, "equity_barrier"))
            & (fx >= _required(fx_barrier, "fx_barrier"))
        )
    raise ValueError(f"unsupported contract kind: {kind}")


def discounted_summary(samples: Any, *, rate: float, maturity: float) -> dict[str, Any]:
    """Return discounted Monte Carlo price and standard error."""

    jnp = _jnp()
    discounted = jnp.exp(-rate * maturity) * jnp.asarray(samples)
    return {
        "price": jnp.mean(discounted),
        "standard_error": jnp.std(discounted, ddof=1) / jnp.sqrt(discounted.size),
    }
