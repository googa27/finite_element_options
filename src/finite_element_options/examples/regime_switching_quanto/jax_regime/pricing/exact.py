"""JAX exact-step simulation oracle for piecewise-constant log diffusions."""

from __future__ import annotations

from typing import Any


def _stack() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
    except ModuleNotFoundError as exc:
        raise ImportError("JAX exact pricing requires finite-element-options[jax-regime].") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, jr


def simulate_regime_paths(
    key: Any,
    initial_probs: Any,
    transition_matrix: Any,
    *,
    paths: int,
    steps: int,
) -> Any:
    """Sample left-continuous discrete regime paths with JAX PRNG keys."""

    jax, jnp, jr = _stack()
    first_key, scan_key = jr.split(key)
    initial = jr.categorical(first_key, jnp.log(initial_probs), shape=(paths,))
    keys = jr.split(scan_key, max(steps - 1, 0))

    def advance(current: Any, current_key: Any) -> tuple[Any, Any]:
        following = jr.categorical(current_key, jnp.log(transition_matrix[current]), axis=-1)
        return following, following

    if steps == 1:
        return initial[:, None]
    _last, rest = jax.lax.scan(advance, initial, keys)
    return jnp.concatenate([initial[:, None], jnp.swapaxes(rest, 0, 1)], axis=1)


def correlated_diffusion(equity_vol: Any, fx_vol: Any, correlation: Any) -> Any:
    """Construct regime-specific lower diffusion matrices."""

    _jax, jnp, _jr = _stack()
    equity_vol = jnp.asarray(equity_vol)
    fx_vol = jnp.asarray(fx_vol)
    correlation = jnp.asarray(correlation)
    matrices = jnp.zeros((equity_vol.shape[0], 2, 2))
    matrices = matrices.at[:, 0, 0].set(equity_vol)
    matrices = matrices.at[:, 1, 0].set(correlation * fx_vol)
    matrices = matrices.at[:, 1, 1].set(fx_vol * jnp.sqrt(jnp.maximum(1.0 - correlation**2, 0.0)))
    return matrices


def simulate_exact_terminal_states(
    regimes: Any,
    brownian_increments: Any,
    drift: Any,
    diffusion: Any,
    maturity: float,
) -> Any:
    """Return exact terminal log returns conditional on grid-aligned regimes."""

    _jax, jnp, _jr = _stack()
    steps = regimes.shape[1]
    dt = maturity / steps
    selected_drift = drift[regimes]
    selected_diffusion = diffusion[regimes]
    stochastic = jnp.einsum("ptij,ptj->pti", selected_diffusion, brownian_increments)
    return jnp.sum(selected_drift * dt + stochastic, axis=1)


def draw_paths_and_increments(
    key: Any,
    initial_probs: Any,
    transition_matrix: Any,
    *,
    paths: int,
    steps: int,
    maturity: float,
) -> tuple[Any, Any]:
    """Draw common regime paths and Brownian increments for exact/Diffrax parity."""

    _jax, jnp, jr = _stack()
    regime_key, brownian_key = jr.split(key)
    regimes = simulate_regime_paths(
        regime_key, initial_probs, transition_matrix, paths=paths, steps=steps
    )
    increments = jr.normal(brownian_key, (paths, steps, 2)) * jnp.sqrt(maturity / steps)
    return regimes, increments
