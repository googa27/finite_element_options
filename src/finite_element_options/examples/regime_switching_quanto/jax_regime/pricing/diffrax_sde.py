"""Diffrax Itô-Euler route for grid-aligned regime-conditioned log SDEs."""

from __future__ import annotations

from typing import Any


def _stack() -> tuple[Any, Any, Any]:
    try:
        import diffrax
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError("Diffrax pricing requires finite-element-options[jax-regime].") from exc
    jax.config.update("jax_enable_x64", True)
    return diffrax, jax, jnp


def simulate_diffrax_terminal_states(
    regimes: Any,
    brownian_increments: Any,
    drift: Any,
    diffusion: Any,
    maturity: float,
) -> Any:
    """Solve each aligned log SDE with Diffrax Euler and common random numbers.

    Brownian increments are represented as a piecewise-linear control solely so
    the Diffrax and exact JAX paths receive identical stochastic increments.
    Euler is conditionally exact here because drift and additive diffusion are
    constant over each regime-aligned interval.
    """

    dfx, jax, jnp = _stack()
    regimes = jnp.asarray(regimes)
    brownian_increments = jnp.asarray(brownian_increments)
    drift = jnp.asarray(drift)
    diffusion = jnp.asarray(diffusion)
    steps = regimes.shape[1]
    times = jnp.linspace(0.0, maturity, steps + 1)

    def solve_one(path: Any, increments: Any) -> Any:
        cumulative = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(increments, axis=0)], axis=0)
        control = dfx.LinearInterpolation(ts=times, ys=cumulative)

        def index(time: Any) -> Any:
            located = jnp.searchsorted(times, time, side="right") - 1
            return jnp.clip(located, 0, steps - 1)

        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda time, state, args: drift[path[index(time)]]),
            dfx.ControlTerm(lambda time, state, args: diffusion[path[index(time)]], control),
        )
        solution = dfx.diffeqsolve(
            terms,
            dfx.Euler(),
            t0=0.0,
            t1=maturity,
            dt0=None,
            y0=jnp.zeros((2,)),
            stepsize_controller=dfx.StepTo(ts=times),
            saveat=dfx.SaveAt(t1=True),
            max_steps=steps,
            throw=True,
        )
        return solution.ys[0]

    return jax.jit(jax.vmap(solve_one))(regimes, brownian_increments)
