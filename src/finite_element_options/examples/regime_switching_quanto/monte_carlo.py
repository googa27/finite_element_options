"""Seeded Monte Carlo oracle for regime-switching quanto research prices."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from finite_element_options.examples.regime_switching_quanto.contracts import (
    ContractSpec,
    MonteCarloPriceResult,
    TwoFactorRegimeModel,
)


def price_contract_monte_carlo(
    model: TwoFactorRegimeModel,
    contract: ContractSpec,
    *,
    maturity: float,
    equity_spot: float,
    fx_spot: float,
    paths: int,
    seed: int,
    steps_per_year: int = 252,
) -> MonteCarloPriceResult:
    """Price by a deterministic seeded daily-transition Monte Carlo oracle.

    Regime transitions use the exact one-step CTMC transition matrix
    ``expm(Q * dt)``.  Conditional on the regime at the start of each step, log
    equity and FX increments use the CLP-domestic risk-neutral drifts from
    :class:`TwoFactorRegimeModel` and correlated Gaussian shocks.  This is a
    research oracle for tests, not a production path engine.
    """

    payoffs = _simulate_terminal_payoffs(
        model,
        contract,
        maturity=maturity,
        equity_spot=equity_spot,
        fx_spot=fx_spot,
        paths=paths,
        seed=seed,
        steps_per_year=steps_per_year,
    )
    discounted = np.exp(-model.domestic_rate * maturity) * payoffs
    return MonteCarloPriceResult(
        price=float(np.mean(discounted)),
        standard_error=float(np.std(discounted, ddof=1) / np.sqrt(paths)),
        paths=int(paths),
        steps=int(np.ceil(steps_per_year * maturity)),
    )


def price_contracts_monte_carlo(
    model: TwoFactorRegimeModel,
    contracts: list[ContractSpec],
    *,
    maturity: float,
    equity_spot: float,
    fx_spot: float,
    paths: int,
    seed: int,
    steps_per_year: int = 252,
) -> list[MonteCarloPriceResult]:
    """Reuse one simulated terminal state to price several contracts."""

    x, y, steps = _simulate_terminal_states(
        model,
        maturity=maturity,
        paths=paths,
        seed=seed,
        steps_per_year=steps_per_year,
    )
    out = []
    discount = np.exp(-model.domestic_rate * maturity)
    for contract in contracts:
        discounted = discount * contract.payoff(x, y, equity_spot=equity_spot, fx_spot=fx_spot)
        out.append(
            MonteCarloPriceResult(
                price=float(np.mean(discounted)),
                standard_error=float(np.std(discounted, ddof=1) / np.sqrt(paths)),
                paths=int(paths),
                steps=steps,
            )
        )
    return out


def _simulate_terminal_payoffs(
    model: TwoFactorRegimeModel,
    contract: ContractSpec,
    *,
    maturity: float,
    equity_spot: float,
    fx_spot: float,
    paths: int,
    seed: int,
    steps_per_year: int,
) -> np.ndarray:
    x, y, _ = _simulate_terminal_states(
        model,
        maturity=maturity,
        paths=paths,
        seed=seed,
        steps_per_year=steps_per_year,
    )
    return contract.payoff(x, y, equity_spot=equity_spot, fx_spot=fx_spot)


def _simulate_terminal_states(
    model: TwoFactorRegimeModel,
    *,
    maturity: float,
    paths: int,
    seed: int,
    steps_per_year: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if maturity <= 0.0 or not np.isfinite(maturity):
        raise ValueError("maturity must be positive and finite")
    if paths < 2:
        raise ValueError("paths must be at least 2")
    if steps_per_year < 1:
        raise ValueError("steps_per_year must be positive")

    rng = np.random.default_rng(seed)
    steps = int(np.ceil(steps_per_year * maturity))
    dt = maturity / steps
    transition = expm(np.asarray(model.generator, dtype=float) * dt)
    transition = np.maximum(transition, 0.0)
    transition = transition / transition.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(transition, axis=1)

    regimes = rng.choice(
        model.n_regimes, size=paths, p=np.asarray(model.current_probabilities, dtype=float)
    )
    x = np.zeros(paths, dtype=float)
    y = np.zeros(paths, dtype=float)
    sig_s = model.scaled_equity_vol
    sig_f = model.scaled_fx_vol
    rho = np.asarray(model.correlation, dtype=float)
    a_s, a_f = model.drifts()
    sqrt_dt = np.sqrt(dt)

    for _ in range(steps):
        z_s = rng.standard_normal(paths)
        z_ind = rng.standard_normal(paths)
        current = regimes.copy()
        x += a_s[current] * dt + sig_s[current] * sqrt_dt * z_s
        z_f = (
            rho[current] * z_s + np.sqrt(np.maximum(1.0 - rho[current] * rho[current], 0.0)) * z_ind
        )
        y += a_f[current] * dt + sig_f[current] * sqrt_dt * z_f
        uniforms = rng.random(paths)
        regimes = _transition_regimes(current, uniforms, cumulative)
    return x, y, steps


def _transition_regimes(
    current: np.ndarray, uniforms: np.ndarray, cumulative: np.ndarray
) -> np.ndarray:
    next_regimes = np.empty_like(current)
    for regime in range(cumulative.shape[0]):
        mask = current == regime
        if np.any(mask):
            next_regimes[mask] = np.searchsorted(cumulative[regime], uniforms[mask], side="right")
    return next_regimes
