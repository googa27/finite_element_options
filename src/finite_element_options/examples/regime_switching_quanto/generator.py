"""Constrained discrete-to-continuous Markov generator conversion."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

TRADING_DAYS = 252


def discrete_to_continuous_generator(
    p_matrix: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[np.ndarray, float]:
    """Derive a nonnegative-off-diagonal CTMC generator from a transition matrix."""

    p = _validate_transition_matrix(p_matrix)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    n_states = p.shape[0]
    if n_states == 1:
        return np.zeros((1, 1), dtype=float), 0.0

    pairs = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]

    def unpack(x: np.ndarray) -> np.ndarray:
        q = np.zeros((n_states, n_states), dtype=float)
        for value, (i, j) in zip(x, pairs, strict=True):
            q[i, j] = max(float(value), 0.0)
        q[np.diag_indices(n_states)] = -q.sum(axis=1)
        return q

    def objective(x: np.ndarray) -> float:
        residual = expm(unpack(x) / float(periods_per_year)) - p
        return float(np.sum(residual * residual))

    best: Any = None
    for start in _generator_initial_guesses(p, pairs, periods_per_year):
        fitted = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * len(pairs),
            options={"ftol": 1.0e-18, "gtol": 1.0e-12, "maxiter": 5000},
        )
        if best is None or fitted.fun < best.fun:
            best = fitted
    q = unpack(best.x)
    residual = float(np.linalg.norm(expm(q / float(periods_per_year)) - p))
    return q, residual


def _validate_transition_matrix(p_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(p_matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("transition matrix must be square")
    if not np.isfinite(p).all() or np.any(p < -1.0e-12):
        raise ValueError("transition matrix must be finite and nonnegative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1.0e-8):
        raise ValueError("transition matrix rows must sum to one")
    return np.clip(p, 0.0, 1.0)


def _generator_initial_guesses(
    p: np.ndarray,
    pairs: list[tuple[int, int]],
    periods_per_year: int,
) -> list[np.ndarray]:
    log_guess = np.real_if_close(logm(p) * float(periods_per_year), tol=1000)
    if np.iscomplexobj(log_guess):
        log_guess = np.real(log_guess)
    guesses = [np.array([max(float(log_guess[i, j]), 0.0) for i, j in pairs])]
    guesses.append(np.array([max(float(p[i, j] * periods_per_year), 0.0) for i, j in pairs]))
    guesses.append(np.full(len(pairs), 0.1, dtype=float))
    return guesses
