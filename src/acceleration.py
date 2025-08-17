"""Acceleration utilities using Numba.

This module contains small prototypes exploring the use of Numba to
speed up identified hot loops such as payoff evaluations over a grid.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fall back to pure Python
    NUMBA_AVAILABLE = False

    def njit(func):  # type: ignore
        """Dummy decorator used when Numba is unavailable."""
        return func


@njit
def call_payoff_grid(s: np.ndarray, k: float) -> np.ndarray:
    """Vectorised intrinsic value of a call option.

    Parameters
    ----------
    s:
        Grid of underlying asset prices.
    k:
        Strike price of the option.
    """
    out = np.empty_like(s)
    for i in range(s.size):
        diff = s[i] - k
        out[i] = diff if diff > 0.0 else 0.0
    return out


@njit
def put_payoff_grid(s: np.ndarray, k: float) -> np.ndarray:
    """Vectorised intrinsic value of a put option."""
    out = np.empty_like(s)
    for i in range(s.size):
        diff = k - s[i]
        out[i] = diff if diff > 0.0 else 0.0
    return out
