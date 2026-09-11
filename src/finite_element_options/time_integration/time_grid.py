"""Pure theta parameter validation and exact output-grid canonicalization."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np


def _validate_theta(value: float, name: str) -> float:
    """Return a validated theta parameter."""

    theta = float(value)
    if not np.isfinite(theta):
        raise ValueError(f"{name} must be finite")
    if theta < 0.0 or theta > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return theta


def _validate_startup_count(value: int, name: str, *, minimum: int) -> int:
    """Require an exact integral schedule count without boolean coercion."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer count")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _validate_time_grid(t: Iterable[float]) -> tuple[float, ...]:
    """Materialize and validate a strictly increasing finite time grid."""

    time_grid = tuple(float(item) for item in t)
    if len(time_grid) < 2:
        raise ValueError("time grid must contain at least two nodes")
    arr = np.asarray(time_grid, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("time grid nodes must be finite")
    with np.errstate(over="ignore"):
        steps = np.diff(arr)
        horizon = arr[-1] - arr[0]
    if not np.isfinite(horizon) or not np.all(np.isfinite(steps)):
        raise ValueError("time grid intervals and horizon must be finite")
    if not np.all(steps > 0.0):
        raise ValueError("time grid nodes must be strictly increasing")
    return time_grid


def _canonical_local_steps(time_grid: tuple[float, ...]) -> tuple[float, ...]:
    """Return local widths, canonicalizing roundoff-uniform grids.

    ``np.linspace`` grids often differ by a few ulps between adjacent
    intervals.  Treating those artifacts as distinct PDE steps defeats sparse
    factorization reuse without adding mathematical information.  Genuinely
    nonuniform grids keep their local widths exactly.
    """

    raw = np.diff(np.asarray(time_grid, dtype=float))
    representative = (time_grid[-1] - time_grid[0]) / (len(time_grid) - 1)
    # An absolute tolerance would replace genuinely unequal small intervals,
    # changing the discretization when the same problem uses different time units.
    if np.allclose(raw, representative, rtol=1.0e-12, atol=0.0):
        return tuple(float(representative) for _ in raw)
    return tuple(float(item) for item in raw)
