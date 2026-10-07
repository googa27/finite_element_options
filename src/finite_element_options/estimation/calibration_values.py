"""Calibration array ownership, bounds, weights and Jacobian diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _coerce_required_array(
    name: str, values: Sequence[float] | np.ndarray
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite")
    return arr.copy()


def _coerce_optional_array(
    name: str,
    values: Sequence[float] | np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have the same shape as quote")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite")
    return arr.copy()


def _normalize_bounds(
    bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return (
            np.full(shape, -np.inf, dtype=float),
            np.full(shape, np.inf, dtype=float),
        )
    lower, upper = bounds
    lower_arr = np.broadcast_to(np.asarray(lower, dtype=float), shape).copy()
    upper_arr = np.broadcast_to(np.asarray(upper, dtype=float), shape).copy()
    if np.any(lower_arr > upper_arr):
        raise ValueError("lower calibration bounds must not exceed upper bounds")
    return lower_arr, upper_arr


def _normalize_weights(
    weights: Sequence[float] | None, shape: tuple[int, ...]
) -> np.ndarray | None:
    if weights is None:
        return None
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != shape:
        raise ValueError("weights must have the same shape as market prices")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative")
    return weights_array


def _jacobian_rank_condition(jacobian: np.ndarray) -> tuple[int, float]:
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or jac.size == 0:
        return 0, np.inf
    singular_values = np.linalg.svd(jac, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0, np.inf
    tolerance = np.finfo(float).eps * max(jac.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank < min(jac.shape) or singular_values[-1] <= tolerance:
        return rank, np.inf
    return rank, float(singular_values[0] / singular_values[-1])
