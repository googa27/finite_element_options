"""Sampling and summary helpers for the OpenTURNS UQ pilot."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import QUANTILE_LEVELS


def summarize_prices(values: np.ndarray) -> dict[str, Any]:
    """Summarize finite propagated price samples."""

    prices = np.asarray(values, dtype=float)
    finite = prices[np.isfinite(prices)]
    quantiles = np.quantile(finite, QUANTILE_LEVELS)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)),
        "quantiles": {
            str(level): float(value)
            for level, value in zip(QUANTILE_LEVELS, quantiles, strict=True)
        },
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def numpy_direct_sample(seed: int, size: int) -> np.ndarray:
    """Draw the five independent marginals using NumPy for direct-reference parity."""

    rng = np.random.default_rng(seed)
    sample = np.empty((size, 5), dtype=float)
    sample[:, :4] = rng.uniform(-1.0, 1.0, size=(size, 4))
    sample[:, 4] = rng.standard_normal(size)
    return sample


__all__ = ["numpy_direct_sample", "summarize_prices"]
