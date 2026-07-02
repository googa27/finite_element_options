"""Greeks computation via JAX automatic differentiation.

The functions here offer an experimental path to evaluate sensitivities
(Delta and Vega) by differentiating the Black--Scholes pricing formula
with respect to spot price and volatility.  When JAX is not available or
proves slower than NumPy, a pure NumPy implementation is used.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jspst

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover
    JAX_AVAILABLE = False


def _bs_price_numpy(s: float, k: float, r: float, q: float, sigma: float, t: float) -> float:
    """Black--Scholes call price using NumPy/SciPy."""
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def _bs_price_jax(s: float, k: float, r: float, q: float, sigma: float, t: float) -> float:
    """Black--Scholes call price in JAX space."""
    d1 = (jnp.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)
    return s * jnp.exp(-q * t) * jspst.norm.cdf(d1) - k * jnp.exp(-r * t) * jspst.norm.cdf(d2)


def _greeks_numpy(s: float, k: float, r: float, q: float, sigma: float, t: float) -> Tuple[float, float]:
    """Return ``delta`` and ``vega`` using analytic derivatives."""
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    delta = np.exp(-q * t) * norm.cdf(d1)
    vega = k * np.exp(-r * t) * norm.pdf(d2) * np.sqrt(t)
    return float(delta), float(vega)


def _greeks_jax(s: float, k: float, r: float, q: float, sigma: float, t: float) -> Tuple[float, float]:
    """Return ``delta`` and ``vega`` via JAX automatic differentiation."""
    price = _bs_price_jax
    delta = jax.grad(lambda _s: price(_s, k, r, q, sigma, t))(s)
    vega = jax.grad(lambda _sigma: price(s, k, r, q, _sigma, t))(sigma)
    return float(delta), float(vega)


def benchmark_greeks(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
) -> dict[str, tuple[float, int]]:
    """Measure runtime and peak memory for both backends."""
    results: dict[str, tuple[float, int]] = {}
    tracemalloc.start()
    start = time.perf_counter()
    _greeks_numpy(s, k, r, q, sigma, t)
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["numpy"] = (runtime, peak)
    if JAX_AVAILABLE:
        tracemalloc.start()
        start = time.perf_counter()
        _greeks_jax(s, k, r, q, sigma, t)
        runtime = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["jax"] = (runtime, peak)
    return results


def compute_greeks(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    *,
    backend: Literal["auto", "numpy", "jax"] = "auto",
) -> Tuple[float, float]:
    """Compute ``delta`` and ``vega`` choosing between backends."""
    if backend == "jax":
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX backend requested but not installed")
        return _greeks_jax(s, k, r, q, sigma, t)
    if backend == "numpy":
        return _greeks_numpy(s, k, r, q, sigma, t)
    stats = benchmark_greeks(s, k, r, q, sigma, t)
    if "jax" in stats and stats["jax"][0] <= stats["numpy"][0] * 1.5:
        return _greeks_jax(s, k, r, q, sigma, t)
    return _greeks_numpy(s, k, r, q, sigma, t)
