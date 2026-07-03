"""Greeks computation via JAX automatic differentiation.

The functions here offer an experimental path to evaluate sensitivities
(Delta and volatility Vega) by differentiating the Black--Scholes pricing
formula with respect to spot price and volatility.  The NumPy path delegates to
the canonical Black-Scholes oracle so edge semantics stay aligned.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from typing import Any, Literal, Tuple

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jspst

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover
    JAX_AVAILABLE = False


def _option(k: float, r: float, q: float) -> EuropeanOptionBs:
    return EuropeanOptionBs(k=k, q=q, mkt=Market(r=r))


def _bs_price_numpy(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> float:
    """Black--Scholes call price using the canonical NumPy oracle."""

    return float(_option(k, r, q).call_from_volatility(t, s, sigma))


def _bs_price_jax(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Any:
    """Black--Scholes call price in JAX space for regular positive-volatility cases."""

    d1 = (jnp.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)
    return s * jnp.exp(-q * t) * jspst.norm.cdf(d1) - k * jnp.exp(
        -r * t
    ) * jspst.norm.cdf(d2)


def _requires_canonical_greek_path(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> bool:
    """Return whether JAX autodiff would hit a singular Black-Scholes expression."""

    values = (s, k, r, q, sigma, t)
    if not all(math.isfinite(value) for value in values):
        return True
    return s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0


def _greeks_numpy(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Tuple[float, float]:
    """Return ``delta`` and volatility ``vega`` using canonical analytic derivatives."""

    option = _option(k, r, q)
    delta = option.call_delta_from_volatility(t, s, sigma)
    vega = option.vega_volatility(t, s, sigma)
    return float(delta), float(vega)


def _greeks_jax(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Tuple[float, float]:
    """Return ``delta`` and volatility ``vega`` via JAX automatic differentiation."""

    if _requires_canonical_greek_path(s, k, r, q, sigma, t):
        return _greeks_numpy(s, k, r, q, sigma, t)
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
    """Compute call ``delta`` and volatility ``vega`` choosing between backends."""

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
