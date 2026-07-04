"""Greeks computation with explicit method, object, and benchmark metadata.

This module deliberately separates three concepts that are easy to conflate:

* analytical Black-Scholes oracle Greeks;
* coordinate-aware numerical recovery on a price grid;
* JAX automatic differentiation of the analytical Black-Scholes formula.

It does not claim automatic differentiation through the FEM assembly or linear
solve.  When JAX is used, timings synchronize asynchronous dispatch and separate
host-device transfer, cold compile/first execution, and warmed execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import tracemalloc
from typing import Any, Literal, Tuple

import numpy as np

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jspst

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    jspst = None  # type: ignore[assignment]
    JAX_AVAILABLE = False

Backend = Literal["auto", "numpy", "jax"]
GreekName = Literal["delta", "vega"]


@dataclass(frozen=True)
class GreekObservation:
    """A single Greek value plus the mathematical object it differentiates."""

    greek: GreekName
    value: float
    method: str
    differentiated_object: str
    input_variable: str
    units: str
    status: Literal["finite", "undefined"]
    oracle_error_abs: float | None = None
    bump_size: float | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class GreekComputationReport:
    """Auditable sensitivity result with backend and runtime metadata."""

    delta: GreekObservation
    vega: GreekObservation
    backend_requested: Backend
    backend_used: Literal["numpy", "jax"]
    differentiated_object: str
    dtype: str
    device: str
    jax_enable_x64: bool | None
    fallback_reason: str | None = None

    def as_tuple(self) -> Tuple[float, float]:
        """Return the legacy ``(delta, vega)`` tuple."""
        return self.delta.value, self.vega.value


@dataclass(frozen=True)
class GridGreekRecovery:
    """Coordinate-aware Greek recovery from a price grid."""

    values: tuple[float, ...]
    method: str
    differentiated_object: str
    input_variable: str
    coordinate_units: str
    stencil: str


@dataclass(frozen=True)
class BackendBenchmark:
    """Runtime benchmark metadata for one Greek backend.

    ``memory_bytes`` preserves the historical tuple contract as host Python peak
    memory measured with ``tracemalloc``. It is deliberately not an accelerator
    memory claim.
    """

    runtime_seconds: float
    transfer_seconds: float | None = None
    compile_seconds: float | None = None
    warmed_seconds: float | None = None
    synchronized: bool = True
    dtype: str = "float64"
    device: str = "cpu"
    jax_enable_x64: bool | None = None
    memory_bytes: int | None = None

    def as_legacy_tuple(self) -> tuple[float, int]:
        """Return the historical ``(runtime_seconds, memory_bytes)`` shape."""
        return self.runtime_seconds, int(self.memory_bytes or 0)

    def __iter__(self):
        """Iterate like the historical two-item benchmark tuple."""
        return iter(self.as_legacy_tuple())

    def __getitem__(self, index: int) -> float | int:
        """Index like the historical two-item benchmark tuple."""
        return self.as_legacy_tuple()[index]

    def __len__(self) -> int:
        """Return the historical benchmark tuple length."""
        return 2


def _time_with_host_peak(call: Any) -> tuple[float, int, Any]:
    """Run ``call`` while measuring elapsed time and host Python peak memory."""

    tracemalloc.start()
    try:
        start = time.perf_counter()
        result = call()
        runtime = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return runtime, int(peak_bytes), result


def _option(k: float, r: float, q: float) -> EuropeanOptionBs:
    return EuropeanOptionBs(k=k, q=q, mkt=Market(r=r))


def _bs_price_numpy(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> float:
    """Black-Scholes call price using the canonical NumPy oracle."""

    return float(_option(k, r, q).call_from_volatility(t, s, sigma))


def _bs_price_jax(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Any:
    """Black-Scholes call price in JAX space for regular positive-volatility cases."""

    assert jnp is not None and jspst is not None
    d1 = (jnp.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)
    return s * jnp.exp(-q * t) * jspst.norm.cdf(d1) - k * jnp.exp(
        -r * t
    ) * jspst.norm.cdf(d2)


def _jax_delta_vega(
    s: Any, k: Any, r: Any, q: Any, sigma: Any, t: Any
) -> tuple[Any, Any]:
    """Return JAX AD delta and volatility vega for the analytical formula."""

    assert jax is not None
    price = _bs_price_jax
    delta = jax.grad(lambda _s: price(_s, k, r, q, sigma, t))(s)
    vega = jax.grad(lambda _sigma: price(s, k, r, q, _sigma, t))(sigma)
    return delta, vega


def _block_until_ready(value: Any) -> Any:
    """Synchronize JAX values before timing or converting them."""

    if isinstance(value, tuple):
        return tuple(_block_until_ready(item) for item in value)
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if JAX_AVAILABLE and jax is not None:  # pragma: no branch - defensive
        return jax.block_until_ready(value)
    return value


def _requires_canonical_greek_path(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> bool:
    """Return whether JAX autodiff would hit a singular or saturated expression."""

    values = (s, k, r, q, sigma, t)
    if not all(math.isfinite(value) for value in values):
        return True
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return True
    sqrt_t = math.sqrt(t)
    variance_time = sigma * sigma * t
    d1 = (math.log(s / k) + (r - q) * t + 0.5 * variance_time) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if not math.isfinite(d1) or not math.isfinite(d2):
        return True
    return max(abs(d1), abs(d2)) > 12.0


def _greeks_numpy(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Tuple[float, float]:
    """Return ``delta`` and volatility ``vega`` using canonical analytic derivatives."""

    option = _option(k, r, q)
    delta = option.call_delta_from_volatility(t, s, sigma)
    vega = option.vega_volatility(t, s, sigma)
    return float(delta), float(vega)


def _greeks_jax_regular(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Tuple[float, float]:
    """Return regular-case JAX AD Greeks after synchronizing device execution."""

    delta, vega = _block_until_ready(_jax_delta_vega(s, k, r, q, sigma, t))
    return float(delta), float(vega)


def _greeks_jax(
    s: float, k: float, r: float, q: float, sigma: float, t: float
) -> Tuple[float, float]:
    """Return Greeks via JAX AD or canonical limits for singular cases."""

    if _requires_canonical_greek_path(s, k, r, q, sigma, t):
        return _greeks_numpy(s, k, r, q, sigma, t)
    return _greeks_jax_regular(s, k, r, q, sigma, t)


def _runtime_metadata(backend: Literal["numpy", "jax"]) -> tuple[str, str, bool | None]:
    """Return dtype, device, and JAX 64-bit configuration for a backend."""

    if backend == "jax" and JAX_AVAILABLE and jax is not None and jnp is not None:
        sample = jnp.asarray(1.0)
        devices = jax.devices()
        device = devices[0].platform if devices else jax.default_backend()
        return (
            str(sample.dtype),
            device,
            bool(getattr(jax.config, "jax_enable_x64", False)),
        )
    return str(np.asarray(1.0, dtype=float).dtype), "cpu", None


def _bump_errors(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    delta: float,
    vega: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compare analytical/AD Greeks with central bump-and-revalue estimates."""

    if _requires_canonical_greek_path(s, k, r, q, sigma, t):
        return None, None, None, None
    spot_bump = max(abs(s) * 1.0e-4, 1.0e-4)
    vol_bump = max(abs(sigma) * 1.0e-4, 1.0e-4)
    vol_down = max(sigma - vol_bump, sigma * 0.5)
    vol_up = sigma + vol_bump
    effective_vol_bump = 0.5 * (vol_up - vol_down)
    bump_delta = (
        _bs_price_numpy(s + spot_bump, k, r, q, sigma, t)
        - _bs_price_numpy(s - spot_bump, k, r, q, sigma, t)
    ) / (2.0 * spot_bump)
    bump_vega = (
        _bs_price_numpy(s, k, r, q, vol_up, t)
        - _bs_price_numpy(s, k, r, q, vol_down, t)
    ) / (2.0 * effective_vol_bump)
    return abs(delta - bump_delta), abs(vega - bump_vega), spot_bump, effective_vol_bump


def _observations(
    *,
    method: str,
    delta: float,
    vega: float,
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    fallback_reason: str | None,
) -> tuple[GreekObservation, GreekObservation]:
    """Build auditable Greek observations for call price sensitivities."""

    delta_error, vega_error, spot_bump, vol_bump = _bump_errors(
        s, k, r, q, sigma, t, delta, vega
    )
    differentiated_object = "Black-Scholes European call price from volatility sigma"
    status: Literal["finite", "undefined"] = "finite"
    return (
        GreekObservation(
            greek="delta",
            value=delta,
            method=method,
            differentiated_object=differentiated_object,
            input_variable="spot",
            units="price / spot",
            status=status,
            oracle_error_abs=delta_error,
            bump_size=spot_bump,
            fallback_reason=fallback_reason,
        ),
        GreekObservation(
            greek="vega",
            value=vega,
            method=method,
            differentiated_object=differentiated_object,
            input_variable="volatility sigma",
            units="price / volatility",
            status=status,
            oracle_error_abs=vega_error,
            bump_size=vol_bump,
            fallback_reason=fallback_reason,
        ),
    )


def benchmark_greeks(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
) -> dict[str, BackendBenchmark]:
    """Measure synchronized runtime metadata for available Greek backends."""

    results: dict[str, BackendBenchmark] = {}
    numpy_runtime, numpy_memory_bytes, _ = _time_with_host_peak(
        lambda: _greeks_numpy(s, k, r, q, sigma, t)
    )
    dtype, device, jax_enable_x64 = _runtime_metadata("numpy")
    results["numpy"] = BackendBenchmark(
        runtime_seconds=numpy_runtime,
        dtype=dtype,
        device=device,
        jax_enable_x64=jax_enable_x64,
        memory_bytes=numpy_memory_bytes,
    )

    if (
        JAX_AVAILABLE
        and jax is not None
        and jnp is not None
        and not _requires_canonical_greek_path(s, k, r, q, sigma, t)
    ):
        jax_numpy = jnp
        transfer_seconds, transfer_memory_bytes, args = _time_with_host_peak(
            lambda: _block_until_ready(
                tuple(jax_numpy.asarray(value) for value in (s, k, r, q, sigma, t))
            )
        )

        compiled = jax.jit(_jax_delta_vega)
        compile_seconds, compile_memory_bytes, _ = _time_with_host_peak(
            lambda: _block_until_ready(compiled(*args))
        )

        warmed_seconds, warmed_memory_bytes, _ = _time_with_host_peak(
            lambda: _block_until_ready(compiled(*args))
        )
        dtype, device, jax_enable_x64 = _runtime_metadata("jax")
        results["jax"] = BackendBenchmark(
            runtime_seconds=transfer_seconds + compile_seconds + warmed_seconds,
            transfer_seconds=transfer_seconds,
            compile_seconds=compile_seconds,
            warmed_seconds=warmed_seconds,
            synchronized=True,
            dtype=dtype,
            device=device,
            jax_enable_x64=jax_enable_x64,
            memory_bytes=max(
                transfer_memory_bytes, compile_memory_bytes, warmed_memory_bytes
            ),
        )
    return results


def recover_grid_delta_report(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    coordinate_units: str = "spot",
) -> GridGreekRecovery:
    """Recover Delta on possibly nonuniform coordinates with ``np.gradient``."""

    value_array = np.asarray(values, dtype=float)
    coordinate_array = np.asarray(coordinates, dtype=float)
    if value_array.ndim != 1 or coordinate_array.ndim != 1:
        raise ValueError("values and coordinates must be one-dimensional")
    if value_array.shape != coordinate_array.shape:
        raise ValueError("values and coordinates must have the same shape")
    if len(value_array) < 3:
        raise ValueError("at least three grid points are required for Delta recovery")
    if not np.all(np.isfinite(value_array)) or not np.all(
        np.isfinite(coordinate_array)
    ):
        raise ValueError("values and coordinates must be finite")
    if np.any(np.diff(coordinate_array) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    recovered = np.gradient(value_array, coordinate_array, edge_order=2)
    return GridGreekRecovery(
        values=tuple(float(item) for item in recovered),
        method="coordinate_aware_np_gradient",
        differentiated_object="price grid values",
        input_variable="mesh coordinate",
        coordinate_units=coordinate_units,
        stencil="second-order edge, coordinate-aware interior",
    )


def recover_grid_delta(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Return coordinate-aware Delta values for a price grid."""

    return np.asarray(
        recover_grid_delta_report(values, coordinates).values, dtype=float
    )


def compute_greeks_report(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    *,
    backend: Backend = "auto",
) -> GreekComputationReport:
    """Compute call Delta/Vega with explicit method and backend diagnostics."""

    if backend not in {"auto", "numpy", "jax"}:
        raise ValueError(f"unsupported backend: {backend}")
    if backend == "jax" and not JAX_AVAILABLE:
        raise RuntimeError("JAX backend requested but not installed")

    regular_jax_path = not _requires_canonical_greek_path(s, k, r, q, sigma, t)
    fallback_reason = None
    if backend == "numpy":
        backend_used: Literal["numpy", "jax"] = "numpy"
        method = "analytical_oracle"
        delta, vega = _greeks_numpy(s, k, r, q, sigma, t)
    elif backend == "jax" and regular_jax_path:
        backend_used = "jax"
        method = "jax_ad_analytical_formula"
        delta, vega = _greeks_jax_regular(s, k, r, q, sigma, t)
    elif backend == "jax":
        backend_used = "numpy"
        method = "analytical_oracle_limit"
        fallback_reason = (
            "JAX AD singular/saturated input; canonical analytical limit used"
        )
        delta, vega = _greeks_numpy(s, k, r, q, sigma, t)
    else:
        stats = benchmark_greeks(s, k, r, q, sigma, t)
        use_jax_auto = (
            "jax" in stats
            and stats["jax"].warmed_seconds is not None
            and stats["jax"].runtime_seconds <= stats["numpy"].runtime_seconds * 1.5
        )
        if use_jax_auto:
            backend_used = "jax"
            method = "jax_ad_analytical_formula"
            delta, vega = _greeks_jax_regular(s, k, r, q, sigma, t)
        else:
            backend_used = "numpy"
            method = (
                "analytical_oracle" if regular_jax_path else "analytical_oracle_limit"
            )
            if not regular_jax_path:
                fallback_reason = (
                    "singular/saturated input; canonical analytical limit used"
                )
            elif "jax" in stats:
                fallback_reason = (
                    "auto backend retained NumPy because synchronized JAX runtime "
                    "exceeded the historical 1.5x threshold"
                )
            delta, vega = _greeks_numpy(s, k, r, q, sigma, t)

    dtype, device, jax_enable_x64 = _runtime_metadata(backend_used)
    delta_obs, vega_obs = _observations(
        method=method,
        delta=delta,
        vega=vega,
        s=s,
        k=k,
        r=r,
        q=q,
        sigma=sigma,
        t=t,
        fallback_reason=fallback_reason,
    )
    return GreekComputationReport(
        delta=delta_obs,
        vega=vega_obs,
        backend_requested=backend,
        backend_used=backend_used,
        differentiated_object="Black-Scholes European call price from volatility sigma",
        dtype=dtype,
        device=device,
        jax_enable_x64=jax_enable_x64,
        fallback_reason=fallback_reason,
    )


def compute_greeks(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    *,
    backend: Backend = "auto",
) -> Tuple[float, float]:
    """Compute call ``delta`` and volatility ``vega`` choosing between backends."""

    return compute_greeks_report(s, k, r, q, sigma, t, backend=backend).as_tuple()
