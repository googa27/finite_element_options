"""Tests for JAX-based Greek computation."""

import math

import numpy as np
import pytest

from finite_element_options.jax_greeks import (
    benchmark_greeks,
    compute_greeks,
    compute_greeks_report,
    recover_grid_delta,
    recover_grid_delta_report,
)


def test_greeks_consistency():
    """JAX and NumPy backends should agree within tolerance."""
    params = dict(s=100.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0)
    delta_jax, vega_jax = compute_greeks(**params, backend="jax")
    delta_np, vega_np = compute_greeks(**params, backend="numpy")
    assert abs(delta_jax - delta_np) < 1e-5
    assert abs(vega_jax - vega_np) < 1e-5


def test_auto_backend():
    """Auto backend should return finite values."""
    params = dict(s=100.0, k=110.0, r=0.03, q=0.01, sigma=0.25, t=0.5)
    delta, vega = compute_greeks(**params)
    assert delta == delta and vega == vega  # NaN check


def test_jax_backend_uses_canonical_limits_for_zero_spot():
    """Explicit JAX requests must match the canonical finite zero-spot limit."""

    params = dict(s=0.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0)
    delta_jax, vega_jax = compute_greeks(**params, backend="jax")
    delta_np, vega_np = compute_greeks(**params, backend="numpy")
    assert math.isfinite(delta_jax)
    assert math.isfinite(vega_jax)
    assert delta_jax == pytest.approx(delta_np)
    assert vega_jax == pytest.approx(vega_np)


def test_jax_backend_uses_canonical_limits_for_subnormal_tail_spot():
    """Positive subnormal tail spots must not be differentiated through JAX log."""

    params = dict(s=1e-40, k=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0)
    delta_jax, vega_jax = compute_greeks(**params, backend="jax")
    delta_np, vega_np = compute_greeks(**params, backend="numpy")
    delta_auto, vega_auto = compute_greeks(**params, backend="auto")
    assert math.isfinite(delta_jax)
    assert math.isfinite(vega_jax)
    assert delta_jax == pytest.approx(delta_np)
    assert vega_jax == pytest.approx(vega_np)
    assert delta_auto == pytest.approx(delta_np)
    assert vega_auto == pytest.approx(vega_np)


def test_jax_backend_delegates_invalid_spot_validation():
    """Invalid non-regular inputs should fail exactly like the canonical oracle."""

    with pytest.raises(ValueError, match="spot"):
        compute_greeks(s=-1.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0, backend="jax")


def test_bump_error_diagnostics_preserve_positive_small_spot() -> None:
    """Finite bump diagnostics must not cross into negative spot values."""

    params = dict(s=1.0e-6, k=1.0e-6, r=0.05, q=0.0, sigma=0.2, t=1.0)
    report = compute_greeks_report(**params, backend="numpy")
    assert report.delta.method == "analytical_oracle"
    assert report.delta.bump_size is not None
    assert 0.0 < report.delta.bump_size < params["s"]
    assert report.delta.oracle_error_abs is not None
    assert math.isfinite(report.delta.oracle_error_abs)


def test_compute_greeks_report_identifies_object_method_units_and_errors() -> None:
    """Diagnostic report should name method, object, inputs, units and oracle error."""

    params = dict(s=100.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0)
    report = compute_greeks_report(**params, backend="jax")
    assert (
        report.differentiated_object
        == "Black-Scholes European call price from volatility sigma"
    )
    assert report.delta.greek == "delta"
    assert report.delta.input_variable == "spot"
    assert report.delta.units == "price / spot"
    assert report.delta.method == "jax_ad_analytical_formula"
    assert report.delta.status == "finite"
    assert report.delta.oracle_error_abs is not None
    assert report.delta.oracle_error_abs < 1.0e-3
    assert report.vega.greek == "vega"
    assert report.vega.input_variable == "volatility sigma"
    assert report.vega.units == "price / volatility"
    assert report.vega.oracle_error_abs is not None
    assert report.vega.oracle_error_abs < 1.0e-2
    assert report.dtype
    assert report.device


def test_jax_report_uses_explicit_analytical_limit_for_expiry_boundary() -> None:
    """Singular expiry inputs should be finite and labeled as analytical limits."""

    params = dict(s=100.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=0.0)
    report = compute_greeks_report(**params, backend="jax")
    assert report.backend_used == "numpy"
    assert report.delta.method == "analytical_oracle_limit"
    assert report.fallback_reason is not None
    assert report.delta.fallback_reason == report.fallback_reason
    assert report.vega.fallback_reason == report.fallback_reason
    assert math.isfinite(report.delta.value)
    assert math.isfinite(report.vega.value)


def test_numpy_report_labels_boundary_inputs_as_analytical_limits() -> None:
    """Requested NumPy reports should still distinguish boundary limit semantics."""

    params = dict(s=100.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=0.0)
    report = compute_greeks_report(**params, backend="numpy")
    assert report.backend_used == "numpy"
    assert report.delta.method == "analytical_oracle_limit"
    assert report.fallback_reason is not None
    assert report.delta.fallback_reason == report.fallback_reason


def test_benchmark_greeks_separates_jax_compile_transfer_and_warm_execution() -> None:
    """JAX benchmark metadata should synchronize and split runtime phases."""

    params = dict(s=100.0, k=100.0, r=0.05, q=0.0, sigma=0.2, t=1.0)
    stats = benchmark_greeks(**params)
    assert "numpy" in stats
    assert stats["numpy"].runtime_seconds >= 0.0
    assert stats["numpy"].memory_bytes is not None
    assert stats["numpy"].memory_bytes >= 0
    assert len(stats["numpy"]) == 2
    assert stats["numpy"][0] == stats["numpy"].runtime_seconds
    assert tuple(stats["numpy"])[1] == int(stats["numpy"].memory_bytes or 0)
    assert stats["numpy"].device == "cpu"
    if "jax" in stats:
        jax_stats = stats["jax"]
        assert jax_stats.synchronized is True
        assert (
            jax_stats.transfer_seconds is not None and jax_stats.transfer_seconds >= 0.0
        )
        assert (
            jax_stats.compile_seconds is not None and jax_stats.compile_seconds >= 0.0
        )
        assert jax_stats.warmed_seconds is not None and jax_stats.warmed_seconds >= 0.0
        assert jax_stats.dtype
        assert jax_stats.device
        assert isinstance(jax_stats.jax_enable_x64, bool)
        assert jax_stats.memory_bytes is not None
        assert jax_stats.memory_bytes >= 0


def test_recover_grid_delta_uses_nonuniform_coordinates() -> None:
    """Grid Delta recovery must use coordinates, not unit-spaced indices."""

    coordinates = np.array([0.0, 0.25, 1.0, 2.0, 4.0], dtype=float)
    values = coordinates**2
    recovered = recover_grid_delta(values, coordinates)
    report = recover_grid_delta_report(values, coordinates)
    np.testing.assert_allclose(recovered, 2.0 * coordinates, atol=1.0e-12)
    assert report.method == "coordinate_aware_np_gradient"
    assert report.differentiated_object == "price grid values"
    assert report.input_variable == "mesh coordinate"
