"""Tests for JAX-based Greek computation."""

from src.jax_greeks import compute_greeks


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
