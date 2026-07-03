"""Black-Scholes oracle semantics for volatility, variance, and limits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.jax_greeks import compute_greeks


def _option(
    rate: float = 0.03, carry: float = 0.01, strike: float = 100.0
) -> EuropeanOptionBs:
    return EuropeanOptionBs(k=strike, q=carry, mkt=Market(r=rate))


def test_volatility_and_variance_apis_are_explicit_and_chain_rule_consistent() -> None:
    option = _option()
    spot = 102.0
    maturity = 1.25
    volatility = 0.21
    variance = volatility**2

    call_from_vol = option.call_from_volatility(maturity, spot, volatility)
    call_from_var = option.call_from_variance(maturity, spot, variance)
    put_from_vol = option.put_from_volatility(maturity, spot, volatility)
    put_from_var = option.put_from_variance(maturity, spot, variance)

    assert call_from_vol == pytest.approx(call_from_var)
    assert put_from_vol == pytest.approx(put_from_var)

    vega_sigma = option.vega_volatility(maturity, spot, volatility)
    sensitivity_variance = option.sensitivity_variance(maturity, spot, variance)
    assert sensitivity_variance == pytest.approx(vega_sigma / (2.0 * volatility))


def test_put_call_parity_and_vectorized_broadcasting() -> None:
    option = _option(rate=-0.01, carry=0.02)
    spots = np.array([75.0, 100.0, 125.0])
    maturity = 0.75
    volatility = 0.18

    calls = option.call_from_volatility(maturity, spots, volatility)
    puts = option.put_from_volatility(maturity, spots, volatility)
    singleton_call = option.call_from_volatility(
        maturity, np.array([100.0]), volatility
    )

    assert calls.shape == spots.shape
    assert puts.shape == spots.shape
    assert isinstance(singleton_call, np.ndarray)
    assert singleton_call.shape == (1,)
    parity = calls - puts
    expected = spots * math.exp(-option.q * maturity) - option.k * math.exp(
        -option.r * maturity
    )
    np.testing.assert_allclose(parity, expected, rtol=1e-12, atol=1e-12)


def test_expiry_zero_volatility_and_zero_spot_limits_are_explicit() -> None:
    option = _option(rate=0.05, carry=0.01)

    assert option.call_from_volatility(0.0, 120.0, 0.3) == pytest.approx(20.0)
    assert option.put_from_volatility(0.0, 80.0, 0.3) == pytest.approx(20.0)
    assert option.call_delta_from_volatility(0.0, option.k, 0.3) == pytest.approx(0.5)
    assert option.put_delta_from_volatility(0.0, option.k, 0.3) == pytest.approx(-0.5)

    maturity = 2.0
    spot = 100.0
    forward = option.forward_price(maturity, spot)
    deterministic_call = max(
        spot * math.exp(-option.q * maturity)
        - option.k * math.exp(-option.r * maturity),
        0.0,
    )
    deterministic_put = max(
        option.k * math.exp(-option.r * maturity)
        - spot * math.exp(-option.q * maturity),
        0.0,
    )
    assert forward > option.k
    assert option.call_from_volatility(maturity, spot, 0.0) == pytest.approx(
        deterministic_call
    )
    assert option.put_from_volatility(maturity, spot, 0.0) == pytest.approx(
        deterministic_put
    )
    assert option.call_delta_from_volatility(maturity, spot, 0.0) == pytest.approx(
        math.exp(-option.q * maturity)
    )

    assert option.call_from_volatility(maturity, 0.0, 0.2) == pytest.approx(0.0)
    assert option.put_from_volatility(maturity, 0.0, 0.2) == pytest.approx(
        option.k * math.exp(-option.r * maturity)
    )


def test_deep_tail_prices_are_finite_and_respect_discounted_bounds() -> None:
    option = _option(rate=0.03, carry=-0.02)
    spots = np.array([1e-12, 1e-6, 1e6, 1e12])
    maturity = 5.0
    volatility = 0.75

    calls = option.call_from_volatility(maturity, spots, volatility)
    puts = option.put_from_volatility(maturity, spots, volatility)

    assert np.all(np.isfinite(calls))
    assert np.all(np.isfinite(puts))
    assert np.all(calls >= 0.0)
    assert np.all(puts >= 0.0)
    np.testing.assert_array_less(calls, spots * math.exp(-option.q * maturity) + 1e-8)
    np.testing.assert_array_less(puts, option.k * math.exp(-option.r * maturity) + 1e-8)


def test_invalid_inputs_and_zero_variance_sensitivity_fail_closed() -> None:
    option = _option()

    with pytest.raises(ValueError, match="spot"):
        option.call_from_volatility(1.0, -1.0, 0.2)
    with pytest.raises(ValueError, match="finite"):
        option.call_from_volatility(1.0, math.inf, 0.2)
    with pytest.raises(ValueError, match="maturity"):
        option.call_from_volatility(-1.0, 100.0, 0.2)
    with pytest.raises(ValueError, match="volatility"):
        option.call_from_volatility(1.0, 100.0, -0.2)
    with pytest.raises(ValueError, match="variance"):
        option.call_from_variance(1.0, 100.0, -0.04)
    with pytest.raises(ValueError, match="variance sensitivity"):
        option.sensitivity_variance(1.0, 100.0, 0.0)


def test_numpy_and_jax_greeks_follow_same_volatility_semantics() -> None:
    params = dict(s=100.0, k=100.0, r=0.05, q=0.01, sigma=0.2, t=1.0)
    option = _option(rate=params["r"], carry=params["q"], strike=params["k"])

    delta_np, vega_np = compute_greeks(**params, backend="numpy")
    assert delta_np == pytest.approx(
        option.call_delta_from_volatility(params["t"], params["s"], params["sigma"])
    )
    assert vega_np == pytest.approx(
        option.vega_volatility(params["t"], params["s"], params["sigma"])
    )

    delta_jax, vega_jax = compute_greeks(**params, backend="jax")
    assert delta_jax == pytest.approx(delta_np, abs=1e-5)
    assert vega_jax == pytest.approx(vega_np, abs=1e-5)
