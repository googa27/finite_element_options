"""Independent one-state analytical oracles for the five European payoffs."""

from __future__ import annotations

import math


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _d1_d2(
    spot: float, strike: float, rate: float, carry: float, volatility: float, maturity: float
) -> tuple[float, float]:
    scale = volatility * math.sqrt(maturity)
    if scale <= 0.0:
        raise ValueError("analytical oracle requires positive volatility and maturity")
    d1 = (math.log(spot / strike) + (rate - carry + 0.5 * volatility**2) * maturity) / scale
    return d1, d1 - scale


def one_state_price(
    kind: str,
    *,
    equity_spot: float,
    fx_spot: float,
    equity_vol: float,
    fx_vol: float,
    correlation: float,
    domestic_rate: float,
    foreign_rate: float,
    dividend_yield: float,
    maturity: float,
    strike: float | None = None,
    payout: float = 1.0,
    fixed_fx: float | None = None,
    equity_barrier: float | None = None,
    fx_barrier: float | None = None,
) -> float:
    """Return an exact one-regime domestic-measure European price."""

    discount = math.exp(-domestic_rate * maturity)
    composite_spot = equity_spot * fx_spot
    composite_variance = equity_vol**2 + fx_vol**2 + 2.0 * correlation * equity_vol * fx_vol
    composite_vol = math.sqrt(max(composite_variance, 0.0))
    if kind in {"composite_call", "composite_put", "composite_digital"}:
        if strike is None:
            raise ValueError("strike is required")
        d1, d2 = _d1_d2(
            composite_spot, strike, domestic_rate, dividend_yield, composite_vol, maturity
        )
        forward_leg = composite_spot * math.exp(-dividend_yield * maturity)
        strike_leg = strike * discount
        if kind == "composite_call":
            return forward_leg * _normal_cdf(d1) - strike_leg * _normal_cdf(d2)
        if kind == "composite_put":
            return strike_leg * _normal_cdf(-d2) - forward_leg * _normal_cdf(-d1)
        return payout * discount * _normal_cdf(d2)
    if kind == "quanto_call":
        if strike is None or fixed_fx is None:
            raise ValueError("strike and fixed_fx are required")
        effective_dividend = (
            dividend_yield + domestic_rate - foreign_rate + correlation * equity_vol * fx_vol
        )
        d1, d2 = _d1_d2(
            equity_spot,
            strike,
            domestic_rate,
            effective_dividend,
            equity_vol,
            maturity,
        )
        return fixed_fx * (
            equity_spot * math.exp(-effective_dividend * maturity) * _normal_cdf(d1)
            - strike * discount * _normal_cdf(d2)
        )
    if kind == "dual_trigger_protection":
        if equity_barrier is None or fx_barrier is None:
            raise ValueError("equity_barrier and fx_barrier are required")
        try:
            from scipy.stats import multivariate_normal
        except ModuleNotFoundError as exc:
            raise ImportError("dual-trigger analytical oracle requires SciPy") from exc
        equity_drift = (
            foreign_rate - dividend_yield - correlation * equity_vol * fx_vol - 0.5 * equity_vol**2
        )
        fx_drift = domestic_rate - foreign_rate - 0.5 * fx_vol**2
        z_equity = (math.log(equity_barrier / equity_spot) - equity_drift * maturity) / (
            equity_vol * math.sqrt(maturity)
        )
        z_fx = (math.log(fx_barrier / fx_spot) - fx_drift * maturity) / (
            fx_vol * math.sqrt(maturity)
        )
        joint_lower = float(
            multivariate_normal.cdf(  # type: ignore[arg-type]
                [z_equity, z_fx], mean=[0.0, 0.0], cov=[[1.0, correlation], [correlation, 1.0]]
            )
        )
        probability = _normal_cdf(z_equity) - joint_lower
        return payout * discount * probability
    raise ValueError(f"unsupported contract kind: {kind}")
