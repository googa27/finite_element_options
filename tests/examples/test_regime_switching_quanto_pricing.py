"""Research pricing tests for the regime-switching quanto example.

The tests are intentionally offline and deterministic.  They characterize the
research-only PDE/MC API before implementation so the scikit-fem engine is added
under a RED/GREEN workflow.
"""

from __future__ import annotations

import json
from math import erf, exp, log, sqrt

import numpy as np
import pytest

from finite_element_options.examples.regime_switching_quanto import (
    ContractSpec,
    FEMGridSpec,
    TwoFactorRegimeModel,
    price_contract_fem,
    price_contract_monte_carlo,
)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _black_scholes_product_call(
    *, spot_product: float, strike: float, maturity: float, rd: float, dividend: float, vol: float
) -> float:
    std = vol * sqrt(maturity)
    d1 = (log(spot_product / strike) + (rd - dividend + 0.5 * vol * vol) * maturity) / std
    d2 = d1 - std
    return spot_product * exp(-dividend * maturity) * _normal_cdf(d1) - strike * exp(
        -rd * maturity
    ) * _normal_cdf(d2)


def _one_regime_model(
    *, sig_s: float = 0.20, sig_f: float = 0.12, rho: float = 0.25
) -> TwoFactorRegimeModel:
    return TwoFactorRegimeModel(
        equity_vol=[sig_s],
        fx_vol=[sig_f],
        correlation=[rho],
        generator=[[0.0]],
        current_probabilities=[1.0],
        domestic_rate=0.03,
        foreign_rate=0.01,
        dividend_yield=0.02,
    )


def test_grid_contract_rejects_nonfinite_domains_and_domains_excluding_origin() -> None:
    with pytest.raises(ValueError, match="finite"):
        FEMGridSpec((float("nan"), 1.0), (-1.0, 1.0), 5, 5, 2)
    with pytest.raises(ValueError, match="contain zero"):
        FEMGridSpec((0.1, 1.0), (-1.0, 1.0), 5, 5, 2)


def test_contract_payoffs_are_vectorized_and_model_validation_documents_q_assumptions() -> None:
    x = np.array([-0.1, 0.0, 0.2])
    y = np.array([0.15, 0.0, -0.05])

    composite_call = ContractSpec(kind="composite_call", strike=1.0)
    np.testing.assert_allclose(
        composite_call.payoff(x, y, equity_spot=2.0, fx_spot=3.0),
        np.maximum(2.0 * np.exp(x) * 3.0 * np.exp(y) - 1.0, 0.0),
    )

    composite_put = ContractSpec(kind="composite_put", strike=6.0)
    np.testing.assert_allclose(
        composite_put.payoff(x, y, equity_spot=2.0, fx_spot=3.0),
        np.maximum(6.0 - 2.0 * np.exp(x) * 3.0 * np.exp(y), 0.0),
    )

    digital = ContractSpec(kind="composite_digital", strike=6.0, payout=2.5)
    np.testing.assert_allclose(
        digital.payoff(x, y, equity_spot=2.0, fx_spot=3.0),
        2.5 * (2.0 * np.exp(x) * 3.0 * np.exp(y) >= 6.0),
    )

    quanto_call = ContractSpec(kind="quanto_call", strike=2.0, fixed_fx=900.0)
    np.testing.assert_allclose(
        quanto_call.payoff(x, y, equity_spot=2.0, fx_spot=3.0),
        900.0 * np.maximum(2.0 * np.exp(x) - 2.0, 0.0),
    )

    protection = ContractSpec(
        kind="dual_trigger_protection", equity_barrier=2.0, fx_barrier=3.0, payout=7.0
    )
    np.testing.assert_allclose(
        protection.payoff(x, y, equity_spot=2.0, fx_spot=3.0),
        7.0 * ((2.0 * np.exp(x) <= 2.0) & (3.0 * np.exp(y) >= 3.0)),
    )

    with pytest.raises(ValueError, match="off-diagonal"):
        TwoFactorRegimeModel(
            equity_vol=[0.2, 0.3],
            fx_vol=[0.1, 0.2],
            correlation=[0.0, 0.1],
            generator=[[-1.0, 1.0], [-0.1, 0.1]],
            current_probabilities=[0.5, 0.5],
            domestic_rate=0.03,
            foreign_rate=0.01,
            dividend_yield=0.0,
        )

    model = TwoFactorRegimeModel(
        equity_vol=[0.2, 0.3],
        fx_vol=[0.1, 0.2],
        correlation=[0.0, -0.2],
        generator=[[-1.0, 1.0], [0.5, -0.5]],
        current_probabilities=[0.25, 0.75],
        domestic_rate=0.03,
        foreign_rate=0.01,
        dividend_yield=0.0,
        volatility_scale=1.1,
    )
    payload = model.to_dict()
    assert payload["volatility_scale"] == 1.1
    assert "same Q generator as fitted P" in payload["measure_note"]
    json.dumps(payload, allow_nan=False)


def test_one_regime_composite_call_fem_matches_product_black_scholes_with_refinement() -> None:
    maturity = 0.5
    model = _one_regime_model()
    contract = ContractSpec(kind="composite_call", strike=1.0)
    reference = _black_scholes_product_call(
        spot_product=1.0,
        strike=1.0,
        maturity=maturity,
        rd=model.domestic_rate,
        dividend=model.dividend_yield,
        vol=sqrt(0.20**2 + 0.12**2 + 2.0 * 0.25 * 0.20 * 0.12),
    )

    coarse = price_contract_fem(
        model,
        contract,
        maturity=maturity,
        equity_spot=1.0,
        fx_spot=1.0,
        grid=FEMGridSpec(x_domain=(-1.4, 1.4), y_domain=(-1.4, 1.4), nx=13, ny=13, time_steps=12),
    )
    fine = price_contract_fem(
        model,
        contract,
        maturity=maturity,
        equity_spot=1.0,
        fx_spot=1.0,
        grid=FEMGridSpec(x_domain=(-1.4, 1.4), y_domain=(-1.4, 1.4), nx=25, ny=25, time_steps=28),
    )

    assert abs(fine.mixture_price - reference) < 7.5e-3
    assert abs(fine.mixture_price - reference) <= abs(coarse.mixture_price - reference) + 2.0e-3
    assert fine.degrees_of_freedom > coarse.degrees_of_freedom
    assert "frozen-diffusion" in fine.boundary_description


def test_composite_call_prices_are_monotone_decreasing_in_strike() -> None:
    model = _one_regime_model(sig_s=0.18, sig_f=0.10, rho=-0.15)
    grid = FEMGridSpec(x_domain=(-1.3, 1.3), y_domain=(-1.3, 1.3), nx=19, ny=19, time_steps=18)

    low = price_contract_fem(
        model,
        ContractSpec(kind="composite_call", strike=0.95),
        maturity=0.4,
        equity_spot=1.0,
        fx_spot=1.0,
        grid=grid,
    )
    high = price_contract_fem(
        model,
        ContractSpec(kind="composite_call", strike=1.05),
        maturity=0.4,
        equity_spot=1.0,
        fx_spot=1.0,
        grid=grid,
    )

    assert low.mixture_price > high.mixture_price
    assert low.per_regime_prices[0] > high.per_regime_prices[0]


def test_multi_regime_fem_agrees_with_seeded_monte_carlo_oracle() -> None:
    model = TwoFactorRegimeModel(
        equity_vol=[0.16, 0.30],
        fx_vol=[0.09, 0.18],
        correlation=[0.20, -0.35],
        generator=[[-1.2, 1.2], [0.8, -0.8]],
        current_probabilities=[0.65, 0.35],
        domestic_rate=0.035,
        foreign_rate=0.015,
        dividend_yield=0.01,
        volatility_scale=0.9,
    )
    contract = ContractSpec(kind="composite_call", strike=1.0)
    maturity = 0.35

    fem = price_contract_fem(
        model,
        contract,
        maturity=maturity,
        equity_spot=1.0,
        fx_spot=1.0,
        grid=FEMGridSpec(x_domain=(-1.5, 1.5), y_domain=(-1.5, 1.5), nx=21, ny=21, time_steps=24),
    )
    mc = price_contract_monte_carlo(
        model,
        contract,
        maturity=maturity,
        equity_spot=1.0,
        fx_spot=1.0,
        paths=80_000,
        seed=20260903,
    )

    assert abs(fem.mixture_price - mc.price) <= 1.5e-2 + 4.0 * mc.standard_error
    assert mc.paths == 80_000
    assert mc.steps == int(np.ceil(252 * maturity))
    assert fem.residual < 1.0e-8


def test_pricing_results_serialize_without_numpy_leakage() -> None:
    model = _one_regime_model()
    contract = ContractSpec(kind="composite_digital", strike=1.0, payout=3.0)
    grid = FEMGridSpec(x_domain=(-1.2, 1.2), y_domain=(-1.2, 1.2), nx=11, ny=11, time_steps=8)

    fem = price_contract_fem(
        model,
        contract,
        maturity=0.25,
        equity_spot=1.0,
        fx_spot=1.0,
        grid=grid,
        return_surface=True,
    )
    mc = price_contract_monte_carlo(
        model,
        contract,
        maturity=0.25,
        equity_spot=1.0,
        fx_spot=1.0,
        paths=5_000,
        seed=99,
    )

    dumped_fem = json.dumps(fem.to_dict(), sort_keys=True, allow_nan=False)
    dumped_mc = json.dumps(mc.to_dict(), sort_keys=True, allow_nan=False)
    assert type(fem.degrees_of_freedom) is int
    assert type(fem.nnz) is int
    assert type(fem.time_steps) is int
    assert fem.factorizations == 2
    assert fem.factorization_reuses == fem.time_steps - fem.factorizations
    assert "ndarray" not in dumped_fem + dumped_mc
    assert "nodal_mixture_surface" not in fem.to_dict()
    assert fem.nodal_mixture_surface is not None
