"""Reduced-form credit-risk analytical model tests."""

from __future__ import annotations

from math import exp

import numpy as np
import pytest

from finite_element_options.problems.credit_risk import (
    CreditRiskIntensitySampler,
    CreditRiskProblem,
    DefaultableZeroCouponClaim,
    RECOVERY_OF_PAR_AT_DEFAULT,
    ReducedFormCreditRiskModel,
    UnsupportedSpatialCreditRiskModel,
)


def test_constant_intensity_defaultable_zcb_matches_closed_form() -> None:
    """Bond value separates into survival and recovery legs exactly."""

    model = ReducedFormCreditRiskModel(r=0.03, default_intensity=0.02)
    claim = DefaultableZeroCouponClaim(notional=2.0, recovery_rate=0.4)
    maturity = 5.0
    rate_sum = model.r + model.default_intensity

    expected_survival_leg = 2.0 * exp(-rate_sum * maturity)
    expected_recovery_leg = 2.0 * 0.4 * 0.02 * (1.0 - exp(-rate_sum * maturity)) / rate_sum
    expected_value = expected_survival_leg + expected_recovery_leg

    outputs = model.value_components(claim, maturity)

    assert outputs.recovery_convention == RECOVERY_OF_PAR_AT_DEFAULT
    assert outputs.survival_probability == pytest.approx(exp(-0.02 * maturity))
    assert outputs.default_probability == pytest.approx(1.0 - exp(-0.02 * maturity))
    assert outputs.survival_leg_pv == pytest.approx(expected_survival_leg)
    assert outputs.recovery_leg_pv == pytest.approx(expected_recovery_leg)
    assert outputs.defaultable_bond_value == pytest.approx(expected_value)
    assert model.defaultable_zero_coupon_value(claim, maturity) == pytest.approx(expected_value)


def test_reduced_form_model_satisfies_scalar_pricing_ode() -> None:
    """Analytical value satisfies dV/dtau = -(r+lambda)V + lambda R N."""

    model = ReducedFormCreditRiskModel(r=0.04, default_intensity=0.03)
    claim = DefaultableZeroCouponClaim(notional=1.5, recovery_rate=0.25)
    tau = 2.0
    eps = 1.0e-6
    v_plus = model.defaultable_zero_coupon_value(claim, tau + eps)
    v_minus = model.defaultable_zero_coupon_value(claim, tau - eps)
    finite_difference = (v_plus - v_minus) / (2.0 * eps)
    value = model.defaultable_zero_coupon_value(claim, tau)

    assert finite_difference == pytest.approx(model.ode_rhs(value, claim), rel=1.0e-7)


def test_zero_intensity_recovery_extremes_and_expiry_limits() -> None:
    """Zero hazard, recovery extremes and maturity-zero limits are explicit."""

    risk_free = ReducedFormCreditRiskModel(r=0.05, default_intensity=0.0)
    claim = DefaultableZeroCouponClaim(notional=3.0, recovery_rate=0.0)
    assert risk_free.defaultable_zero_coupon_value(claim, 4.0) == pytest.approx(
        3.0 * exp(-0.05 * 4.0)
    )
    assert risk_free.default_probability(4.0) == pytest.approx(0.0)
    assert risk_free.value_components(claim, 4.0).credit_loss_value == pytest.approx(0.0)

    zero_rate_model = ReducedFormCreditRiskModel(r=0.0, default_intensity=0.7)
    zero_recovery = DefaultableZeroCouponClaim(notional=1.0, recovery_rate=0.0)
    full_recovery = DefaultableZeroCouponClaim(notional=1.0, recovery_rate=1.0)
    assert zero_rate_model.defaultable_zero_coupon_value(zero_recovery, 2.0) == pytest.approx(
        exp(-0.7 * 2.0)
    )
    assert zero_rate_model.defaultable_zero_coupon_value(full_recovery, 2.0) == pytest.approx(1.0)
    assert zero_rate_model.defaultable_zero_coupon_value(zero_recovery, 0.0) == pytest.approx(1.0)


def test_invalid_credit_risk_inputs_fail_before_assembly() -> None:
    """Invalid parameters are rejected at construction or maturity validation."""

    with pytest.raises(ValueError, match="default_intensity"):
        ReducedFormCreditRiskModel(default_intensity=-0.01)
    with pytest.raises(ValueError, match="recovery_rate"):
        DefaultableZeroCouponClaim(recovery_rate=1.01)
    with pytest.raises(ValueError, match="notional"):
        DefaultableZeroCouponClaim(notional=0.0)
    with pytest.raises(ValueError, match="maturity"):
        ReducedFormCreditRiskModel().default_probability(-1.0)


def test_credit_risk_problem_separates_outputs_and_fails_closed_for_fem() -> None:
    """Problem preset exposes analytical outputs and no fake spatial boundary."""

    problem = CreditRiskProblem(r=0.01, default_intensity=0.2, recovery_rate=0.3)
    outputs = problem.value_components(1.25)

    assert outputs.defaultable_bond_value == pytest.approx(problem.value(1.25))
    assert outputs.loss_given_default == pytest.approx(0.7)
    assert outputs.recovery_rate == pytest.approx(0.3)

    with pytest.raises(UnsupportedSpatialCreditRiskModel, match="no spatial FEM state"):
        problem.reduced_form_model.A(1.0)
    with pytest.raises(UnsupportedSpatialCreditRiskModel, match="not a call/put payoff"):
        problem.claim.put_payoff(1.0)
    with pytest.raises(UnsupportedSpatialCreditRiskModel, match="no spatial boundary facets"):
        problem.boundary_condition.apply(None, None, None, 0.0)  # type: ignore[arg-type]


def test_intensity_sampler_is_parameter_uncertainty_not_jump_dynamics() -> None:
    """The old jump-dynamics label is replaced by explicit intensity sampling."""

    model = ReducedFormCreditRiskModel(r=0.03, default_intensity=0.02)
    sampler = CreditRiskIntensitySampler(base_model=model, log_std=0.0)
    sampled = sampler.sample(np.random.default_rng(0))
    assert sampled.default_intensity == pytest.approx(model.default_intensity)
    assert sampled.r == pytest.approx(model.r)

    import finite_element_options.problems.credit_risk as credit_risk

    assert not hasattr(credit_risk, "CreditRiskJumpDynamics")
    assert not hasattr(credit_risk, "CreditRiskPayoff")
