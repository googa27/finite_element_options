"""Explicit reduced-form credit-risk claim model.

The supported model is a constant-intensity defaultable zero-coupon
claim with fractional recovery of par paid at default.  Because a constant
hazard-rate claim has no spatial state, this module intentionally provides an
analytical/ODE reference instead of manufacturing an artificial FEM dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite
from typing import Final, NoReturn

import numpy as np
import skfem as fem

from finite_element_options.core.interfaces import (
    ArrayLikeFloat,
    BoundaryCondition,
    DynamicsModel,
    Payoff,
    SpaceDiscretization,
)

from .base import Problem

RECOVERY_OF_PAR_AT_DEFAULT: Final[str] = "fractional_recovery_of_par_at_default"
_NUMERICAL_ZERO: Final[float] = 1.0e-14


class UnsupportedSpatialCreditRiskModel(ValueError):
    """Raised when a state-free credit model is routed to a spatial FEM solver."""


@dataclass(frozen=True)
class DefaultableZeroCouponOutputs:
    """Analytical outputs for a constant-intensity defaultable zero coupon."""

    maturity: float
    notional: float
    recovery_rate: float
    default_intensity: float
    risk_free_rate: float
    survival_probability: float
    default_probability: float
    survival_leg_pv: float
    recovery_leg_pv: float
    defaultable_bond_value: float
    default_free_value: float
    credit_loss_value: float
    loss_given_default: float
    recovery_convention: str = RECOVERY_OF_PAR_AT_DEFAULT


@dataclass(frozen=True)
class DefaultableZeroCouponClaim(Payoff):
    r"""Unit claim paying notional at maturity if alive and recovery at default.

    The recovery convention is fractional recovery of par paid at default:

    .. math::

       V(\tau) = N e^{-(r+\lambda)\tau}
              + R N \lambda \int_0^\tau e^{-(r+\lambda)s}\,ds.

    This is not an equity call/put payoff.  The inherited call/put methods are
    implemented only to fail closed if generic option infrastructure attempts to
    treat the claim as a spatial option payoff.
    """

    notional: float = 1.0
    recovery_rate: float = 0.4
    recovery_convention: str = RECOVERY_OF_PAR_AT_DEFAULT

    def __post_init__(self) -> None:
        """Validate recovery and notional conventions."""

        if not isfinite(self.notional) or self.notional <= 0.0:
            raise ValueError("notional must be finite and strictly positive")
        if not isfinite(self.recovery_rate) or not 0.0 <= self.recovery_rate <= 1.0:
            raise ValueError("recovery_rate must be finite and in [0, 1]")
        if self.recovery_convention != RECOVERY_OF_PAR_AT_DEFAULT:
            raise ValueError(
                "only fractional_recovery_of_par_at_default is currently supported"
            )

    @property
    def terminal_payoff(self) -> float:
        """Cash paid at maturity when no default occurs."""

        return self.notional

    @property
    def recovery_cashflow(self) -> float:
        """Cash paid at the default event under the recovery convention."""

        return self.notional * self.recovery_rate

    @property
    def loss_given_default(self) -> float:
        """Par loss amount at default."""

        return self.notional * (1.0 - self.recovery_rate)

    def _unsupported_option_route(self) -> NoReturn:
        raise UnsupportedSpatialCreditRiskModel(
            "DefaultableZeroCouponClaim is a reduced-form credit claim, not a "
            "call/put payoff. Use ReducedFormCreditRiskModel.value_components() "
            "or CreditRiskProblem.value() instead of option-space pricing."
        )

    def call_payoff(self, s: ArrayLikeFloat) -> ArrayLikeFloat:  # pylint: disable=unused-argument
        """Fail closed for generic call-option routes."""

        self._unsupported_option_route()

    def put_payoff(self, s: ArrayLikeFloat) -> ArrayLikeFloat:  # pylint: disable=unused-argument
        """Fail closed for generic put-option routes."""

        self._unsupported_option_route()

    def call(
        self, th: float, s: ArrayLikeFloat, variance: ArrayLikeFloat
    ) -> ArrayLikeFloat:  # pylint: disable=unused-argument
        """Fail closed for generic call-option routes."""

        self._unsupported_option_route()

    def put(
        self, th: float, s: ArrayLikeFloat, variance: ArrayLikeFloat
    ) -> ArrayLikeFloat:  # pylint: disable=unused-argument
        """Fail closed for generic put-option routes."""

        self._unsupported_option_route()


@dataclass(frozen=True)
class ReducedFormCreditRiskModel:
    """Constant-rate reduced-form model for a defaultable zero-coupon claim.

    Let ``tau`` be time to maturity, ``r`` the risk-free short rate and
    ``lambda`` the default intensity.  The value under fractional recovery of
    par paid at default is the analytical solution of

    ``dV/dtau = -(r + lambda) V + lambda * recovery_cashflow``

    with terminal condition ``V(0) = notional``.
    """

    r: float = 0.03
    default_intensity: float = 0.02
    q: float = 0.0

    def __post_init__(self) -> None:
        """Validate constant-rate parameters."""

        if not isfinite(self.r):
            raise ValueError("risk-free rate r must be finite")
        if not isfinite(self.default_intensity) or self.default_intensity < 0.0:
            raise ValueError("default_intensity must be finite and nonnegative")
        if not isfinite(self.q):
            raise ValueError("q must be finite")

    @property
    def lamb(self) -> float:
        """Legacy-compatible alias for the default intensity."""

        return self.default_intensity

    def _validate_maturity(self, maturity: float) -> None:
        if not isfinite(maturity) or maturity < 0.0:
            raise ValueError("maturity must be finite and nonnegative")

    def risk_free_discount(self, maturity: float) -> float:
        """Risk-free discount factor for ``maturity``."""

        self._validate_maturity(maturity)
        return exp(-self.r * maturity)

    def survival_probability(self, maturity: float) -> float:
        """Risk-neutral survival probability to maturity."""

        self._validate_maturity(maturity)
        return exp(-self.default_intensity * maturity)

    def default_probability(self, maturity: float) -> float:
        """Risk-neutral event probability by maturity."""

        return 1.0 - self.survival_probability(maturity)

    def survival_leg_pv(
        self, claim: DefaultableZeroCouponClaim, maturity: float
    ) -> float:
        """PV of notional paid at maturity conditional on survival."""

        self._validate_maturity(maturity)
        return claim.terminal_payoff * exp(
            -(self.r + self.default_intensity) * maturity
        )

    def recovery_leg_pv(
        self, claim: DefaultableZeroCouponClaim, maturity: float
    ) -> float:
        """PV of recovery cash paid at default time."""

        self._validate_maturity(maturity)
        rate_sum = self.r + self.default_intensity
        if abs(rate_sum) < _NUMERICAL_ZERO:
            default_annuity = maturity
        else:
            default_annuity = (1.0 - exp(-rate_sum * maturity)) / rate_sum
        return claim.recovery_cashflow * self.default_intensity * default_annuity

    def defaultable_zero_coupon_value(
        self, claim: DefaultableZeroCouponClaim, maturity: float
    ) -> float:
        """Analytical price of the supported defaultable zero-coupon claim."""

        return self.survival_leg_pv(claim, maturity) + self.recovery_leg_pv(
            claim, maturity
        )

    def default_free_value(
        self, claim: DefaultableZeroCouponClaim, maturity: float
    ) -> float:
        """Price of the same notional with no default risk."""

        return claim.notional * self.risk_free_discount(maturity)

    def credit_loss_value(
        self, claim: DefaultableZeroCouponClaim, maturity: float
    ) -> float:
        """PV gap between the default-free and defaultable claim values."""

        return self.default_free_value(
            claim, maturity
        ) - self.defaultable_zero_coupon_value(claim, maturity)

    def ode_rhs(self, value: float, claim: DefaultableZeroCouponClaim) -> float:
        """Right-hand side of the scalar pricing ODE in time-to-maturity."""

        if not isfinite(value):
            raise ValueError("value must be finite")
        return (
            -(self.r + self.default_intensity) * value
            + self.default_intensity * claim.recovery_cashflow
        )

    def value_components(
        self, claim: DefaultableZeroCouponClaim, maturity: float
    ) -> DefaultableZeroCouponOutputs:
        """Return separated bond, loss, survival and recovery outputs."""

        survival_leg = self.survival_leg_pv(claim, maturity)
        recovery_leg = self.recovery_leg_pv(claim, maturity)
        bond_value = survival_leg + recovery_leg
        default_free = self.default_free_value(claim, maturity)
        return DefaultableZeroCouponOutputs(
            maturity=maturity,
            notional=claim.notional,
            recovery_rate=claim.recovery_rate,
            default_intensity=self.default_intensity,
            risk_free_rate=self.r,
            survival_probability=self.survival_probability(maturity),
            default_probability=self.default_probability(maturity),
            survival_leg_pv=survival_leg,
            recovery_leg_pv=recovery_leg,
            defaultable_bond_value=bond_value,
            default_free_value=default_free,
            credit_loss_value=default_free - bond_value,
            loss_given_default=claim.loss_given_default,
        )

    def mean_variance(self, th: float, v: float) -> float:  # pylint: disable=unused-argument
        """Return zero variance for the scalar ODE reference."""

        return 0.0

    def _unsupported_spatial_route(self) -> NoReturn:
        raise UnsupportedSpatialCreditRiskModel(
            "constant-intensity credit risk has no spatial FEM state; use the "
            "closed-form/ODE reference on ReducedFormCreditRiskModel instead"
        )

    def A(self, *coords) -> list[list[float]]:  # pylint: disable=unused-argument
        """Reject artificial diffusion coefficients for this state-free model."""

        self._unsupported_spatial_route()

    def dA(self, *coords) -> list[float]:  # pylint: disable=unused-argument
        """Reject artificial diffusion-divergence coefficients."""

        self._unsupported_spatial_route()

    def b(self, *coords) -> list[float]:  # pylint: disable=unused-argument
        """Reject artificial drift coefficients for this state-free model."""

        self._unsupported_spatial_route()

    def boundary_term(
        self,
        is_call: bool,
        payoff: Payoff,  # pylint: disable=unused-argument
    ) -> fem.LinearForm:
        """Reject natural boundary assembly for a state-free ODE model."""

        self._unsupported_spatial_route()


@dataclass(frozen=True)
class CreditRiskIntensitySampler:
    """Uncertainty sampler for constant default intensities.

    This is parameter uncertainty sampling, not a jump generator.  It returns a
    new reduced-form model whose constant intensity is lognormally perturbed.
    """

    base_model: ReducedFormCreditRiskModel
    log_std: float = 0.1

    def __post_init__(self) -> None:
        """Validate sampler parameters."""

        if not isfinite(self.log_std) or self.log_std < 0.0:
            raise ValueError("log_std must be finite and nonnegative")
        if self.base_model.default_intensity <= 0.0 and self.log_std > 0.0:
            raise ValueError(
                "positive base default_intensity is required for lognormal sampling"
            )

    def sample(self, rng: np.random.Generator) -> ReducedFormCreditRiskModel:
        """Return a model with sampled default intensity."""

        if self.log_std == 0.0:
            sampled_intensity = self.base_model.default_intensity
        else:
            sampled_intensity = float(
                rng.lognormal(
                    mean=np.log(self.base_model.default_intensity),
                    sigma=self.log_std,
                )
            )
        return ReducedFormCreditRiskModel(
            r=self.base_model.r,
            default_intensity=sampled_intensity,
            q=self.base_model.q,
        )


@dataclass(frozen=True)
class NoSpatialBoundaryCondition(BoundaryCondition):
    """Boundary marker for reduced-form claims without a spatial domain."""

    def apply(self, space: SpaceDiscretization, A, b, th: float) -> tuple:  # pylint: disable=unused-argument
        """Reject FEM boundary enforcement for state-free credit claims."""

        raise UnsupportedSpatialCreditRiskModel(
            "constant-intensity credit claims have no spatial boundary facets"
        )


@dataclass
class CreditRiskProblem(Problem):
    """Problem preset for the supported reduced-form defaultable bond."""

    dynamics: DynamicsModel = field(init=False)
    payoff: Payoff = field(init=False)
    boundary_condition: BoundaryCondition = field(init=False)
    r: float = 0.03
    default_intensity: float = 0.02
    recovery_rate: float = 0.4
    notional: float = 1.0
    recovery_convention: str = RECOVERY_OF_PAR_AT_DEFAULT

    def __post_init__(self) -> None:
        """Instantiate the analytical reduced-form model and claim."""

        dynamics = ReducedFormCreditRiskModel(
            r=self.r,
            default_intensity=self.default_intensity,
        )
        payoff = DefaultableZeroCouponClaim(
            notional=self.notional,
            recovery_rate=self.recovery_rate,
            recovery_convention=self.recovery_convention,
        )
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "payoff", payoff)
        object.__setattr__(self, "boundary_condition", NoSpatialBoundaryCondition())

    @property
    def reduced_form_model(self) -> ReducedFormCreditRiskModel:
        """Return the typed reduced-form model for this preset."""

        assert isinstance(self.dynamics, ReducedFormCreditRiskModel)
        return self.dynamics

    @property
    def claim(self) -> DefaultableZeroCouponClaim:
        """Return the typed defaultable zero-coupon claim for this preset."""

        assert isinstance(self.payoff, DefaultableZeroCouponClaim)
        return self.payoff

    def value(self, maturity: float) -> float:
        """Return analytical defaultable zero-coupon value."""

        return self.reduced_form_model.defaultable_zero_coupon_value(
            self.claim, maturity
        )

    def value_components(self, maturity: float) -> DefaultableZeroCouponOutputs:
        """Return separated value, probability and loss outputs."""

        return self.reduced_form_model.value_components(self.claim, maturity)
