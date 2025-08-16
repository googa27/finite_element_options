"""Illustrative credit risk problem with basic defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pydantic as pyd

from src.core.interfaces import Payoff, DynamicsModel, StochasticDynamicsModel
from src.core.market import Market
from src.space.boundary import DirichletBC

from .base import Problem


class CreditRiskDynamics(pyd.BaseModel):
    """Toy dynamics with constant default intensity."""

    r: float
    lamb: float

    def mean_variance(self, th, _):  # pylint: disable=unused-argument
        """No variance in the simple intensity model."""
        return 0.0

    def A(self, s):  # pylint: disable=unused-argument
        """No diffusion term for the intensity process."""
        return [[0.0]]

    def dA(self, s):  # pylint: disable=unused-argument
        """Divergence of the diffusion matrix."""
        return [0.0]

    def b(self, s):  # pylint: disable=unused-argument
        """Drift capturing the default intensity."""
        return [-self.lamb * s]


class CreditRiskJumpDynamics(CreditRiskDynamics):
    """Default intensity with lognormal jump risk."""

    jump_vol: float = 0.1

    def sample(self, rng: np.random.Generator) -> CreditRiskDynamics:
        """Return dynamics with randomised default intensity."""
        new_lamb = rng.lognormal(mean=np.log(self.lamb), sigma=self.jump_vol)
        return CreditRiskDynamics(r=self.r, lamb=new_lamb)


@dataclass(frozen=True)
class CreditRiskPayoff(Payoff):
    """Simple payoff for a defaultable zero-coupon bond."""

    recovery: float
    mkt: Market

    def call_payoff(self, s: float) -> float:  # pylint: disable=unused-argument
        """No call-style payoff; return zero."""
        return 0.0

    def put_payoff(self, s: float) -> float:
        """Loss given default modeled as ``1 - recovery``."""
        return 1.0 - self.recovery

    def call(self, th: float, s: float, v: float) -> float:  # pylint: disable=unused-argument
        """Call price is zero for the bond."""
        return 0.0

    def put(self, th: float, s: float, v: float) -> float:  # pylint: disable=unused-argument
        """Discounted expected loss at maturity."""
        return self.mkt.discount_factor(th) * self.put_payoff(s)


@dataclass
class CreditRiskProblem(Problem):
    """Credit risk problem with default intensity and recovery rate."""

    r: float = 0.03
    default_intensity: float = 0.02
    recovery: float = 0.4
    boundaries: Iterable[str] = field(default_factory=lambda: ("default", "survival"))

    def __post_init__(self) -> None:
        """Instantiate default dynamics, payoff and boundary condition."""

        dynamics: DynamicsModel = CreditRiskDynamics(r=self.r, lamb=self.default_intensity)
        market = Market(r=self.r)
        payoff = CreditRiskPayoff(recovery=self.recovery, mkt=market)
        boundary_condition = DirichletBC(self.boundaries)
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "payoff", payoff)
        object.__setattr__(self, "boundary_condition", boundary_condition)
