"""Problem definition for vanilla option pricing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.boundary import DirichletBC

from .base import Problem


@dataclass
class OptionPricingProblem(Problem):
    """Convenience problem with Black–Scholes defaults.

    Parameters
    ----------
    k:
        Strike price of the option.
    r:
        Risk-free rate.
    q:
        Continuous dividend yield.
    sigma:
        Volatility of the underlying asset.
    is_call:
        Whether the option is a call (``True``) or put (``False``).
    boundaries:
        Collection of boundary set names where Dirichlet conditions are
        enforced.  Defaults to ``("left", "right")`` which correspond to
        the extremes of the one dimensional domain.
    """

    k: float = 1.0
    r: float = 0.03
    q: float = 0.0
    sigma: float = 0.2
    is_call: bool = True
    boundaries: Iterable[str] = field(default_factory=lambda: ("left", "right"))

    def __post_init__(self) -> None:
        """Instantiate default dynamics, payoff and boundary condition."""

        dynamics = DynamicsParametersBlackScholes(r=self.r, q=self.q, sig=self.sigma)
        market = Market(r=self.r)
        payoff = EuropeanOptionBs(k=self.k, q=self.q, mkt=market)
        boundary_condition = DirichletBC(self.boundaries)
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "payoff", payoff)
        object.__setattr__(self, "boundary_condition", boundary_condition)
