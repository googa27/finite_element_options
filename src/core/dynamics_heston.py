"""Heston model dynamics utilities.

This module contains only numerical code and is free of any side effects or UI
logic so that it can be reused from both Streamlit and command line
interfaces.
"""

import pydantic as pyd
import numpy as np
import skfem as fem

from .config import Config
from .interfaces import DynamicsModel, Payoff


class _ModelDynamicsMeta(type(pyd.BaseModel), type(DynamicsModel)):
    """Metaclass combining ``BaseModel`` and ``DynamicsModel`` protocols."""


class DynamicsParametersHeston(
    pyd.BaseModel, DynamicsModel, metaclass=_ModelDynamicsMeta
):
    r"""Parameters for the two-dimensional Heston stochastic volatility model.

    The spot price ``S_t`` and variance ``V_t`` satisfy

    .. math::
       \begin{aligned}
       dS_t &= (r-q)S_t\,dt + \sqrt{V_t} S_t\,dW^S_t,\\
       dV_t &= \kappa(\theta - V_t)\,dt + \sigma\sqrt{V_t}\,dW^V_t,\\
       \end{aligned}

    with correlation ``\rho = d\langle W^S, W^V \rangle_t``.
    """

    r: float
    q: float
    kappa: float
    theta: float
    sig: float
    rho: float

    def cir_number(self) -> float:
        """Return the Cox–Ingersoll–Ross (CIR) parameter."""
        return 2 * self.kappa * self.theta / self.sig ** 2

    def cir_message(self) -> str:
        """Human readable message about the CIR parameter."""
        return f"CIR Parameter (must be greater than 1): {self.cir_number():.2f}"

    def mean_variance(self, th, v, config: Config | None = None):
        r"""Return ``\mathbb{E}[V_{t+th} \mid V_t = v]`` under CIR dynamics.

        Parameters
        ----------
        th:
            Time horizon.
        v:
            Initial variance.
        config:
            Optional numerical configuration. If not provided a default
            :class:`~src.core.config.Config` instance is used.
        """
        cfg = config or Config()
        x = self.kappa * th + cfg.eps
        return -np.expm1(-x) / x * (v - self.theta) + self.theta

    # def mean_variance(self, th, v):
    #     return v + th*0

    # def mean_variance(self, th, v):
    #     return (alp.CIRProcess(theta=self.kappa,
    #                            mu=self.theta,
    #                            sigma=self.sig,
    #                            initial=v)
    #             .get_marginal(t=th + cfg.eps)
    #             .mean()
    #             )

    def A(self, x, y) -> list[list]:
        '''
        Covariance matrix appearing in feynman kac formula.
        '''
        return [[x**2*y, self.rho*self.sig*x*y],
                [self.rho*self.sig*x*y, self.sig**2*y]]

    def dA(self, x, y) -> list:
        '''
        Divergence of A
        '''
        return [2*x*y + self.rho*self.sig*x,
                self.rho*self.sig*y + self.sig**2]

    def b(self, x, y) -> list:
        '''
        Drift vector in feynman kac formula
        '''
        return [(self.r - self.q)*x,
                self.kappa*(self.theta - y)]

    def boundary_term(self, is_call: bool, payoff: Payoff) -> fem.LinearForm:  # pylint: disable=unused-argument
        """Return the natural boundary contribution, which is zero."""

        @fem.LinearForm
        def zero(_v, _w):  # pylint: disable=unused-argument
            return 0.0

        return zero

    # def b(self, x, y) -> list:
    #     '''
    #     Drift vector in feynman kac formula
    #     '''
    #     return [self.r - self.q + 0*x,
    #             self.kappa*(self.theta - y)]
