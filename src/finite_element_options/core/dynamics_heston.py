"""Heston model dynamics utilities.

This module contains only numerical code and is free of any side effects or UI
logic so that it can be reused from both Streamlit and command line
interfaces.
"""

import pydantic as pyd
import skfem as fem

from .cir import (
    cir_conditional_mean,
    cir_time_average_mean,
    cir_variance_domain_diagnostics,
    feller_ratio,
    validate_cir_variance_parameters,
)
from .config import Config
from .interfaces import DynamicsModel, Payoff


class _ModelDynamicsMeta(type(pyd.BaseModel), type(DynamicsModel)):  # type: ignore[misc]
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

    @pyd.model_validator(mode="after")
    def _validate_cir_parameters(self) -> "DynamicsParametersHeston":
        validate_cir_variance_parameters(
            kappa=self.kappa,
            theta=self.theta,
            volatility_of_variance=self.sig,
            rho=self.rho,
        )
        return self

    @property
    def variance_volatility(self) -> float:
        """Return the volatility-of-variance parameter ``sigma_v``."""

        return self.sig

    def cir_number(self) -> float:
        """Return the Cox–Ingersoll–Ross (CIR) Feller ratio."""
        return feller_ratio(
            kappa=self.kappa,
            theta=self.theta,
            volatility_of_variance=self.variance_volatility,
        )

    def cir_message(self) -> str:
        """Human readable message about the CIR parameter."""
        return f"CIR Parameter (must be greater than or equal to 1): {self.cir_number():.2f}"

    def mean_variance(self, th, v, config: Config | None = None):
        r"""Return expected average variance over ``[t, t+th]``.

        Black-Scholes Dirichlet boundary oracles consume a constant variance
        for the whole remaining horizon, so the effective variance is the CIR
        time-average mean, not the terminal conditional mean.
        """
        del config
        return cir_time_average_mean(
            kappa=self.kappa,
            theta=self.theta,
            horizon=th,
            initial_variance=v,
        )

    def terminal_mean_variance(self, th, v):
        r"""Return ``\mathbb{E}[V_{t+th} \mid V_t = v]`` under CIR dynamics."""
        return cir_conditional_mean(
            kappa=self.kappa,
            theta=self.theta,
            horizon=th,
            initial_variance=v,
        )

    def variance_domain_diagnostics(
        self,
        *,
        horizon: float,
        initial_variance,
        tail_mass: float = 1.0e-6,
    ) -> dict:
        """Return conservative variance-domain truncation diagnostics."""

        return cir_variance_domain_diagnostics(
            kappa=self.kappa,
            theta=self.theta,
            volatility_of_variance=self.variance_volatility,
            horizon=horizon,
            initial_variance=initial_variance,
            tail_mass=tail_mass,
        )

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

    def discount(self, state, time):  # pylint: disable=unused-argument
        """Return the constant short-rate reaction coefficient."""

        return self.r

    def source(self, state, time):  # pylint: disable=unused-argument
        """Return the default zero running source/load."""

        return 0.0

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
