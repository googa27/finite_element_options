"""Extended Heston model dynamics with stochastic interest rate."""

import pydantic as pyd
import numpy as np
import skfem as fem

from .config import Config
from .interfaces import DynamicsModel, Payoff


class _ModelDynamicsMeta(type(pyd.BaseModel), type(DynamicsModel)):
    """Metaclass combining ``BaseModel`` and ``DynamicsModel`` protocols."""


class DynamicsParametersHeston3D(
    pyd.BaseModel, DynamicsModel, metaclass=_ModelDynamicsMeta
):
    """Heston dynamics for stock, variance and short rate."""

    r: float
    q: float
    kappa: float
    theta: float
    sig_v: float
    rho: float
    kappa_r: float
    theta_r: float
    sig_r: float

    def mean_variance(self, th, v, config: Config | None = None):
        """Mean of the variance process under Heston dynamics.

        Parameters
        ----------
        th:
            Time horizon.
        v:
            Initial variance.
        config:
            Optional numerical configuration providing ``eps``.
        """
        cfg = config or Config()
        x = self.kappa * th + cfg.eps
        return -np.expm1(-x) / x * (v - self.theta) + self.theta

    def A(self, s, v, r_val):  # pylint: disable=unused-argument
        """Covariance matrix for the three-dimensional system."""
        zero = np.zeros_like(s)
        ones = np.ones_like(s)
        return [
            [s ** 2 * v, self.rho * self.sig_v * s * v, zero],
            [self.rho * self.sig_v * s * v, self.sig_v ** 2 * v, zero],
            [zero, zero, self.sig_r ** 2 * ones],
        ]

    def dA(self, s, v, r_val):  # pylint: disable=unused-argument
        """Divergence of ``A``."""
        zero = np.zeros_like(s)
        return [
            2 * s * v + self.rho * self.sig_v * s,
            self.rho * self.sig_v * v + self.sig_v ** 2,
            zero,
        ]

    def b(self, s, v, r_val):
        """Drift vector in the Feynman–Kac representation."""
        return [
            (r_val - self.q) * s,
            self.kappa * (self.theta - v),
            self.kappa_r * (self.theta_r - r_val),
        ]

    def boundary_term(self, is_call: bool, payoff: Payoff) -> fem.LinearForm:  # pylint: disable=unused-argument
        """Return the natural boundary contribution, which is zero."""

        @fem.LinearForm
        def zero(_v, _w):  # pylint: disable=unused-argument
            return 0.0

        return zero
