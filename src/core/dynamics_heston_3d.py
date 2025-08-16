"""Extended Heston model dynamics with stochastic interest rate."""

import pydantic as pyd
import numpy as np
import CONFIG as CFG


class DynamicsParametersHeston3D(pyd.BaseModel):
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

    def mean_variance(self, th, v):
        """Mean of the variance process under Heston dynamics."""
        x = self.kappa * th + CFG.EPS
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
