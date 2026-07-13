"""Extended Heston model dynamics with stochastic interest rate."""

import pydantic as pyd
import numpy as np
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

    @pyd.model_validator(mode="after")
    def _validate_cir_parameters(self) -> "DynamicsParametersHeston3D":
        validate_cir_variance_parameters(
            kappa=self.kappa,
            theta=self.theta,
            volatility_of_variance=self.sig_v,
            rho=self.rho,
        )
        if not np.isfinite(float(self.kappa_r)) or self.kappa_r < 0.0:
            raise ValueError("kappa_r must be finite and non-negative")
        if not np.isfinite(float(self.theta_r)):
            raise ValueError("theta_r must be finite")
        if not np.isfinite(float(self.sig_r)) or self.sig_r < 0.0:
            raise ValueError("sig_r must be finite and non-negative")
        return self

    @property
    def variance_volatility(self) -> float:
        """Return the volatility-of-variance parameter ``sigma_v``."""

        return self.sig_v

    def cir_number(self) -> float:
        """Return the Cox–Ingersoll–Ross (CIR) Feller ratio."""

        return feller_ratio(
            kappa=self.kappa,
            theta=self.theta,
            volatility_of_variance=self.variance_volatility,
        )

    def mean_variance(self, th, v, config: Config | None = None):
        r"""Return expected average variance over ``[t, t+th]``.

        The 3D model shares the same CIR variance process as 2D Heston; use
        the time-average mean for finite-horizon boundary values and keep the
        terminal conditional moments in ``variance_domain_diagnostics``.
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

    def discount(self, state, time):  # pylint: disable=unused-argument
        """Return the short-rate state coordinate for reaction/discount terms."""

        state_array = np.asarray(state, dtype=float)
        if state_array.ndim == 0 or state_array.shape[0] < 3:
            raise ValueError("Heston3D discount requires state coordinates (s, v, r)")
        return state_array[2]

    def source(self, state, time):  # pylint: disable=unused-argument
        """Return the default zero running source/load."""

        return 0.0

    def boundary_term(self, is_call: bool, payoff: Payoff) -> fem.LinearForm:  # pylint: disable=unused-argument
        """Return the natural boundary contribution, which is zero."""

        @fem.LinearForm
        def zero(_v, _w):  # pylint: disable=unused-argument
            return 0.0

        return zero
