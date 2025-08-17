"""Black-Scholes dynamics in 1D."""

import pydantic as pyd
import skfem as fem

from .interfaces import DynamicsModel, Payoff


class _ModelDynamicsMeta(type(pyd.BaseModel), type(DynamicsModel)):
    """Metaclass combining ``BaseModel`` and ``DynamicsModel`` protocols."""


class DynamicsParametersBlackScholes(
    pyd.BaseModel, DynamicsModel, metaclass=_ModelDynamicsMeta
):
    """Parameters for the one-dimensional Black-Scholes model."""

    r: float
    q: float
    sig: float

    def mean_variance(self, th, _):  # pylint: disable=unused-argument
        """Return the constant variance ``sig^2``."""
        return self.sig ** 2

    def A(self, s):
        """Diffusion matrix (1x1) for the stock price."""
        return [[self.sig ** 2 * s ** 2]]

    def dA(self, s):
        """Divergence of the diffusion matrix."""
        return [2 * self.sig ** 2 * s]

    def b(self, s):
        """Drift vector in the Feynman–Kac formulation."""
        return [(self.r - self.q) * s]

    def boundary_term(self, is_call: bool, payoff: Payoff) -> fem.LinearForm:  # pylint: disable=unused-argument
        """Return the natural boundary contribution, which is zero."""

        @fem.LinearForm
        def zero(_v, _w):  # pylint: disable=unused-argument
            return 0.0

        return zero
