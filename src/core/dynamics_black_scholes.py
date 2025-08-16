"""Black-Scholes dynamics in 1D."""

import pydantic as pyd


class DynamicsParametersBlackScholes(pyd.BaseModel):
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
