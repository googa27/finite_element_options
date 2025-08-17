"""Finite element forms expressed as injectable strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import skfem as fem
import skfem.helpers as fhl

from src.core.interfaces import DynamicsModel, Payoff
from src.transform import CoordinateTransform


class Forms(ABC):
    """Interface defining the variational forms used by the solver."""

    @abstractmethod
    def id_bil(self):
        r"""Return the mass bilinear form ``\int u v\,dx``."""

    @abstractmethod
    def l_bil(self):
        """Return the bilinear form representing the PDE operator."""

    @abstractmethod
    def b_lin(self):
        """Return the linear form associated with natural boundaries."""


class PDEForms(Forms):
    """Default forms constructed from a dynamics model and payoff."""

    def __init__(
        self,
        *,
        is_call: bool,
        payoff: Payoff,
        dynamics: DynamicsModel,
        transform: CoordinateTransform | None = None,
    ):
        """Store option type, payoff and dynamics descriptors.

        Parameters
        ----------
        is_call:
            ``True`` for call options and ``False`` for puts.
        payoff:
            Payoff function describing the terminal condition.
        dynamics:
            Model supplying drift, diffusion and rate terms.
        transform:
            Optional coordinate transform applied before evaluation.
        """

        self.is_call = is_call
        self.payoff = payoff
        self.dynamics = dynamics
        self.transform = transform or CoordinateTransform()

    @staticmethod
    def id_bil():
        r"""Return the mass bilinear form ``\int u v\,dx``."""

        @fem.BilinearForm
        def _id(u, v, _):
            return u * v

        return _id

    def l_bil(self):
        """Return the bilinear form representing the PDE operator."""

        @fem.BilinearForm
        def _l(u, v, w):
            coords = self.transform.untransform_state(w.x)
            A = self.dynamics.A(*coords)
            dA = self.dynamics.dA(*coords)
            b = self.dynamics.b(*coords)
            mu = [b_i - dA_i / 2 for b_i, dA_i in zip(b, dA)]
            return (
                -(1 / 2) * fhl.dot(fhl.grad(v), fhl.mul(A, fhl.grad(u)))
                + v * fhl.dot(mu, fhl.grad(u))
                - self.dynamics.r * v * u
            )

        return _l

    def b_lin(self):
        """Return the linear form associated with natural boundaries."""

        if not hasattr(self.dynamics, "boundary_term"):
            @fem.LinearForm
            def zero(v, w):  # pylint: disable=unused-argument
                return 0.0

            return zero

        return self.dynamics.boundary_term(self.is_call, self.payoff)
