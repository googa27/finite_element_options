"""Spatial solver assembling finite element operators."""

from __future__ import annotations

import numpy as np
import skfem as fem

from .forms import Forms
from .boundary import apply_dirichlet
import CONFIG as CFG


class SpaceSolver:
    """Assemble spatial operators and boundary terms for the PDE."""

    def __init__(self, mesh: fem.Mesh, dynh, bsopt, is_call: bool):
        self.mesh = mesh
        self.dynh = dynh
        self.bsopt = bsopt
        self.is_call = is_call
        self.Vh = fem.CellBasis(mesh, CFG.ELEM)
        self.dVh = fem.FacetBasis(mesh, CFG.ELEM)
        self.forms = Forms(is_call=is_call, bsopt=bsopt, dynh=dynh)
        self.mass = self.forms.id_bil().assemble(self.Vh)
        self.stiffness = self.forms.l_bil().assemble(self.Vh)

    def initial_condition(self) -> np.ndarray:
        """Initial spatial values from the payoff."""
        return self.Vh.project(
            lambda x: self.bsopt.call_payoff(x[0]) * self.is_call
            + self.bsopt.put_payoff(x[0]) * (not self.is_call)
        )

    def matrices(self, theta: float, dt: float):
        """Return the system matrices for the θ-scheme."""
        A = self.mass - theta * dt * self.stiffness
        B = self.mass + (1 - theta) * dt * self.stiffness
        return A, B

    def boundary_term(self, th: float) -> np.ndarray:
        """Assemble the natural boundary contribution at time ``th``."""
        return self.forms.b_lin().assemble(self.dVh, th=th)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return Dirichlet values at time ``th``."""
        return self.Vh.project(
            lambda x: (
                self.bsopt.call(
                    th, x[0], self.dynh.mean_variance(th, x[1])
                )
                if self.is_call
                else self.bsopt.put(
                    th, x[0], self.dynh.mean_variance(th, x[1])
                )
            )
        )

    def apply_dirichlet(self, A, b, dirichlet_bcs, u_dirichlet):
        """Apply Dirichlet boundary conditions to ``A`` and ``b``."""
        return apply_dirichlet(A, b, self.Vh, dirichlet_bcs, u_dirichlet)
