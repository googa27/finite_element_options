"""Assembly utilities for spatial discretisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import skfem as fem

from ..config import ELEM

from .forms import Forms
from . import boundary


@dataclass
class SpaceSolver:
    """Handle spatial assembly of the option pricing problem."""

    prm: any

    def __post_init__(self) -> None:
        self.forms = Forms(self.prm)
        self.Th = self.prm.mesh
        self.Vh = fem.CellBasis(self.Th, ELEM)
        self.dVh = fem.FacetBasis(self.Th, ELEM)
        self.I = self.forms.id_bil().assemble(self.Vh)
        self.L = self.forms.l_bil().assemble(self.Vh)
        self.v0 = self.Vh.project(
            lambda x: self.prm.bsopt.call_payoff(x[0]) * self.prm.is_call
            + self.prm.bsopt.put_payoff(x[0]) * (not self.prm.is_call)
        )

    def rhs(self, th: float) -> np.ndarray:
        """Assemble inhomogeneous boundary contribution at time ``th``."""

        return self.forms.b_lin().assemble(self.dVh, th=th)

    def apply_dirichlet(self, A, b, th: float):
        """Apply Dirichlet boundary conditions if configured."""

        if self.prm.dirichlet_bcs:
            u_dirichlet = self.Vh.project(
                lambda x: self.prm.bsopt.call(
                    th, x[0], self.prm.dh.mean_variance(th, x[1])
                )
                if self.prm.is_call
                else self.prm.bsopt.put(
                    th, x[0], self.prm.dh.mean_variance(th, x[1])
                )
            )
            A, b = boundary.apply_dirichlet(
                self.Vh, A, b, self.prm.dirichlet_bcs, u_dirichlet
            )
        return A, b
