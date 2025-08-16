"""Boundary utilities for spatial discretization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import skfem as fem

from src.core.interfaces import BoundaryCondition, SpaceDiscretization


def apply_dirichlet(A, b, Vh, bcs, x):
    """Apply Dirichlet boundary conditions to system ``(A, b)``."""
    return fem.enforce(A, b, x=x, D=Vh.get_dofs(bcs))


@dataclass
class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition applied on a set of boundaries."""

    boundaries: Iterable[str]

    def apply(self, space: SpaceDiscretization, A, b, th: float):
        """Enforce Dirichlet values at time ``th`` on ``boundaries``."""
        if not list(self.boundaries):
            return A, b
        u_dirichlet = space.dirichlet(th)
        return apply_dirichlet(A, b, space.Vh, self.boundaries, u_dirichlet)
