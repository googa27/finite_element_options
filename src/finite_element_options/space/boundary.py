"""Boundary utilities for spatial discretization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import skfem as fem

from finite_element_options.core.interfaces import BoundaryCondition, SpaceDiscretization


def _materialize_boundaries(boundaries: Iterable[str]) -> tuple[str, ...]:
    """Return a validated tuple of boundary names."""

    names = tuple(str(boundary) for boundary in boundaries)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate boundary facet(s): {', '.join(duplicates)}")
    return names


def _available_boundary_names(Vh) -> set[str] | None:
    """Return named facets advertised by a basis mesh, if available."""

    mesh = getattr(Vh, "mesh", None)
    if mesh is None:
        return None
    boundaries = getattr(mesh, "boundaries", None) or {}
    return set(boundaries)


def validate_boundary_names(Vh, boundaries: Iterable[str]) -> tuple[str, ...]:
    """Validate named boundary facets before algebraic enforcement."""

    names = _materialize_boundaries(boundaries)
    if not names:
        return names
    available = _available_boundary_names(Vh)
    if available is None:
        return names
    missing = sorted(set(names) - available)
    if missing:
        available_msg = ", ".join(sorted(available)) if available else "<none>"
        raise ValueError(
            "Unknown boundary facet(s): "
            f"{', '.join(missing)}; available facets: {available_msg}"
        )
    return names


def apply_dirichlet(A, b, Vh, bcs, x):
    """Apply Dirichlet boundary conditions to system ``(A, b)``."""

    boundaries = validate_boundary_names(Vh, bcs)
    if not boundaries:
        return A, b
    return fem.enforce(A, b, x=x, D=Vh.get_dofs(boundaries))


@dataclass
class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition applied on a set of named facets."""

    boundaries: Iterable[str]

    def __post_init__(self) -> None:
        """Materialize boundary iterables exactly once."""

        self.boundaries = _materialize_boundaries(self.boundaries)

    def apply(self, space: SpaceDiscretization, A, b, th: float):
        """Enforce Dirichlet values at time ``th`` on ``boundaries``."""

        if not self.boundaries:
            return A, b
        u_dirichlet = space.dirichlet(th)
        return apply_dirichlet(A, b, space.Vh, self.boundaries, u_dirichlet)
