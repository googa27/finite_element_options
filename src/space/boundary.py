"""Spatial boundary condition utilities."""

from typing import Iterable, Tuple

import skfem as fem


def apply_dirichlet(
    Vh: fem.Basis,
    A,
    b,
    boundaries: Iterable[str],
    values,
) -> Tuple:
    """Enforce Dirichlet conditions on matrix ``A`` and vector ``b``.

    Parameters
    ----------
    Vh:
        The finite element basis associated with the mesh.
    A, b:
        System matrix and right-hand side vector to be modified in-place.
    boundaries:
        Iterable of boundary names where Dirichlet conditions apply.
    values:
        Projected boundary values.
    """
    return fem.enforce(A, b, x=values, D=Vh.get_dofs(boundaries))
