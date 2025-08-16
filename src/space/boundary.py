"""Boundary utilities for spatial discretization."""

import skfem as fem


def apply_dirichlet(A, b, Vh, bcs, x):
    """Apply Dirichlet boundary conditions.

    Parameters
    ----------
    A, b:
        System matrix and right-hand side.
    Vh:
        Cell basis used to compute Dirichlet degrees of freedom.
    bcs:
        Iterable of boundary names to apply Dirichlet conditions on.
    x:
        Array of Dirichlet values corresponding to Vh degrees of freedom.
    """
    return fem.enforce(A, b, x=x, D=Vh.get_dofs(bcs))
