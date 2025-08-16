"""Mesh utilities for finite element domain construction."""

from typing import Sequence

import numpy as np
import skfem as fem
import CONFIG as CFG


def create_mesh(extents: Sequence[float], refine: int) -> fem.Mesh:
    """Create a tensor-product mesh of configurable dimension.

    Parameters
    ----------
    extents:
        Sequence of domain maxima for each spatial dimension.
    refine:
        Number of uniform refinement steps.

    Returns
    -------
    skfem.Mesh
        Refined mesh of appropriate dimension (line, triangle or tetrahedron).
    """

    dim = len(extents)
    grids = [np.linspace(0.0, e, 2) for e in extents]
    if dim == 1:
        mesh = fem.MeshLine(np.linspace(0.0, extents[0], 2)).refined(refine)
        CFG.ELEM = fem.ElementLineP2()
    elif dim == 2:
        mesh = (
            fem.MeshTri()
            .init_tensor(x=grids[0], y=grids[1])
            .refined(refine)
        )
        CFG.ELEM = fem.ElementTriP2()
    elif dim == 3:
        mesh = (
            fem.MeshTet()
            .init_tensor(x=grids[0], y=grids[1], z=grids[2])
            .refined(refine)
        )
        CFG.ELEM = fem.ElementTetP1()
    else:
        raise ValueError("Only 1D, 2D and 3D meshes are supported.")
    return mesh


def create_rectangular_mesh(
    s_max: float, v_max: float, refine: int
) -> fem.Mesh:
    """Backward-compatible 2D mesh creation helper."""

    return create_mesh([s_max, v_max], refine)
