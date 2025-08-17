"""Mesh utilities for finite element domain construction."""

from typing import Sequence

import numpy as np
import skfem as fem

from src.core.config import Config


def create_mesh(
    extents: Sequence[float], refine: int, config: Config | None = None
) -> tuple[fem.Mesh, Config]:
    """Create a tensor-product mesh of configurable dimension.

    Parameters
    ----------
    extents:
        Sequence of domain maxima for each spatial dimension.
    refine:
        Number of uniform refinement steps.
    config:
        Optional configuration object updated with the element type.

    Returns
    -------
    tuple[fem.Mesh, Config]
        Refined mesh and the configuration containing the chosen element.
    """

    cfg = config or Config()
    dim = len(extents)
    grids = [np.linspace(0.0, e, 2) for e in extents]
    if dim == 1:
        mesh = fem.MeshLine(np.linspace(0.0, extents[0], 2)).refined(refine)
        cfg.elem = fem.ElementLineP2()
    elif dim == 2:
        mesh = (
            fem.MeshTri()
            .init_tensor(x=grids[0], y=grids[1])
            .refined(refine)
        )
        cfg.elem = fem.ElementTriP2()
    elif dim == 3:
        mesh = (
            fem.MeshTet()
            .init_tensor(x=grids[0], y=grids[1], z=grids[2])
            .refined(refine)
        )
        cfg.elem = fem.ElementTetP1()
    else:
        raise ValueError("Only 1D, 2D and 3D meshes are supported.")
    return mesh, cfg


def create_rectangular_mesh(
    s_max: float, v_max: float, refine: int, config: Config | None = None
) -> tuple[fem.Mesh, Config]:
    """Backward-compatible 2D mesh creation helper."""

    return create_mesh([s_max, v_max], refine, config)
