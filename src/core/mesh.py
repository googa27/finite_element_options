"""Mesh utilities for finite element domain construction."""

import numpy as np
import skfem as fem


def create_rectangular_mesh(s_max: float, v_max: float, refine: int) -> fem.MeshTri:
    """Create a rectangular ``MeshTri`` over ``[0, s_max] x [0, v_max]``.

    Parameters
    ----------
    s_max:
        Maximum underlying price.
    v_max:
        Maximum variance.
    refine:
        Number of uniform refinement steps.

    Returns
    -------
    ``skfem.MeshTri``
        Refined mesh with boundary markers ``s_min``, ``s_max``, ``v_min`` and
        ``v_max``.
    """

    mesh = (
        fem.MeshTri()
        .init_tensor(x=np.linspace(0, s_max, 2), y=np.linspace(0, v_max, 2))
        .refined(refine)
        .with_boundaries(
            {
                "s_min": lambda x: x[0] == 0,
                "s_max": lambda x: x[0] == s_max,
                "v_min": lambda x: x[1] == 0,
                "v_max": lambda x: x[1] == v_max,
            }
        )
    )
    return mesh
