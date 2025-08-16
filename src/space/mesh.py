"""Mesh construction utilities."""

from __future__ import annotations

import numpy as np
import skfem as fem


class MeshHandler:
    """Create and manage computational meshes."""

    def __init__(self, s_max: float, v_max: float, refine: int = 2) -> None:
        self.s_max = s_max
        self.v_max = v_max
        self.refine = refine

    def mesh_init(self) -> fem.MeshTri:
        """Initialise a tensor-product triangular mesh."""

        mesh = (
            fem.MeshTri()
            .init_tensor(
                x=np.linspace(0, self.s_max, 2),
                y=np.linspace(0, self.v_max, 2),
            )
            .refined(self.refine)
            .with_boundaries(
                {
                    "s_min": lambda x: x[0] == 0,
                    "s_max": lambda x: x[0] == self.s_max,
                    "v_min": lambda x: x[1] == 0,
                    "v_max": lambda x: x[1] == self.v_max,
                }
            )
        )
        return mesh
