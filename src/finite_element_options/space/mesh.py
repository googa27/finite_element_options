"""Mesh utilities for finite element domain construction."""

from typing import Sequence

import skfem as fem

from finite_element_options.core.config import Config
from finite_element_options.space.domain import (
    DomainAxis,
    DomainSpec,
    attach_domain_metadata,
    ensure_domain_spec,
)


def create_mesh(
    extents: DomainSpec | Sequence[float | Sequence[float] | DomainAxis],
    refine: int,
    config: Config | None = None,
) -> tuple[fem.Mesh, Config]:
    """Create a tensor-product mesh with explicit domain metadata.

    ``extents`` remains backward-compatible with the legacy API: a numeric
    sequence means ``[0, upper]`` per axis.  It may also contain
    :class:`~finite_element_options.space.domain.DomainAxis` records or
    ``(lower, upper)`` pairs, allowing negative rate/OU states and transformed
    coordinates without hidden zero lower bounds.
    """

    cfg = config or Config()
    domain = ensure_domain_spec(extents)
    dim = domain.dimension
    grids = domain.tensor_endpoints()
    if dim == 1:
        mesh = fem.MeshLine(grids[0]).refined(refine)
        cfg.elem = fem.ElementLineP2()
    elif dim == 2:
        mesh = fem.MeshTri().init_tensor(x=grids[0], y=grids[1]).refined(refine)
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
    mesh = mesh.with_boundaries(domain.boundary_predicates())
    return attach_domain_metadata(mesh, domain), cfg


def create_rectangular_mesh(
    s_max: float, v_max: float, refine: int, config: Config | None = None
) -> tuple[fem.Mesh, Config]:
    """Backward-compatible 2D mesh creation helper."""

    return create_mesh([s_max, v_max], refine, config)
