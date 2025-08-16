"""Spatial discretisation utilities."""

from .forms import Forms
from .mesh import MeshHandler  # keeps mesh utilities available
from .boundary import apply_dirichlet
from .solver import SpaceSolver

__all__ = [
    "Forms",
    "MeshHandler",
    "apply_dirichlet",
    "SpaceSolver",
]
