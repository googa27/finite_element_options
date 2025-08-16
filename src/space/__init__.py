"""Spatial discretisation helpers and solvers."""

from .mesh import create_mesh, create_rectangular_mesh
from .forms import Forms
from .solver import SpaceSolver
from .boundary import apply_dirichlet
from .adaptive import AdaptiveMesh

__all__ = [
    "create_mesh",
    "create_rectangular_mesh",
    "Forms",
    "SpaceSolver",
    "apply_dirichlet",
    "AdaptiveMesh",
]
