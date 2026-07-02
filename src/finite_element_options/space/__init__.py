"""Spatial discretisation helpers and solvers."""

from .mesh import create_mesh, create_rectangular_mesh
from .forms import Forms, PDEForms
from .solver import SpaceSolver
from .boundary import apply_dirichlet, DirichletBC
from .adaptive import AdaptiveMesh

__all__ = [
    "create_mesh",
    "create_rectangular_mesh",
    "Forms",
    "PDEForms",
    "SpaceSolver",
    "apply_dirichlet",
    "DirichletBC",
    "AdaptiveMesh",
]
