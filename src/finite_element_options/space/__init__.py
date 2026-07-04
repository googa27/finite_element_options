"""Spatial discretisation helpers and solvers."""

from .mesh import create_mesh, create_rectangular_mesh
from .domain import DomainAxis, DomainSpec, ensure_domain_spec
from .forms import Forms, PDEForms
from .solver import SpaceSolver
from .boundary import apply_dirichlet, DirichletBC
from .adaptive import AdaptiveDiagnostics, AdaptiveMesh, AdaptiveResult, mesh_measure

__all__ = [
    "create_mesh",
    "create_rectangular_mesh",
    "DomainAxis",
    "DomainSpec",
    "ensure_domain_spec",
    "Forms",
    "PDEForms",
    "SpaceSolver",
    "apply_dirichlet",
    "DirichletBC",
    "AdaptiveDiagnostics",
    "AdaptiveMesh",
    "AdaptiveResult",
    "mesh_measure",
]
