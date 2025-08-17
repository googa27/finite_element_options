"""Experimental FEniCS-based spatial solver using UFL forms.

This spike mirrors the existing :class:`SpaceSolver` implemented with
scikit-fem but leverages the FEniCSx runtime and UFL to assemble the
same PDE operators.  The implementation currently targets the
one-dimensional Black--Scholes equation with constant coefficients and
serves as a starting point for broader support.

The solver exposes a minimal API compatible with the :class:`ThetaScheme`
stepper, allowing drop-in benchmarking against the scikit-fem backend.
FEniCS is treated as an optional dependency; instantiation will raise an
:class:`ImportError` if the library is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # pragma: no cover - optional dependency
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, mesh
    import ufl
except Exception:  # pragma: no cover - imported lazily
    fem = mesh = ufl = MPI = PETSc = None

from src.core.interfaces import DynamicsModel, Payoff, SpaceDiscretization


@dataclass
class FenicsSolver(SpaceDiscretization):
    """Finite element spatial discretisation using FEniCSx."""

    domain: tuple[float, float]
    num_elements: int
    dynamics: DynamicsModel
    payoff: Payoff
    is_call: bool = True

    def __post_init__(self) -> None:
        """Create mesh and assemble static operators after initialisation."""
        if fem is None:  # pragma: no cover - optional dependency
            raise ImportError("FEniCSx is required for FenicsSolver")

        # create interval mesh and function space
        self.mesh = mesh.create_interval(
            MPI.COMM_WORLD, self.num_elements, list(self.domain)
        )
        self.Vh = fem.FunctionSpace(self.mesh, ("Lagrange", 1))
        self._assemble_operators()

    # ------------------------------------------------------------------
    def _assemble_operators(self) -> None:
        """Assemble mass and stiffness matrices via UFL forms."""

        u = ufl.TrialFunction(self.Vh)
        v = ufl.TestFunction(self.Vh)
        x = ufl.SpatialCoordinate(self.mesh)

        sig = self.dynamics.sig
        r = self.dynamics.r
        q = getattr(self.dynamics, "q", 0.0)

        mass_form = u * v * ufl.dx
        pde_form = (
            0.5 * sig**2 * x[0] ** 2 * ufl.grad(u)[0] * ufl.grad(v)[0]
            + (r - q) * x[0] * ufl.grad(u)[0] * v
            - r * u * v
        ) * ufl.dx

        self.mass = fem.petsc.assemble_matrix(fem.form(mass_form))
        self.mass.assemble()
        self.stiffness = fem.petsc.assemble_matrix(fem.form(pde_form))
        self.stiffness.assemble()

    # ------------------------------------------------------------------
    def initial_condition(self) -> np.ndarray:
        """Project the terminal payoff onto the nodal grid."""
        coords = self.Vh.tabulate_dof_coordinates().reshape(-1)
        if self.is_call:
            return np.array([self.payoff.call_payoff(s) for s in coords])
        return np.array([self.payoff.put_payoff(s) for s in coords])

    def matrices(self, theta: float, dt: float):
        """Return PETSc system matrices for the θ-scheme."""
        A = self.mass.copy()
        A.axpy(-theta * dt, self.stiffness)
        B = self.mass.copy()
        B.axpy((1 - theta) * dt, self.stiffness)
        return A, B

    def boundary_term(self, _th: float) -> np.ndarray:  # pragma: no cover
        """Return natural boundary term (zero for Black--Scholes)."""
        return np.zeros(self.Vh.dofmap.index_map.size_local)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return Dirichlet values at time ``th``."""
        r = self.dynamics.r
        k = self.payoff.k
        s_max = self.domain[1]
        if self.is_call:
            left = 0.0
            right = s_max - k * np.exp(-r * th)
        else:
            left = k * np.exp(-r * th)
            right = 0.0
        vals = np.zeros(self.Vh.dofmap.index_map.size_local)
        vals[0] = left
        vals[-1] = right
        return vals

    def apply_dirichlet(self, A, b, _dirichlet_bcs, u_dirichlet):
        """Apply Dirichlet conditions to PETSc matrices."""
        fem.petsc.apply_lifting(b, [A], bcs=[], x0=None)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(A, b, [])
        with A.localForm() as loc:
            loc.set(0.0)
            loc.diagonal().set(1.0)
        b.array[:] = u_dirichlet
        return A, b

    # ------------------------------------------------------------------
    def solve(self, t: Iterable[float], theta: float = 0.5) -> np.ndarray:
        """Solve the PDE using a θ-scheme time stepper."""
        dt = t[1] - t[0]
        A, B = self.matrices(theta, dt)
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")

        u = PETSc.Vec().createMPI(A.getSize()[0])
        u.array = self.initial_condition()
        sol = np.empty((len(t), u.getSize()))
        sol[0] = u.array

        for i, th in enumerate(t[:-1]):
            b = B @ u
            u_d = self.dirichlet(th + dt)
            A_bc, b_bc = self.apply_dirichlet(A.copy(), b, [], u_d)
            ksp.solve(b_bc, u)
            sol[i + 1] = u.array
        return sol
