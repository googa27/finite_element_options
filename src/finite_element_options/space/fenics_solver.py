"""Experimental FEniCSx spatial solver using supported DOLFINx/PETSc APIs.

This optional backend mirrors the one-dimensional Black--Scholes scikit-fem
route while keeping FEniCSx/PETSc semantics explicit: boundary degrees of
freedom are located geometrically, Dirichlet conditions are passed through
DOLFINx boundary-condition objects, the PETSc KSP always solves the boundary-
modified matrix, and divergence is surfaced as an exception with diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import default_scalar_type, fem, mesh
    import ufl
except Exception:  # pragma: no cover - imported lazily
    fem = mesh = ufl = MPI = PETSc = default_scalar_type = None

from finite_element_options.core.interfaces import (
    DynamicsModel,
    Payoff,
    SpaceDiscretization,
)


@dataclass
class FenicsSolver(SpaceDiscretization):
    """Finite element spatial discretisation using FEniCSx."""

    domain: tuple[float, float]
    num_elements: int
    dynamics: DynamicsModel
    payoff: Payoff
    is_call: bool = True
    petsc_options_prefix: str | None = None
    mesh: Any = field(init=False, repr=False)
    Vh: Any = field(init=False, repr=False)
    mass: Any = field(init=False, repr=False)
    stiffness: Any = field(init=False, repr=False)
    _left_dofs: Any = field(init=False, repr=False)
    _right_dofs: Any = field(init=False, repr=False)
    _mass_expr: Any = field(init=False, repr=False)
    _operator_expr: Any = field(init=False, repr=False)
    _active_system_form: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create mesh, locate boundary DOFs, and assemble static operators."""
        if fem is None or mesh is None or ufl is None:  # pragma: no cover
            raise ImportError("FEniCSx is required for FenicsSolver")
        left, right = (float(self.domain[0]), float(self.domain[1]))
        if not np.isfinite([left, right]).all() or not left < right:
            raise ValueError("domain must be a finite increasing (left, right) pair")
        if self.num_elements < 2:
            raise ValueError("num_elements must be at least 2")
        self.domain = (left, right)
        self.petsc_options_prefix = (
            self.petsc_options_prefix or f"feo_fenicsx_{id(self):x}_"
        )

        self.mesh = mesh.create_interval(
            MPI.COMM_WORLD, self.num_elements, list(self.domain)
        )
        self.Vh = self._function_space(("Lagrange", 1))
        self._left_dofs, self._right_dofs = self._locate_boundary_dofs()
        self._assemble_operators()

    def _function_space(self, element: tuple[str, int]) -> Any:
        """Create a DOLFINx function space across supported API names."""
        if hasattr(fem, "functionspace"):
            return fem.functionspace(self.mesh, element)
        return fem.FunctionSpace(self.mesh, element)  # pragma: no cover - old API

    def _locate_boundary_dofs(self) -> tuple[Any, Any]:
        """Locate endpoint boundary DOFs geometrically, including MPI ownership."""
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        left_facets = mesh.locate_entities_boundary(
            self.mesh, fdim, lambda x: np.isclose(x[0], self.domain[0])
        )
        right_facets = mesh.locate_entities_boundary(
            self.mesh, fdim, lambda x: np.isclose(x[0], self.domain[1])
        )
        left_dofs = fem.locate_dofs_topological(self.Vh, fdim, left_facets)
        right_dofs = fem.locate_dofs_topological(self.Vh, fdim, right_facets)
        if len(left_dofs) == 0 or len(right_dofs) == 0:
            raise RuntimeError("failed to locate FEniCSx endpoint boundary DOFs")
        return left_dofs, right_dofs

    def _assemble_operators(self) -> None:
        """Assemble unconstrained mass and PDE operator forms via UFL."""
        trial = ufl.TrialFunction(self.Vh)
        test = ufl.TestFunction(self.Vh)
        x = ufl.SpatialCoordinate(self.mesh)

        sigma = self.dynamics.sig
        rate = self.dynamics.r
        carry = getattr(self.dynamics, "q", 0.0)

        self._mass_expr = trial * test * ufl.dx
        self._operator_expr = (
            0.5 * sigma**2 * x[0] ** 2 * ufl.grad(trial)[0] * ufl.grad(test)[0]
            + (rate - carry) * x[0] * ufl.grad(trial)[0] * test
            - rate * trial * test
        ) * ufl.dx

        self.mass = self._assemble_matrix(self._mass_expr)
        self.stiffness = self._assemble_matrix(self._operator_expr)

    def _assemble_matrix(self, expression: Any, bcs: Sequence[Any] = ()) -> Any:
        """Assemble a PETSc matrix from a UFL expression and optional BCs."""
        matrix = fem.petsc.assemble_matrix(fem.form(expression), bcs=list(bcs))
        matrix.assemble()
        return matrix

    def _system_form(self, theta: float, dt: float) -> Any:
        """Return the θ-step left-hand-side DOLFINx form."""
        return fem.form(self._mass_expr - theta * dt * self._operator_expr)

    def _rhs_form(self, theta: float, dt: float) -> Any:
        """Return the θ-step right-hand-side matrix expression."""
        return fem.form(self._mass_expr + (1.0 - theta) * dt * self._operator_expr)

    def _system_matrix(self, theta: float, dt: float, bcs: Sequence[Any]) -> tuple[Any, Any]:
        """Assemble the boundary-modified left-hand-side matrix."""
        system_form = self._system_form(theta, dt)
        matrix = fem.petsc.assemble_matrix(system_form, bcs=list(bcs))
        matrix.assemble()
        return matrix, system_form

    def _rhs_matrix(self, theta: float, dt: float) -> Any:
        """Assemble the unconstrained right-hand-side matrix."""
        matrix = fem.petsc.assemble_matrix(self._rhs_form(theta, dt), bcs=[])
        matrix.assemble()
        return matrix

    def _boundary_values(self, th: float) -> tuple[float, float]:
        """Return left and right Black--Scholes Dirichlet endpoint values."""
        rate = self.dynamics.r
        strike = self.payoff.k
        upper = self.domain[1]
        discount = np.exp(-rate * th)
        if self.is_call:
            return 0.0, float(upper - strike * discount)
        return float(strike * discount), 0.0

    def _scalar(self, value: float) -> Any:
        """Cast a Python float to the scalar type expected by DOLFINx."""
        if default_scalar_type is not None:
            return default_scalar_type(value)
        return PETSc.ScalarType(value)  # pragma: no cover - defensive fallback

    def _dirichlet_bcs(self, th: float) -> list[Any]:
        """Create supported DOLFINx DirichletBC objects for endpoint DOFs."""
        left, right = self._boundary_values(th)
        return [
            fem.dirichletbc(self._scalar(left), self._left_dofs, self.Vh),
            fem.dirichletbc(self._scalar(right), self._right_dofs, self.Vh),
        ]

    def _local_dof_count(self) -> int:
        """Return the local scalar DOF count for this MPI rank."""
        index_map = self.Vh.dofmap.index_map
        block_size = self.Vh.dofmap.index_map_bs
        return int(index_map.size_local * block_size)

    def _dof_coordinates(self) -> np.ndarray:
        """Return scalar DOF coordinates for this rank."""
        coordinates = self.Vh.tabulate_dof_coordinates()
        return np.asarray(coordinates, dtype=float).reshape((-1, coordinates.shape[-1]))

    def _owned_array(self, vec: Any) -> np.ndarray:
        """Copy the owned local PETSc vector entries as a NumPy array."""
        return np.asarray(vec.array, dtype=float).copy()

    def _initial_petsc_vec(self) -> Any:
        """Create the initial PETSc vector from local terminal payoff values."""
        index_map = self.Vh.dofmap.index_map
        block_size = self.Vh.dofmap.index_map_bs
        local_size = int(index_map.size_local * block_size)
        global_size = int(index_map.size_global * block_size)
        vec = PETSc.Vec().createMPI((local_size, global_size), comm=self.mesh.comm)
        vec.array[:] = self.initial_condition()
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return vec

    def initial_condition(self) -> np.ndarray:
        """Project the terminal payoff onto local FEniCSx DOF coordinates."""
        coords = self._dof_coordinates()[:, 0]
        if self.is_call:
            return np.asarray([self.payoff.call_payoff(s) for s in coords], dtype=float)
        return np.asarray([self.payoff.put_payoff(s) for s in coords], dtype=float)

    def matrices(
        self,
        theta: float,
        dt: float,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[Any, Any]:
        """Return unconstrained PETSc matrices for a θ-scheme interval."""
        del start, end
        return self._assemble_matrix(
            self._mass_expr - theta * dt * self._operator_expr
        ), self._rhs_matrix(theta, dt)

    def boundary_term(self, _th: float) -> np.ndarray:  # pragma: no cover
        """Return natural boundary/source term; zero for this Black--Scholes route."""
        return np.zeros(self._local_dof_count(), dtype=float)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return endpoint Dirichlet values at geometrically matched local DOFs."""
        coords = self._dof_coordinates()[:, 0]
        left_value, right_value = self._boundary_values(th)
        values = np.zeros(coords.shape[0], dtype=float)
        left_mask = np.isclose(coords, self.domain[0])
        right_mask = np.isclose(coords, self.domain[1])
        values[left_mask] = left_value
        values[right_mask] = right_value
        return values

    def apply_dirichlet(
        self,
        A: Any,
        b: Any,
        dirichlet_bcs: Iterable[Any],
        u_dirichlet: np.ndarray,
    ) -> tuple[Any, Any]:
        """Apply supported DOLFINx Dirichlet lifting and RHS value injection."""
        del u_dirichlet
        bcs = list(dirichlet_bcs)
        if self._active_system_form is None:
            raise RuntimeError("active system form is required for DOLFINx BC lifting")
        fem.petsc.apply_lifting(b, [self._active_system_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)
        return A, b

    def _new_ksp(self, A_bc: Any) -> Any:
        """Create a PETSc KSP whose operator is the boundary-modified matrix."""
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOptionsPrefix(str(self.petsc_options_prefix))
        ksp.setOperators(A_bc)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        return ksp

    @staticmethod
    def _check_ksp_converged(ksp: Any) -> None:
        """Raise with PETSc diagnostics when KSP fails to converge."""
        reason = int(ksp.getConvergedReason())
        if reason > 0:
            return
        iterations = int(ksp.getIterationNumber())
        residual = float(ksp.getResidualNorm())
        prefix = ksp.getOptionsPrefix()
        raise RuntimeError(
            "PETSc KSP failed to converge: "
            f"prefix={prefix!r}, reason={reason}, iterations={iterations}, "
            f"residual={residual:.6e}"
        )

    def solve(self, t: Iterable[float], theta: float = 0.5) -> np.ndarray:
        """Solve the PDE using a θ-scheme with per-step BC assembly."""
        times = np.asarray(tuple(t), dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("time grid must contain at least two entries")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise ValueError("time grid must be finite and strictly increasing")
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")

        u = self._initial_petsc_vec()
        local_size = len(self._owned_array(u))
        solution = np.empty((len(times), local_size), dtype=float)
        solution[0] = self._owned_array(u)

        for i, (start, end) in enumerate(zip(times[:-1], times[1:], strict=True)):
            dt = float(end - start)
            bcs = self._dirichlet_bcs(float(end))
            A_bc, system_form = self._system_matrix(theta, dt, bcs)
            self._active_system_form = system_form
            rhs_matrix = self._rhs_matrix(theta, dt)
            b = rhs_matrix @ u
            _, b_bc = self.apply_dirichlet(A_bc, b, bcs, self.dirichlet(float(end)))
            ksp = self._new_ksp(A_bc)
            ksp.solve(b_bc, u)
            self._check_ksp_converged(ksp)
            u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solution[i + 1] = self._owned_array(u)

        self._active_system_form = None
        return solution

    def domain_diagnostics(
        self, *, horizon: float, tail_mass: float = 1.0e-6
    ) -> dict[str, object]:
        """Return minimal public domain diagnostics for this optional backend."""
        del tail_mass
        return {
            "horizon": float(horizon),
            "backend": "fenicsx",
            "coordinate_system": "spot",
            "state_domain": [
                {
                    "name": "spot",
                    "lower": self.domain[0],
                    "upper": self.domain[1],
                    "scale": "linear",
                    "truncation_policy": "finite-interval",
                    "tail_mass": None,
                }
            ],
            "boundary_facets": ("left", "right"),
            "mesh_dimension": 1,
            "mesh_elements": int(self.num_elements),
            "petsc_options_prefix": self.petsc_options_prefix,
        }
