"""Time-stepping algorithms for the option PDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
import skfem as fem  # type: ignore[import-untyped]

from finite_element_options.core.interfaces import BoundaryCondition, SpaceDiscretization
from finite_element_options.time_integration.lcp import (
    DiscreteLCP,
    LCPConvergenceError,
    LCPDiagnostics,
    ProjectedSORSolver,
    ProjectedSORSolverSettings,
)


@dataclass(frozen=True)
class LinearSolveDiagnostics:
    """Diagnostics for one theta-scheme solve and its linear-system cache."""

    linear_solver: str
    factorization_reuse_enabled: bool
    factorization_count: int
    factorization_reuse_count: int
    solve_count: int
    max_linear_residual_abs: float
    assembly_cache_key: str
    factorization_cache_key: str
    stage_timings_sec: dict[str, float]

    def to_public_dict(self) -> dict[str, str | bool | int | float | dict[str, float]]:
        """Return JSON-safe public diagnostics."""

        return {
            "linear_solver": self.linear_solver,
            "factorization_reuse_enabled": self.factorization_reuse_enabled,
            "factorization_count": self.factorization_count,
            "factorization_reuse_count": self.factorization_reuse_count,
            "solve_count": self.solve_count,
            "max_linear_residual_abs": self.max_linear_residual_abs,
            "assembly_cache_key": self.assembly_cache_key,
            "factorization_cache_key": self.factorization_cache_key,
            "stage_timings_sec": dict(self.stage_timings_sec),
        }


@dataclass(frozen=True)
class _InternalThetaStep:
    """One internal theta step, possibly part of a subdivided output interval."""

    output_index: int
    start: float
    end: float
    dt: float
    theta: float


class TimeStepper(ABC):
    """Abstract base class for time-stepping schemes."""

    @abstractmethod
    def solve(
        self,
        t: Iterable[float],
        space: SpaceDiscretization,
        boundary_condition: BoundaryCondition | None = None,
        is_american: bool = False,
    ) -> np.ndarray:
        """Return the time-space solution array."""
        raise NotImplementedError


class ThetaScheme(TimeStepper):
    """General θ-scheme stepping (θ=1 implicit Euler, θ=0 explicit Euler)."""

    def __init__(
        self,
        theta: float = 0.5,
        *,
        linear_solver: str = "scipy_direct",
        reuse_factorization: bool = True,
        startup_theta: float | None = None,
        startup_steps: int = 0,
        startup_substeps: int = 1,
        lcp_solver_settings: ProjectedSORSolverSettings | None = None,
    ):
        """Store θ and sparse-direct solver-cache policy.

        ``startup_theta``/``startup_steps``/``startup_substeps`` encode the same
        step API for pure-theta and Rannacher-style startup.  For example,
        ``ThetaScheme(theta=0.5, startup_theta=1.0, startup_steps=2,
        startup_substeps=2)`` replaces the first two Crank-Nicolson intervals
        by four backward-Euler half-steps.
        """

        if linear_solver != "scipy_direct":
            msg = (
                "Only linear_solver='scipy_direct' is currently supported by the "
                "released FEM route; AMG/PETSc/banded policies fail closed in the "
                "capability manifest."
            )
            raise ValueError(msg)
        self.theta = _validate_theta(theta, "theta")
        self.startup_theta = (
            None if startup_theta is None else _validate_theta(startup_theta, "startup_theta")
        )
        if startup_steps < 0:
            raise ValueError("startup_steps must be non-negative")
        if startup_substeps < 1:
            raise ValueError("startup_substeps must be at least one")
        self.startup_steps = int(startup_steps)
        self.startup_substeps = int(startup_substeps)
        self.linear_solver = linear_solver
        self.reuse_factorization = reuse_factorization
        self.lcp_solver = ProjectedSORSolver(lcp_solver_settings)
        self.last_lcp_diagnostics: list[LCPDiagnostics] = []
        self.last_solve_diagnostics = LinearSolveDiagnostics(
            linear_solver=linear_solver,
            factorization_reuse_enabled=reuse_factorization,
            factorization_count=0,
            factorization_reuse_count=0,
            solve_count=0,
            max_linear_residual_abs=0.0,
            assembly_cache_key="not_run",
            factorization_cache_key="not_run",
            stage_timings_sec={"factorization": 0.0, "solve": 0.0},
        )
        self.last_domain_diagnostics: dict[str, object] = {}
        self.last_time_grid_diagnostics: dict[str, object] = {}

    def solve(
        self,
        t: Iterable[float],
        space: SpaceDiscretization,
        boundary_condition: BoundaryCondition | None = None,
        is_american: bool = False,
    ) -> np.ndarray:
        """Return solution grid for the supplied time nodes ``t``."""

        time_grid = _validate_time_grid(t)
        internal_steps = self._internal_steps(time_grid)
        if hasattr(space, "domain_diagnostics"):
            self.last_domain_diagnostics = space.domain_diagnostics(
                horizon=time_grid[-1] - time_grid[0]
            )
        else:
            self.last_domain_diagnostics = {}
        self.last_time_grid_diagnostics = _time_grid_diagnostics(
            time_grid=time_grid,
            internal_steps=internal_steps,
            startup_theta=self.startup_theta,
            startup_steps=self.startup_steps,
            startup_substeps=self.startup_substeps,
        )

        v_tsv = np.empty((len(time_grid), space.Vh.N))
        current_values = np.asarray(space.initial_condition(), dtype=float)
        v_tsv[0] = current_values
        self.last_lcp_diagnostics = []

        factorized_solvers: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        factorization_count = 0
        factorization_reuse_count = 0
        solve_count = 0
        max_residual = 0.0
        factorization_time = 0.0
        solve_time = 0.0
        assembly_cache_keys: list[str] = []
        factorization_cache_keys: list[str] = []

        for step in internal_steps:
            A, B = _space_matrices(space, step)
            b_previous = B @ current_values
            b_inhom = step.theta * space.boundary_term(step.end) + (
                1.0 - step.theta
            ) * space.boundary_term(step.start)
            b = b_previous + step.dt * b_inhom

            if boundary_condition:
                A_enf, b_enf = boundary_condition.apply(space, A, b, step.end)
            else:
                A_enf, b_enf = A, b

            cache_key = _theta_cache_key(space, A_enf, step.dt, step.theta, boundary_condition)
            assembly_cache_keys.append(cache_key)

            if is_american:
                lcp_result = self.lcp_solver.solve(
                    DiscreteLCP(
                        matrix=A_enf,
                        rhs=np.asarray(b_enf, dtype=float),
                        obstacle=v_tsv[0],
                    ),
                    initial=current_values,
                    fail_on_nonconvergence=False,
                )
                self.last_lcp_diagnostics.append(lcp_result.diagnostics)
                solve_time += lcp_result.diagnostics.solve_time_sec
                solve_count += 1
                max_residual = max(
                    max_residual,
                    lcp_result.diagnostics.projected_residual_max,
                )
                if not lcp_result.success:
                    raise LCPConvergenceError(lcp_result.diagnostics)
                current_values = np.asarray(lcp_result.values, dtype=float)
            else:
                if self.reuse_factorization:
                    if cache_key in factorized_solvers:
                        factorization_reuse_count += 1
                        solver = factorized_solvers[cache_key]
                    else:
                        started = perf_counter()
                        factorized_matrix = sps.csc_matrix(A_enf)
                        lu = spla.splu(factorized_matrix)
                        solver = lu.solve
                        factorized_solvers[cache_key] = solver
                        factorization_time += perf_counter() - started
                        factorization_count += 1
                        factorization_cache_keys.append(_matrix_cache_key(factorized_matrix))
                    started = perf_counter()
                    next_values = solver(np.asarray(b_enf, dtype=float))
                    solve_time += perf_counter() - started
                else:
                    started = perf_counter()
                    next_values = fem.solve(A_enf, b_enf)
                    solve_time += perf_counter() - started
                    factorization_count += 1
                    factorization_cache_keys.append(_matrix_cache_key(sps.csr_matrix(A_enf)))

                current_values = np.asarray(next_values, dtype=float)
                solve_count += 1
                residual = np.asarray(A_enf @ current_values - b_enf, dtype=float)
                if residual.size:
                    max_residual = max(max_residual, float(np.max(np.abs(residual))))

            if np.isclose(step.end, time_grid[step.output_index]):
                v_tsv[step.output_index] = current_values

        self.last_solve_diagnostics = LinearSolveDiagnostics(
            linear_solver=self.linear_solver,
            factorization_reuse_enabled=self.reuse_factorization,
            factorization_count=factorization_count,
            factorization_reuse_count=factorization_reuse_count,
            solve_count=solve_count,
            max_linear_residual_abs=max_residual,
            assembly_cache_key=_sequence_cache_key(assembly_cache_keys),
            factorization_cache_key=_sequence_cache_key(factorization_cache_keys),
            stage_timings_sec={
                "factorization": factorization_time,
                "solve": solve_time,
            },
        )
        return v_tsv

    def _internal_steps(self, time_grid: tuple[float, ...]) -> tuple[_InternalThetaStep, ...]:
        """Return internal steps after optional startup subdivision."""

        steps: list[_InternalThetaStep] = []
        local_steps = _canonical_local_steps(time_grid)
        for interval_index, (start, end, width) in enumerate(
            zip(time_grid[:-1], time_grid[1:], local_steps)
        ):
            use_startup = self.startup_theta is not None and interval_index < self.startup_steps
            theta = self.theta
            if use_startup:
                if self.startup_theta is None:  # pragma: no cover - guarded above
                    raise RuntimeError("startup_theta unexpectedly missing")
                theta = self.startup_theta
            substeps = self.startup_substeps if use_startup else 1
            dt = width / substeps
            for substep in range(substeps):
                fraction_start = substep / substeps
                fraction_end = (substep + 1) / substeps
                sub_start = start + fraction_start * (end - start)
                sub_end = start + fraction_end * (end - start)
                steps.append(
                    _InternalThetaStep(
                        output_index=interval_index + 1,
                        start=sub_start,
                        end=sub_end,
                        dt=dt,
                        theta=theta,
                    )
                )
        return tuple(steps)


def _validate_theta(value: float, name: str) -> float:
    """Return a validated theta parameter."""

    theta = float(value)
    if not np.isfinite(theta):
        raise ValueError(f"{name} must be finite")
    if theta < 0.0 or theta > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return theta


def _validate_time_grid(t: Iterable[float]) -> tuple[float, ...]:
    """Materialize and validate a strictly increasing finite time grid."""

    time_grid = tuple(float(item) for item in t)
    if len(time_grid) < 2:
        raise ValueError("time grid must contain at least two nodes")
    arr = np.asarray(time_grid, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("time grid nodes must be finite")
    steps = np.diff(arr)
    if not np.all(steps > 0.0):
        raise ValueError("time grid nodes must be strictly increasing")
    return time_grid


def _canonical_local_steps(time_grid: tuple[float, ...]) -> tuple[float, ...]:
    """Return local widths, canonicalizing roundoff-uniform grids.

    ``np.linspace`` grids often differ by a few ulps between adjacent
    intervals.  Treating those artifacts as distinct PDE steps defeats sparse
    factorization reuse without adding mathematical information.  Genuinely
    nonuniform grids keep their local widths exactly.
    """

    raw = np.diff(np.asarray(time_grid, dtype=float))
    representative = (time_grid[-1] - time_grid[0]) / (len(time_grid) - 1)
    if np.allclose(raw, representative, rtol=1.0e-12, atol=1.0e-15):
        return tuple(float(representative) for _ in raw)
    return tuple(float(item) for item in raw)


def _time_grid_diagnostics(
    *,
    time_grid: tuple[float, ...],
    internal_steps: tuple[_InternalThetaStep, ...],
    startup_theta: float | None,
    startup_steps: int,
    startup_substeps: int,
) -> dict[str, object]:
    """Return public diagnostics for result-history time orientation."""

    local_steps = _canonical_local_steps(time_grid)
    uniform = bool(np.allclose(local_steps, local_steps[0], rtol=1.0e-12, atol=1.0e-15))
    return {
        "time_grid": time_grid,
        "time_orientation": "increasing",
        "time_convention": "forward_in_supplied_coordinate",
        "time_step_count": len(time_grid) - 1,
        "output_time_count": len(time_grid),
        "local_time_steps": local_steps,
        "uniform_time_grid": uniform,
        "internal_step_count": len(internal_steps),
        "internal_time_steps": tuple(step.dt for step in internal_steps),
        "theta_schedule": tuple(step.theta for step in internal_steps),
        "startup_theta": startup_theta,
        "startup_steps": startup_steps,
        "startup_substeps": startup_substeps,
    }


def _matrix_cache_key(matrix) -> str:
    """Return a deterministic cache key for a sparse matrix structure and data."""

    sparse = sps.csr_matrix(matrix)
    digest = sha256()
    digest.update(str(sparse.shape).encode("ascii"))
    digest.update(np.asarray(sparse.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(sparse.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(sparse.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _sequence_cache_key(keys: Iterable[str]) -> str:
    """Return a deterministic digest for a sequence of cache keys."""

    digest = sha256()
    for key in keys:
        digest.update(str(key).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()


def _space_matrices(space: SpaceDiscretization, step: _InternalThetaStep):
    """Return spatial matrices through the explicit endpoint-aware protocol."""

    return space.matrices(step.theta, step.dt, start=step.start, end=step.end)


def _theta_cache_key(
    space: SpaceDiscretization,
    matrix,
    dt: float,
    theta: float,
    boundary_condition: BoundaryCondition | None,
) -> str:
    """Return the invalidation key for an invariant theta-step system."""

    boundary = type(boundary_condition).__name__ if boundary_condition else "none"
    mesh_nelements = getattr(getattr(space, "mesh", None), "nelements", "unknown")
    config = getattr(space, "config", None)
    element = getattr(config, "elem", None)
    payload = {
        "matrix": _matrix_cache_key(matrix),
        "dt": f"{dt:.17g}",
        "theta": f"{theta:.17g}",
        "dofs": str(space.Vh.N),
        "mesh_nelements": str(mesh_nelements),
        "element": type(element).__name__,
        "boundary_condition": boundary,
    }
    digest = sha256()
    for key, value in sorted(payload.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"=")
        digest.update(str(value).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()


class CrankNicolson(ThetaScheme):
    """Crank–Nicolson scheme with θ=1/2."""

    def __init__(self):
        """Initialize with ``θ = 1/2``."""
        super().__init__(theta=0.5)
