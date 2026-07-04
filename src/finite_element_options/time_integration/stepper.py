"""Time-stepping algorithms for the option PDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Iterable

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
import skfem as fem  # type: ignore[import-untyped]

from finite_element_options.core.interfaces import BoundaryCondition, SpaceDiscretization


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
    ):
        """Store θ and sparse-direct solver-cache policy."""

        if linear_solver != "scipy_direct":
            msg = (
                "Only linear_solver='scipy_direct' is currently supported by the "
                "released FEM route; AMG/PETSc/banded policies fail closed in the "
                "capability manifest."
            )
            raise ValueError(msg)
        self.theta = theta
        self.linear_solver = linear_solver
        self.reuse_factorization = reuse_factorization
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

    def solve(
        self,
        t: Iterable[float],
        space: SpaceDiscretization,
        boundary_condition: BoundaryCondition | None = None,
        is_american: bool = False,
    ) -> np.ndarray:
        """Return solution grid for the supplied time nodes ``t``."""
        t = tuple(float(item) for item in t)
        dt = t[1] - t[0]
        if hasattr(space, "domain_diagnostics"):
            self.last_domain_diagnostics = space.domain_diagnostics(
                horizon=t[-1] - t[0]
            )
        else:
            self.last_domain_diagnostics = {}
        v_tsv = np.empty((len(t), space.Vh.N))
        v_tsv[0] = space.initial_condition()
        A, B = space.matrices(self.theta, dt)
        assembly_cache_key = _theta_cache_key(space, A, dt, self.theta, boundary_condition)
        factorization_cache_key = "not_factorized"
        factorized_solver = None
        factorized_matrix = None
        factorization_count = 0
        factorization_reuse_count = 0
        solve_count = 0
        max_residual = 0.0
        factorization_time = 0.0
        solve_time = 0.0

        for i, th_i in enumerate(t[:-1]):
            b_previous = B @ v_tsv[i]
            b_inhom = self.theta * space.boundary_term(th_i + dt) + (
                1 - self.theta
            ) * space.boundary_term(th_i)
            b = b_previous + dt * b_inhom

            if boundary_condition:
                A_enf, b_enf = boundary_condition.apply(space, A, b, th_i)
            else:
                A_enf, b_enf = A, b

            if self.reuse_factorization:
                if factorized_solver is None:
                    started = perf_counter()
                    factorized_matrix = sps.csc_matrix(A_enf)
                    lu = spla.splu(factorized_matrix)
                    factorized_solver = lu.solve
                    factorization_time += perf_counter() - started
                    factorization_count += 1
                    factorization_cache_key = _matrix_cache_key(factorized_matrix)
                else:
                    factorization_reuse_count += 1
                started = perf_counter()
                next_values = factorized_solver(b_enf)
                solve_time += perf_counter() - started
            else:
                started = perf_counter()
                next_values = fem.solve(A_enf, b_enf)
                solve_time += perf_counter() - started
                factorization_count += 1
                factorized_matrix = sps.csr_matrix(A_enf)
                factorization_cache_key = _matrix_cache_key(factorized_matrix)

            v_tsv[i + 1] = next_values
            solve_count += 1
            residual = np.asarray(A_enf @ next_values - b_enf, dtype=float)
            if residual.size:
                max_residual = max(max_residual, float(np.max(np.abs(residual))))

            if is_american:
                v_tsv[i + 1] = np.maximum(v_tsv[i + 1], v_tsv[0])

        self.last_solve_diagnostics = LinearSolveDiagnostics(
            linear_solver=self.linear_solver,
            factorization_reuse_enabled=self.reuse_factorization,
            factorization_count=factorization_count,
            factorization_reuse_count=factorization_reuse_count,
            solve_count=solve_count,
            max_linear_residual_abs=max_residual,
            assembly_cache_key=assembly_cache_key,
            factorization_cache_key=factorization_cache_key,
            stage_timings_sec={
                "factorization": factorization_time,
                "solve": solve_time,
            },
        )
        return v_tsv


def _matrix_cache_key(matrix) -> str:
    """Return a deterministic cache key for a sparse matrix structure and data."""

    sparse = sps.csr_matrix(matrix)
    digest = sha256()
    digest.update(str(sparse.shape).encode("ascii"))
    digest.update(np.asarray(sparse.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(sparse.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(sparse.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


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
