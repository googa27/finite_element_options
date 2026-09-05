"""Discrete lower-obstacle LCP solvers for American exercise."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sps


@dataclass(frozen=True)
class DiscreteLCP:
    """Discrete lower-obstacle linear complementarity problem.

    The convention is

    ``values >= obstacle``, ``matrix @ values - rhs >= 0`` and
    ``(values - obstacle) * (matrix @ values - rhs) = 0``.
    """

    matrix: Any
    rhs: np.ndarray
    obstacle: np.ndarray


@dataclass(frozen=True)
class ProjectedSORSolverSettings:
    """Configuration for the reference projected-SOR LCP solver."""

    tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    relaxation: float = 0.5

    def __post_init__(self) -> None:
        """Validate the projected-SOR numerical controls."""

        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("LCP tolerance must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("LCP max_iterations must be at least one")
        if not np.isfinite(self.relaxation) or not (0.0 < self.relaxation < 2.0):
            raise ValueError("projected-SOR relaxation must lie in (0, 2)")


@dataclass(frozen=True)
class LCPDiagnostics:
    """Auditable residuals and exercise-set metadata for one LCP solve."""

    success: bool
    iterations: int
    tolerance: float
    relaxation: float
    primal_violation_max: float
    dual_violation_max: float
    complementarity_max: float
    projected_residual_max: float
    max_update: float
    exercise_count: int
    exercise_set: tuple[bool, ...]
    message: str
    solve_time_sec: float
    solver: str = "projected_sor"
    backend_reason: str = ""
    linear_iterations: int = 0

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-safe diagnostic payload."""

        return {
            "success": self.success,
            "iterations": self.iterations,
            "tolerance": self.tolerance,
            "relaxation": self.relaxation,
            "primal_violation_max": self.primal_violation_max,
            "dual_violation_max": self.dual_violation_max,
            "complementarity_max": self.complementarity_max,
            "projected_residual_max": self.projected_residual_max,
            "max_update": self.max_update,
            "exercise_count": self.exercise_count,
            "exercise_set": self.exercise_set,
            "message": self.message,
            "solve_time_sec": self.solve_time_sec,
            "solver": self.solver,
            "backend_reason": self.backend_reason,
            "linear_iterations": self.linear_iterations,
        }


@dataclass(frozen=True)
class LCPResult:
    """Result returned by a lower-obstacle LCP solve."""

    values: np.ndarray
    success: bool
    diagnostics: LCPDiagnostics


class LCPSolver(Protocol):
    """Structural contract for interchangeable lower-obstacle solvers."""

    def solve(
        self,
        problem: DiscreteLCP,
        *,
        initial: np.ndarray | None = None,
        fail_on_nonconvergence: bool = True,
    ) -> LCPResult:
        """Solve one discrete LCP under the canonical convention."""

        ...


class LCPConvergenceError(RuntimeError):
    """Raised when an American-exercise LCP fails to converge."""

    def __init__(self, diagnostics: LCPDiagnostics):
        """Attach failed LCP diagnostics to the exception."""

        super().__init__(diagnostics.message)
        self.diagnostics = diagnostics


class ProjectedSORSolver:
    """Reference projected-SOR solver for sparse lower-obstacle LCPs."""

    def __init__(self, settings: ProjectedSORSolverSettings | None = None):
        """Store projected-SOR settings, using conservative defaults if absent."""

        self.settings = settings or ProjectedSORSolverSettings()

    def solve(
        self,
        problem: DiscreteLCP,
        *,
        initial: np.ndarray | None = None,
        fail_on_nonconvergence: bool = True,
    ) -> LCPResult:
        """Solve ``problem`` and optionally raise on nonconvergence."""

        matrix, rhs, obstacle = validate_discrete_lcp(problem)
        started = perf_counter()
        if initial is None:
            values = obstacle.copy()
        else:
            values = np.maximum(np.asarray(initial, dtype=float).copy(), obstacle)
            if values.shape != rhs.shape:
                raise ValueError("initial LCP iterate must match rhs shape")
        previous = values.copy()
        diagnostics = _diagnostics(
            matrix,
            rhs,
            obstacle,
            values,
            success=False,
            iterations=0,
            settings=self.settings,
            max_update=np.inf,
            message="projected SOR not yet run",
            solve_time_sec=0.0,
        )
        for iteration in range(1, self.settings.max_iterations + 1):
            previous[:] = values
            _projected_sor_sweep(matrix, rhs, obstacle, values, self.settings.relaxation)
            max_update = float(np.max(np.abs(values - previous))) if values.size else 0.0
            diagnostics = _diagnostics(
                matrix,
                rhs,
                obstacle,
                values,
                success=False,
                iterations=iteration,
                settings=self.settings,
                max_update=max_update,
                message="projected SOR did not converge",
                solve_time_sec=perf_counter() - started,
            )
            if _has_converged(diagnostics, self.settings.tolerance):
                diagnostics = _diagnostics(
                    matrix,
                    rhs,
                    obstacle,
                    values,
                    success=True,
                    iterations=iteration,
                    settings=self.settings,
                    max_update=max_update,
                    message="projected SOR converged",
                    solve_time_sec=perf_counter() - started,
                )
                return LCPResult(values=values.copy(), success=True, diagnostics=diagnostics)

        diagnostics = _diagnostics(
            matrix,
            rhs,
            obstacle,
            values,
            success=False,
            iterations=self.settings.max_iterations,
            settings=self.settings,
            max_update=diagnostics.max_update,
            message=(
                f"projected SOR did not converge within {self.settings.max_iterations} iterations"
            ),
            solve_time_sec=perf_counter() - started,
        )
        result = LCPResult(values=values.copy(), success=False, diagnostics=diagnostics)
        if fail_on_nonconvergence:
            raise LCPConvergenceError(diagnostics)
        return result


def _validate_problem(problem: DiscreteLCP) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    matrix = sps.csr_matrix(problem.matrix, dtype=float)
    rhs = np.asarray(problem.rhs, dtype=float)
    obstacle = np.asarray(problem.obstacle, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("LCP matrix must be square")
    if rhs.ndim != 1 or obstacle.ndim != 1:
        raise ValueError("LCP rhs and obstacle must be one-dimensional")
    if matrix.shape[0] != rhs.shape[0] or rhs.shape != obstacle.shape:
        raise ValueError("LCP matrix, rhs, and obstacle dimensions must match")
    if matrix.shape[0] == 0:
        raise ValueError("LCP must contain at least one degree of freedom")
    if not (np.all(np.isfinite(rhs)) and np.all(np.isfinite(obstacle))):
        raise ValueError("LCP rhs and obstacle must be finite")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("LCP matrix entries must be finite")
    diagonal = matrix.diagonal()
    if np.any(diagonal <= 0.0):
        raise ValueError("projected SOR requires a strictly positive matrix diagonal")
    return matrix, rhs, obstacle


def _projected_sor_sweep(
    matrix: sps.csr_matrix,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    values: np.ndarray,
    relaxation: float,
) -> None:
    diagonal = matrix.diagonal()
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        cols = matrix.indices[start:end]
        data = matrix.data[start:end]
        coupled = float(data @ values[cols] - diagonal[row] * values[row])
        gauss_seidel = (rhs[row] - coupled) / diagonal[row]
        relaxed = values[row] + relaxation * (gauss_seidel - values[row])
        values[row] = max(obstacle[row], relaxed)


def validate_discrete_lcp(
    problem: DiscreteLCP,
) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """Validate and canonicalize one lower-obstacle LCP for backend adapters."""

    return _validate_problem(problem)


def evaluate_lcp_solution(
    problem: DiscreteLCP,
    values: np.ndarray,
    *,
    tolerance: float,
    success: bool,
    iterations: int,
    max_update: float,
    message: str,
    solve_time_sec: float,
    solver: str,
    relaxation: float = 0.0,
    backend_reason: str = "",
    linear_iterations: int = 0,
) -> LCPDiagnostics:
    """Evaluate one candidate against the canonical lower-obstacle convention."""

    matrix, rhs, obstacle = _validate_problem(problem)
    candidate = np.asarray(values, dtype=float)
    if candidate.shape != rhs.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("candidate LCP values must be finite and match rhs shape")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("LCP diagnostic tolerance must be finite and positive")
    residual = np.asarray(matrix @ candidate - rhs, dtype=float)
    slack = candidate - obstacle
    primal_violation = np.maximum(obstacle - candidate, 0.0)
    dual_violation = np.maximum(-residual, 0.0)
    complementarity = np.abs(slack * residual)
    projected_residual = np.minimum(slack, residual)
    exercise_set = tuple(bool(item) for item in candidate <= obstacle + tolerance)
    return LCPDiagnostics(
        success=bool(success),
        iterations=int(iterations),
        tolerance=float(tolerance),
        relaxation=float(relaxation),
        primal_violation_max=_max_or_zero(primal_violation),
        dual_violation_max=_max_or_zero(dual_violation),
        complementarity_max=_max_or_zero(complementarity),
        projected_residual_max=_max_or_zero(np.abs(projected_residual)),
        max_update=float(max_update),
        exercise_count=sum(exercise_set),
        exercise_set=exercise_set,
        message=message,
        solve_time_sec=float(solve_time_sec),
        solver=solver,
        backend_reason=backend_reason,
        linear_iterations=int(linear_iterations),
    )


def _diagnostics(
    matrix: sps.csr_matrix,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    values: np.ndarray,
    *,
    success: bool,
    iterations: int,
    settings: ProjectedSORSolverSettings,
    max_update: float,
    message: str,
    solve_time_sec: float,
) -> LCPDiagnostics:
    return evaluate_lcp_solution(
        DiscreteLCP(matrix=matrix, rhs=rhs, obstacle=obstacle),
        values,
        tolerance=settings.tolerance,
        success=success,
        iterations=iterations,
        max_update=max_update,
        message=message,
        solve_time_sec=solve_time_sec,
        solver="projected_sor",
        relaxation=settings.relaxation,
    )


def _has_converged(diagnostics: LCPDiagnostics, tolerance: float) -> bool:
    return (
        diagnostics.primal_violation_max <= tolerance
        and diagnostics.dual_violation_max <= tolerance
        and diagnostics.complementarity_max <= tolerance
        and diagnostics.projected_residual_max <= tolerance
        and diagnostics.max_update <= max(1.0, diagnostics.projected_residual_max) * tolerance
    )


def _max_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


__all__ = [
    "DiscreteLCP",
    "LCPConvergenceError",
    "LCPDiagnostics",
    "LCPResult",
    "LCPSolver",
    "ProjectedSORSolver",
    "ProjectedSORSolverSettings",
    "evaluate_lcp_solution",
    "validate_discrete_lcp",
]
