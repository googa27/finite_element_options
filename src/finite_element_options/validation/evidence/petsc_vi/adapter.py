"""Single-rank PETSc SNES-VI adapter over scikit-fem/SciPy LCP assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from finite_element_options.time_integration.lcp import (
    DiscreteLCP,
    LCPConvergenceError,
    LCPResult,
    evaluate_lcp_solution,
    validate_discrete_lcp,
)


EXTERNAL_PROFILE_HINT = (
    "PETSc is an explicit external profile; follow docs/PETSC_VI_ASSESSMENT.md "
    "to build matched PETSc/petsc4py versions"
)


@dataclass(frozen=True, slots=True)
class PetscVISolverSettings:
    """Explicit single-rank SNES VI and nested KSP controls."""

    tolerance: float = 1.0e-10
    max_iterations: int = 100
    snes_type: str = "vinewtonrsls"
    ksp_type: str = "preonly"
    pc_type: str = "lu"

    def __post_init__(self) -> None:
        """Reject unsupported or weakened external-solver controls."""

        if not isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("PETSc VI tolerance must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("PETSc VI max_iterations must be positive")
        if self.snes_type not in {"vinewtonrsls", "vinewtonssls"}:
            raise ValueError("PETSc VI snes_type must be vinewtonrsls or vinewtonssls")
        if self.ksp_type != "preonly" or self.pc_type != "lu":
            raise ValueError("the validated single-rank PETSc profile requires preonly+lu")


class PetscVISolver:
    """Solve canonical lower-obstacle LCPs using PETSc SNES VI."""

    def __init__(self, settings: PetscVISolverSettings | None = None):
        """Store explicit solver settings without importing PETSc eagerly."""

        self.settings = settings or PetscVISolverSettings()
        self.last_backend_evidence: dict[str, object] = {}

    def solve(
        self,
        problem: DiscreteLCP,
        *,
        initial: np.ndarray | None = None,
        fail_on_nonconvergence: bool = True,
    ) -> LCPResult:
        """Solve one LCP on ``COMM_SELF`` and return canonical diagnostics."""

        try:
            from petsc4py import PETSc
        except ImportError as exc:  # pragma: no cover - external profile test
            raise ModuleNotFoundError(EXTERNAL_PROFILE_HINT) from exc

        matrix, rhs, obstacle = validate_discrete_lcp(problem)
        initial_values = obstacle.copy() if initial is None else np.asarray(initial, dtype=float)
        if initial_values.shape != rhs.shape or not np.all(np.isfinite(initial_values)):
            raise ValueError("initial LCP iterate must be finite and match rhs shape")
        initial_values = np.maximum(initial_values.copy(), obstacle)
        started = perf_counter()
        petsc_matrix = PETSc.Mat().createAIJ(
            size=matrix.shape,
            csr=(matrix.indptr, matrix.indices, matrix.data),
            comm=PETSc.COMM_SELF,
        )
        petsc_matrix.assemble()
        rhs_vector = PETSc.Vec().createWithArray(rhs.copy(), comm=PETSc.COMM_SELF)
        solution = PETSc.Vec().createWithArray(initial_values.copy(), comm=PETSc.COMM_SELF)
        residual = solution.duplicate()
        lower = PETSc.Vec().createWithArray(obstacle.copy(), comm=PETSc.COMM_SELF)
        upper = solution.duplicate()
        upper.set(PETSc.INFINITY)

        def form_function(_snes: Any, values: Any, output: Any) -> None:
            petsc_matrix.mult(values, output)
            output.axpy(-1.0, rhs_vector)

        def form_jacobian(_snes: Any, _values: Any, jacobian: Any, preconditioner: Any) -> Any:
            jacobian.assemble()
            if preconditioner.handle != jacobian.handle:
                preconditioner.assemble()
            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

        snes = PETSc.SNES().create(PETSc.COMM_SELF)
        snes.setType(self.settings.snes_type)
        snes.setFunction(form_function, residual)
        snes.setJacobian(form_jacobian, petsc_matrix, petsc_matrix)
        snes.setVariableBounds(lower, upper)
        snes.setTolerances(
            rtol=self.settings.tolerance,
            atol=self.settings.tolerance,
            stol=self.settings.tolerance,
            max_it=self.settings.max_iterations,
        )
        ksp = snes.getKSP()
        ksp.setType(self.settings.ksp_type)
        ksp.getPC().setType(self.settings.pc_type)
        try:
            snes.solve(None, solution)
            values = np.asarray(solution.getArray(readonly=True), dtype=float).copy()
            reason_code = int(snes.getConvergedReason())
            reason = _reason_name(PETSc.SNES.ConvergedReason, reason_code, "SNES")
            ksp_reason_code = int(ksp.getConvergedReason())
            diagnostics = evaluate_lcp_solution(
                problem,
                values,
                tolerance=self.settings.tolerance,
                success=reason_code > 0,
                iterations=int(snes.getIterationNumber()),
                max_update=float(np.max(np.abs(values - initial_values))),
                message=(
                    "PETSc SNES VI converged" if reason_code > 0 else "PETSc SNES VI diverged"
                ),
                solve_time_sec=perf_counter() - started,
                solver=f"petsc_snes_{self.settings.snes_type}",
                backend_reason=reason,
                linear_iterations=int(snes.getLinearSolveIterations()),
            )
            residual_success = (
                max(
                    diagnostics.primal_violation_max,
                    diagnostics.dual_violation_max,
                    diagnostics.complementarity_max,
                    diagnostics.projected_residual_max,
                )
                <= self.settings.tolerance
            )
            canonical_success = reason_code > 0 and residual_success
            diagnostics = replace(
                diagnostics,
                success=canonical_success,
                message=(
                    "PETSc SNES VI converged with canonical residuals"
                    if canonical_success
                    else "PETSc SNES VI failed convergence or canonical residual gate"
                ),
            )
            self.last_backend_evidence = {
                "snes_reason": reason,
                "snes_reason_code": reason_code,
                "snes_iterations": diagnostics.iterations,
                "function_norm": float(snes.getFunctionNorm()),
                "linear_iterations": diagnostics.linear_iterations,
                "ksp_type": ksp.getType(),
                "pc_type": ksp.getPC().getType(),
                "ksp_reason": _reason_name(
                    PETSc.KSP.ConvergedReason,
                    ksp_reason_code,
                    "KSP",
                ),
                "ksp_reason_code": ksp_reason_code,
                "matrix_nnz": int(matrix.nnz),
                "matrix_csr_input_bytes": int(
                    matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                ),
                "matrix_memory_bytes": float(petsc_matrix.getInfo()["memory"]),
                "comm_size": int(PETSc.COMM_SELF.getSize()),
            }
            result = LCPResult(values=values, success=diagnostics.success, diagnostics=diagnostics)
            if not result.success and fail_on_nonconvergence:
                raise LCPConvergenceError(diagnostics)
            return result
        finally:
            for obj in (snes, upper, lower, residual, solution, rhs_vector, petsc_matrix):
                obj.destroy()


def petsc_runtime_doctor() -> dict[str, object]:
    """Execute real KSP, SNES-VI, and TS functionality on ``COMM_SELF``."""

    try:
        import petsc4py
        from petsc4py import PETSc
    except ImportError as exc:  # pragma: no cover - external profile test
        raise ModuleNotFoundError(EXTERNAL_PROFILE_HINT) from exc

    ksp = _ksp_doctor(PETSc)
    vi_problem = DiscreteLCP(
        matrix=np.array([[2.0, -1.0], [-1.0, 2.0]]),
        rhs=np.array([0.0, 0.0]),
        obstacle=np.array([1.0, 0.0]),
    )
    vi_solver = PetscVISolver()
    vi_result = vi_solver.solve(vi_problem)
    ts = _ts_doctor(PETSc)
    return {
        "petsc4py_version": petsc4py.__version__,
        "petsc_version": ".".join(str(item) for item in PETSc.Sys.getVersion()),
        "comm_size": int(PETSc.COMM_WORLD.getSize()),
        "scalar_type": np.dtype(PETSc.ScalarType).name,
        "ksp": ksp,
        "snes_vi": {
            **vi_solver.last_backend_evidence,
            "converged": vi_result.success,
            "projected_residual_max": vi_result.diagnostics.projected_residual_max,
        },
        "ts": ts,
        "passed": bool(ksp["converged"] and vi_result.success and ts["converged"]),
    }


def _ksp_doctor(PETSc: Any) -> dict[str, object]:
    matrix_values = np.array([[4.0, 1.0], [1.0, 3.0]])
    matrix = PETSc.Mat().createDense((2, 2), array=matrix_values, comm=PETSc.COMM_SELF)
    rhs = PETSc.Vec().createWithArray(np.array([1.0, 2.0]), comm=PETSc.COMM_SELF)
    solution = rhs.duplicate()
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setOperators(matrix)
    try:
        ksp.solve(rhs, solution)
        values = np.asarray(solution.getArray(readonly=True), dtype=float).copy()
        residual = matrix_values @ values - np.array([1.0, 2.0])
        reason_code = int(ksp.getConvergedReason())
        return {
            "converged": reason_code > 0 and float(np.max(np.abs(residual))) <= 1.0e-12,
            "reason": _reason_name(PETSc.KSP.ConvergedReason, reason_code, "KSP"),
            "iterations": int(ksp.getIterationNumber()),
            "residual_linf": float(np.max(np.abs(residual))),
        }
    finally:
        for obj in (ksp, solution, rhs, matrix):
            obj.destroy()


def _ts_doctor(PETSc: Any) -> dict[str, object]:
    solution = PETSc.Vec().createWithArray(np.array([1.0]), comm=PETSc.COMM_SELF)
    residual = solution.duplicate()
    jacobian = PETSc.Mat().createAIJ((1, 1), nnz=1, comm=PETSc.COMM_SELF)
    jacobian.setUp()

    def implicit_function(
        _ts: Any, _time: float, values: Any, derivative: Any, output: Any
    ) -> None:
        output.array_w[:] = derivative.array_r + values.array_r

    def implicit_jacobian(
        _ts: Any,
        _time: float,
        _values: Any,
        _derivative: Any,
        shift: float,
        operator: Any,
        preconditioner: Any,
    ) -> Any:
        operator.zeroEntries()
        operator.setValue(0, 0, shift + 1.0)
        operator.assemble()
        if preconditioner.handle != operator.handle:
            preconditioner.assemble()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    ts = PETSc.TS().create(PETSc.COMM_SELF)
    ts.setType("beuler")
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
    ts.setIFunction(implicit_function, residual)
    ts.setIJacobian(implicit_jacobian, jacobian, jacobian)
    ts.setTime(0.0)
    ts.setTimeStep(0.01)
    ts.setMaxTime(0.1)
    ts.setMaxSteps(20)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
    try:
        ts.solve(solution)
        observed = float(solution.getArray(readonly=True)[0])
        expected = float(np.exp(-0.1))
        reason_code = int(ts.getConvergedReason())
        return {
            "converged": reason_code > 0,
            "reason": _reason_name(PETSc.TS.ConvergedReason, reason_code, "TS"),
            "steps": int(ts.getStepNumber()),
            "final_time": float(ts.getTime()),
            "absolute_error": abs(observed - expected),
        }
    finally:
        for obj in (ts, jacobian, residual, solution):
            obj.destroy()


def _reason_name(enum_type: Any, reason_code: int, prefix: str) -> str:
    for name in dir(enum_type):
        if not name.isupper() or name in {"ITERATING", "CONVERGED_ITERATING"}:
            continue
        try:
            if int(getattr(enum_type, name)) == reason_code:
                return f"{prefix}_{name}"
        except (TypeError, ValueError):
            continue
    return f"{prefix}_UNKNOWN_{reason_code}"


__all__ = [
    "EXTERNAL_PROFILE_HINT",
    "PetscVISolver",
    "PetscVISolverSettings",
    "petsc_runtime_doctor",
]
