"""Real, non-skipped PETSc VI tests for the explicit external profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

pytestmark = pytest.mark.petsc_external
pytest.importorskip("petsc4py", reason="follow docs/PETSC_VI_ASSESSMENT.md external profile")

from finite_element_options.time_integration.lcp import (  # noqa: E402
    DiscreteLCP,
    LCPConvergenceError,
    ProjectedSORSolver,
    ProjectedSORSolverSettings,
)
from finite_element_options.validation.evidence.petsc_vi import (  # noqa: E402
    PetscVIAssessmentConfig,
    PetscVISolver,
    PetscVISolverSettings,
    petsc_runtime_doctor,
    run_petsc_vi_assessment,
)


def test_petsc_snes_vi_matches_projected_sor_on_coupled_lcp() -> None:
    """SNES VI must solve the same lower-obstacle complementarity convention."""

    problem = DiscreteLCP(
        matrix=sps.csr_matrix([[2.0, -1.0], [-1.0, 2.0]]),
        rhs=np.array([0.0, 0.0]),
        obstacle=np.array([1.0, 0.0]),
    )
    reference = ProjectedSORSolver(
        ProjectedSORSolverSettings(tolerance=1.0e-10, max_iterations=500, relaxation=1.0)
    ).solve(problem)
    result = PetscVISolver(PetscVISolverSettings(tolerance=1.0e-10)).solve(problem)

    assert result.success is True
    assert result.diagnostics.solver == "petsc_snes_vinewtonrsls"
    assert result.diagnostics.backend_reason.startswith("SNES_CONVERGED")
    assert result.diagnostics.linear_iterations >= 1
    assert result.values == pytest.approx(reference.values, abs=1.0e-10)
    assert result.diagnostics.projected_residual_max <= 1.0e-10


def test_petsc_vi_typed_nonconvergence_and_failure_path() -> None:
    """A singular incompatible LCP must return diagnostics or raise the typed error."""

    problem = DiscreteLCP(
        matrix=sps.csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        rhs=np.array([1.0, 1.0]),
        obstacle=np.array([0.0, 0.0]),
    )
    solver = PetscVISolver(PetscVISolverSettings(tolerance=1.0e-14, max_iterations=2))
    result = solver.solve(problem, fail_on_nonconvergence=False)
    assert result.success is False
    assert result.diagnostics.backend_reason.startswith("SNES_DIVERGED")
    with pytest.raises(LCPConvergenceError):
        solver.solve(problem)


def test_petsc_runtime_doctor_executes_ksp_snes_vi_and_ts() -> None:
    """Installed external evidence must exercise KSP, SNES VI, and TS for real."""

    report = petsc_runtime_doctor()
    assert report["passed"] is True
    assert report["comm_size"] == 1
    assert report["ksp"]["converged"] is True
    assert report["snes_vi"]["converged"] is True
    assert report["ts"]["converged"] is True
    assert report["ts"]["absolute_error"] <= 1.0e-3


def test_small_american_put_assessment_runs_both_real_backends() -> None:
    """The external profile must execute full American time stepping without skips."""

    report = run_petsc_vi_assessment(
        config=PetscVIAssessmentConfig(
            refinement_level=5,
            time_steps=24,
            repeats=3,
            grid_abs_tolerance=2.0e-6,
            price_abs_tolerance=2.0e-6,
            delta_abs_tolerance=1.0e-4,
            gamma_abs_tolerance=2.0e-3,
        ),
        root=Path(__file__).resolve().parents[2],
    )
    assert report["trigger"]["triggered"] is True
    assert report["projected_sor"]["solve_count"] == 24
    assert report["petsc_snes_vi"]["solve_count"] == 24
    assert report["decision"]["checks"]["equal_discretization_parity"] is True
    assert report["decision"]["checks"]["typed_failure"] is True
