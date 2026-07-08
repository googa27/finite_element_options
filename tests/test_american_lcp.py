"""American-option LCP regressions for issue #41."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import scipy.sparse as sps

from finite_element_options.time_integration.lcp import (
    DiscreteLCP,
    LCPConvergenceError,
    ProjectedSORSolver,
    ProjectedSORSolverSettings,
)
from finite_element_options.time_integration.stepper import ThetaScheme


@dataclass
class _FakeBasis:
    N: int = 2


@dataclass
class _AmericanTwoNodeSpace:
    """Minimal two-node system where post-step clipping violates dual feasibility."""

    Vh: _FakeBasis = field(default_factory=_FakeBasis)
    matrix_calls: list[tuple[float, float]] = field(default_factory=list)
    boundary_term_calls: list[float] = field(default_factory=list)

    def initial_condition(self) -> np.ndarray:
        return np.array([1.0, 0.0])

    def matrices(
        self,
        theta: float,
        dt: float,
        *,
        start: float | None = None,
        end: float | None = None,
    ):
        self.matrix_calls.append((theta, dt))
        del start, end
        return (
            sps.csr_matrix([[2.0, -1.0], [-1.0, 2.0]]),
            sps.csr_matrix([[0.0, 0.0], [0.0, 0.0]]),
        )

    def boundary_term(self, th: float) -> np.ndarray:
        self.boundary_term_calls.append(th)
        return np.zeros(2)

    def domain_diagnostics(self, *, horizon: float, tail_mass: float = 1.0e-6) -> dict:
        return {"horizon": horizon, "tail_mass": tail_mass}

    def dirichlet(self, th: float) -> np.ndarray:
        return np.array([th, th])

    def apply_dirichlet(self, A, b, boundaries, u_dirichlet):
        del boundaries, u_dirichlet
        return A, b


def test_projected_sor_solves_coupled_lcp_instead_of_post_step_clipping() -> None:
    matrix = sps.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.array([0.0, 0.0])
    obstacle = np.array([1.0, 0.0])
    clipped_unconstrained = np.maximum(np.linalg.solve(matrix.toarray(), rhs), obstacle)

    result = ProjectedSORSolver(
        ProjectedSORSolverSettings(tolerance=1.0e-10, max_iterations=500)
    ).solve(DiscreteLCP(matrix=matrix, rhs=rhs, obstacle=obstacle))

    assert result.success is True
    np.testing.assert_allclose(clipped_unconstrained, np.array([1.0, 0.0]))
    np.testing.assert_allclose(result.values, np.array([1.0, 0.5]), atol=1.0e-8)
    assert result.diagnostics.primal_violation_max <= 1.0e-10
    assert result.diagnostics.dual_violation_max <= 1.0e-10
    assert result.diagnostics.complementarity_max <= 1.0e-10
    assert result.diagnostics.exercise_count == 1
    assert result.diagnostics.exercise_set == (True, False)


def test_projected_sor_does_not_ignore_scaled_projected_residual() -> None:
    problem = DiscreteLCP(
        matrix=sps.csr_matrix([[1000.0]]),
        rhs=np.array([0.01]),
        obstacle=np.array([0.0]),
    )
    solver = ProjectedSORSolver(
        ProjectedSORSolverSettings(
            tolerance=1.0e-8,
            max_iterations=13,
            relaxation=1.5,
        )
    )

    result = solver.solve(problem, fail_on_nonconvergence=False)

    assert result.success is False
    assert result.diagnostics.projected_residual_max > result.diagnostics.tolerance
    assert result.diagnostics.dual_violation_max == pytest.approx(0.0)
    assert result.diagnostics.complementarity_max <= result.diagnostics.tolerance


def test_projected_sor_returns_failed_result_and_can_raise_on_nonconvergence() -> None:
    problem = DiscreteLCP(
        matrix=sps.csr_matrix([[2.0, -1.0], [-1.0, 2.0]]),
        rhs=np.array([0.0, 0.0]),
        obstacle=np.array([1.0, 0.0]),
    )
    solver = ProjectedSORSolver(ProjectedSORSolverSettings(tolerance=1.0e-14, max_iterations=1))

    result = solver.solve(problem, fail_on_nonconvergence=False)

    assert result.success is False
    assert result.diagnostics.iterations == 1
    assert result.diagnostics.dual_violation_max > 0.0
    assert result.diagnostics.message.startswith("projected SOR did not converge")
    with pytest.raises(LCPConvergenceError) as excinfo:
        solver.solve(problem)
    assert excinfo.value.diagnostics.success is False
    assert excinfo.value.diagnostics.iterations == result.diagnostics.iterations
    assert excinfo.value.diagnostics.exercise_set == result.diagnostics.exercise_set


def test_theta_scheme_american_path_uses_lcp_solver_and_records_residuals() -> None:
    space = _AmericanTwoNodeSpace()
    stepper = ThetaScheme(
        theta=1.0,
        lcp_solver_settings=ProjectedSORSolverSettings(
            tolerance=1.0e-10,
            max_iterations=500,
        ),
    )

    values = stepper.solve([0.0, 1.0], space, is_american=True)

    np.testing.assert_allclose(values[-1], np.array([1.0, 0.5]), atol=1.0e-8)
    assert stepper.last_lcp_diagnostics[-1].success is True
    assert stepper.last_lcp_diagnostics[-1].primal_violation_max <= 1.0e-10
    assert stepper.last_lcp_diagnostics[-1].dual_violation_max <= 1.0e-10
    assert stepper.last_lcp_diagnostics[-1].complementarity_max <= 1.0e-10
    assert stepper.last_lcp_diagnostics[-1].exercise_count == 1
    assert stepper.last_solve_diagnostics.solve_count == 1


def test_american_stepper_fails_explicitly_when_lcp_does_not_converge() -> None:
    stepper = ThetaScheme(
        theta=1.0,
        lcp_solver_settings=ProjectedSORSolverSettings(
            tolerance=1.0e-14,
            max_iterations=1,
        ),
    )

    with pytest.raises(LCPConvergenceError, match="projected SOR did not converge"):
        stepper.solve([0.0, 1.0], _AmericanTwoNodeSpace(), is_american=True)
    assert stepper.last_lcp_diagnostics[-1].success is False


def test_rannacher_startup_schedule_is_reported_for_american_lcp_solves() -> None:
    stepper = ThetaScheme(
        theta=0.5,
        startup_theta=1.0,
        startup_steps=1,
        startup_substeps=2,
        lcp_solver_settings=ProjectedSORSolverSettings(tolerance=1.0e-10),
    )

    stepper.solve([0.0, 0.2, 0.5], _AmericanTwoNodeSpace(), is_american=True)

    assert stepper.last_time_grid_diagnostics["theta_schedule"] == pytest.approx((1.0, 1.0, 0.5))
    assert len(stepper.last_lcp_diagnostics) == 3
