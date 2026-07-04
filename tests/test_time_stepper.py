"""Time-grid and theta-schedule regression tests for issue #40."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import scipy.sparse as sps

from finite_element_options.core.interfaces import BoundaryCondition
from finite_element_options.time_integration.stepper import ThetaScheme


@dataclass
class _FakeBasis:
    N: int = 1


@dataclass
class _ScalarSpace:
    """Minimal scalar semidiscretization used to expose time-stepping semantics."""

    matrix_mode: str = "growth"
    source_scale: float = 0.0
    Vh: _FakeBasis = field(default_factory=_FakeBasis)
    matrix_calls: list[tuple[float, float]] = field(default_factory=list)
    boundary_term_calls: list[float] = field(default_factory=list)

    def initial_condition(self) -> np.ndarray:
        """Return a deterministic scalar initial state."""

        return np.array([1.0])

    def matrices(self, theta: float, dt: float):
        """Record each local theta/dt pair and return scalar matrices."""

        self.matrix_calls.append((theta, dt))
        if self.matrix_mode == "growth":
            return sps.csr_matrix([[1.0]]), sps.csr_matrix([[1.0 + dt]])
        if self.matrix_mode == "dt_system":
            return sps.csr_matrix([[1.0 + dt]]), sps.csr_matrix([[1.0]])
        return sps.csr_matrix([[1.0]]), sps.csr_matrix([[1.0]])

    def boundary_term(self, th: float) -> np.ndarray:
        """Return a scalar source and record its evaluation time."""

        self.boundary_term_calls.append(th)
        return np.array([self.source_scale * th])

    def domain_diagnostics(self, *, horizon: float, tail_mass: float = 1.0e-6) -> dict:
        """Return a minimal diagnostics payload."""

        return {"horizon": horizon, "tail_mass": tail_mass}

    def dirichlet(self, th: float) -> np.ndarray:
        """Return a deterministic scalar Dirichlet payload."""

        return np.array([th])

    def apply_dirichlet(self, A, b, boundaries, u_dirichlet):
        """Return the input system unchanged for protocol compatibility."""

        return A, b


@dataclass
class _RecordingBoundary(BoundaryCondition):
    times: list[float] = field(default_factory=list)

    def apply(self, space, A, b, th: float):
        """Record the Dirichlet enforcement time and leave the system unchanged."""

        self.times.append(th)
        return A, b


def test_nonuniform_time_grid_uses_each_local_dt_once() -> None:
    space = _ScalarSpace(matrix_mode="growth")
    stepper = ThetaScheme(theta=0.5)

    solution = stepper.solve(iter([0.0, 0.25, 1.0, 1.5]), space)

    assert [dt for _, dt in space.matrix_calls] == pytest.approx([0.25, 0.75, 0.5])
    assert solution[:, 0] == pytest.approx([1.0, 1.25, 2.1875, 3.28125])
    assert stepper.last_time_grid_diagnostics["local_time_steps"] == pytest.approx(
        (0.25, 0.75, 0.5)
    )
    assert not stepper.last_time_grid_diagnostics["uniform_time_grid"]
    assert stepper.last_solve_diagnostics.solve_count == 3


@pytest.mark.parametrize(
    "grid, message",
    [
        ([0.0], "at least two"),
        ([0.0, 0.5, 0.5], "strictly increasing"),
        ([0.0, np.nan], "finite"),
    ],
)
def test_time_grid_validation_rejects_ambiguous_or_invalid_nodes(grid, message: str) -> None:
    stepper = ThetaScheme(theta=0.5)

    with pytest.raises(ValueError, match=message):
        stepper.solve(grid, _ScalarSpace())


def test_dirichlet_boundary_is_enforced_at_new_time_node() -> None:
    space = _ScalarSpace(matrix_mode="identity", source_scale=2.0)
    boundary = _RecordingBoundary()
    stepper = ThetaScheme(theta=0.25)

    stepper.solve([0.0, 0.4, 1.0], space, boundary_condition=boundary)

    assert boundary.times == pytest.approx([0.4, 1.0])
    assert space.boundary_term_calls == pytest.approx([0.4, 0.0, 1.0, 0.4])


def test_factorization_cache_keys_follow_distinct_step_systems() -> None:
    space = _ScalarSpace(matrix_mode="dt_system")
    stepper = ThetaScheme(theta=1.0, reuse_factorization=True)

    stepper.solve([0.0, 0.25, 0.75, 1.0], space)

    diagnostics = stepper.last_solve_diagnostics
    assert [dt for _, dt in space.matrix_calls] == pytest.approx([0.25, 0.5, 0.25])
    assert diagnostics.factorization_count == 2
    assert diagnostics.factorization_reuse_count == 1
    assert diagnostics.solve_count == 3


def test_roundoff_uniform_linspace_reuses_one_invariant_system() -> None:
    space = _ScalarSpace(matrix_mode="dt_system")
    stepper = ThetaScheme(theta=1.0, reuse_factorization=True)

    stepper.solve(np.linspace(0.0, 1.0, 31), space)

    diagnostics = stepper.last_solve_diagnostics
    assert len({dt for _, dt in space.matrix_calls}) == 1
    assert diagnostics.factorization_count == 1
    assert diagnostics.factorization_reuse_count == 29
    assert stepper.last_time_grid_diagnostics["uniform_time_grid"]


def test_rannacher_startup_subdivides_initial_intervals_with_backward_euler() -> None:
    space = _ScalarSpace(matrix_mode="growth")
    stepper = ThetaScheme(theta=0.5, startup_theta=1.0, startup_steps=2, startup_substeps=2)

    solution = stepper.solve([0.0, 0.2, 0.4, 0.8], space)

    assert [theta for theta, _ in space.matrix_calls] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 0.5]
    )
    assert [dt for _, dt in space.matrix_calls] == pytest.approx(
        [0.1, 0.1, 0.1, 0.1, 0.4]
    )
    assert solution.shape == (4, 1)
    assert stepper.last_solve_diagnostics.solve_count == 5
    assert stepper.last_time_grid_diagnostics["theta_schedule"] == pytest.approx(
        (1.0, 1.0, 1.0, 1.0, 0.5)
    )
    assert stepper.last_time_grid_diagnostics["internal_time_steps"] == pytest.approx(
        (0.1, 0.1, 0.1, 0.1, 0.4)
    )
