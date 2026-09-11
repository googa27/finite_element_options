"""Time-unit invariance and pre-assembly refusal for theta grids (issue 152)."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sparse

from finite_element_options.time_integration.stepper import ThetaScheme


class ScalarGrowth:
    """One-dof explicit Euler oracle; equivalent generators use inverse time units."""

    Vh = SimpleNamespace(N=1)

    def __init__(self, rate):
        self.rate = rate
        self.widths = []

    def initial_condition(self):
        return np.array([1.0])

    def matrices(self, theta, dt, *, start, end):
        self.widths.append(dt)
        return sparse.eye(1, format="csr"), sparse.csr_matrix([[1 + self.rate * dt]])

    def boundary_term(self, time):
        return np.zeros(1)


@pytest.mark.parametrize("scale", [1e-300, 1e-100, 1e-15, 1.0, 1e100, 1e300])
def test_nonuniform_steps_are_invariant_under_time_units(scale):
    space = ScalarGrowth(1 / scale)
    stepper = ThetaScheme(theta=0)
    grid = np.array([0.0, 0.1, 1.1]) * scale
    result = stepper.solve(grid, space)
    np.testing.assert_array_equal(space.widths, np.diff(grid))
    np.testing.assert_allclose(result[:, 0], [1, 1.1, 2.2], rtol=2e-15, atol=0)
    assert not stepper.last_time_grid_diagnostics["uniform_time_grid"]


@pytest.mark.parametrize("scale", [1e-300, 1e-15, 1.0, 1e100])
def test_roundoff_uniform_grid_preserves_one_factorization(scale):
    space = ScalarGrowth(1 / scale)
    stepper = ThetaScheme(theta=0)
    stepper.solve(np.linspace(0, scale, 31), space)
    assert len(set(space.widths)) == 1
    assert stepper.last_time_grid_diagnostics["uniform_time_grid"]
    assert stepper.last_solve_diagnostics.factorization_count == 1
    assert stepper.last_solve_diagnostics.factorization_reuse_count == 29


@pytest.mark.parametrize("grid", [[-1e308, 1e308], [-1e308, 0, 1e308]])
def test_nonrepresentable_interval_or_horizon_refuses_before_assembly(grid):
    space = ScalarGrowth(0)
    with pytest.raises(ValueError, match="finite"):
        ThetaScheme().solve(grid, space)
    assert space.widths == []
