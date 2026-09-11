"""Real P1 reaction FEM proves bounded retention without changing theta arithmetic."""

from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import skfem as fem

from finite_element_options.core.config import Config
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme


class ReactionForms:
    def __init__(self, *, slope=0.2):
        self.slope = slope
        self.rate = 0.2
        self.assembled = []

    def id_bil(self):
        return fem.BilinearForm(lambda u, v, w: u * v)

    def operator_form(self, th):
        self.assembled.append(th)
        coefficient = self.rate + self.slope * th
        return fem.BilinearForm(lambda u, v, w: -coefficient * u * v)

    def b_lin(self):
        return fem.LinearForm(lambda v, w: 0 * v)

    def source_lin(self, th):
        return fem.LinearForm(lambda v, w: 0 * v)


def space(*, capacity=2, slope=0.2):
    payoff = SimpleNamespace(
        call_payoff=lambda x: np.ones_like(x),
        put_payoff=lambda x: np.ones_like(x),
        call=lambda *args: None,
        put=lambda *args: None,
    )
    return SpaceSolver(
        fem.MeshLine(np.linspace(0, 1, 65)),
        object(),
        payoff,
        True,
        forms=ReactionForms(slope=slope),
        config=Config(elem=fem.ElementLineP1()),
        operator_cache_size=capacity,
    )


@pytest.mark.parametrize("steps", [20, 80])
@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_actual_operator_retention_is_bounded_and_evicted_matrix_is_released(
    steps, capacity
):
    spatial = space(capacity=capacity)
    refs = []
    for time in np.linspace(0.01, 1, steps):
        matrix = spatial.operator_matrix(time)
        refs.append(weakref.ref(matrix))
    del matrix
    assert sum(ref() is not None for ref in refs) <= capacity
    info = spatial.operator_cache_info()
    assert info.entries <= capacity and info.peak_entries <= capacity
    assert info.misses == steps + 1  # constructor initializes stiffness at zero


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_bounded_and_unbounded_capacity_histories_are_bitwise_equal(capacity):
    grid = np.linspace(0, 1, 81)
    baseline = ThetaScheme(factorization_cache_size=100).solve(
        grid, space(capacity=100)
    )
    stepper = ThetaScheme(factorization_cache_size=capacity)
    values = stepper.solve(grid, space(capacity=capacity))
    np.testing.assert_array_equal(values, baseline)
    expected = 1.0
    for start, end in zip(grid[:-1], grid[1:]):
        dt = 1 / 80
        expected *= (1 - 0.5 * dt * (0.2 + 0.2 * start)) / (
            1 + 0.5 * dt * (0.2 + 0.2 * end)
        )
    np.testing.assert_allclose(values[-1], expected, rtol=2e-12, atol=2e-12)
    info = stepper.last_solve_diagnostics
    assert info.factorization_cache_peak_entries <= capacity
    assert info.factorization_count == 80 and info.factorization_reuse_count == 0
    assert info.factorization_cache_entries_at_completion <= capacity


def test_constant_two_width_and_startup_systems_preserve_useful_reuse():
    constant = ThetaScheme()
    spatial = space(slope=0)
    first = constant.solve(np.linspace(0, 1, 31), spatial)
    assert constant.last_solve_diagnostics.factorization_count == 1
    assert constant.last_solve_diagnostics.factorization_reuse_count == 29
    second = constant.solve(np.linspace(0, 1, 31), spatial)
    np.testing.assert_array_equal(first, second)
    assert (
        constant.last_solve_diagnostics.factorization_count == 1
    )  # cache belongs to this solve
    repeated = ThetaScheme()
    repeated.solve([0, 0.25, 0.75, 1], space(slope=0))
    assert repeated.last_solve_diagnostics.factorization_count == 2
    assert repeated.last_solve_diagnostics.factorization_reuse_count == 1
    options = dict(startup_theta=1.0, startup_steps=2, startup_substeps=2)
    a = ThetaScheme(**options, factorization_cache_size=100).solve(
        np.linspace(0, 1, 11), space(slope=0)
    )
    b = ThetaScheme(**options).solve(np.linspace(0, 1, 11), space(slope=0))
    np.testing.assert_array_equal(a, b)


def test_explicit_invalidation_refreshes_stiffness_and_cached_time_operators():
    spatial = space()
    initial = spatial.stiffness.copy()
    previous = spatial.operator_matrix(0.5).copy()
    spatial.forms.rate += 1
    spatial.invalidate_operator_cache()
    assert not np.array_equal(spatial.stiffness.data, initial.data)
    assert not np.array_equal(spatial.operator_matrix(0.5).data, previous.data)
    assert spatial.operator_cache_info().clears == 1
    np.testing.assert_array_equal(
        spatial.stiffness.toarray(), spatial.operator_matrix(0).toarray()
    )


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_factor_payloads_are_released_after_eviction_and_after_each_solve(
    monkeypatch, capacity
):
    from finite_element_options.time_integration import stepper as module

    original = module.spla.splu
    refs = []
    retained_before_factor = []

    class TrackedFactor:
        def __init__(self, matrix):
            self.factor = original(matrix)

        def solve(self, rhs):
            return self.factor.solve(rhs)

    def factor(matrix):
        retained_before_factor.append(sum(ref() is not None for ref in refs))
        result = TrackedFactor(matrix)
        refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(module.spla, "splu", factor)
    solver = ThetaScheme(factorization_cache_size=capacity)
    solver.solve(np.linspace(0, 1, 81), space(capacity=capacity))
    assert len(refs) == 80
    # The previous active solve may retain one factor when retention is disabled.
    assert max(retained_before_factor) <= max(1, capacity)
    assert all(ref() is None for ref in refs)  # no factor escapes this solve
    assert solver.last_solve_diagnostics.factorization_cache_eviction_count == (
        80 - capacity if capacity else 0
    )


def test_evicted_factor_is_rebuilt_and_failure_does_not_poison_next_solve(monkeypatch):
    from finite_element_options.time_integration import stepper as module

    grid = [0, 0.125, 0.375, 0.875, 1]
    reference = ThetaScheme(factorization_cache_size=10).solve(grid, space(slope=0))
    solver = ThetaScheme()
    result = solver.solve(grid, space(slope=0))
    np.testing.assert_array_equal(result, reference)
    assert solver.last_solve_diagnostics.factorization_count == 4
    assert solver.last_solve_diagnostics.factorization_cache_eviction_count == 2
    original = module.spla.splu
    calls = 0

    def fail_second(matrix):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic factorization failure")
        return original(matrix)

    monkeypatch.setattr(module.spla, "splu", fail_second)
    with pytest.raises(RuntimeError, match="synthetic factorization failure"):
        solver.solve(grid, space(slope=0))
    monkeypatch.setattr(module.spla, "splu", original)
    np.testing.assert_array_equal(solver.solve(grid, space(slope=0)), reference)
    assert solver.last_solve_diagnostics.factorization_count == 4


def test_changed_enforced_matrix_is_not_reused_under_the_same_step_width():
    class ChangingBoundary:
        def apply(self, spatial, matrix, rhs, time):
            return matrix * (1 + time), rhs

    grid = np.linspace(0, 1, 11)
    a = ThetaScheme(factorization_cache_size=100)
    b = ThetaScheme()
    np.testing.assert_array_equal(
        a.solve(grid, space(slope=0), ChangingBoundary()),
        b.solve(grid, space(slope=0), ChangingBoundary()),
    )
    assert b.last_solve_diagnostics.factorization_count == 10
    assert b.last_solve_diagnostics.factorization_reuse_count == 0


def test_refinement_invalidates_old_basis_operators_and_rebuilds_mass():
    from finite_element_options.space.adaptive import AdaptiveMesh

    spatial = space()
    old_stiffness = weakref.ref(spatial.stiffness)
    old_operator = weakref.ref(spatial.operator_matrix(0.5))
    old_size = spatial.Vh.N
    spatial.adapt = AdaptiveMesh(fem.ElementLineP1(), criterion="gradient")
    values = np.sin(spatial.Vh.doflocs[0] * np.pi)
    result = spatial.refine_with_transfer(values)
    assert result.mesh is spatial.mesh
    assert spatial.Vh.N > old_size
    assert spatial.mass.shape == spatial.stiffness.shape == (spatial.Vh.N, spatial.Vh.N)
    assert old_stiffness() is None and old_operator() is None
    assert spatial.operator_cache_info().clears == 1
    assert spatial.operator_cache_info().entries == 1
    assert spatial.operator_matrix(0.5).shape == spatial.mass.shape


def test_zero_capacity_and_disabled_reuse_preserve_explicit_policies():
    grid = np.linspace(0, 1, 11)
    zero = ThetaScheme(factorization_cache_size=0)
    zero.solve(grid, space(slope=0))
    assert zero.last_solve_diagnostics.factorization_count == 10
    assert zero.last_solve_diagnostics.factorization_reuse_count == 0
    disabled = ThetaScheme(reuse_factorization=False, factorization_cache_size=2)
    disabled.solve(grid, space(slope=0))
    info = disabled.last_solve_diagnostics
    assert info.factorization_cache_capacity == 0
    assert info.factorization_cache_peak_entries == 0
    assert not info.factorization_reuse_enabled
    assert info.factorization_count == 10


@pytest.mark.parametrize("capacity", [-1, True, False, 1.5, 2.0, None, "2"])
def test_invalid_capacities_are_refused_before_spatial_assembly(monkeypatch, capacity):
    def unexpected(*args, **kwargs):
        raise AssertionError("assembly must not start")

    monkeypatch.setattr(fem, "CellBasis", unexpected)
    with pytest.raises(ValueError, match="capacity"):
        space(capacity=capacity)
    with pytest.raises(ValueError, match="capacity"):
        ThetaScheme(factorization_cache_size=capacity)
