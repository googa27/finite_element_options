"""Regression tests for the Black-Scholes finite-difference reference backend.

Issue #51 narrows this backend to a small, independently auditable 1D
Black-Scholes oracle.  These tests guard the route gates, boundary algebra, time
semantics, payoff evaluation, and result diagnostics required before FEM parity
uses it as evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.fdsolver import FDSolver, solve_system, vega


def _model(rate: float = 0.03, carry: float = 0.01, sigma: float = 0.2):
    dynamics = DynamicsParametersBlackScholes(r=rate, q=carry, sig=sigma)
    option = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    return dynamics, option


def test_fd_reference_entry_points_emit_targeted_compatibility_warnings() -> None:
    dynamics, option = _model()
    s_grid = np.linspace(0.0, 2.0, 41)
    message = "compatibility-only.*0.3.0.*2026-10-31.*finite_difference_options"

    with pytest.warns(DeprecationWarning, match=message):
        FDSolver(s_grid, dynamics, option, is_call=True)
    with pytest.warns(DeprecationWarning, match=message):
        solve_system(s_grid, np.linspace(0.0, 1.0, 4), dynamics, option, is_call=True)
    with pytest.warns(DeprecationWarning, match=message):
        vega(np.ones((3, 3)), 0.1)


@pytest.mark.parametrize(
    "grid, message",
    [
        (np.array([0.0, 1.0]), "at least three"),
        (np.array([-0.1, 0.0, 0.1, 0.2]), "non-negative"),
        (np.array([0.0, 1.0, 0.5, 2.0]), "strictly increasing"),
        (np.array([0.0, 0.5, np.nan, 1.5]), "finite"),
        (np.array([0.0, 0.5, 1.4, 2.0]), "uniform"),
    ],
)
def test_fd_reference_rejects_invalid_or_nonuniform_spot_grids(grid, message) -> None:
    dynamics, option = _model()

    with pytest.raises(ValueError, match=message):
        FDSolver(grid, dynamics, option, is_call=True)


@pytest.mark.parametrize(
    "time_grid, message",
    [
        ([0.0], "at least two"),
        ([0.0, 0.25, np.nan], "finite"),
        ([0.0, 0.5, 0.4], "strictly increasing"),
        ([0.0, 0.25, 0.75, 1.0], "uniform"),
    ],
)
def test_solve_system_rejects_invalid_or_nonuniform_time_grids(
    time_grid, message
) -> None:
    dynamics, option = _model()
    s_grid = np.linspace(0.0, 2.0, 41)

    with pytest.raises(ValueError, match=message):
        solve_system(s_grid, time_grid, dynamics, option, is_call=True)


def test_fd_reference_call_boundary_includes_dividend_carry() -> None:
    dynamics, option = _model(rate=0.05, carry=0.02)
    s_grid = np.linspace(0.0, 3.0, 61)
    solver = FDSolver(s_grid, dynamics, option, is_call=True)
    tau = 0.75

    boundary = solver.dirichlet(tau)

    assert boundary[0] == 0.0
    assert boundary[-1] == pytest.approx(
        s_grid[-1] * np.exp(-dynamics.q * tau) - option.k * np.exp(-dynamics.r * tau)
    )


def test_fd_reference_call_boundary_never_returns_negative_far_field() -> None:
    dynamics, option = _model(rate=0.03, carry=1.00)
    s_grid = np.linspace(0.0, 2.0, 41)
    solver = FDSolver(s_grid, dynamics, option, is_call=True)

    boundary = solver.dirichlet(1.0)

    assert boundary[0] == 0.0
    assert boundary[-1] == 0.0


def test_dirichlet_elimination_adjusts_interior_rhs_and_zeroes_boundary_columns() -> (
    None
):
    dynamics, option = _model(rate=0.05, carry=0.02)
    solver = FDSolver(np.linspace(0.0, 2.0, 5), dynamics, option, is_call=True)
    matrix = sps.csr_matrix(
        np.array(
            [
                [10.0, 2.0, 0.0, 0.0, 0.0],
                [3.0, 11.0, 4.0, 0.0, 5.0],
                [0.0, 6.0, 12.0, 7.0, 0.0],
                [8.0, 0.0, 9.0, 13.0, 10.0],
                [0.0, 0.0, 0.0, 14.0, 15.0],
            ]
        )
    )
    rhs = np.array([1.0, 20.0, 30.0, 40.0, 50.0])
    boundary = np.array([2.0, 0.0, 0.0, 0.0, 3.0])

    enforced, adjusted = solver.apply_dirichlet(matrix, rhs, [], boundary)
    dense = enforced.toarray()

    assert adjusted[1] == pytest.approx(20.0 - 3.0 * 2.0 - 5.0 * 3.0)
    assert adjusted[3] == pytest.approx(40.0 - 8.0 * 2.0 - 10.0 * 3.0)
    assert dense[1, 0] == dense[1, -1] == 0.0
    assert dense[3, 0] == dense[3, -1] == 0.0
    assert dense[0, 0] == dense[-1, -1] == 1.0
    assert adjusted[0] == 2.0
    assert adjusted[-1] == 3.0


class _BrokenVectorizedPayoff:
    k = 1.0

    def call_payoff(self, s):
        if isinstance(s, np.ndarray):
            raise ValueError("negative spot domain bug should propagate")
        return max(s - self.k, 0.0)

    def put_payoff(self, s):
        return max(self.k - s, 0.0)


class _ScalarOnlyPayoff:
    k = 1.0

    def call_payoff(self, s):
        if isinstance(s, np.ndarray):
            raise TypeError("scalar only")
        return max(s - self.k, 0.0)

    def put_payoff(self, s):
        if isinstance(s, np.ndarray):
            raise TypeError("scalar only")
        return max(self.k - s, 0.0)


def test_payoff_domain_value_errors_are_not_swallowed_as_scalar_fallback() -> None:
    dynamics, _ = _model()
    solver = FDSolver(
        np.linspace(0.0, 2.0, 5), dynamics, _BrokenVectorizedPayoff(), is_call=True
    )

    with pytest.raises(ValueError, match="domain bug"):
        solver.initial_condition()


def test_scalar_only_payoff_type_error_still_uses_explicit_scalar_fallback() -> None:
    dynamics, _ = _model()
    solver = FDSolver(
        np.linspace(0.0, 2.0, 5), dynamics, _ScalarOnlyPayoff(), is_call=True
    )

    np.testing.assert_allclose(solver.initial_condition(), [0.0, 0.0, 0.0, 0.5, 1.0])


def test_solve_system_reports_time_orientation_residual_and_factorization_reuse() -> (
    None
):
    dynamics, option = _model(rate=0.03, carry=0.01)
    s_grid = np.linspace(0.0, 2.0, 81)
    tau_grid = np.linspace(0.0, 1.0, 41)

    result = solve_system(s_grid, tau_grid, dynamics, option, is_call=True)

    assert result.attrs["fd_backend"] == "black_scholes_uniform_1d"
    assert result.attrs["time_orientation"] == "tau_time_to_maturity_forward"
    assert result.attrs["time_step_count"] == len(tau_grid) - 1
    assert result.attrs["factorization_reuse_count"] == len(tau_grid) - 1
    assert result.attrs["convergence_status"] == "solved"
    assert result.attrs["coordinate_units"] == {"time": "year", "space": "spot"}
    assert result.attrs["max_linear_residual_abs"] < 1.0e-9


def test_fd_reference_black_scholes_convergence_improves_with_refinement() -> None:
    dynamics, option = _model(rate=0.03, carry=0.0)
    tau = 0.5
    strike = option.k
    errors = []
    for n_space, n_time in [(81, 41), (161, 81)]:
        s_grid = np.linspace(0.0, 3.0, n_space)
        tau_grid = np.linspace(0.0, tau, n_time)
        result = solve_system(s_grid, tau_grid, dynamics, option, is_call=True)
        numerical = result.sel(time=tau, space=strike, method="nearest").item()
        exact = option.call(tau, strike, dynamics.sig**2)
        errors.append(abs(numerical - exact))

    assert errors[1] < 0.75 * errors[0]


def test_vega_requires_an_explicit_volatility_axis() -> None:
    with pytest.raises(ValueError, match="volatility axis"):
        vega(np.ones(5), 0.1)
