"""State/time-dependent coefficient regressions for issue #36."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import scipy.sparse as sps

from finite_element_options.core.dynamics_heston import DynamicsParametersHeston
from finite_element_options.core.dynamics_heston_3d import DynamicsParametersHeston3D
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme


@dataclass
class _FakeBasis:
    N: int = 1


@dataclass
class _ManufacturedScalarSpace:
    """Scalar manufactured ODE route for reaction/source time convergence."""

    Vh: _FakeBasis = field(default_factory=_FakeBasis)

    @staticmethod
    def exact(time: float) -> float:
        return float(np.exp(0.3 * time) + 0.1 * np.sin(1.7 * time))

    @staticmethod
    def exact_prime(time: float) -> float:
        return float(0.3 * np.exp(0.3 * time) + 0.17 * np.cos(1.7 * time))

    @staticmethod
    def discount(time: float) -> float:
        return 0.2 + 0.1 * time

    def initial_condition(self) -> np.ndarray:
        return np.array([self.exact(0.0)])

    def matrices(
        self,
        theta: float,
        dt: float,
        *,
        start: float | None = None,
        end: float | None = None,
    ):
        start_time = 0.0 if start is None else float(start)
        end_time = start_time if end is None else float(end)
        operator_start = -self.discount(start_time)
        operator_end = -self.discount(end_time)
        return (
            sps.csr_matrix([[1.0 - theta * dt * operator_end]]),
            sps.csr_matrix([[1.0 + (1.0 - theta) * dt * operator_start]]),
        )

    def boundary_term(self, th: float) -> np.ndarray:
        time = float(th)
        source = self.exact_prime(time) + self.discount(time) * self.exact(time)
        return np.array([source])

    def domain_diagnostics(self, *, horizon: float, tail_mass: float = 1.0e-6) -> dict:
        return {"horizon": horizon, "tail_mass": tail_mass}

    def dirichlet(self, th: float) -> np.ndarray:  # pylint: disable=unused-argument
        return np.array([0.0])

    def apply_dirichlet(self, A, b, boundaries, u_dirichlet):  # pylint: disable=unused-argument
        return A, b


@dataclass
class _LinearCoefficientDynamics:
    """Minimal one-dimensional model with variable discount/source fields."""

    r: float = 0.03
    q: float = 0.0
    sig: float = 0.0

    def mean_variance(self, th, v, config=None):  # pylint: disable=unused-argument
        return np.zeros_like(v, dtype=float)

    def A(self, s):
        return [[np.zeros_like(s, dtype=float)]]

    def dA(self, s):
        return [np.zeros_like(s, dtype=float)]

    def b(self, s):
        return [np.zeros_like(s, dtype=float)]

    def discount(self, state, time):
        """State/time-dependent reaction coefficient c(x,t)."""

        return 0.05 + 0.10 * np.asarray(state[0], dtype=float) + 0.20 * float(time)

    def source(self, state, time):
        """State/time-dependent source field f(x,t)."""

        return 1.0 + np.asarray(state[0], dtype=float) + 2.0 * float(time)


def _payoff(rate: float = 0.03) -> EuropeanOptionBs:
    return EuropeanOptionBs(k=0.4, q=0.0, mkt=Market(r=rate))


def test_discount_field_is_assembled_from_quadrature_state_and_time() -> None:
    dynamics = _LinearCoefficientDynamics()
    mesh, config = create_mesh([1.0], refine=2)
    space = SpaceSolver(mesh, dynamics, _payoff(dynamics.r), is_call=True, config=config)

    operator_t0 = space.operator_matrix(0.0)
    operator_t1 = space.operator_matrix(1.0)
    constant_frozen_operator = space.forms.operator_form(0.0).assemble(space.Vh)

    assert operator_t0.shape == space.mass.shape
    assert operator_t1.shape == space.mass.shape
    assert np.max(np.abs((operator_t1 - operator_t0).toarray())) > 0.0
    assert np.max(np.abs((operator_t0 - constant_frozen_operator).toarray())) == pytest.approx(0.0)
    assert space.last_coefficient_diagnostics["discount_field"] == "callable"


def test_source_field_is_assembled_as_time_dependent_cell_load() -> None:
    dynamics = _LinearCoefficientDynamics()
    mesh, config = create_mesh([1.0], refine=2)
    space = SpaceSolver(mesh, dynamics, _payoff(dynamics.r), is_call=True, config=config)

    load_t0 = space.boundary_term(0.0)
    load_t1 = space.boundary_term(1.0)

    assert load_t0.shape == (space.Vh.N,)
    assert load_t1.shape == (space.Vh.N,)
    assert np.max(np.abs(load_t1 - load_t0)) > 0.0
    assert space.last_coefficient_diagnostics["source_field"] == "callable"


def test_theta_scheme_refreshes_time_dependent_operators_per_endpoint() -> None:
    dynamics = _LinearCoefficientDynamics()
    mesh, config = create_mesh([1.0], refine=2)
    space = SpaceSolver(mesh, dynamics, _payoff(dynamics.r), is_call=True, config=config)
    stepper = ThetaScheme(theta=0.5)

    stepper.solve([0.0, 0.25, 1.0], space)

    assert space.matrix_time_calls == pytest.approx([(0.0, 0.25), (0.25, 1.0)])
    assert len({round(item, 12) for pair in space.matrix_time_calls for item in pair}) == 3
    assert stepper.last_solve_diagnostics.solve_count == 2


def test_crank_nicolson_time_dependent_reaction_source_converges_quadratically() -> None:
    errors = []
    final_time = 1.0
    for steps in (20, 40, 80):
        space = _ManufacturedScalarSpace()
        solution = ThetaScheme(theta=0.5).solve(np.linspace(0.0, final_time, steps + 1), space)
        errors.append(abs(float(solution[-1, 0]) - space.exact(final_time)))

    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5


def test_heston3d_discount_uses_short_rate_state_coordinate() -> None:
    dynamics = DynamicsParametersHeston3D(
        r=0.03,
        q=0.01,
        kappa=1.7,
        theta=0.04,
        sig_v=0.35,
        rho=-0.25,
        kappa_r=0.8,
        theta_r=0.03,
        sig_r=0.01,
    )
    state = np.array(
        [
            [1.0, 1.0],
            [0.04, 0.04],
            [0.01, 0.07],
        ]
    )

    np.testing.assert_allclose(dynamics.discount(state, time=0.5), np.array([0.01, 0.07]))


def test_heston3d_constant_rate_discount_matches_2d_heston_limit() -> None:
    heston_2d = DynamicsParametersHeston(
        r=0.03, q=0.01, kappa=1.7, theta=0.04, sig=0.35, rho=-0.25
    )
    heston_3d = DynamicsParametersHeston3D(
        r=0.03,
        q=0.01,
        kappa=1.7,
        theta=0.04,
        sig_v=0.35,
        rho=-0.25,
        kappa_r=1.0,
        theta_r=0.03,
        sig_r=0.0,
    )
    state_2d = np.array([[0.8, 1.0, 1.2], [0.02, 0.04, 0.08]])
    state_3d = np.vstack([state_2d, np.full(state_2d.shape[1], heston_2d.r)])

    expected = np.full(state_2d.shape[1], heston_2d.discount(state_2d, time=0.5))
    np.testing.assert_allclose(heston_3d.discount(state_3d, time=0.5), expected)
