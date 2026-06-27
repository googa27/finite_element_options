import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.fdsolver import FDSolver, solve_system, delta  # noqa: E402
from src.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)  # noqa: E402
from src.core.market import Market  # noqa: E402
from src.core.vanilla_bs import EuropeanOptionBs  # noqa: E402


def test_fd_black_scholes_price():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    s_grid = np.linspace(0.0, 2.0, 101)
    t = np.linspace(0.0, 1.0, 50)
    v = solve_system(s_grid, t, dh, bsopt, is_call=True)

    s0 = 1.0
    idx = np.argmin(np.abs(s_grid - s0))
    price_num = v.sel(time=t[-1], space=s0, method="nearest").item()
    price_exact = bsopt.call(t[-1], s0, dh.sig ** 2)
    assert price_num == pytest.approx(price_exact, rel=1e-2)

    d_num = delta(v.sel(time=t[-1]).values, s_grid[1] - s_grid[0])[idx]
    d_exact = bsopt.call_delta(t[-1], s0, dh.sig ** 2)
    assert d_num == pytest.approx(d_exact, rel=2e-2)


def test_fd_solver_initial_condition_matches_vectorized_payoff_formula():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    s_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)

    call_solver = FDSolver(s_grid, dh, bsopt, is_call=True)
    put_solver = FDSolver(s_grid, dh, bsopt, is_call=False)

    np.testing.assert_allclose(call_solver.initial_condition(), np.maximum(s_grid - bsopt.k, 0.0))
    np.testing.assert_allclose(put_solver.initial_condition(), np.maximum(bsopt.k - s_grid, 0.0))


def test_fd_solver_initial_condition_accepts_noncontiguous_grid_views():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    s_grid = np.linspace(0.0, 2.0, 11, dtype=float)[::2]

    solver = FDSolver(s_grid, dh, bsopt, is_call=True)

    assert not s_grid.flags.c_contiguous
    np.testing.assert_allclose(solver.initial_condition(), np.maximum(s_grid - bsopt.k, 0.0))


class _CappedPayoffWithStrike:
    k = 1.0

    def call_payoff(self, s):
        return np.minimum(np.maximum(np.asarray(s) - self.k, 0.0), 0.25)

    def put_payoff(self, s):
        return np.minimum(np.maximum(self.k - np.asarray(s), 0.0), 0.25)


class _ScalarOnlyPayoffWithStrike:
    k = 1.0

    def call_payoff(self, s):
        return max(s - self.k, 0.0)

    def put_payoff(self, s):
        return max(self.k - s, 0.0)


def test_fd_solver_initial_condition_honors_custom_payoff_semantics():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    s_grid = np.array([0.0, 1.0, 1.5, 2.0], dtype=float)
    solver = FDSolver(s_grid, dh, _CappedPayoffWithStrike(), is_call=True)

    np.testing.assert_allclose(solver.initial_condition(), [0.0, 0.0, 0.25, 0.25])


def test_fd_solver_initial_condition_falls_back_for_scalar_only_payoffs():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    s_grid = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    solver = FDSolver(s_grid, dh, _ScalarOnlyPayoffWithStrike(), is_call=True)

    np.testing.assert_allclose(solver.initial_condition(), [0.0, 0.0, 0.0, 0.5])
