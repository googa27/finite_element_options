"""Tests for the experimental FEniCS solver."""

import numpy as np
import pytest

from src.space import fenics_solver
from src.space.fenics_solver import FenicsSolver
from src.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs

HAS_FENICS = fenics_solver.fem is not None


@pytest.mark.skipif(not HAS_FENICS, reason="FEniCSx not installed")
def test_fenics_black_scholes() -> None:
    """Verify that the FEniCS solver reproduces analytic prices."""
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    solver = FenicsSolver(domain=(0.0, 2.0), num_elements=16, dynamics=dh, payoff=bsopt, is_call=True)
    t = np.linspace(0.0, 1.0, 5)
    v = solver.solve(t)
    node = np.argmin(np.abs(solver.Vh.tabulate_dof_coordinates().ravel() - 1.0))
    price_num = v[-1, node]
    price_exact = bsopt.call(t[-1], 1.0, dh.sig ** 2)
    assert price_num == pytest.approx(price_exact, rel=1e-2)
