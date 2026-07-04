import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finite_element_options.core.dynamics_black_scholes import (  # noqa: E402
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.market import Market  # noqa: E402
from finite_element_options.core.vanilla_bs import EuropeanOptionBs  # noqa: E402
from finite_element_options.space.mesh import create_mesh  # noqa: E402
from finite_element_options.space.solver import SpaceSolver  # noqa: E402
from finite_element_options.space.boundary import DirichletBC  # noqa: E402
from finite_element_options.time_integration.stepper import ThetaScheme  # noqa: E402


def test_black_scholes_price():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 5)
    mesh, cfg = create_mesh([2.0], 3)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True, config=cfg)
    stepper = ThetaScheme(theta=0.5)
    bc = DirichletBC([])
    v_tsv = stepper.solve(t, space, boundary_condition=bc)

    s0 = 1.0
    node = np.argmin(np.abs(space.Vh.doflocs[0] - s0))
    price_num = v_tsv[-1, node]
    price_exact = bsopt.call(t[-1], s0, dh.sig ** 2)

    assert price_num == pytest.approx(price_exact, rel=1e-2)


def test_black_scholes_initial_condition_matches_payoff():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    mesh, cfg = create_mesh([2.0], 2)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True, config=cfg)

    initial = space.initial_condition()
    expected = space.Vh.project(
        lambda x: bsopt.call_payoff(space.transform.untransform_state(x)[0])
    )
    np.testing.assert_allclose(initial, expected)


def test_black_scholes_dirichlet_matches_price():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    mesh, cfg = create_mesh([2.0], 2)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True, config=cfg)

    th = 0.5
    dirichlet_vals = space.dirichlet(th)
    th_phys = float(np.asarray(space.transform.untransform_time(th)))
    state = space.transform.untransform_state(space.Vh.doflocs)
    spots = state[0]
    variance = dh.mean_variance(th_phys, np.zeros_like(spots))
    expected = bsopt.call(th_phys, spots, variance)
    np.testing.assert_allclose(dirichlet_vals, expected)
