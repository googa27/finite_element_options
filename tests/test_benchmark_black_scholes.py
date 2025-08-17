"""Benchmark the Black-Scholes solver on a standard grid."""

import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.mesh import create_mesh
from src.space.solver import SpaceSolver
from src.space.boundary import DirichletBC
from src.time.stepper import ThetaScheme


def test_black_scholes_benchmark(benchmark) -> None:
    """Run solver benchmark and validate numerical accuracy."""
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 5)
    mesh, cfg = create_mesh([2.0], 3)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True, config=cfg)
    stepper = ThetaScheme(theta=0.5)
    bc = DirichletBC([])

    def run():
        return stepper.solve(t, space, boundary_condition=bc)

    v_tsv = benchmark(run)

    s0 = 1.0
    node = np.argmin(np.abs(space.Vh.doflocs[0] - s0))
    price_num = v_tsv[-1, node]
    price_exact = bsopt.call(t[-1], s0, dh.sig ** 2)

    assert price_num == pytest.approx(price_exact, rel=1e-2)
