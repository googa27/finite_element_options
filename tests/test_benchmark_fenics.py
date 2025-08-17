"""Benchmark FEniCS solver against scikit-fem backend."""

import numpy as np
import pytest

from src.space import fenics_solver
from src.space.fenics_solver import FenicsSolver
from src.space.solver import SpaceSolver
from src.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.mesh import create_mesh
from src.time.stepper import ThetaScheme
from src.space.boundary import DirichletBC

HAS_FENICS = fenics_solver.fem is not None


@pytest.mark.skipif(not HAS_FENICS, reason="FEniCSx not installed")
def test_fenics_solver_benchmark(benchmark) -> None:
    """Compare solve times between FEniCS and scikit-fem implementations."""
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 5)

    mesh_sk = create_mesh([2.0], 3)
    space_sk = SpaceSolver(mesh_sk, dh, bsopt, is_call=True)
    stepper = ThetaScheme(theta=0.5)
    bc = DirichletBC([])

    def run_sk():
        return stepper.solve(t, space_sk, boundary_condition=bc)

    benchmark(run_sk)

    solver = FenicsSolver(domain=(0.0, 2.0), num_elements=mesh_sk.nelements, dynamics=dh, payoff=bsopt, is_call=True)

    def run_fn():
        return solver.solve(t)

    benchmark(run_fn)
