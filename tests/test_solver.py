import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.dynamics_heston import DynamicsParametersHeston
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.mesh import create_rectangular_mesh
from src.space.solver import SpaceSolver
from src.time.stepper import ThetaScheme


def test_solver_runs():
    dh = DynamicsParametersHeston(r=0.03, q=0.03, kappa=0.5, theta=0.5, sig=0.2, rho=0.5)
    mkt = Market(r=0.03)
    bsopt = EuropeanOptionBs(k=0.4, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 3)
    mesh = create_rectangular_mesh(1.0, 1.0, 1)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True)
    stepper = ThetaScheme(theta=0.5)
    v_tsv = stepper.solve(t, space, dirichlet_bcs=[])
    assert v_tsv.shape[0] == 3
