import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.dynamics_heston import DynamicsParametersHeston
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.core.mesh import create_rectangular_mesh
from src.core.solver import solve_pde


def test_solver_runs():
    dh = DynamicsParametersHeston(r=0.03, q=0.03, kappa=0.5, theta=0.5, sig=0.2, rho=0.5)
    mkt = Market(r=0.03)
    bsopt = EuropeanOptionBs(k=0.4, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 3)
    mesh = create_rectangular_mesh(1.0, 1.0, 1)
    v_tsv = solve_pde(t, mesh, dh, bsopt, is_call=True, dirichlet_bcs=[], lam=0.5)
    assert v_tsv.shape[0] == 3
