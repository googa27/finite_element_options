import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.dynamics_heston import DynamicsParametersHeston  # noqa: E402
from src.core.market import Market  # noqa: E402
from src.core.vanilla_bs import EuropeanOptionBs  # noqa: E402
from src.space.mesh import create_mesh  # noqa: E402
from src.space.solver import SpaceSolver  # noqa: E402
from src.space.boundary import DirichletBC  # noqa: E402
from src.time.stepper import ThetaScheme  # noqa: E402


def test_solver_runs():
    dh = DynamicsParametersHeston(
        r=0.03, q=0.03, kappa=0.5, theta=0.5, sig=0.2, rho=0.5
    )
    mkt = Market(r=0.03)
    bsopt = EuropeanOptionBs(k=0.4, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 3)
    mesh = create_mesh([1.0, 1.0], 1)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True)
    stepper = ThetaScheme(theta=0.5)
    bc = DirichletBC([])
    v_tsv = stepper.solve(t, space, boundary_condition=bc)
    assert v_tsv.shape[0] == 3
