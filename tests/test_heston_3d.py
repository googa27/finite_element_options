import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.dynamics_heston_3d import (  # noqa: E402
    DynamicsParametersHeston3D,
)
from src.core.market import Market  # noqa: E402
from src.core.vanilla_bs import EuropeanOptionBs  # noqa: E402
from src.space.mesh import create_mesh  # noqa: E402
from src.space.solver import SpaceSolver  # noqa: E402
from src.time.stepper import ThetaScheme  # noqa: E402


def test_heston_3d_runs():
    dh = DynamicsParametersHeston3D(
        r=0.03,
        q=0.02,
        kappa=1.0,
        theta=0.04,
        sig_v=0.2,
        rho=0.0,
        kappa_r=0.5,
        theta_r=0.03,
        sig_r=0.1,
    )
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 0.5, 3)
    mesh = create_mesh([1.0, 1.0, 1.0], 1)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True)
    stepper = ThetaScheme(theta=0.5)
    v_tsv = stepper.solve(t, space, dirichlet_bcs=[])
    assert v_tsv.shape[0] == 3
