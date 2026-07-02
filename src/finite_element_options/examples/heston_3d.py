"""Example: solving the 3D Heston PDE with stochastic interest rate."""

import numpy as np

from finite_element_options.core.dynamics_heston_3d import DynamicsParametersHeston3D
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.space.boundary import DirichletBC
from finite_element_options.time_integration.stepper import ThetaScheme


def price_call():
    """Solve the 3D Heston PDE and return the option value grid."""
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
    t = np.linspace(0.0, 0.5, 10)
    mesh, cfg = create_mesh([1.0, 1.0, 1.0], 1)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True, config=cfg)
    stepper = ThetaScheme(theta=0.5)
    return stepper.solve(t, space, boundary_condition=DirichletBC([]))


if __name__ == "__main__":
    grid = price_call()
    print(grid[-1])
