"""Example: pricing a European call option in the 1D Black-Scholes model."""

import numpy as np

from src.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.mesh import create_mesh
from src.space.solver import SpaceSolver
from src.time.stepper import ThetaScheme


def price_call():
    """Solve the Black-Scholes PDE and return the option value grid."""
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    t = np.linspace(0.0, 1.0, 20)
    mesh = create_mesh([2.0], 4)
    space = SpaceSolver(mesh, dh, bsopt, is_call=True)
    stepper = ThetaScheme(theta=0.5)
    return stepper.solve(t, space, dirichlet_bcs=[])


if __name__ == "__main__":
    grid = price_call()
    print(grid[-1])
