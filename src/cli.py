"""Command line interface for batch option pricing runs."""

import argparse
import numpy as np

from .core.dynamics_heston import DynamicsParametersHeston
from .core.market import Market
from .core.vanilla_bs import EuropeanOptionBs
from .space.mesh import create_mesh
from .space.solver import SpaceSolver
from .time.stepper import ThetaScheme


def main(args=None):
    parser = argparse.ArgumentParser(description="Run Heston option solver")
    parser.add_argument("--k", type=float, default=0.4, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Maturity")
    parser.add_argument("--r", type=float, default=0.03, help="Risk free rate")
    parser.add_argument("--q", type=float, default=0.03, help="Dividend yield")
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--s-max", type=float, default=1.0, dest="s_max")
    parser.add_argument("--v-max", type=float, default=1.0, dest="v_max")
    parser.add_argument("--nt", type=int, default=10)
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument(
        "--call", action="store_true", help="Price a call option"
    )
    parser.add_argument("--american", action="store_true")

    ns = parser.parse_args(args=args)

    dh = DynamicsParametersHeston(
        r=ns.r, q=ns.q, kappa=ns.kappa, theta=ns.theta, sig=ns.sig, rho=ns.rho
    )
    mkt = Market(r=ns.r)
    bsopt = EuropeanOptionBs(ns.k, dh.q, mkt)

    t = np.linspace(0, ns.T, ns.nt)
    mesh = create_mesh([ns.s_max, ns.v_max], ns.refine)
    space = SpaceSolver(mesh, dh, bsopt, is_call=ns.call)
    stepper = ThetaScheme(theta=ns.lam)

    v_tsv = stepper.solve(
        t,
        space,
        dirichlet_bcs=None,
        is_american=ns.american,
    )

    print(v_tsv[-1])


if __name__ == "__main__":
    main()
