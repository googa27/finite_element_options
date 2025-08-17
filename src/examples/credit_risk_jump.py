"""Credit-risk bond pricing with jump intensity and Monte Carlo comparison."""

from __future__ import annotations

import numpy as np

from src.problems.credit_risk import CreditRiskJumpDynamics, CreditRiskPayoff
from src.core.market import Market
from src.space.mesh import create_mesh
from src.space.solver import SpaceSolver
from src.space.boundary import DirichletBC
from src.time.stepper import ThetaScheme


def pde_monte_carlo(n_paths: int = 10) -> tuple[float, float]:
    """Compare PDE estimate against analytic Monte Carlo baseline."""
    rng = np.random.default_rng(0)
    dyn = CreditRiskJumpDynamics(r=0.03, lamb=0.02, jump_vol=0.3)
    payoff = CreditRiskPayoff(recovery=0.4, mkt=Market(r=dyn.r))
    T = 1.0
    t = np.linspace(0.0, T, 5)
    mesh, cfg = create_mesh([1.0], 1)
    stepper = ThetaScheme(theta=0.5)

    pde_vals = []
    mc_vals = []

    for _ in range(n_paths):
        sampled = dyn.sample(rng)
        space = SpaceSolver(mesh, sampled, payoff, is_call=False, config=cfg)
        u = stepper.solve(t, space, boundary_condition=DirichletBC([]))
        pde_vals.append(float(u[-1][0]))
        lamb = sampled.lamb
        mc_vals.append(
            np.exp(-dyn.r * T) * (1 - payoff.recovery) * (1 - np.exp(-lamb * T))
        )

    return float(np.mean(pde_vals)), float(np.mean(mc_vals))


if __name__ == "__main__":  # pragma: no cover - example script
    pde, mc = pde_monte_carlo()
    print("PDE estimate:", pde)
    print("Monte Carlo baseline:", mc)
