"""Reusable FEM Black--Scholes surface solve for numerical evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from ...core.dynamics_black_scholes import DynamicsParametersBlackScholes
from ...core.market import Market
from ...core.vanilla_bs import EuropeanOptionBs
from ...space.boundary import DirichletBC
from ...space.mesh import create_mesh
from ...space.solver import SpaceSolver
from ...time_integration.stepper import ThetaScheme


@dataclass(frozen=True, slots=True)
class FEMBlackScholesSurfacePoint:
    """One point extracted from a single numerical FEM price surface."""

    spot: float
    degrees_of_freedom: int
    price: float
    delta: float
    gamma: float


def solve_black_scholes_surface(
    spots: Iterable[float],
    *,
    refinement_level: int = 6,
    time_steps: int = 80,
    strike: float = 100.0,
) -> tuple[FEMBlackScholesSurfacePoint, ...]:
    """Solve once and extract value/Delta/Gamma at each requested physical spot."""

    requested = tuple(float(spot) for spot in spots)
    if not requested or any(not isfinite(spot) or spot <= 0.0 for spot in requested):
        raise ValueError("spots must contain positive finite values")
    if refinement_level < 1 or time_steps < 1 or not isfinite(strike) or strike <= 0.0:
        raise ValueError("refinement_level, time_steps, and strike must be positive")

    dynamics = DynamicsParametersBlackScholes(r=0.05, q=0.0, sig=0.2)
    option = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    times = np.linspace(0.0, 1.0, time_steps + 1)
    mesh, config = create_mesh([4.0], refinement_level)
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 4.0),
        }
    )
    space = SpaceSolver(mesh, dynamics, option, is_call=True, config=config)
    with np.errstate(divide="ignore", invalid="ignore"):
        solution = ThetaScheme(theta=0.5).solve(
            times,
            space,
            boundary_condition=DirichletBC(["left", "right"]),
        )

    order = np.argsort(space.Vh.doflocs[0])
    coordinates = np.asarray(space.Vh.doflocs[0][order], dtype=float)
    values = np.asarray(solution[-1][order], dtype=float)
    points = tuple(
        _extract_surface_point(coordinates, values, spot=spot, strike=strike)
        for spot in requested
    )
    if any(
        not all(isfinite(value) for value in (point.price, point.delta, point.gamma))
        for point in points
    ):
        raise FloatingPointError("non-finite FEM Black-Scholes surface result")
    return points


def _extract_surface_point(
    coordinates: np.ndarray,
    values: np.ndarray,
    *,
    spot: float,
    strike: float,
) -> FEMBlackScholesSurfacePoint:
    target = spot / strike
    center = int(np.argmin(np.abs(coordinates - target)))
    if center == 0 or center == len(coordinates) - 1:
        raise ValueError("requested spot must have neighboring FEM degrees of freedom")
    local_x = coordinates[center - 1 : center + 2]
    local_values = values[center - 1 : center + 2]
    a, b, c = np.polyfit(local_x, local_values, deg=2)
    normalized_price = float(a * target * target + b * target + c)
    delta = float(2.0 * a * target + b)
    gamma = float(2.0 * a / strike)
    return FEMBlackScholesSurfacePoint(
        spot=spot,
        degrees_of_freedom=len(coordinates),
        price=strike * normalized_price,
        delta=delta,
        gamma=gamma,
    )


__all__ = ["FEMBlackScholesSurfacePoint", "solve_black_scholes_surface"]
