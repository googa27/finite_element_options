"""Problem abstractions supplying PDE components."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.interfaces import DynamicsModel, Payoff, BoundaryCondition


@dataclass
class Problem:
    """Container bundling PDE ingredients.

    Concrete problems provide default instances for the dynamics model,
    payoff and boundary condition strategy used by the solver.  The
    attributes are typed according to the core interfaces so they can be
    consumed directly by the space and time modules.
    """

    dynamics: DynamicsModel
    payoff: Payoff
    boundary_condition: BoundaryCondition
