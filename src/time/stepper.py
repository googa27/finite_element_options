"""Time-stepping algorithms for the option PDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import skfem as fem

from src.space.solver import SpaceSolver


class TimeStepper(ABC):
    """Abstract base class for time-stepping schemes."""

    @abstractmethod
    def solve(
        self,
        t: Iterable[float],
        space: SpaceSolver,
        dirichlet_bcs=None,
        is_american: bool = False,
    ) -> np.ndarray:
        """Return the time-space solution array."""
        raise NotImplementedError


class ThetaScheme(TimeStepper):
    """General θ-scheme stepping (θ=1 implicit Euler, θ=0 explicit Euler)."""

    def __init__(self, theta: float = 0.5):
        self.theta = theta

    def solve(
        self,
        t: Iterable[float],
        space: SpaceSolver,
        dirichlet_bcs=None,
        is_american: bool = False,
    ) -> np.ndarray:
        dt = t[1] - t[0]
        v_tsv = np.empty((len(t), space.Vh.N))
        v_tsv[0] = space.initial_condition()
        A, B = space.matrices(self.theta, dt)

        for i, th_i in enumerate(t[:-1]):
            b_previous = B @ v_tsv[i]
            b_inhom = (
                self.theta * space.boundary_term(th_i + dt)
                + (1 - self.theta) * space.boundary_term(th_i)
            )
            b = b_previous + dt * b_inhom

            if dirichlet_bcs:
                u_dirichlet = space.dirichlet(th_i)
                A_enf, b_enf = space.apply_dirichlet(A, b, dirichlet_bcs, u_dirichlet)
                v_tsv[i + 1] = fem.solve(A_enf, b_enf)
            else:
                v_tsv[i + 1] = fem.solve(A, b)

            if is_american:
                v_tsv[i + 1] = np.maximum(v_tsv[i + 1], v_tsv[0])

        return v_tsv


class CrankNicolson(ThetaScheme):
    """Crank–Nicolson scheme with θ=1/2."""

    def __init__(self):
        super().__init__(theta=0.5)
