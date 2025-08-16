"""Generic :math:`\theta`-scheme time-stepper."""

from __future__ import annotations

import numpy as np
import skfem as fem

from .base import TimeStepper


class ThetaScheme(TimeStepper):
    """Time-stepping scheme parameterised by ``theta``.

    ``theta=0``   corresponds to explicit Euler,
    ``theta=1``   to implicit Euler,
    ``theta=0.5`` to the Crank-Nicolson scheme.
    """

    def __init__(self, theta: float) -> None:
        self.theta = theta

    def step(self, space, v_prev: np.ndarray, t_i: float, dt: float) -> np.ndarray:
        A = space.I - self.theta * dt * space.L
        B = space.I + (1.0 - self.theta) * dt * space.L
        b_previous = B @ v_prev
        b_inhom = (
            self.theta * space.rhs(t_i + dt)
            + (1.0 - self.theta) * space.rhs(t_i)
        )
        b = b_previous + dt * b_inhom
        A, b = space.apply_dirichlet(A, b, t_i)
        return fem.solve(A, b)
