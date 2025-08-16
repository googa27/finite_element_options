"""Time-stepping interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class TimeStepper(ABC):
    """Abstract base class for time-stepping schemes."""

    @abstractmethod
    def step(self, space, v_prev: np.ndarray, t_i: float, dt: float) -> np.ndarray:
        """Advance the solution by one time step."""
        raise NotImplementedError
