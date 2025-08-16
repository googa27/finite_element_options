"""Base classes for parameter calibration.

This module defines the :class:`Calibrator` abstract base class which stores
market data as NumPy arrays.  Subclasses are expected to implement
:method:`model_prices` to generate model-implied prices for a vector of model
parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Calibrator(ABC):
    """Abstract optimizer matching model prices to market data.

    Parameters
    ----------
    strikes:
        Array of strike prices.
    maturities:
        Array of option maturities with the same shape as ``strikes``.
    prices:
        Observed market option prices.
    """

    strikes: np.ndarray
    maturities: np.ndarray
    prices: np.ndarray

    def __post_init__(self) -> None:  # noqa: D401 - short explanation
        """Validate input array shapes."""
        self.strikes = np.asarray(self.strikes, dtype=float)
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.prices = np.asarray(self.prices, dtype=float)
        if not (
            self.strikes.shape == self.maturities.shape == self.prices.shape
        ):
            raise ValueError("Input arrays must share the same shape")

    @abstractmethod
    def model_prices(self, params: Sequence[float]) -> np.ndarray:
        """Return model prices for the supplied parameters."""

    def residuals(self, params: Sequence[float]) -> np.ndarray:
        """Difference between model and market prices."""
        return self.model_prices(params) - self.prices

    def calibrate(self, initial_guess: Sequence[float]) -> np.ndarray:
        """Calibrate model parameters via least squares.

        Parameters
        ----------
        initial_guess:
            Initial parameter vector for the solver.

        Returns
        -------
        numpy.ndarray
            Optimized parameter vector.
        """

        result = least_squares(self.residuals, x0=np.asarray(initial_guess))
        return result.x
