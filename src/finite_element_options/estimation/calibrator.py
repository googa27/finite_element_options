"""Base classes for parameter calibration.

This module defines the :class:`Calibrator` abstract base class which stores
market data in a :class:`pandas.DataFrame` while maintaining NumPy arrays
internally for performance. Subclasses are expected to implement
``model_prices`` to generate model-implied prices for a vector of model
parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class Calibrator(ABC):
    """Abstract optimiser matching model prices to market data.

    Parameters
    ----------
    market_data:
        DataFrame with ``strike``, ``maturity`` and ``price`` columns.
    """

    market_data: pd.DataFrame
    strikes: np.ndarray = field(init=False)
    maturities: np.ndarray = field(init=False)
    prices: np.ndarray = field(init=False)

    def __post_init__(self) -> None:  # noqa: D401 - short explanation
        """Validate market data and extract NumPy arrays."""
        required = {"strike", "maturity", "price"}
        if not required.issubset(self.market_data.columns):
            raise ValueError("market_data must contain strike, maturity and price")
        df = self.market_data[sorted(required)].astype(float)
        self.strikes = df["strike"].to_numpy()
        self.maturities = df["maturity"].to_numpy()
        self.prices = df["price"].to_numpy()

    @abstractmethod
    def model_prices(self, params: Sequence[float]) -> np.ndarray:
        """Return model prices for the supplied parameters."""

    def residuals(self, params: Sequence[float]) -> np.ndarray:
        """Difference between model and market prices."""
        return self.model_prices(params) - self.prices

    # ------------------------------------------------------------------
    # DataFrame helpers
    def model_prices_df(self, params: Sequence[float]) -> pd.DataFrame:
        """Return model prices as a DataFrame.

        Parameters
        ----------
        params:
            Model parameter vector.

        Returns
        -------
        pandas.DataFrame
            DataFrame with ``strike``, ``maturity`` and ``model_price`` columns.
        """

        prices = self.model_prices(params)
        return pd.DataFrame(
            {
                "strike": self.strikes,
                "maturity": self.maturities,
                "model_price": prices,
            }
        )

    def residuals_df(self, params: Sequence[float]) -> pd.DataFrame:
        """Return residuals between model and market prices as a DataFrame."""
        df = self.model_prices_df(params)
        df["residual"] = df["model_price"] - self.prices
        return df

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
