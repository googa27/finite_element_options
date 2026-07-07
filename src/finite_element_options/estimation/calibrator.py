"""Base classes and diagnostics for parameter calibration.

This module defines the :class:`Calibrator` abstract base class and the
:class:`CalibrationResult` value object. Calibration methods return structured
optimizer diagnostics instead of a bare parameter vector so convergence, rank,
bounds and failure states are explicit at the call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ParameterVector: TypeAlias = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """Structured result returned by calibration routines.

    Parameters are not sufficient evidence of convergence. Callers must inspect
    ``success``, ``status``, ``message``, residuals and Jacobian diagnostics
    before using a fitted model in a numerical route.
    """

    parameters: np.ndarray
    success: bool
    status: int
    message: str
    residuals: np.ndarray
    cost: float
    optimality: float
    jacobian_rank: int
    jacobian_condition: float
    bounds: tuple[np.ndarray, np.ndarray]
    active_mask: np.ndarray
    nfev: int
    njev: int | None
    method: str
    parameter_names: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)


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
    parameter_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:  # noqa: D401 - short explanation
        """Validate market data and extract NumPy arrays."""
        required = {"strike", "maturity", "price"}
        missing = required.difference(self.market_data.columns)
        if missing:
            raise ValueError(
                f"market_data must contain strike, maturity and price; missing {sorted(missing)}"
            )
        df = self.market_data.loc[:, ["strike", "maturity", "price"]].astype(float)
        self.strikes = df["strike"].to_numpy()
        self.maturities = df["maturity"].to_numpy()
        self.prices = df["price"].to_numpy()
        if not (self.strikes.shape == self.maturities.shape == self.prices.shape):
            raise ValueError("strike, maturity and price arrays must have matching shape")

    @abstractmethod
    def model_prices(self, params: ParameterVector) -> np.ndarray:
        """Return model prices for the supplied parameters."""

    def residuals(self, params: ParameterVector) -> np.ndarray:
        """Difference between model and market prices."""
        return np.asarray(self.model_prices(params), dtype=float) - self.prices

    # ------------------------------------------------------------------
    # DataFrame helpers
    def model_prices_df(self, params: ParameterVector) -> pd.DataFrame:
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

    def residuals_df(self, params: ParameterVector) -> pd.DataFrame:
        """Return residuals between model and market prices as a DataFrame."""
        df = self.model_prices_df(params)
        df["residual"] = df["model_price"] - self.prices
        return df

    def calibrate(
        self,
        initial_guess: ParameterVector,
        *,
        bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None = None,
        weights: Sequence[float] | None = None,
        loss: str = "linear",
    ) -> CalibrationResult:
        """Calibrate model parameters via SciPy least squares.

        Parameters
        ----------
        initial_guess:
            Initial parameter vector for the solver.
        bounds:
            Optional lower/upper bounds accepted by ``scipy.optimize.least_squares``.
        weights:
            Optional residual weights. The weighted residuals drive the optimizer;
            the result stores unweighted residuals for auditability.
        loss:
            SciPy robust loss name.

        Returns
        -------
        CalibrationResult
            Parameters plus termination, residual, rank, conditioning and bounds
            diagnostics. A result object, rather than a bare vector, forces callers
            to inspect fit quality.
        """

        x0 = np.asarray(initial_guess, dtype=float)
        if x0.ndim != 1:
            raise ValueError("initial_guess must be a one-dimensional parameter vector")
        normalized_bounds = self._normalize_bounds(bounds, x0.shape)
        weights_array = self._normalize_weights(weights)

        def objective(params: np.ndarray) -> np.ndarray:
            raw_residuals = self.residuals(params)
            if raw_residuals.shape != self.prices.shape:
                raise ValueError(
                    "model_prices must return an array with the same shape as market prices"
                )
            if weights_array is None:
                return raw_residuals
            return raw_residuals * weights_array

        result = least_squares(
            objective,
            x0=x0,
            bounds=normalized_bounds,
            loss=loss,
        )
        residuals = self.residuals(result.x)
        rank, condition = self._jacobian_rank_condition(result.jac)
        return CalibrationResult(
            parameters=np.asarray(result.x, dtype=float),
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            residuals=np.asarray(residuals, dtype=float),
            cost=float(result.cost),
            optimality=float(result.optimality),
            jacobian_rank=rank,
            jacobian_condition=condition,
            bounds=normalized_bounds,
            active_mask=np.asarray(result.active_mask, dtype=int),
            nfev=int(result.nfev),
            njev=None if result.njev is None else int(result.njev),
            method="scipy.least_squares",
            parameter_names=self.parameter_names,
        )

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None,
        shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        if bounds is None:
            return (
                np.full(shape, -np.inf, dtype=float),
                np.full(shape, np.inf, dtype=float),
            )
        lower, upper = bounds
        lower_arr = np.broadcast_to(np.asarray(lower, dtype=float), shape).copy()
        upper_arr = np.broadcast_to(np.asarray(upper, dtype=float), shape).copy()
        if np.any(lower_arr > upper_arr):
            raise ValueError("lower calibration bounds must not exceed upper bounds")
        return lower_arr, upper_arr

    def _normalize_weights(self, weights: Sequence[float] | None) -> np.ndarray | None:
        if weights is None:
            return None
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != self.prices.shape:
            raise ValueError("weights must have the same shape as market prices")
        if np.any(weights_array < 0):
            raise ValueError("weights must be non-negative")
        return weights_array

    @staticmethod
    def _jacobian_rank_condition(jacobian: np.ndarray) -> tuple[int, float]:
        jac = np.asarray(jacobian, dtype=float)
        if jac.ndim != 2 or jac.size == 0:
            return 0, np.inf
        singular_values = np.linalg.svd(jac, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] == 0:
            return 0, np.inf
        tolerance = np.finfo(float).eps * max(jac.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))
        if rank < min(jac.shape) or singular_values[-1] <= tolerance:
            return rank, np.inf
        return rank, float(singular_values[0] / singular_values[-1])
