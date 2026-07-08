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
from typing import ClassVar, Protocol, TypeAlias

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ParameterVector: TypeAlias = Sequence[float] | np.ndarray


class PricingFunction(Protocol):
    """Callable pricing-engine contract for pricing-model calibration."""

    def __call__(
        self,
        params: np.ndarray,
        dataset: PricingCalibrationDataset,
    ) -> np.ndarray:
        """Return model quotes for ``params`` on ``dataset``."""
        raise NotImplementedError


_ALLOWED_RESIDUAL_UNITS = frozenset({"price", "implied_volatility"})
_ALLOWED_WEIGHT_POLICIES = frozenset({"none", "explicit", "bid_ask", "vega"})


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


class CalibrationPricingError(ValueError):
    """Raised when a pricing engine violates the calibration contract."""


@dataclass(frozen=True)
class CalibrationObjective:
    """Objective controls for diagnosable pricing calibration.

    ``residual_units`` names the unit returned by the pricing engine and market
    quotes. ``weight_policy`` controls how residuals are scaled before entering
    SciPy's optimizer, while ``robust_loss`` is passed through to
    :func:`scipy.optimize.least_squares`.
    """

    residual_units: str = "price"
    weight_policy: str = "none"
    robust_loss: str = "linear"
    f_scale: float = 1.0
    min_scale: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate objective metadata before an optimizer is allocated."""

        if self.residual_units not in _ALLOWED_RESIDUAL_UNITS:
            raise ValueError(f"residual_units must be one of {sorted(_ALLOWED_RESIDUAL_UNITS)}")
        if self.weight_policy not in _ALLOWED_WEIGHT_POLICIES:
            raise ValueError(f"weight_policy must be one of {sorted(_ALLOWED_WEIGHT_POLICIES)}")
        if not np.isfinite(self.f_scale) or self.f_scale <= 0.0:
            raise ValueError("f_scale must be finite and strictly positive")
        if not np.isfinite(self.min_scale) or self.min_scale <= 0.0:
            raise ValueError("min_scale must be finite and strictly positive")

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly objective metadata."""

        return {
            "residual_units": self.residual_units,
            "weight_policy": self.weight_policy,
            "robust_loss": self.robust_loss,
            "f_scale": self.f_scale,
            "min_scale": self.min_scale,
        }


@dataclass(frozen=True)
class PricingCalibrationDataset:
    """Vectorized market quotes used by pricing-model calibration.

    The dataset is intentionally explicit about spot, strike, maturity, rate,
    carry and quote units. Optional bid/ask, explicit weights and vega columns
    are validated before a requested objective uses them.
    """

    spot: np.ndarray
    strike: np.ndarray
    maturity: np.ndarray
    rate: np.ndarray
    carry: np.ndarray
    quote: np.ndarray
    quote_units: str = "price"
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    weights: np.ndarray | None = None
    vega: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce and validate all observation arrays."""

        if self.quote_units not in _ALLOWED_RESIDUAL_UNITS:
            raise ValueError(f"quote_units must be one of {sorted(_ALLOWED_RESIDUAL_UNITS)}")
        shape: tuple[int, ...] | None = None
        for name in ("spot", "strike", "maturity", "rate", "carry", "quote"):
            values = self._coerce_required_array(name, getattr(self, name))
            object.__setattr__(self, name, values)
            if shape is None:
                shape = values.shape
            elif values.shape != shape:
                raise ValueError("pricing calibration arrays must share one shape")
        assert shape is not None
        if shape[0] == 0:
            raise ValueError("pricing calibration dataset must not be empty")
        if np.any(self.spot <= 0.0):
            raise ValueError("spot values must be strictly positive")
        if np.any(self.strike <= 0.0):
            raise ValueError("strike values must be strictly positive")
        if np.any(self.maturity <= 0.0):
            raise ValueError("maturity values must be strictly positive")
        for name in ("bid", "ask", "weights", "vega"):
            values = self._coerce_optional_array(name, getattr(self, name), shape)
            object.__setattr__(self, name, values)

    @staticmethod
    def _coerce_required_array(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} values must be finite")
        return arr.copy()

    @staticmethod
    def _coerce_optional_array(
        name: str,
        values: Sequence[float] | np.ndarray | None,
        shape: tuple[int, ...],
    ) -> np.ndarray | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=float)
        if arr.shape != shape:
            raise ValueError(f"{name} must have the same shape as quote")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} values must be finite")
        return arr.copy()

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        quote_units: str = "price",
        metadata: Mapping[str, object] | None = None,
    ) -> "PricingCalibrationDataset":
        """Build a calibration dataset from a typed market quote DataFrame."""

        required = {"spot", "strike", "maturity", "rate", "carry", "quote"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"market frame is missing {sorted(missing)}")

        def optional(*columns: str) -> np.ndarray | None:
            for column in columns:
                if column in frame.columns:
                    return frame[column].to_numpy(dtype=float)
            return None

        return cls(
            spot=frame["spot"].to_numpy(dtype=float),
            strike=frame["strike"].to_numpy(dtype=float),
            maturity=frame["maturity"].to_numpy(dtype=float),
            rate=frame["rate"].to_numpy(dtype=float),
            carry=frame["carry"].to_numpy(dtype=float),
            quote=frame["quote"].to_numpy(dtype=float),
            quote_units=quote_units,
            bid=optional("bid"),
            ask=optional("ask"),
            weights=optional("weights", "weight"),
            vega=optional("vega"),
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the common observation-vector shape."""

        return self.quote.shape


@dataclass
class PricingModelCalibrator:
    """Bounded, weighted and diagnosable pricing-model calibrator.

    The pricing function is injected so FEM/PDE/Fourier engines can be validated
    separately. Calibration only claims success when SciPy converges, every
    pricing call returns finite values with the correct shape, and fitted
    parameters satisfy declared bounds.
    """

    dataset: PricingCalibrationDataset
    pricing_function: PricingFunction
    parameter_names: tuple[str, ...]
    bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None = None
    method_name: str = "scipy.least_squares.pricing"
    metadata: Mapping[str, object] = field(default_factory=dict)
    _pricing_evaluations: int = field(default=0, init=False)
    _pricing_failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate parameter metadata."""

        if not self.parameter_names:
            raise ValueError("parameter_names must not be empty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique")

    def calibrate(
        self,
        initial_guess: ParameterVector,
        *,
        objective: CalibrationObjective | None = None,
        holdout_mask: Sequence[bool] | np.ndarray | None = None,
        candidate_initial_guesses: Sequence[ParameterVector] = (),
        max_nfev: int | None = None,
    ) -> CalibrationResult:
        """Run bounded least-squares calibration with explicit diagnostics."""

        if objective is None:
            objective = CalibrationObjective(residual_units=self.dataset.quote_units)
        elif objective.residual_units != self.dataset.quote_units:
            raise ValueError("objective residual_units must match dataset quote_units")
        x0 = np.asarray(initial_guess, dtype=float)
        if x0.ndim != 1:
            raise ValueError("initial_guess must be a one-dimensional parameter vector")
        if x0.shape != (len(self.parameter_names),):
            raise ValueError("initial_guess shape must match parameter_names")
        if not np.all(np.isfinite(x0)):
            raise ValueError("initial_guess must be finite")
        normalized_bounds = Calibrator._normalize_bounds(self.bounds, x0.shape)
        train_mask, holdout = self._split_masks(holdout_mask)
        weights, weight_diagnostics = self._objective_weights(objective, train_mask)
        self._pricing_evaluations = 0
        self._pricing_failures = 0

        def residual_objective(params: np.ndarray) -> np.ndarray:
            prices = self._evaluate_prices(params)
            raw_residuals = prices - self.dataset.quote
            if not np.all(np.isfinite(raw_residuals)):
                self._pricing_failures += 1
                raise CalibrationPricingError("nonfinite calibration residuals")
            return raw_residuals[train_mask] * weights

        optimizer_results = [
            least_squares(
                residual_objective,
                x0=start,
                bounds=normalized_bounds,
                loss=objective.robust_loss,
                f_scale=objective.f_scale,
                max_nfev=max_nfev,
            )
            for start in self._candidate_initial_guesses(x0, candidate_initial_guesses)
        ]
        optimizer_result = min(
            optimizer_results,
            key=lambda item: (not bool(item.success), float(item.cost)),
        )
        multistart_diagnostics = tuple(
            {
                "start_index": index,
                "success": bool(candidate.success),
                "status": int(candidate.status),
                "cost": float(candidate.cost),
                "optimality": float(candidate.optimality),
                "nfev": int(candidate.nfev),
            }
            for index, candidate in enumerate(optimizer_results)
        )
        fitted_prices = self._evaluate_prices(optimizer_result.x)
        residuals = np.asarray(fitted_prices - self.dataset.quote, dtype=float)
        rank, condition = Calibrator._jacobian_rank_condition(optimizer_result.jac)
        in_bounds = self._parameters_within_bounds(optimizer_result.x, normalized_bounds)
        success = bool(optimizer_result.success and in_bounds and self._pricing_failures == 0)
        if not success and optimizer_result.success and not in_bounds:
            message = "optimizer converged outside declared parameter bounds"
        else:
            message = str(optimizer_result.message)
        diagnostics = self._diagnostics(
            objective=objective,
            train_mask=train_mask,
            holdout_mask=holdout,
            residuals=residuals,
            weighted_training_residuals=residuals[train_mask] * weights,
            weight_diagnostics=weight_diagnostics,
            optimizer_result=optimizer_result,
            in_bounds=in_bounds,
            multistart_diagnostics=multistart_diagnostics,
        )
        return CalibrationResult(
            parameters=np.asarray(optimizer_result.x, dtype=float),
            success=success,
            status=int(optimizer_result.status),
            message=message,
            residuals=residuals,
            cost=float(optimizer_result.cost),
            optimality=float(optimizer_result.optimality),
            jacobian_rank=rank,
            jacobian_condition=condition,
            bounds=normalized_bounds,
            active_mask=np.asarray(optimizer_result.active_mask, dtype=int),
            nfev=int(optimizer_result.nfev),
            njev=None if optimizer_result.njev is None else int(optimizer_result.njev),
            method=self.method_name,
            parameter_names=self.parameter_names,
            diagnostics=diagnostics,
            provenance=dict(self.metadata),
        )

    def _candidate_initial_guesses(
        self,
        initial_guess: np.ndarray,
        candidate_initial_guesses: Sequence[ParameterVector],
    ) -> tuple[np.ndarray, ...]:
        """Return validated deterministic multi-start initial guesses."""

        starts = [np.asarray(initial_guess, dtype=float)]
        for index, candidate in enumerate(candidate_initial_guesses, start=1):
            candidate_array = np.asarray(candidate, dtype=float)
            if candidate_array.shape != initial_guess.shape:
                raise ValueError(
                    f"candidate_initial_guesses[{index}] must match initial_guess shape"
                )
            if not np.all(np.isfinite(candidate_array)):
                raise ValueError(f"candidate_initial_guesses[{index}] must be finite")
            starts.append(candidate_array)
        return tuple(starts)

    def _evaluate_prices(self, params: np.ndarray) -> np.ndarray:
        try:
            prices = np.asarray(
                self.pricing_function(np.asarray(params, dtype=float), self.dataset),
                dtype=float,
            )
        except Exception as exc:  # pragma: no cover - source exception is engine-specific
            self._pricing_failures += 1
            raise CalibrationPricingError(f"pricing engine failed: {exc}") from exc
        self._pricing_evaluations += 1
        if prices.shape != self.dataset.shape:
            self._pricing_failures += 1
            raise CalibrationPricingError("pricing engine returned the wrong shape")
        if not np.all(np.isfinite(prices)):
            self._pricing_failures += 1
            raise CalibrationPricingError("pricing engine returned nonfinite prices")
        return prices

    def _split_masks(
        self,
        holdout_mask: Sequence[bool] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if holdout_mask is None:
            holdout = np.zeros(self.dataset.shape, dtype=bool)
        else:
            holdout = np.asarray(holdout_mask, dtype=bool)
            if holdout.shape != self.dataset.shape:
                raise ValueError("holdout_mask must have the same shape as quote")
        train = ~holdout
        if not np.any(train):
            raise ValueError("at least one training observation is required")
        return train, holdout

    def _objective_weights(
        self,
        objective: CalibrationObjective,
        train_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        policy = objective.weight_policy
        source = policy
        if policy == "none":
            all_weights = np.ones(self.dataset.shape, dtype=float)
        elif policy == "explicit":
            if self.dataset.weights is None:
                raise ValueError("explicit weight_policy requires a weight column")
            all_weights = self.dataset.weights
        elif policy == "bid_ask":
            if self.dataset.bid is None or self.dataset.ask is None:
                raise ValueError("bid_ask weight_policy requires bid and ask columns")
            if np.any(self.dataset.bid >= self.dataset.ask):
                raise ValueError("bid must be less than ask for bid_ask weights")
            all_weights = 1.0 / np.maximum(self.dataset.ask - self.dataset.bid, objective.min_scale)
        elif policy == "vega":
            if self.dataset.vega is None:
                raise ValueError("vega weight_policy requires a vega column")
            if np.any(self.dataset.vega <= 0.0):
                raise ValueError("vega values must be strictly positive")
            scaled_vega = np.maximum(self.dataset.vega, objective.min_scale)
            if objective.residual_units == "price":
                all_weights = 1.0 / scaled_vega
            else:
                all_weights = scaled_vega
        else:  # pragma: no cover - objective validates this before dispatch
            raise ValueError(f"unsupported weight policy {policy}")
        if not np.all(np.isfinite(all_weights)) or np.any(all_weights <= 0.0):
            raise ValueError("calibration weights must be finite and strictly positive")
        weights = np.asarray(all_weights[train_mask], dtype=float)
        return weights, {
            "source": source,
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
        }

    @staticmethod
    def _parameters_within_bounds(
        parameters: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        lower, upper = bounds
        return bool(np.all(parameters >= lower - 1.0e-10) and np.all(parameters <= upper + 1.0e-10))

    def _diagnostics(
        self,
        *,
        objective: CalibrationObjective,
        train_mask: np.ndarray,
        holdout_mask: np.ndarray,
        residuals: np.ndarray,
        weighted_training_residuals: np.ndarray,
        weight_diagnostics: Mapping[str, object],
        optimizer_result: object,
        in_bounds: bool,
        multistart_diagnostics: tuple[Mapping[str, object], ...],
    ) -> dict[str, object]:
        train_residuals = residuals[train_mask]
        holdout_residuals = residuals[holdout_mask]
        holdout_rmse = (
            None if holdout_residuals.size == 0 else float(np.sqrt(np.mean(holdout_residuals**2)))
        )
        return {
            "objective": objective.as_dict(),
            "observations": {
                "total_count": int(residuals.size),
                "training_count": int(np.sum(train_mask)),
                "holdout_count": int(np.sum(holdout_mask)),
                "quote_units": self.dataset.quote_units,
            },
            "weights": dict(weight_diagnostics),
            "fit": {
                "training_rmse": float(np.sqrt(np.mean(train_residuals**2))),
                "holdout_rmse": holdout_rmse,
                "max_abs_residual": float(np.max(np.abs(residuals))),
                "weighted_training_norm": float(np.linalg.norm(weighted_training_residuals)),
            },
            "pricing_engine": {
                "pricing_evaluations": int(self._pricing_evaluations),
                "pricing_failures": int(self._pricing_failures),
            },
            "optimizer": {
                "success": bool(getattr(optimizer_result, "success")),
                "status": int(getattr(optimizer_result, "status")),
                "message": str(getattr(optimizer_result, "message")),
                "in_declared_bounds": bool(in_bounds),
                "start_count": len(multistart_diagnostics),
                "multi_start": [dict(item) for item in multistart_diagnostics],
            },
        }


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
