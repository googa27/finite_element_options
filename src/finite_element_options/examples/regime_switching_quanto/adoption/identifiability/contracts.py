"""JSON-safe contracts and objective for iminuit identifiability evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Any, Literal

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.examples.regime_switching_quanto._types import json_safe

PARAMETERS = ("equity_vol", "correlation")
SCHEMA_VERSION = "regime_iminuit_identifiability.v1"
OBJECTIVE_NAME = "market_calibration_objective"
TARGET_SOURCE = "public-synthetic"

MissingReason = Literal["not_run", "optimizer_failure", "not_available", "invalid"]


@dataclass(frozen=True)
class ParameterBounds:
    """Closed calibration bounds for equity volatility and correlation."""

    equity_vol: tuple[float, float]
    correlation: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate parameter bounds."""

        _validate_bounds("equity_vol", self.equity_vol, lower_positive=True)
        _validate_bounds("correlation", self.correlation, lower_positive=False)
        if self.correlation[0] < -1.0 or self.correlation[1] > 1.0:
            raise ValueError("correlation bounds must lie inside [-1, 1]")

    def for_parameter(self, name: str) -> tuple[float, float]:
        """Return bounds for an optimizer parameter name."""

        if name == "equity_vol":
            return self.equity_vol
        if name == "correlation":
            return self.correlation
        raise KeyError(f"unknown parameter {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class ProfileGrid:
    """Explicit deterministic profile-likelihood grid for one parameter."""

    parameter: str
    lower: float
    upper: float
    points: int

    def __post_init__(self) -> None:
        """Validate a bounded profile grid."""

        if self.parameter not in PARAMETERS:
            raise ValueError(f"unsupported profile parameter {self.parameter!r}")
        _require_finite("profile lower", self.lower)
        _require_finite("profile upper", self.upper)
        if float(self.lower) >= float(self.upper):
            raise ValueError("profile lower must be less than upper")
        if int(self.points) < 5 or int(self.points) > 101:
            raise ValueError("profile points must be between 5 and 101")

    def values(self) -> list[float]:
        """Return deterministic grid values including both endpoints."""

        if self.points == 1:
            return [float(self.lower)]
        step = (float(self.upper) - float(self.lower)) / float(self.points - 1)
        return [float(self.lower) + step * i for i in range(self.points)]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class QuantoOptionTarget:
    """Immutable public-synthetic fixed-FX quanto call calibration target."""

    instrument_id: str
    evaluation_date: date
    maturity_date: date
    spot: float
    strike: float
    fixed_fx: float
    domestic_rate: float
    foreign_rate: float
    dividend_yield: float
    fx_vol: float
    target_price: float
    price_std: float
    source: str = TARGET_SOURCE
    option_type: str = "call"

    def __post_init__(self) -> None:
        """Validate date, source, positivity, range, and finite-value laws."""

        if not str(self.instrument_id).strip():
            raise ValueError("instrument_id must be nonempty")
        if self.source != TARGET_SOURCE:
            raise ValueError("source must be public-synthetic")
        if self.option_type != "call":
            raise ValueError("only call targets are supported")
        if not isinstance(self.evaluation_date, date) or not isinstance(self.maturity_date, date):
            raise TypeError("evaluation_date and maturity_date must be datetime.date")
        if self.maturity_date <= self.evaluation_date:
            raise ValueError("maturity_date must be after evaluation_date")
        for field in ("spot", "strike", "fixed_fx", "price_std"):
            _require_positive_finite(field, getattr(self, field))
        for field in ("domestic_rate", "foreign_rate", "dividend_yield"):
            _require_finite(field, getattr(self, field))
        _require_nonnegative_finite("fx_vol", self.fx_vol)
        _require_nonnegative_finite("target_price", self.target_price)

    @property
    def year_fraction(self) -> float:
        """Return Actual365Fixed-equivalent year fraction."""

        return (self.maturity_date - self.evaluation_date).days / 365.0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe immutable target payload."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class CalibrationCase:
    """Immutable bounded iminuit calibration case."""

    case_id: str
    description: str
    targets: tuple[QuantoOptionTarget, ...]
    initial_equity_vol: float
    initial_correlation: float
    bounds: ParameterBounds
    edm_threshold: float
    bound_contact_tolerance: float
    minos_cl: float
    minos_ncall: int
    profile_grids: tuple[ProfileGrid, ...]
    finite_difference_step: float
    synthetic_truth: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate supplied targets and numerical controls."""

        if not str(self.case_id).strip():
            raise ValueError("case_id must be nonempty")
        if not self.targets:
            raise ValueError("calibration targets must be nonempty")
        _require_positive_finite("initial_equity_vol", self.initial_equity_vol)
        _require_finite("initial_correlation", self.initial_correlation)
        _require_positive_finite("edm_threshold", self.edm_threshold)
        _require_positive_finite("bound_contact_tolerance", self.bound_contact_tolerance)
        _require_positive_finite("minos_cl", self.minos_cl)
        if not 0.0 < float(self.minos_cl) < 1.0:
            raise ValueError("minos_cl must lie in (0, 1)")
        if int(self.minos_ncall) < 1:
            raise ValueError("minos_ncall must be positive")
        _require_positive_finite("finite_difference_step", self.finite_difference_step)
        _validate_initial_inside_bounds(self)
        if {grid.parameter for grid in self.profile_grids} != set(PARAMETERS):
            raise ValueError("profile_grids must contain exactly both free parameters")
        for grid in self.profile_grids:
            bounds = self.bounds.for_parameter(grid.parameter)
            if grid.lower < bounds[0] or grid.upper > bounds[1]:
                raise ValueError("profile grids must lie inside parameter bounds")

    @property
    def initial_values(self) -> dict[str, float]:
        """Return initial optimizer values by public parameter name."""

        return {
            "equity_vol": float(self.initial_equity_vol),
            "correlation": float(self.initial_correlation),
        }

    def to_input_dict(self) -> dict[str, Any]:
        """Return JSON-safe input fields, including synthetic-truth metadata only."""

        return {
            "case_id": self.case_id,
            "description": self.description,
            "targets": [target.to_dict() for target in self.targets],
            "initial_values": self.initial_values,
            "bounds": self.bounds.to_dict(),
            "edm_threshold": self.edm_threshold,
            "bound_contact_tolerance": self.bound_contact_tolerance,
            "minos_cl": self.minos_cl,
            "minos_ncall": self.minos_ncall,
            "profile_grids": [grid.to_dict() for grid in self.profile_grids],
            "finite_difference_step": self.finite_difference_step,
            "synthetic_truth_metadata": json_safe(self.synthetic_truth or {}),
        }


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """JSON-safe objective evaluation with call accounting and diagnostics."""

    chi2: float | str
    finite: bool
    call_count: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe evaluation payload without NaN values."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class IdentifiabilityResult:
    """JSON-safe public result for one profile-likelihood identifiability run."""

    schema_version: str
    case_input_hash: str
    case: dict[str, Any]
    objective_name: str
    optimizer: dict[str, Any]
    minimum: dict[str, Any]
    hesse: dict[str, Any]
    minos: dict[str, Any]
    boundary_contact: dict[str, Any]
    finite_difference: dict[str, Any]
    profiles: dict[str, Any]
    identification: dict[str, Any]
    objective_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result representation."""

        return json_safe(asdict(self))


class WeightedQuantoCalibrationObjective:
    """Deterministic weighted least-squares fixed-FX quanto objective."""

    def __init__(self, case: CalibrationCase) -> None:
        """Store a validated calibration case and reset call diagnostics."""

        case.__post_init__()
        self.case = case
        self.call_count = 0
        self.failure_counts: dict[str, int] = {}
        self.last_diagnostic: dict[str, Any] = {"status": "not_evaluated"}

    def __call__(self, equity_vol: float, correlation: float) -> float:
        """Return chi-square or infinity for iminuit."""

        return float(self.evaluate(equity_vol, correlation).chi2)

    def evaluate(self, equity_vol: float, correlation: float) -> ObjectiveEvaluation:
        """Evaluate chi-square and record fail-closed diagnostics."""

        self.call_count += 1
        diagnostic = self._parameter_diagnostic(equity_vol, correlation)
        if diagnostic:
            return self._failed_evaluation(diagnostic)
        chi2 = 0.0
        contributions: list[dict[str, Any]] = []
        try:
            for target in self.case.targets:
                model_price = quanto_call_price(target, equity_vol, correlation)
                residual = (model_price - float(target.target_price)) / float(target.price_std)
                if not math.isfinite(residual):
                    return self._failed_evaluation(
                        {"reason": "nonfinite_residual", "instrument_id": target.instrument_id}
                    )
                term = residual * residual
                chi2 += term
                contributions.append(
                    {
                        "instrument_id": target.instrument_id,
                        "model_price": model_price,
                        "target_price": target.target_price,
                        "standardized_residual": residual,
                        "chi2_contribution": term,
                    }
                )
        except (OverflowError, ValueError) as exc:
            return self._failed_evaluation(
                {"reason": "pricing_exception", "exception_type": type(exc).__name__}
            )
        if not math.isfinite(chi2):
            return self._failed_evaluation({"reason": "nonfinite_chi2"})
        self.last_diagnostic = {"status": "finite", "contributions": contributions}
        return ObjectiveEvaluation(
            chi2=float(chi2),
            finite=True,
            call_count=self.call_count,
            diagnostics=self.last_diagnostic,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe aggregate objective diagnostics."""

        return {
            "call_count": self.call_count,
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "last_diagnostic": json_safe(self.last_diagnostic),
        }

    def _parameter_diagnostic(self, equity_vol: float, correlation: float) -> dict[str, Any] | None:
        for name, value in (
            ("equity_vol", equity_vol),
            ("correlation", correlation),
        ):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return {"reason": "nonfinite_parameter", "parameter": name}
            if not math.isfinite(numeric):
                return {"reason": "nonfinite_parameter", "parameter": name}
            lower, upper = self.case.bounds.for_parameter(name)
            if numeric < lower or numeric > upper:
                return {
                    "reason": "out_of_bounds_parameter",
                    "parameter": name,
                    "value": numeric,
                    "bounds": [lower, upper],
                }
        return None

    def _failed_evaluation(self, diagnostic: dict[str, Any]) -> ObjectiveEvaluation:
        reason = str(diagnostic.get("reason", "unknown"))
        self.failure_counts[reason] = self.failure_counts.get(reason, 0) + 1
        self.last_diagnostic = {"status": "failed", **diagnostic}
        return ObjectiveEvaluation(
            chi2=math.inf,
            finite=False,
            call_count=self.call_count,
            diagnostics=self.last_diagnostic,
        )


def quanto_call_price(target: QuantoOptionTarget, equity_vol: float, correlation: float) -> float:
    """Return fixed-FX quanto call price from the core analytical BS oracle."""

    q_eff = (
        float(target.dividend_yield)
        + float(target.domestic_rate)
        - float(target.foreign_rate)
        + float(correlation) * float(equity_vol) * float(target.fx_vol)
    )
    option = EuropeanOptionBs(
        k=float(target.strike),
        q=q_eff,
        mkt=Market(r=float(target.domestic_rate)),
    )
    raw = option.call_from_volatility(
        target.year_fraction,
        float(target.spot),
        float(equity_vol),
    )
    return float(raw) * float(target.fixed_fx)


def _validate_initial_inside_bounds(case: CalibrationCase) -> None:
    for name, value in case.initial_values.items():
        lower, upper = case.bounds.for_parameter(name)
        if value < lower or value > upper:
            raise ValueError(f"initial {name} must lie inside bounds")


def _validate_bounds(field: str, value: tuple[float, float], *, lower_positive: bool) -> None:
    if len(value) != 2:
        raise ValueError(f"{field} bounds must have length 2")
    lower, upper = float(value[0]), float(value[1])
    _require_finite(f"{field} lower bound", lower)
    _require_finite(f"{field} upper bound", upper)
    if lower >= upper:
        raise ValueError(f"{field} lower bound must be less than upper bound")
    if lower_positive and lower <= 0.0:
        raise ValueError(f"{field} lower bound must be positive")


def _require_finite(field: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")


def _require_positive_finite(field: str, value: float) -> None:
    _require_finite(field, value)
    if float(value) <= 0.0:
        raise ValueError(f"{field} must be positive")


def _require_nonnegative_finite(field: str, value: float) -> None:
    _require_finite(field, value)
    if float(value) < 0.0:
        raise ValueError(f"{field} must be non-negative")


__all__ = [
    "CalibrationCase",
    "IdentifiabilityResult",
    "MissingReason",
    "OBJECTIVE_NAME",
    "ObjectiveEvaluation",
    "PARAMETERS",
    "ParameterBounds",
    "ProfileGrid",
    "QuantoOptionTarget",
    "SCHEMA_VERSION",
    "TARGET_SOURCE",
    "WeightedQuantoCalibrationObjective",
    "quanto_call_price",
]
