"""JSON-safe QuantLib oracle contracts for quanto adoption evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Any, Literal

from finite_element_options.examples.regime_switching_quanto._types import json_safe

OracleKind = Literal["vanilla", "fixed_fx_quanto"]
SUPPORTED_CALENDARS = ("TARGET",)
SUPPORTED_DAY_COUNTS = ("Actual365Fixed",)
SUPPORTED_BUSINESS_DAY_CONVENTIONS = ("Unadjusted",)
SUPPORTED_RATE_COMPOUNDING = ("Continuous",)
SUPPORTED_EXERCISES = ("European",)
SUPPORTED_OPTION_TYPES = ("call",)
SUPPORTED_KINDS = ("vanilla", "fixed_fx_quanto")


class QuantLibConventionError(ValueError):
    """Actionable failure for unsupported QuantLib oracle conventions."""

    def __init__(self, *, field: str, received: object, supported: tuple[str, ...]) -> None:
        """Store the unsupported field, received value, and supported values."""

        self.field = field
        self.received = received
        self.supported = supported
        super().__init__(
            f"unsupported QuantLib oracle convention for {field}: "
            f"received {received!r}; supported values: {', '.join(supported)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic representation."""

        return {
            "field": self.field,
            "received": json_safe(self.received),
            "supported": list(self.supported),
        }


class QuantLibReductionError(ValueError):
    """Actionable failure for inconsistent one-regime oracle reductions."""

    def __init__(
        self,
        *,
        field: str,
        received: object,
        expected: object,
        kind: str,
    ) -> None:
        """Store the inconsistent field, received value, expected value, and kind."""

        self.field = field
        self.received = received
        self.expected = expected
        self.kind = kind
        super().__init__(
            f"invalid QuantLib oracle {kind!r} reduction for {field}: "
            f"received {received!r}; expected {expected!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic representation."""

        return {
            "field": self.field,
            "received": json_safe(self.received),
            "expected": json_safe(self.expected),
            "kind": json_safe(self.kind),
        }


@dataclass(frozen=True)
class QuantLibOracleSpec:
    """Immutable stdlib-only input contract for vanilla or fixed-FX quanto calls."""

    evaluation_date: date
    maturity_date: date
    spot: float
    strike: float
    equity_vol: float
    domestic_rate: float
    foreign_rate: float
    dividend_yield: float
    fx_vol: float
    correlation: float
    fixed_fx: float
    kind: OracleKind = "vanilla"
    calendar: str = "TARGET"
    day_count: str = "Actual365Fixed"
    business_day_convention: str = "Unadjusted"
    rate_compounding: str = "Continuous"
    exercise: str = "European"
    option_type: str = "call"

    def __post_init__(self) -> None:
        """Validate finite numeric laws and supported conventions."""

        _ensure_supported("calendar", self.calendar, SUPPORTED_CALENDARS)
        _ensure_supported("day_count", self.day_count, SUPPORTED_DAY_COUNTS)
        _ensure_supported(
            "business_day_convention",
            self.business_day_convention,
            SUPPORTED_BUSINESS_DAY_CONVENTIONS,
        )
        _ensure_supported("rate_compounding", self.rate_compounding, SUPPORTED_RATE_COMPOUNDING)
        _ensure_supported("exercise", self.exercise, SUPPORTED_EXERCISES)
        _ensure_supported("option_type", self.option_type, SUPPORTED_OPTION_TYPES)
        _ensure_supported("kind", self.kind, SUPPORTED_KINDS)
        if not isinstance(self.evaluation_date, date) or not isinstance(self.maturity_date, date):
            raise TypeError("evaluation_date and maturity_date must be datetime.date values")
        if self.maturity_date <= self.evaluation_date:
            raise ValueError("maturity_date must be after evaluation_date")
        _require_positive_finite("spot", self.spot)
        _require_positive_finite("strike", self.strike)
        _require_positive_finite("equity_vol", self.equity_vol)
        _require_nonnegative_finite("fx_vol", self.fx_vol)
        _require_positive_finite("fixed_fx", self.fixed_fx)
        for field in ("domestic_rate", "foreign_rate", "dividend_yield"):
            _require_finite(field, getattr(self, field))
        if not math.isfinite(float(self.correlation)) or abs(float(self.correlation)) > 1.0:
            raise ValueError("correlation must be finite and lie in [-1, 1]")
        _ensure_reduction_invariants(self)

    @property
    def quanto_adjustment(self) -> float:
        """Return rho * sigma_S * sigma_FX for the spec."""

        if self.kind == "vanilla":
            return 0.0
        return float(self.correlation) * float(self.equity_vol) * float(self.fx_vol)

    @property
    def effective_dividend_yield(self) -> float:
        """Return the BSM-equivalent dividend yield for this oracle kind."""

        if self.kind == "vanilla":
            return float(self.dividend_yield)
        return (
            float(self.dividend_yield)
            + float(self.domestic_rate)
            - float(self.foreign_rate)
            + self.quanto_adjustment
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe immutable spec payload."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class QuantLibOracleResult:
    """JSON-safe output from the isolated QuantLib oracle adapter."""

    schema_version: str
    spec: dict[str, Any]
    price: float
    quantlib_npv: float
    analytical_price: float
    analytical_absolute_error: float
    analytical_tolerance: float
    analytical_passed: bool
    year_fraction: float
    quanto_adjustment: float
    effective_dividend_yield: float
    quantlib_version: str
    conventions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result representation."""

        return json_safe(asdict(self))


def _ensure_supported(field: str, received: object, supported: tuple[str, ...]) -> None:
    if received not in supported:
        raise QuantLibConventionError(field=field, received=received, supported=supported)


def _ensure_reduction_invariants(spec: QuantLibOracleSpec) -> None:
    if spec.kind != "vanilla":
        return
    expected_by_field = {
        "foreign_rate": float(spec.domestic_rate),
        "fx_vol": 0.0,
        "correlation": 0.0,
        "fixed_fx": 1.0,
    }
    for field, expected in expected_by_field.items():
        received = float(getattr(spec, field))
        if received != expected:
            raise QuantLibReductionError(
                field=field,
                received=received,
                expected=expected,
                kind=spec.kind,
            )


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
    "OracleKind",
    "QuantLibConventionError",
    "QuantLibOracleResult",
    "QuantLibOracleSpec",
    "QuantLibReductionError",
    "SUPPORTED_BUSINESS_DAY_CONVENTIONS",
    "SUPPORTED_CALENDARS",
    "SUPPORTED_DAY_COUNTS",
    "SUPPORTED_EXERCISES",
    "SUPPORTED_KINDS",
    "SUPPORTED_OPTION_TYPES",
    "SUPPORTED_RATE_COMPOUNDING",
]
