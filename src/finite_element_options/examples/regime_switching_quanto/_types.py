"""Typed result records for regime-switching quanto research calibration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from finite_element_options.validation.evidence.serialization import json_safe


class FrozenMapping(Mapping[Any, Any]):
    """Recursively immutable mapping used by hash-bound public evidence contracts."""

    __slots__ = ("_items",)
    _items: tuple[tuple[Any, Any], ...]

    def __init__(self, value: Mapping[Any, Any]) -> None:
        """Freeze a mapping and all nested values."""

        object.__setattr__(
            self,
            "_items",
            tuple((key, deep_freeze(item)) for key, item in value.items()),
        )

    def __getitem__(self, key: Any) -> Any:
        """Return the immutable value stored for ``key``."""

        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys in source insertion order."""

        return (key for key, _ in self._items)

    def __len__(self) -> int:
        """Return the number of stored key-value pairs."""

        return len(self._items)

    def __hash__(self) -> int:
        """Return an order-independent content hash consistent with mapping equality."""

        return hash(frozenset(self._items))

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject attribute mutation after construction."""

        raise TypeError("FrozenMapping is immutable")

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenMapping:
        """Return self because the complete object graph is immutable."""

        return self


def deep_freeze(value: Any) -> Any:
    """Return recursively immutable mappings/tuples without changing scalar values."""

    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, np.ndarray):
        return deep_freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class DataQualityConfig:
    """Bounds and minimum sample controls for level-to-return preparation."""

    sp500_min: float = 100.0
    sp500_max: float = 20_000.0
    usdclp_min: float = 100.0
    usdclp_max: float = 2_000.0
    min_return_rows: int = 20
    date_column: str = "date"

    def __post_init__(self) -> None:
        """Validate explicit bounds."""

        if self.sp500_min <= 0 or self.usdclp_min <= 0:
            raise ValueError("level lower bounds must be positive")
        if self.sp500_min >= self.sp500_max:
            raise ValueError("sp500_min must be less than sp500_max")
        if self.usdclp_min >= self.usdclp_max:
            raise ValueError("usdclp_min must be less than usdclp_max")
        if self.min_return_rows < 1:
            raise ValueError("min_return_rows must be at least 1")


@dataclass(frozen=True)
class RegimeSummary:
    """Annualized weighted bivariate diffusion summary for one regime."""

    label: int
    occupancy: float
    daily_mean: list[float]
    annual_mean: list[float]
    annual_covariance: list[list[float]]
    annual_volatility: list[float]
    correlation: list[list[float]]
    composite_vol: float


@dataclass(frozen=True)
class RegimeCandidateResult:
    """Information criterion and duration diagnostics for a candidate k."""

    k_regimes: int
    llf: float
    aic: float
    bic: float
    converged: bool
    expected_durations: list[float | None]
    occupancies: list[float]
    autoregressive_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(self)


@dataclass(frozen=True)
class MarkovSwitchingDiffusionResult:
    """Research-only fitted discrete transition and diffusion diagnostics."""

    k_regimes: int
    regimes: list[RegimeSummary]
    transition_matrix: list[list[float]]
    continuous_time_generator: list[list[float]]
    generator_residual: float
    current_probabilities: list[float]
    occupancies: list[float]
    expected_durations: list[float | None]
    llf: float
    aic: float
    bic: float
    converged: bool
    residual_diagnostics: dict[str, Any]
    var1_diagnostics: dict[str, Any]
    autoregressive_order: int = 0
    ar_coefficients: list[list[float]] = field(default_factory=list)
    fit_attempt_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    model_note: str = field(
        default=(
            "P is the fitted daily transition matrix; generator is derived from P. "
            "Autoregression filters the physical-measure composite return but is not "
            "carried into the risk-neutral log-diffusion pricing dynamics."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with no NumPy or pandas objects."""

        return json_safe(self)
