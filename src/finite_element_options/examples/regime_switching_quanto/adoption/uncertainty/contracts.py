"""JSON-safe uncertainty contracts for the OpenTURNS FEM pilot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
    canonical_json_sha256,
    quantize_json_floats,
)
from finite_element_options.examples.regime_switching_quanto._types import (
    deep_freeze,
    json_safe,
)

ComponentName = Literal["data", "parameter", "model_form", "numerical", "monte_carlo"]
ComponentRole = Literal["fem_perturbation", "additive_validation_estimator_error"]
COMPONENT_NAMES: tuple[ComponentName, ...] = (
    "data",
    "parameter",
    "model_form",
    "numerical",
    "monte_carlo",
)
QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_lower_sha256(value: object) -> bool:
    """Return whether ``value`` is exactly one lowercase SHA-256 hex digest."""

    return (
        isinstance(value, str) and len(value) == 64 and all(ch in _LOWER_HEX_DIGITS for ch in value)
    )


def _freeze_fields(instance: Any, *names: str) -> None:
    """Recursively freeze named public evidence fields on a frozen dataclass."""

    for name in names:
        object.__setattr__(instance, name, deep_freeze(getattr(instance, name)))


def _quantized_dict(value: Any) -> dict[str, Any]:
    """Return ten-significant-digit JSON evidence for cross-platform replay."""

    payload = quantize_json_floats(json_safe(value), significant_digits=10)
    if not isinstance(payload, dict):
        raise TypeError("evidence payload must serialize to a dictionary")
    return payload


@dataclass(frozen=True)
class UncertaintyComponent:
    """One immutable named uncertainty component in normalized coordinates."""

    name: ComponentName
    distribution: str
    scale_or_range: Mapping[str, float | str]
    units: str
    role: ComponentRole
    source_identity: str
    source_hash: str
    perturbs_fem_model: bool
    additive_validation_estimator_error: bool
    description: str

    def __post_init__(self) -> None:
        """Validate component names, finite ranges, and hash-bound provenance."""

        _freeze_fields(self, "scale_or_range")
        if self.name not in COMPONENT_NAMES:
            raise ValueError(f"unexpected component name: {self.name}")
        if not _is_lower_sha256(self.source_hash):
            raise ValueError(f"component {self.name} must record a SHA-256 source hash")
        for key, value in self.scale_or_range.items():
            if isinstance(value, str):
                if not value:
                    raise ValueError(f"empty scale/range value for {self.name}.{key}")
            elif not np.isfinite(float(value)):
                raise ValueError(f"non-finite scale/range value for {self.name}.{key}")
        if self.perturbs_fem_model == self.additive_validation_estimator_error:
            raise ValueError(f"component {self.name} must choose exactly one perturbation role")
        expected_role = (
            "fem_perturbation" if self.perturbs_fem_model else "additive_validation_estimator_error"
        )
        if self.role != expected_role:
            raise ValueError(f"component {self.name} role must be {expected_role}")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe component data with no OpenTURNS objects."""

        return _quantized_dict(asdict(self))


@dataclass(frozen=True)
class UQPilotConfig:
    """Canonical public-synthetic pilot controls."""

    sample_seed: int = 134_101
    sample_size: int = 64
    sobol_seed: int = 134_201
    sobol_base_size: int = 128
    direct_seed: int = 134_301
    direct_size: int = 64
    component_seed: int = 134_401
    component_size: int = 33
    additive_sobol_seed: int = 134_501
    additive_sobol_base_size: int = 1024

    def __post_init__(self) -> None:
        """Validate deterministic sizes used by the bounded pilot."""

        for name, value in asdict(self).items():
            if name.endswith("_seed") and int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            if name.endswith("_size") and int(value) < 8:
                raise ValueError(f"{name} must be at least 8")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe controls."""

        return _quantized_dict(asdict(self))


@dataclass(frozen=True)
class UQCalibration:
    """Baseline deterministic and seeded-MC scale calibration."""

    baseline_price_fine: float
    baseline_price_coarse: float
    baseline_price_oracle: float
    fine_oracle_abs_error: float
    coarse_oracle_abs_error: float
    oracle_identity: str
    domain_error_grid: Mapping[str, Any]
    domain_error_grid_hash: str
    domain_max_fine_oracle_abs_error: float
    domain_max_error_input: Mapping[str, float]
    domain_error_safety_factor: float
    numerical_half_width: float
    numerical_formula: str
    mc_price: float
    mc_standard_error: float
    mc_seed: int
    mc_paths: int
    mc_steps: int
    mc_steps_per_year: int
    fine_grid: Mapping[str, Any]
    coarse_grid: Mapping[str, Any]
    fine_grid_hash: str
    coarse_grid_hash: str
    baseline_model_hash: str
    payoff_hash: str
    oracle_hash: str

    def __post_init__(self) -> None:
        """Require finite, separated numerical and Monte Carlo scale records."""

        _freeze_fields(
            self,
            "domain_error_grid",
            "domain_max_error_input",
            "fine_grid",
            "coarse_grid",
        )
        finite_values = (
            self.baseline_price_fine,
            self.baseline_price_coarse,
            self.baseline_price_oracle,
            self.fine_oracle_abs_error,
            self.coarse_oracle_abs_error,
            self.domain_max_fine_oracle_abs_error,
            self.domain_error_safety_factor,
            self.numerical_half_width,
            self.mc_price,
            self.mc_standard_error,
        )
        if not all(np.isfinite(value) for value in finite_values):
            raise ValueError("calibration values must be finite")
        if self.numerical_half_width < 0.0 or self.mc_standard_error <= 0.0:
            raise ValueError("calibration scales must be non-negative/positive as documented")
        if self.fine_oracle_abs_error < 0.0 or self.coarse_oracle_abs_error < 0.0:
            raise ValueError("analytical-oracle absolute errors must be non-negative")
        if self.domain_max_fine_oracle_abs_error <= 0.0 or self.domain_error_safety_factor < 1.0:
            raise ValueError(
                "domain error calibration and safety factor must be positive/conservative"
            )
        if self.numerical_half_width < max(
            self.fine_oracle_abs_error,
            self.coarse_oracle_abs_error,
            self.domain_error_safety_factor * self.domain_max_fine_oracle_abs_error,
        ):
            raise ValueError(
                "numerical half-width must cover baseline and domain analytical errors"
            )
        if not self.oracle_identity:
            raise ValueError("oracle_identity must be non-empty")
        for value in (
            self.fine_grid_hash,
            self.coarse_grid_hash,
            self.baseline_model_hash,
            self.payoff_hash,
            self.oracle_hash,
            self.domain_error_grid_hash,
        ):
            if not _is_lower_sha256(value):
                raise ValueError("calibration hashes must be lowercase SHA-256 hex strings")
        for label, grid, digest in (
            ("fine", self.fine_grid, self.fine_grid_hash),
            ("coarse", self.coarse_grid, self.coarse_grid_hash),
        ):
            payload = dict(grid)
            embedded = payload.pop("hash", None)
            if embedded != digest or canonical_json_sha256(payload) != digest:
                raise ValueError(f"{label} grid payload does not match its digest")
        if canonical_json_sha256(self.domain_error_grid) != self.domain_error_grid_hash:
            raise ValueError("domain error grid payload does not match its digest")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe calibration data."""

        return _quantized_dict(asdict(self))


@dataclass(frozen=True)
class UQPropagationResult:
    """JSON-safe OpenTURNS propagation and sensitivity result."""

    prices: Mapping[str, Any]
    first_order_sobol: Mapping[ComponentName, float]
    total_order_sobol: Mapping[ComponentName, float]
    sobol_intervals: Mapping[str, Mapping[ComponentName, Mapping[str, float]]]
    sobol_validation: Mapping[str, Any]
    component_variance: Mapping[ComponentName, Mapping[str, float | int]]
    finite_count: int
    sample_seed: int
    sample_size: int
    sobol_seed: int
    sobol_base_size: int
    openturns_version: str
    distribution_constructor: str

    def __post_init__(self) -> None:
        """Freeze all nested propagation evidence after construction."""

        _freeze_fields(
            self,
            "prices",
            "first_order_sobol",
            "total_order_sobol",
            "sobol_intervals",
            "sobol_validation",
            "component_variance",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe propagation data."""

        return _quantized_dict(asdict(self))


@dataclass(frozen=True)
class UQParityResult:
    """Direct NumPy reference parity diagnostics."""

    direct_seed: int
    direct_size: int
    direct_prices: Mapping[str, Any]
    differences: Mapping[str, float]
    tolerances: Mapping[str, float]
    passed: bool
    tolerance_formula: str

    def __post_init__(self) -> None:
        """Freeze parity summaries and thresholds."""

        _freeze_fields(self, "direct_prices", "differences", "tolerances")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe parity data."""

        return _quantized_dict(asdict(self))


@dataclass(frozen=True)
class AdditiveSobolRecovery:
    """Known-variance synthetic Sobol recovery gate."""

    expected_first: Mapping[ComponentName, float]
    estimated_first: Mapping[ComponentName, float]
    estimated_total: Mapping[ComponentName, float]
    max_abs_error_first: float
    max_abs_error_total: float
    tolerance: float
    passed: bool
    seed: int
    base_size: int

    def __post_init__(self) -> None:
        """Freeze reference and estimated Sobol mappings."""

        _freeze_fields(self, "expected_first", "estimated_first", "estimated_total")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe recovery data."""

        return _quantized_dict(asdict(self))


@dataclass(frozen=True)
class UQPilotResult:
    """Complete canonical OpenTURNS UQ pilot artifact."""

    schema_version: str
    issue: str
    scope: str
    decision: Mapping[str, Any]
    study_input_hash: str
    component_names: tuple[ComponentName, ...]
    components: tuple[UncertaintyComponent, ...]
    calibration: UQCalibration
    propagation: UQPropagationResult
    direct_reference: UQParityResult
    additive_sobol_recovery: AdditiveSobolRecovery
    attribution_table: Mapping[ComponentName, Mapping[str, Any]]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Enforce exactly the five named components and no model-risk bucket."""

        _freeze_fields(self, "decision", "components", "attribution_table", "provenance")
        names = tuple(component.name for component in self.components)
        if names != COMPONENT_NAMES or self.component_names != COMPONENT_NAMES:
            raise ValueError(f"pilot must expose exactly {COMPONENT_NAMES}, got {names}")
        if "model_risk" in names or "model_risk" in self.attribution_table:
            raise ValueError("undifferentiated model_risk bucket is forbidden")

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-safe artifact data."""

        payload = asdict(self)
        payload["components"] = [component.to_dict() for component in self.components]
        return _quantized_dict(payload)


__all__ = [
    "AdditiveSobolRecovery",
    "COMPONENT_NAMES",
    "ComponentName",
    "QUANTILE_LEVELS",
    "UQCalibration",
    "UQParityResult",
    "UQPilotConfig",
    "UQPilotResult",
    "UQPropagationResult",
    "UncertaintyComponent",
]
