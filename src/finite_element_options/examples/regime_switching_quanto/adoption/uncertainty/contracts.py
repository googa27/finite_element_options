"""JSON-safe uncertainty contracts for the OpenTURNS FEM pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from finite_element_options.examples.regime_switching_quanto._types import json_safe

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


@dataclass(frozen=True)
class UncertaintyComponent:
    """One immutable named uncertainty component in normalized coordinates."""

    name: ComponentName
    distribution: str
    scale_or_range: dict[str, float | str]
    units: str
    role: ComponentRole
    source_identity: str
    source_hash: str
    perturbs_fem_model: bool
    additive_validation_estimator_error: bool
    description: str

    def __post_init__(self) -> None:
        """Validate component names, finite ranges, and hash-bound provenance."""

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

        return json_safe(asdict(self))


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

        return json_safe(asdict(self))


@dataclass(frozen=True)
class UQCalibration:
    """Baseline deterministic and seeded-MC scale calibration."""

    baseline_price_fine: float
    baseline_price_coarse: float
    numerical_half_width: float
    numerical_formula: str
    mc_price: float
    mc_standard_error: float
    mc_seed: int
    mc_paths: int
    mc_steps: int
    mc_steps_per_year: int
    fine_grid: dict[str, Any]
    coarse_grid: dict[str, Any]
    fine_grid_hash: str
    coarse_grid_hash: str
    baseline_model_hash: str
    payoff_hash: str

    def __post_init__(self) -> None:
        """Require finite, separated numerical and Monte Carlo scale records."""

        finite_values = (
            self.baseline_price_fine,
            self.baseline_price_coarse,
            self.numerical_half_width,
            self.mc_price,
            self.mc_standard_error,
        )
        if not all(np.isfinite(value) for value in finite_values):
            raise ValueError("calibration values must be finite")
        if self.numerical_half_width < 0.0 or self.mc_standard_error <= 0.0:
            raise ValueError("calibration scales must be non-negative/positive as documented")
        for value in (
            self.fine_grid_hash,
            self.coarse_grid_hash,
            self.baseline_model_hash,
            self.payoff_hash,
        ):
            if not _is_lower_sha256(value):
                raise ValueError("calibration hashes must be lowercase SHA-256 hex strings")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe calibration data."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class UQPropagationResult:
    """JSON-safe OpenTURNS propagation and sensitivity result."""

    prices: dict[str, Any]
    first_order_sobol: dict[ComponentName, float]
    total_order_sobol: dict[ComponentName, float]
    sobol_intervals: dict[str, dict[ComponentName, dict[str, float]]]
    sobol_validation: dict[str, Any]
    component_variance: dict[ComponentName, dict[str, float | int]]
    finite_count: int
    sample_seed: int
    sample_size: int
    sobol_seed: int
    sobol_base_size: int
    openturns_version: str
    distribution_constructor: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe propagation data."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class UQParityResult:
    """Direct NumPy reference parity diagnostics."""

    direct_seed: int
    direct_size: int
    direct_prices: dict[str, Any]
    differences: dict[str, float]
    tolerances: dict[str, float]
    passed: bool
    tolerance_formula: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe parity data."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class AdditiveSobolRecovery:
    """Known-variance synthetic Sobol recovery gate."""

    expected_first: dict[ComponentName, float]
    estimated_first: dict[ComponentName, float]
    estimated_total: dict[ComponentName, float]
    max_abs_error_first: float
    max_abs_error_total: float
    tolerance: float
    passed: bool
    seed: int
    base_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe recovery data."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class UQPilotResult:
    """Complete canonical OpenTURNS UQ pilot artifact."""

    schema_version: str
    issue: str
    scope: str
    decision: dict[str, Any]
    study_input_hash: str
    component_names: tuple[ComponentName, ...]
    components: list[UncertaintyComponent]
    calibration: UQCalibration
    propagation: UQPropagationResult
    direct_reference: UQParityResult
    additive_sobol_recovery: AdditiveSobolRecovery
    attribution_table: dict[ComponentName, dict[str, Any]]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        """Enforce exactly the five named components and no model-risk bucket."""

        names = tuple(component.name for component in self.components)
        if names != COMPONENT_NAMES or self.component_names != COMPONENT_NAMES:
            raise ValueError(f"pilot must expose exactly {COMPONENT_NAMES}, got {names}")
        if "model_risk" in names or "model_risk" in self.attribution_table:
            raise ValueError("undifferentiated model_risk bucket is forbidden")

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-safe artifact data."""

        payload = asdict(self)
        payload["components"] = [component.to_dict() for component in self.components]
        return json_safe(payload)


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
