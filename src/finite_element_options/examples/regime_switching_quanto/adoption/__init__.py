"""Facade for optional adoption dependency boundaries."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "OPTIONAL_DEPENDENCIES": "optional",
    "OptionalDependency": "optional",
    "optional_dependency_registry": "optional",
    "require_optional": "optional",
    "quantlib_evaluation_date": "quantlib_state",
    "QuantLibConventionError": "quantlib_oracle",
    "QuantLibOracleResult": "quantlib_oracle",
    "QuantLibOracleSpec": "quantlib_oracle",
    "QuantLibReductionError": "quantlib_oracle",
    "price_quantlib_oracle": "quantlib_oracle",
    "run_quantlib_oracle_matrix": "quantlib_oracle",
    "VolatilityBenchmarkConfig": "volatility_benchmark",
    "VolatilityBenchmarkResult": "volatility_benchmark",
    "run_volatility_benchmark": "volatility_benchmark",
    "CalibrationCase": "identifiability",
    "IdentifiabilityResult": "identifiability",
    "IdentifiabilityStudyResult": "identifiability",
    "ParameterBounds": "identifiability",
    "ProfileGrid": "identifiability",
    "QuantoOptionTarget": "identifiability",
    "WeightedQuantoCalibrationObjective": "identifiability",
    "canonical_identifiability_input_hash": "identifiability",
    "run_iminuit_identifiability": "identifiability",
    "run_iminuit_identifiability_study": "identifiability",
}

__all__ = [
    "OPTIONAL_DEPENDENCIES",
    "OptionalDependency",
    "optional_dependency_registry",
    "quantlib_evaluation_date",
    "QuantLibConventionError",
    "QuantLibOracleResult",
    "QuantLibOracleSpec",
    "QuantLibReductionError",
    "price_quantlib_oracle",
    "run_quantlib_oracle_matrix",
    "require_optional",
    "VolatilityBenchmarkConfig",
    "VolatilityBenchmarkResult",
    "run_volatility_benchmark",
    "CalibrationCase",
    "IdentifiabilityResult",
    "IdentifiabilityStudyResult",
    "ParameterBounds",
    "ProfileGrid",
    "QuantoOptionTarget",
    "WeightedQuantoCalibrationObjective",
    "canonical_identifiability_input_hash",
    "run_iminuit_identifiability",
    "run_iminuit_identifiability_study",
]


def __getattr__(name: str) -> Any:
    """Load adoption boundary helpers only when requested."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily available facade names to interactive users."""

    return sorted(set(globals()) | set(__all__))
