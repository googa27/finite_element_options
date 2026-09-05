"""Lazy facade for the OpenTURNS FEM uncertainty pilot."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .contracts import (
    AdditiveSobolRecovery,
    UQCalibration,
    UQParityResult,
    UQPilotConfig,
    UQPilotResult,
    UQPropagationResult,
    UncertaintyComponent,
)

_EXPORT_MODULES = {
    "BASELINE_FULL_CORRELATION": "cases",
    "BASELINE_SIGMA": "cases",
    "BASELINE_SPOT": "cases",
    "COMPONENT_NAMES": "contracts",
    "QUANTILE_LEVELS": "contracts",
    "SCOPE_STATEMENT": "cases",
    "baseline_contract": "cases",
    "baseline_model": "cases",
    "build_components": "cases",
    "calibrate_scales": "cases",
    "canonical_study_input": "cases",
    "canonical_uq_input_hash": "cases",
    "evaluate_response": "cases",
    "map_normalized_inputs": "cases",
    "openturns_seeded": "openturns_adapter",
    "run_openturns_uq_pilot": "pilot",
    "verify_predecessor_hashes": "pilot",
}

__all__ = [
    "AdditiveSobolRecovery",
    "BASELINE_FULL_CORRELATION",
    "BASELINE_SIGMA",
    "BASELINE_SPOT",
    "COMPONENT_NAMES",
    "QUANTILE_LEVELS",
    "SCOPE_STATEMENT",
    "UQCalibration",
    "UQParityResult",
    "UQPilotConfig",
    "UQPilotResult",
    "UQPropagationResult",
    "UncertaintyComponent",
    "baseline_contract",
    "baseline_model",
    "build_components",
    "calibrate_scales",
    "canonical_study_input",
    "canonical_uq_input_hash",
    "evaluate_response",
    "map_normalized_inputs",
    "openturns_seeded",
    "run_openturns_uq_pilot",
    "verify_predecessor_hashes",
]


def __getattr__(name: str) -> Any:
    """Load uncertainty helpers only when requested."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily available facade names."""

    return sorted(set(globals()) | set(__all__))
