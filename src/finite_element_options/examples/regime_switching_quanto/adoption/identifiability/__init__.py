"""Lazy facade for iminuit profile-likelihood identifiability diagnostics."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "CalibrationCase": "contracts",
    "FORMULA_STATEMENT": "cases",
    "IdentifiabilityResult": "contracts",
    "IdentifiabilityStudyResult": "cases",
    "OBJECTIVE_NAME": "contracts",
    "ObjectiveEvaluation": "contracts",
    "PARAMETERS": "contracts",
    "ParameterBounds": "contracts",
    "ProfileGrid": "contracts",
    "QuantoOptionTarget": "contracts",
    "SCHEMA_VERSION": "contracts",
    "SCOPE_STATEMENT": "cases",
    "STUDY_SCHEMA_VERSION": "cases",
    "TARGET_SOURCE": "contracts",
    "WeightedQuantoCalibrationObjective": "contracts",
    "canonical_identifiability_input_hash": "cases",
    "default_identifiability_cases": "cases",
    "quanto_call_price": "contracts",
    "run_iminuit_identifiability": "adapter",
    "run_iminuit_identifiability_study": "cases",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load identifiability helpers only when requested."""

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
