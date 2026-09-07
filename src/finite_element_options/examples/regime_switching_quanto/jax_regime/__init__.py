"""Lazy public facade for the isolated JAX-native regime study."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EvidenceHash": ".contracts",
    "HMMParameterSummary": ".contracts",
    "JaxRegimeStudyConfig": ".contracts",
    "NumPyroPriorConfig": ".contracts",
    "PDPObservationBatch": ".contracts",
    "PDPPreprocessingAudit": ".contracts",
    "PosteriorDiagnosticSummary": ".contracts",
    "PriceEstimate": ".contracts",
    "PromotionDecision": ".contracts",
    "load_pdp_observations": ".data",
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public symbols without importing the optional JAX stack eagerly."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return facade exports for deterministic capability discovery."""

    return sorted((*globals(), *__all__))
