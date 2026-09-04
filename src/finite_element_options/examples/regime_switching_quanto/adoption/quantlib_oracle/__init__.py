"""Lazy facade for the QuantLib vanilla/quanto oracle package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .contracts import (
    QuantLibConventionError,
    QuantLibOracleResult,
    QuantLibOracleSpec,
    QuantLibReductionError,
)

_EXPORT_MODULES = {
    "ANALYTICAL_TOLERANCE": "adapter",
    "MATRIX_SCHEMA_VERSION": "matrix",
    "SCOPE_STATEMENT": "matrix",
    "MatrixCase": "matrix",
    "MatrixRunResult": "matrix",
    "analytical_oracle_price": "adapter",
    "canonical_matrix_input_hash": "matrix",
    "default_matrix_cases": "matrix",
    "price_quantlib_oracle": "adapter",
    "run_quantlib_oracle_matrix": "matrix",
}

__all__ = [
    "ANALYTICAL_TOLERANCE",
    "MATRIX_SCHEMA_VERSION",
    "SCOPE_STATEMENT",
    "MatrixCase",
    "MatrixRunResult",
    "QuantLibConventionError",
    "QuantLibOracleResult",
    "QuantLibOracleSpec",
    "QuantLibReductionError",
    "analytical_oracle_price",
    "canonical_matrix_input_hash",
    "default_matrix_cases",
    "price_quantlib_oracle",
    "run_quantlib_oracle_matrix",
]


def __getattr__(name: str) -> Any:
    """Load implementation modules only when their names are requested."""

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
