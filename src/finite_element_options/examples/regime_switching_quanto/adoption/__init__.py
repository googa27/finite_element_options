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
    "VolatilityBenchmarkConfig": "volatility_benchmark",
    "VolatilityBenchmarkResult": "volatility_benchmark",
    "run_volatility_benchmark": "volatility_benchmark",
}

__all__ = [
    "OPTIONAL_DEPENDENCIES",
    "OptionalDependency",
    "optional_dependency_registry",
    "quantlib_evaluation_date",
    "require_optional",
    "VolatilityBenchmarkConfig",
    "VolatilityBenchmarkResult",
    "run_volatility_benchmark",
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
