"""Research-only regime-switching quanto APIs with lazy optional imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "ContractSpec": "contracts",
    "DataQualityConfig": "quality",
    "FEMGridSpec": "contracts",
    "FEMPriceResult": "contracts",
    "MarkovSwitchingDiffusionResult": "_types",
    "MonteCarloPriceResult": "contracts",
    "RegimeCandidateResult": "_types",
    "RegimeSummary": "_types",
    "TwoFactorRegimeModel": "contracts",
    "discrete_to_continuous_generator": "generator",
    "fit_markov_switching_joint_diffusion": "fitting",
    "fit_regime_candidates": "fitting",
    "prepare_joint_log_returns": "quality",
    "price_contract_fem": "fem",
    "price_contract_monte_carlo": "monte_carlo",
    "price_contracts_monte_carlo": "monte_carlo",
}

__all__ = [
    "ContractSpec",
    "DataQualityConfig",
    "FEMGridSpec",
    "FEMPriceResult",
    "MarkovSwitchingDiffusionResult",
    "MonteCarloPriceResult",
    "RegimeCandidateResult",
    "RegimeSummary",
    "TwoFactorRegimeModel",
    "discrete_to_continuous_generator",
    "fit_markov_switching_joint_diffusion",
    "fit_regime_candidates",
    "prepare_joint_log_returns",
    "price_contract_fem",
    "price_contract_monte_carlo",
    "price_contracts_monte_carlo",
]


def __getattr__(name: str) -> Any:
    """Load example components only when requested.

    Pricing remains importable from the base wheel.  Data preparation and
    statistical fitting name the calibration extra when their optional stack
    is absent.
    """

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"pandas", "statsmodels"}:
            raise ImportError(
                f"{name} requires the optional calibration stack; install "
                "finite-element-options[calibration]."
            ) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily available public names to interactive users."""

    return sorted(set(globals()) | set(__all__))
