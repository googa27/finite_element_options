"""Lazy estimation, deterministic calibration, and Bayesian-profile facade."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_CALIBRATOR_EXPORTS = {
    "CalibrationObjective",
    "CalibrationPricingError",
    "CalibrationResult",
    "Calibrator",
    "PricingCalibrationDataset",
    "PricingModelCalibrator",
}
_HESTON_EXPORTS = {
    "HestonCalibrator",
    "HestonConstraintReport",
    "HestonMCMCDiagnosticReport",
    "HestonMCMCDiagnosticThresholds",
    "HestonPricingCalibrator",
    "PyMCCalibrator",
    "StatsmodelsCalibrator",
    "SyntheticSurfaceCalibrator",
    "build_heston_bayesian_calibration_result",
    "evaluate_heston_mcmc_diagnostics",
    "sample_calibration",
    "sample_pymc_calibration",
    "sample_statsmodels_calibration",
    "validate_heston_posterior_draws",
}


def __getattr__(name: str) -> Any:
    """Load deterministic/PyMC compatibility exports only when requested."""

    if name in _CALIBRATOR_EXPORTS:
        return getattr(import_module(".calibrator", __name__), name)
    if name in _HESTON_EXPORTS:
        return getattr(import_module(".heston", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the compatibility surface without importing optional stacks."""

    return sorted(set(globals()) | set(__all__))


__all__ = (
    "CalibrationObjective",
    "CalibrationPricingError",
    "CalibrationResult",
    "Calibrator",
    "HestonCalibrator",
    "HestonConstraintReport",
    "HestonMCMCDiagnosticReport",
    "HestonMCMCDiagnosticThresholds",
    "HestonPricingCalibrator",
    "PricingCalibrationDataset",
    "PricingModelCalibrator",
    "PyMCCalibrator",
    "StatsmodelsCalibrator",
    "SyntheticSurfaceCalibrator",
    "build_heston_bayesian_calibration_result",
    "evaluate_heston_mcmc_diagnostics",
    "sample_calibration",
    "sample_pymc_calibration",
    "sample_statsmodels_calibration",
    "validate_heston_posterior_draws",
)
