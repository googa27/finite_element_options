"""Tests for calibration utilities."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
import warnings
from collections.abc import Mapping, Sequence as StringSequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finite_element_options.estimation import (
    CalibrationResult,
    HestonCalibrator,
    PyMCCalibrator,
    StatsmodelsCalibrator,
    SyntheticSurfaceCalibrator,
)


def _surface() -> tuple[pd.DataFrame, np.ndarray]:
    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes_grid, maturities_grid = np.meshgrid(s, t)
    strikes = strikes_grid.ravel()
    maturities = maturities_grid.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    return data, true_params


def test_synthetic_calibration_returns_full_diagnostics() -> None:
    data, true_params = _surface()
    calibrator = SyntheticSurfaceCalibrator(data)
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])

    result = calibrator.calibrate(initial)

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert result.parameters.shape == true_params.shape
    assert np.allclose(result.parameters, true_params, atol=1e-2)
    assert result.residuals.shape == data["price"].shape
    assert np.linalg.norm(result.residuals) < 1e-8
    assert result.jacobian_rank == len(true_params)
    assert np.isfinite(result.jacobian_condition)
    assert result.bounds[0].shape == true_params.shape
    assert result.bounds[1].shape == true_params.shape
    assert result.nfev > 0
    assert result.status != 0
    assert result.message
    assert result.method == "scipy.least_squares"


def test_calibration_reports_rank_deficiency_without_claiming_full_identification() -> (
    None
):
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    strikes = np.full(8, 100.0)
    maturities = np.full(8, 0.5)
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    calibrator = SyntheticSurfaceCalibrator(data)

    result = calibrator.calibrate(true_params + np.array([0.1, -0.2, 0.1, -0.1, 0.2]))

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert result.jacobian_rank < len(true_params)
    assert result.jacobian_condition == np.inf


def test_statsmodels_calibrator_is_deprecated_scipy_shim_without_global_monkeypatch() -> (
    None
):
    data, true_params = _surface()
    calibrator = StatsmodelsCalibrator(data)
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])

    regression = None
    had_regression_results = False
    prior_regression_results = None
    if importlib.util.find_spec("statsmodels") is not None:
        import statsmodels.regression as regression  # type: ignore[import-untyped,no-redef]

        had_regression_results = hasattr(regression, "RegressionResults")
        prior_regression_results = getattr(regression, "RegressionResults", None)

    with pytest.warns(DeprecationWarning, match="delegates to SciPy"):
        result = calibrator.calibrate(initial)

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert np.allclose(result.parameters, true_params, atol=1e-2)
    if regression is not None:
        assert hasattr(regression, "RegressionResults") is had_regression_results
        assert (
            getattr(regression, "RegressionResults", None) is prior_regression_results
        )


def test_heston_named_calibrator_fails_closed_until_real_heston_engine_exists() -> None:
    data, true_params = _surface()
    calibrator = HestonCalibrator(data)

    with pytest.raises(NotImplementedError, match="real Heston pricing engine"):
        calibrator.calibrate(true_params)


def test_core_import_does_not_require_statsmodels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: StringSequence[str] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("statsmodels"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("finite_element_options", None)

    module = importlib.import_module("finite_element_options")

    assert module.__name__ == "finite_element_options"


def test_estimation_source_has_no_private_statsmodels_nonlinls_import() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "finite_element_options"
    offenders = [
        path
        for path in src_root.rglob("*.py")
        if "statsmodels.miscmodels.nonlinls" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_pymc_calibration_recovers_parameters() -> None:
    data, true_params = _surface()
    calibrator = PyMCCalibrator(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calibrator.calibrate(draws=200, chains=2, random_seed=123)
    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert np.allclose(result.parameters, true_params, atol=0.15)
