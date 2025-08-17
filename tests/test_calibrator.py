"""Tests for calibration utilities."""

import numpy as np

from src.estimation import (
    HestonCalibrator,
    StatsmodelsCalibrator,
    PyMCCalibrator,
)


def _surface() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes, maturities = np.meshgrid(s, t)
    strikes, maturities = strikes.ravel(), maturities.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = HestonCalibrator.price_formula(strikes, maturities, true_params)
    return strikes, maturities, prices, true_params


def test_heston_calibration_recovers_parameters() -> None:
    strikes, maturities, prices, true_params = _surface()
    calibrator = HestonCalibrator(strikes, maturities, prices)
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    estimated = calibrator.calibrate(initial)
    assert np.allclose(estimated, true_params, atol=1e-2)


def test_statsmodels_calibration_recovers_parameters() -> None:
    strikes, maturities, prices, true_params = _surface()
    calibrator = StatsmodelsCalibrator(strikes, maturities, prices)
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    estimated = calibrator.calibrate(initial)
    assert np.allclose(estimated, true_params, atol=1e-2)


def test_pymc_calibration_recovers_parameters() -> None:
    strikes, maturities, prices, true_params = _surface()
    calibrator = PyMCCalibrator(strikes, maturities, prices)
    estimated = calibrator.calibrate(draws=200, chains=2, random_seed=123)
    assert np.allclose(estimated, true_params, atol=0.15)
