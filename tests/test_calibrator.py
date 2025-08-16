"""Tests for calibration utilities."""

import numpy as np

from src.estimation import HestonCalibrator


def test_heston_calibration_recovers_parameters() -> None:
    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes, maturities = np.meshgrid(s, t)
    strikes, maturities = strikes.ravel(), maturities.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = HestonCalibrator.price_formula(strikes, maturities, true_params)
    calibrator = HestonCalibrator(strikes, maturities, prices)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    estimated = calibrator.calibrate(initial_guess)
    assert np.allclose(estimated, true_params, atol=1e-2)
