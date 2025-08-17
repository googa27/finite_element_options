"""Synthetic calibration examples for Heston parameters."""

from __future__ import annotations

import numpy as np

from src.estimation import (
    HestonCalibrator,
    StatsmodelsCalibrator,
    PyMCCalibrator,
)


def synthetic_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a toy option surface and corresponding prices."""

    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes, maturities = np.meshgrid(s, t)
    strikes, maturities = strikes.ravel(), maturities.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = HestonCalibrator.price_formula(strikes, maturities, true_params)
    return strikes, maturities, prices, true_params


def run_statsmodels() -> np.ndarray:
    """Estimate parameters via :class:`StatsmodelsCalibrator`."""

    strikes, maturities, prices, true_params = synthetic_surface()
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    calib = StatsmodelsCalibrator(strikes, maturities, prices)
    return calib.calibrate(initial)


def run_pymc() -> np.ndarray:
    """Estimate parameters via :class:`PyMCCalibrator`."""

    strikes, maturities, prices, _ = synthetic_surface()
    calib = PyMCCalibrator(strikes, maturities, prices)
    return calib.calibrate(draws=500, chains=2, random_seed=123)

