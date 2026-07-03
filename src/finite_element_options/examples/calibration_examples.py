"""Synthetic calibration examples.

These examples intentionally use :class:`SyntheticSurfaceCalibrator`; they are
not production Heston calibration routes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from finite_element_options.estimation import (
    CalibrationResult,
    PyMCCalibrator,
    StatsmodelsCalibrator,
    SyntheticSurfaceCalibrator,
)


def synthetic_surface() -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a toy option surface and corresponding prices."""

    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes_grid, maturities_grid = np.meshgrid(s, t)
    strikes = strikes_grid.ravel()
    maturities = maturities_grid.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    return data, true_params


def run_scipy() -> CalibrationResult:
    """Estimate synthetic fixture parameters via SciPy least squares."""

    data, true_params = synthetic_surface()
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    calib = SyntheticSurfaceCalibrator(data)
    return calib.calibrate(initial)


def run_statsmodels_shim() -> CalibrationResult:
    """Estimate synthetic parameters through the deprecated Statsmodels shim."""

    data, true_params = synthetic_surface()
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    calib = StatsmodelsCalibrator(data)
    with warnings.catch_warnings():
        warnings.simplefilter("default", DeprecationWarning)
        return calib.calibrate(initial)


def run_pymc() -> CalibrationResult:
    """Estimate synthetic parameters via :class:`PyMCCalibrator`."""

    data, _ = synthetic_surface()
    calib = PyMCCalibrator(data)
    return calib.calibrate(draws=500, chains=2, random_seed=123)
