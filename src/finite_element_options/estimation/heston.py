"""Calibration adapters and synthetic surface fixtures.

The historical module name is retained for compatibility, but the Heston-named
calibrator is fail-closed until a real Heston pricing engine is wired in. The
toy formula used by tests and examples lives behind explicitly synthetic class
names so it cannot be mistaken for production Heston calibration.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .calibrator import CalibrationResult, Calibrator, ParameterVector

_SYNTHETIC_PARAMETER_NAMES = (
    "level",
    "strike_slope",
    "maturity_slope",
    "sqrt_strike_slope",
    "maturity_quadratic",
)


class SyntheticSurfaceCalibrator(Calibrator):
    """Calibrate a documented synthetic option-surface fixture.

    This class is intentionally not a Heston model. It exists for deterministic
    examples and tests while real Heston calibration remains blocked behind a
    supported pricing engine and model-risk diagnostics.
    """

    parameter_names = _SYNTHETIC_PARAMETER_NAMES

    @staticmethod
    def price_formula(
        strikes: np.ndarray, maturities: np.ndarray, params: ParameterVector
    ) -> np.ndarray:
        """Synthetic pricing formula used for fixtures only.

        Parameters
        ----------
        strikes, maturities:
            Arrays defining the option surface.
        params:
            Sequence ``[level, strike_slope, maturity_slope,
            sqrt_strike_slope, maturity_quadratic]``.
        """

        level, strike_slope, maturity_slope, sqrt_strike_slope, maturity_quadratic = (
            params
        )
        return (
            level
            + 1e-2 * strike_slope * strikes
            + maturity_slope * maturities
            + 1e-1 * sqrt_strike_slope * np.sqrt(strikes)
            + maturity_quadratic * maturities**2
        )

    def model_prices(self, params: ParameterVector) -> np.ndarray:
        """Return prices implied by ``params`` across the market grid."""
        return self.price_formula(self.strikes, self.maturities, params)


class HestonCalibrator(Calibrator):
    """Fail-closed placeholder for real Heston calibration.

    The previous implementation used a toy polynomial under a Heston name. That
    is model-risk unsafe, so this compatibility class refuses to calibrate until
    a real Heston pricing route and diagnostics are implemented.
    """

    parameter_names = ("v0", "kappa", "theta", "sigma", "rho")

    @staticmethod
    def _unsupported_message() -> str:
        return (
            "HestonCalibrator requires a real Heston pricing engine and model-risk "
            "diagnostics; use SyntheticSurfaceCalibrator only for synthetic fixtures."
        )

    def model_prices(self, params: ParameterVector) -> np.ndarray:
        """Refuse to price through a toy Heston route."""
        raise NotImplementedError(self._unsupported_message())

    def calibrate(
        self,
        initial_guess: ParameterVector,
        *,
        bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None = None,
        weights: Sequence[float] | None = None,
        loss: str = "linear",
    ) -> CalibrationResult:
        """Refuse Heston calibration until a real model implementation exists."""
        del initial_guess, bounds, weights, loss
        raise NotImplementedError(self._unsupported_message())


class StatsmodelsCalibrator(SyntheticSurfaceCalibrator):
    """Deprecated compatibility shim for the removed Statsmodels NLS adapter."""

    def calibrate(
        self,
        initial_guess: ParameterVector,
        *,
        bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None = None,
        weights: Sequence[float] | None = None,
        loss: str = "linear",
    ) -> CalibrationResult:
        """Delegate to the supported SciPy least-squares adapter."""
        warnings.warn(
            "StatsmodelsCalibrator no longer uses private statsmodels NLS APIs; "
            "it delegates to SciPy least_squares and will be removed after the "
            "compatibility window.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().calibrate(
            initial_guess,
            bounds=bounds,
            weights=weights,
            loss=loss,
        )


class PyMCCalibrator(SyntheticSurfaceCalibrator):
    """Bayesian calibration of the synthetic fixture with :mod:`pymc`.

    This is not a Heston calibration route. It returns posterior means plus a
    structured diagnostic shell for fixture experiments only.
    """

    def calibrate(  # type: ignore[override]
        self,
        draws: int = 1000,
        chains: int = 2,
        tune: int | None = None,
        random_seed: int | None = 123,
        target_accept: float = 0.9,
    ) -> CalibrationResult:
        """Return posterior means for the synthetic fixture parameters."""

        import pymc as pm

        if tune is None:
            tune = draws

        strikes, maturities = self.strikes, self.maturities

        with pm.Model():
            level = pm.Normal("level", mu=0.04, sigma=0.1)
            strike_slope = pm.Normal("strike_slope", mu=1.0, sigma=0.5)
            maturity_slope = pm.Normal("maturity_slope", mu=0.04, sigma=0.1)
            sqrt_strike_slope = pm.HalfNormal("sqrt_strike_slope", sigma=0.3)
            maturity_quadratic = pm.Uniform("maturity_quadratic", lower=-1.0, upper=1.0)
            params = pm.math.stack(
                [
                    level,
                    strike_slope,
                    maturity_slope,
                    sqrt_strike_slope,
                    maturity_quadratic,
                ]
            )
            mu = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, params)
            pm.Normal("obs", mu=mu, sigma=0.01, observed=self.prices)
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                target_accept=target_accept,
                progressbar=False,
            )

        posterior = trace.posterior
        means = np.asarray(
            [posterior[var].mean().item() for var in _SYNTHETIC_PARAMETER_NAMES],
            dtype=float,
        )
        residuals = self.residuals(means)
        return CalibrationResult(
            parameters=means,
            success=True,
            status=1,
            message="PyMC posterior mean computed for synthetic fixture",
            residuals=np.asarray(residuals, dtype=float),
            cost=float(0.5 * np.dot(residuals, residuals)),
            optimality=float("nan"),
            jacobian_rank=0,
            jacobian_condition=np.inf,
            bounds=(
                np.full(means.shape, -np.inf, dtype=float),
                np.full(means.shape, np.inf, dtype=float),
            ),
            active_mask=np.zeros(means.shape, dtype=int),
            nfev=int(draws * chains),
            njev=None,
            method="pymc.sample",
            parameter_names=_SYNTHETIC_PARAMETER_NAMES,
        )


def _synthetic_market_data() -> tuple[pd.DataFrame, np.ndarray]:
    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes_grid, maturities_grid = np.meshgrid(s, t)
    strikes = strikes_grid.ravel()
    maturities = maturities_grid.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    return data, true_params


def sample_calibration() -> CalibrationResult:
    """Run a toy calibration against explicitly synthetic market data."""

    data, true_params = _synthetic_market_data()
    calibrator = SyntheticSurfaceCalibrator(data)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)


def sample_statsmodels_calibration() -> CalibrationResult:
    """Example calibration through the deprecated Statsmodels shim."""

    data, true_params = _synthetic_market_data()
    calibrator = StatsmodelsCalibrator(data)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)


def sample_pymc_calibration() -> CalibrationResult:
    """Example Bayesian calibration returning synthetic posterior means."""

    data, _ = _synthetic_market_data()
    calibrator = PyMCCalibrator(data)
    return calibrator.calibrate(draws=500, chains=2, random_seed=123)
