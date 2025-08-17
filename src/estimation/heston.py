"""Heston model calibration utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.miscmodels.nonlinls import NonlinearLS
import pymc as pm

from .calibrator import Calibrator


class HestonCalibrator(Calibrator):
    """Calibrate simplified Heston parameters using least squares.

    The pricing formula is intentionally simplistic and serves only as a
    placeholder.  It allows the calibration workflow to be demonstrated with
    synthetic data without relying on a full Heston implementation.
    """

    @staticmethod
    def price_formula(
        strikes: np.ndarray, maturities: np.ndarray, params: Sequence[float]
    ) -> np.ndarray:
        """Synthetic pricing formula used for demo purposes.

        Parameters
        ----------
        strikes, maturities:
            Arrays defining the option surface.
        params:
            Sequence ``[v0, kappa, theta, sigma, rho]``.
        """

        v0, kappa, theta, sigma, rho = params
        return (
            v0
            + 1e-2 * kappa * strikes
            + theta * maturities
            + 1e-1 * sigma * np.sqrt(strikes)
            + rho * maturities**2
        )

    def model_prices(self, params: Sequence[float]) -> np.ndarray:
        """Return prices implied by ``params`` across the market grid."""
        return self.price_formula(self.strikes, self.maturities, params)


class StatsmodelsCalibrator(HestonCalibrator):
    """Calibrate using ``statsmodels`` nonlinear least squares."""

    def calibrate(self, initial_guess: Sequence[float]) -> np.ndarray:  # type: ignore[override]
        """Return parameters estimated via ``statsmodels`` NLS solver."""
        strikes, maturities = self.strikes, self.maturities

        class _NLS(NonlinearLS):
            def _predict(self, params: Sequence[float]) -> np.ndarray:
                return HestonCalibrator.price_formula(strikes, maturities, params)

        # Work around missing RegressionResults symbol in statsmodels.miscmodels
        from statsmodels.regression.linear_model import RegressionResults
        import statsmodels.regression as regression

        if not hasattr(regression, "RegressionResults"):
            regression.RegressionResults = RegressionResults

        model = _NLS(endog=self.prices)
        res = model.fit(start_value=np.asarray(initial_guess), nparams=len(initial_guess))
        return res.params


class PyMCCalibrator(HestonCalibrator):
    """Bayesian calibration with :mod:`pymc` and simple priors."""

    def calibrate(  # type: ignore[override]
        self,
        draws: int = 1000,
        chains: int = 2,
        tune: int | None = None,
        random_seed: int | None = 123,
        target_accept: float = 0.9,
    ) -> np.ndarray:
        """Return posterior means of the Heston parameters."""

        if tune is None:
            tune = draws

        strikes, maturities = self.strikes, self.maturities

        with pm.Model():
            v0 = pm.Normal("v0", mu=0.04, sigma=0.1)
            kappa = pm.Normal("kappa", mu=1.0, sigma=0.5)
            theta = pm.Normal("theta", mu=0.04, sigma=0.1)
            sigma = pm.HalfNormal("sigma", sigma=0.3)
            rho = pm.Uniform("rho", lower=-1.0, upper=1.0)
            params = pm.math.stack([v0, kappa, theta, sigma, rho])
            mu = HestonCalibrator.price_formula(strikes, maturities, params)
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
        means = [posterior[var].mean().item() for var in ["v0", "kappa", "theta", "sigma", "rho"]]
        return np.asarray(means)


def sample_calibration() -> np.ndarray:
    """Run a toy calibration against synthetic market data.

    Returns
    -------
    numpy.ndarray
        Estimated parameter vector close to the ground truth.
    """

    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes, maturities = np.meshgrid(s, t)
    strikes, maturities = strikes.ravel(), maturities.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = HestonCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    calibrator = HestonCalibrator(data)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)


def sample_statsmodels_calibration() -> np.ndarray:
    """Example calibration using :class:`StatsmodelsCalibrator`."""

    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes, maturities = np.meshgrid(s, t)
    strikes, maturities = strikes.ravel(), maturities.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = HestonCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    calibrator = StatsmodelsCalibrator(data)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)


def sample_pymc_calibration() -> np.ndarray:
    """Example Bayesian calibration returning posterior means."""

    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes, maturities = np.meshgrid(s, t)
    strikes, maturities = strikes.ravel(), maturities.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = HestonCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    calibrator = PyMCCalibrator(data)
    return calibrator.calibrate(draws=500, chains=2, random_seed=123)
