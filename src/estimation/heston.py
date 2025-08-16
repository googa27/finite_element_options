"""Heston model calibration utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

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
    calibrator = HestonCalibrator(strikes, maturities, prices)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)
