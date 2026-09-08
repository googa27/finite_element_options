"""Deterministic statsmodels VAR baseline on the identical bivariate split."""

from __future__ import annotations

import math
from typing import Any


def fit_statsmodels_var_baseline(observations: Any, train_count: int) -> dict[str, Any]:
    """Fit VAR(1) on training data and score observed-lag hold-out predictions."""

    try:
        import numpy as np
        from statsmodels.tsa.api import VAR
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The deterministic VAR comparator requires finite-element-options[jax-regime]."
        ) from exc

    values = np.asarray(observations, dtype=float)
    train = values[:train_count]
    if values.ndim != 2 or values.shape[1] != 2 or train_count < 30:
        raise ValueError("VAR baseline requires at least 30 bivariate training observations")
    fitted = VAR(train).fit(maxlags=1, trend="c")
    covariance = np.asarray(fitted.sigma_u, dtype=float)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(covariance).all():
        raise ValueError("statsmodels VAR residual covariance must be finite positive definite")
    inverse = np.linalg.inv(covariance)
    scores: list[float] = []
    for index in range(train_count, len(values)):
        forecast = np.asarray(fitted.forecast(values[index - 1 : index], steps=1)[0])
        residual = values[index] - forecast
        quadratic = float(residual @ inverse @ residual)
        scores.append(-0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + quadratic))
    return {
        "engine": "statsmodels_VAR_1",
        "train_log_likelihood": float(fitted.llf),
        "heldout_log_likelihood": float(np.sum(scores)),
        "heldout_mean_log_score": float(np.mean(scores)),
        "lags": int(fitted.k_ar),
        "observed_lags_used_for_sequential_scoring": True,
        "residual_covariance_percent_squared_daily": covariance.tolist(),
        "finite": bool(np.isfinite(scores).all()),
    }
