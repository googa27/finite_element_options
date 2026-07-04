"""Cox-Ingersoll-Ross moment and truncation diagnostics."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

ArrayLikeFloat = float | np.ndarray


def _as_float_array(name: str, value: ArrayLikeFloat) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _maybe_scalar(result: np.ndarray, *values: ArrayLikeFloat) -> float | np.ndarray:
    if all(np.asarray(value).ndim == 0 for value in values):
        return float(np.asarray(result))
    return result


def validate_cir_variance_parameters(
    *, kappa: float, theta: float, volatility_of_variance: float, rho: float
) -> None:
    """Validate Heston variance-process parameters fail-closed.

    ``kappa=0`` is admitted as the martingale limit
    ``dV_t = sigma sqrt(V_t) dW_t``.  The volatility of variance must remain
    strictly positive because the CIR/Feller diagnostics divide by ``sigma^2``.
    """

    values = {
        "kappa": kappa,
        "theta": theta,
        "volatility_of_variance": volatility_of_variance,
        "rho": rho,
    }
    for name, value in values.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if theta < 0.0:
        raise ValueError("theta must be non-negative")
    if volatility_of_variance <= 0.0:
        raise ValueError("volatility_of_variance must be strictly positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1")


def cir_conditional_mean(
    *,
    kappa: float,
    theta: float,
    horizon: ArrayLikeFloat,
    initial_variance: ArrayLikeFloat,
) -> float | np.ndarray:
    r"""Return ``E[V_{t+tau} | V_t=v]`` for the CIR variance process.

    For

    .. math:: dV_t = \kappa(\theta - V_t)dt + \sigma\sqrt{V_t}dW_t,

    the first conditional moment is

    .. math:: \theta + (v-\theta)e^{-\kappa\tau}.

    The ``kappa=0`` limit is the martingale mean ``v``.  No artificial epsilon
    is injected, so ``tau=0`` returns the input variance exactly.
    """

    tau = _as_float_array("horizon", horizon)
    v0 = _as_float_array("initial_variance", initial_variance)
    if np.any(tau < 0.0):
        raise ValueError("horizon must be non-negative")
    if np.any(v0 < 0.0):
        raise ValueError("initial_variance must be non-negative")
    if kappa == 0.0:
        result = np.broadcast_to(v0, np.broadcast_shapes(np.shape(tau), np.shape(v0))).astype(float)
    else:
        decay = np.exp(-kappa * tau)
        result = theta + (v0 - theta) * decay
        result = np.where(tau == 0.0, v0, result)
    return _maybe_scalar(np.asarray(result, dtype=float), horizon, initial_variance)


def cir_time_average_mean(
    *,
    kappa: float,
    theta: float,
    horizon: ArrayLikeFloat,
    initial_variance: ArrayLikeFloat,
) -> float | np.ndarray:
    r"""Return ``E[tau^{-1}\int_t^{t+tau} V_s ds | V_t=v]`` for CIR.

    This is the effective constant variance used by Black-Scholes boundary
    oracles over a finite horizon.  It is distinct from the terminal
    conditional mean returned by :func:`cir_conditional_mean`.
    """

    tau = _as_float_array("horizon", horizon)
    v0 = _as_float_array("initial_variance", initial_variance)
    if np.any(tau < 0.0):
        raise ValueError("horizon must be non-negative")
    if np.any(v0 < 0.0):
        raise ValueError("initial_variance must be non-negative")
    x = kappa * tau
    factor = np.divide(
        -np.expm1(-x),
        x,
        out=np.ones_like(x, dtype=float),
        where=x != 0.0,
    )
    result = theta + (v0 - theta) * factor
    result = np.where(x == 0.0, v0, result)
    return _maybe_scalar(np.asarray(result, dtype=float), horizon, initial_variance)


def cir_conditional_variance(
    *,
    kappa: float,
    theta: float,
    volatility_of_variance: float,
    horizon: ArrayLikeFloat,
    initial_variance: ArrayLikeFloat,
) -> float | np.ndarray:
    r"""Return ``Var[V_{t+tau} | V_t=v]`` for the CIR variance process."""

    tau = _as_float_array("horizon", horizon)
    v0 = _as_float_array("initial_variance", initial_variance)
    if np.any(tau < 0.0):
        raise ValueError("horizon must be non-negative")
    if np.any(v0 < 0.0):
        raise ValueError("initial_variance must be non-negative")
    sigma2 = volatility_of_variance * volatility_of_variance
    if kappa == 0.0:
        result = v0 * sigma2 * tau
    else:
        one_minus_decay = -np.expm1(-kappa * tau)
        decay = np.exp(-kappa * tau)
        result = (
            v0 * sigma2 * decay * one_minus_decay / kappa
            + theta * sigma2 * one_minus_decay**2 / (2.0 * kappa)
        )
    return _maybe_scalar(np.asarray(result, dtype=float), horizon, initial_variance)


def feller_ratio(*, kappa: float, theta: float, volatility_of_variance: float) -> float:
    """Return the CIR Feller ratio ``2*kappa*theta/sigma^2``."""

    return float(2.0 * kappa * theta / (volatility_of_variance**2))


def cir_variance_domain_diagnostics(
    *,
    kappa: float,
    theta: float,
    volatility_of_variance: float,
    horizon: float,
    initial_variance: ArrayLikeFloat,
    tail_mass: float = 1.0e-6,
) -> dict[str, Any]:
    """Return conservative variance-domain truncation diagnostics.

    The domain is a deterministic, auditable Chebyshev tail bound around the
    exact first two CIR moments.  It is intentionally conservative; later
    quantile-based policies can replace it once distribution-backed evidence is
    added.  The returned ``estimated_omitted_mass`` is the requested one-sided
    bound, not an empirical calibration.
    """

    if not np.isfinite(float(tail_mass)) or not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass must be finite and in (0, 1)")
    validate_cir_variance_parameters(
        kappa=kappa,
        theta=theta,
        volatility_of_variance=volatility_of_variance,
        rho=0.0,
    )
    mean = np.asarray(
        cir_conditional_mean(
            kappa=kappa,
            theta=theta,
            horizon=horizon,
            initial_variance=initial_variance,
        ),
        dtype=float,
    )
    variance = np.asarray(
        cir_conditional_variance(
            kappa=kappa,
            theta=theta,
            volatility_of_variance=volatility_of_variance,
            horizon=horizon,
            initial_variance=initial_variance,
        ),
        dtype=float,
    )
    mean_min = float(np.min(mean))
    mean_max = float(np.max(mean))
    variance_max = max(0.0, float(np.max(variance)))
    radius = sqrt(variance_max / tail_mass) if variance_max > 0.0 else 0.0
    ratio = feller_ratio(
        kappa=kappa, theta=theta, volatility_of_variance=volatility_of_variance
    )
    initial_variance_array = _as_float_array("initial_variance", initial_variance)
    initial_variance_min = float(np.min(initial_variance_array))
    initial_variance_max = float(np.max(initial_variance_array))
    return {
        "policy": "cir-chebyshev-tail-bound",
        "horizon": float(horizon),
        "kappa": float(kappa),
        "theta": float(theta),
        "volatility_of_variance": float(volatility_of_variance),
        "initial_variance_min": initial_variance_min,
        "initial_variance_max": initial_variance_max,
        "mean_variance_min": mean_min,
        "mean_variance_max": mean_max,
        "variance_of_variance_max": variance_max,
        "domain_lower": 0.0,
        "domain_upper": max(0.0, initial_variance_max, mean_max + radius),
        "tail_mass": float(tail_mass),
        "estimated_omitted_mass": float(tail_mass),
        "feller_ratio": ratio,
        "feller_condition_satisfied": bool(ratio >= 1.0),
    }
