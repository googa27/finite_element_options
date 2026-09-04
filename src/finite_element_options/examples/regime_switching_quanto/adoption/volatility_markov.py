"""Statsmodels Markov AR(2) baseline helpers for volatility benchmarks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
    CandidateFailure,
    MarkovBaselineResult,
    MarkovHighVolatilityProbability,
    RollingBoundary,
    VolatilityBenchmarkConfig,
)
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_metrics import (
    candidate_result,
    parameter_stability,
    params_dict,
)


def fit_markov_baseline(
    response: np.ndarray,
    boundaries: list[RollingBoundary],
    cfg: VolatilityBenchmarkConfig,
) -> MarkovBaselineResult:
    """Fit the real statsmodels MarkovAutoregression baseline on rolling blocks."""

    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
    except ModuleNotFoundError as exc:
        raise ImportError("Markov baseline requires finite-element-options[calibration].") from exc
    observations: list[float] = []
    means: list[float] = []
    variances: list[float] = []
    logs: list[float] = []
    vars_: list[float] = []
    params: list[dict[str, float]] = []
    converged: list[bool] = []
    try:
        for block_index, boundary in enumerate(boundaries):
            train = response[boundary.train_start : boundary.train_end]
            with temporary_numpy_seed(cfg.seed + 2000 + block_index):
                model = MarkovAutoregression(
                    train,
                    k_regimes=cfg.markov_regimes,
                    order=cfg.markov_order,
                    trend="c",
                    switching_ar=True,
                    switching_trend=True,
                    switching_variance=True,
                )
                fitted = model.fit(
                    method="bfgs",
                    maxiter=cfg.markov_maxiter,
                    disp=False,
                    search_reps=cfg.markov_search_reps,
                    search_iter=cfg.markov_search_iter,
                )
            params.append(params_dict(fitted.params, fitted.model.param_names))
            converged.append(
                bool((getattr(fitted, "mle_retvals", {}) or {}).get("converged", False))
            )
            holdout = response[boundary.holdout_start : boundary.holdout_end]
            mean, variance, log_score, var_value = markov_forecast(
                fitted, train, holdout, cfg.var_alpha
            )
            observations.extend(float(item) for item in holdout)
            means.extend(float(item) for item in mean)
            variances.extend(float(item) for item in variance)
            logs.extend(float(item) for item in log_score)
            vars_.extend(float(item) for item in var_value)
    except Exception as exc:  # pragma: no cover - numeric optimizer fallback
        return MarkovBaselineResult(
            "statsmodels MarkovAutoregression",
            "gaussian-mixture",
            cfg.markov_regimes,
            cfg.markov_order,
            False,
            CandidateFailure("optimizer", str(exc), len(params)),
            len(params),
            None,
            None,
            None,
            parameter_stability(params),
            markov_note(),
        )
    candidate = candidate_result(
        "statsmodels MarkovAutoregression",
        "gaussian-mixture",
        all(converged),
        observations,
        means,
        variances,
        logs,
        vars_,
        cfg,
        params,
    )
    return MarkovBaselineResult(
        candidate.family,
        candidate.distribution,
        cfg.markov_regimes,
        cfg.markov_order,
        candidate.converged,
        candidate.failure,
        candidate.fit_count,
        candidate.qlike,
        candidate.mean_predictive_log_score,
        candidate.var,
        candidate.parameter_stability,
        markov_note(),
    )


def full_markov_high_volatility_probability(
    response: np.ndarray, cfg: VolatilityBenchmarkConfig
) -> MarkovHighVolatilityProbability:
    """Return smoothed probability of the highest-variance Markov regime or failure."""

    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    except ModuleNotFoundError as exc:
        return MarkovHighVolatilityProbability(None, CandidateFailure("dependency", str(exc), 0))
    try:
        with temporary_numpy_seed(cfg.seed + 3000):
            fitted = MarkovAutoregression(
                response,
                k_regimes=cfg.markov_regimes,
                order=cfg.markov_order,
                trend="c",
                switching_ar=True,
                switching_trend=True,
                switching_variance=True,
            ).fit(
                method="bfgs",
                maxiter=cfg.markov_maxiter,
                disp=False,
                search_reps=cfg.markov_search_reps,
                search_iter=cfg.markov_search_iter,
            )
        converged = bool((getattr(fitted, "mle_retvals", {}) or {}).get("converged", False))
        if not converged:
            return MarkovHighVolatilityProbability(
                None,
                CandidateFailure(
                    "optimizer",
                    "full-sample Markov high-volatility regime fit did not converge",
                    1,
                ),
            )
        probs = np.asarray(fitted.smoothed_marginal_probabilities, dtype=float)
        names = list(fitted.model.param_names)
        params = np.asarray(fitted.params, dtype=float)
        variances = np.array([params[names.index(f"sigma2[{i}]")] for i in range(fitted.k_regimes)])
        aligned = np.full(len(response), np.nan, dtype=float)
        aligned[cfg.markov_order :] = probs[:, int(np.argmax(variances))]
        if np.any(~np.isfinite(aligned[cfg.markov_order :])):
            return MarkovHighVolatilityProbability(
                None,
                CandidateFailure(
                    "metric_nonfinite",
                    "nonfinite full-sample high-volatility probabilities",
                    1,
                ),
            )
        return MarkovHighVolatilityProbability(aligned.tolist(), None)
    except (
        FloatingPointError,
        IndexError,
        KeyError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return MarkovHighVolatilityProbability(
            None, CandidateFailure("optimizer", f"{type(exc).__name__}: {exc}", 1)
        )


def markov_forecast(
    fitted: Any, train: np.ndarray, holdout: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate sequential one-step Markov AR(2) forecasts on observed holdout data.

    At each hold-out date, the fitted transition matrix first propagates the
    latest filtered regime probabilities into a one-step-ahead prior. The
    forecast then uses only the two most recent observed response values
    (training data or earlier hold-out observations). The current hold-out
    observation updates the regime probabilities by Bayes' rule only after the
    mean, variance, log density, and VaR forecast for that date have been emitted.
    """

    names = list(fitted.model.param_names)
    values = np.asarray(fitted.params, dtype=float)
    k = int(fitted.k_regimes)
    const = np.array([values[names.index(f"const[{i}]")] for i in range(k)], dtype=float)
    sig2 = np.maximum(np.array([values[names.index(f"sigma2[{i}]")] for i in range(k)]), 1.0e-12)
    ar = np.array(
        [[values[names.index(f"ar.L{lag}[{i}]")] for lag in (1, 2)] for i in range(k)],
        dtype=float,
    )
    transition = statsmodels_row_stochastic_transition(fitted)
    prob = normalize(np.asarray(fitted.smoothed_marginal_probabilities, dtype=float)[-1])
    previous = [float(train[-1]), float(train[-2])]
    means: list[float] = []
    variances: list[float] = []
    log_scores: list[float] = []
    vars_: list[float] = []
    for observed in np.asarray(holdout, dtype=float):
        prob = normalize(prob @ transition)
        component_mean = const + ar[:, 0] * previous[0] + ar[:, 1] * previous[1]
        mean = float(np.dot(prob, component_mean))
        variance = float(np.dot(prob, sig2 + (component_mean - mean) ** 2))
        variances.append(max(variance, 1.0e-12))
        means.append(mean)
        log_scores.append(mixture_logpdf(float(observed), prob, component_mean, sig2))
        vars_.append(gaussian_mixture_var(prob, component_mean, sig2, alpha))
        prob = update_markov_probabilities(prob, float(observed), component_mean, sig2)
        previous = [float(observed), previous[0]]
    return (
        np.asarray(means, dtype=float),
        np.asarray(variances, dtype=float),
        np.asarray(log_scores, dtype=float),
        np.asarray(vars_, dtype=float),
    )


def markov_note() -> str:
    """Document Markov block forecast assumptions."""

    return (
        "Statsmodels MarkovAutoregression(k=3, AR(2)) is fitted on each rolling "
        "training block. Hold-out evaluation is sequential one-step-ahead: fitted "
        "statsmodels transition probabilities are transposed into an explicit "
        "row-stochastic previous-regime to next-regime matrix, propagated before "
        "each forecast, scored as a Gaussian regime mixture, and then filtered with "
        "the observed hold-out value by Bayes' rule. AR(2) lags are always observed "
        "training or earlier hold-out responses; no forecast means are fed back as "
        "pseudo-observations. VaR is the exact Gaussian-mixture quantile."
    )


def statsmodels_row_stochastic_transition(fitted: Any) -> np.ndarray:
    """Return a row-stochastic transition matrix from statsmodels orientation.

    Statsmodels exposes ``regime_transition[i, j, t]`` as the probability of
    next regime ``i`` conditional on previous regime ``j``. This repository uses
    row vectors, so row ``j`` / column ``i`` is needed before multiplying
    ``filtered @ transition``.
    """

    raw = np.asarray(fitted.regime_transition[:, :, 0], dtype=float)
    expected = (int(fitted.k_regimes), int(fitted.k_regimes))
    if raw.shape != expected:
        raise ValueError(f"unexpected Markov transition shape {raw.shape}, expected {expected}")
    return normalize_rows(raw.T)


def update_markov_probabilities(
    prior: np.ndarray, observed: float, component_mean: np.ndarray, component_variance: np.ndarray
) -> np.ndarray:
    """Filter Markov regime probabilities after observing one hold-out value."""

    log_terms = np.log(np.maximum(prior, 1.0e-300)) + normal_logpdf(
        observed, component_mean, component_variance
    )
    normalized_log_terms = log_terms - logsumexp(log_terms)
    return normalize(np.exp(normalized_log_terms))


def mixture_logpdf(
    observed: float, weights: np.ndarray, component_mean: np.ndarray, component_variance: np.ndarray
) -> float:
    """Return the predictive log density of a Gaussian regime mixture."""

    log_terms = np.log(np.maximum(weights, 1.0e-300)) + normal_logpdf(
        observed, component_mean, component_variance
    )
    return float(logsumexp(log_terms))


def gaussian_mixture_var(
    weights: np.ndarray, component_mean: np.ndarray, component_variance: np.ndarray, alpha: float
) -> float:
    """Return the exact alpha quantile of a univariate Gaussian mixture."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    probabilities = normalize(weights)
    means = np.asarray(component_mean, dtype=float)
    variances = np.maximum(np.asarray(component_variance, dtype=float), 1.0e-12)
    std = np.sqrt(variances)

    def cdf_minus_alpha(value: float) -> float:
        z = (value - means) / std
        return float(np.dot(probabilities, norm.cdf(z)) - alpha)

    mixture_mean = float(np.dot(probabilities, means))
    mixture_variance = float(np.dot(probabilities, variances + (means - mixture_mean) ** 2))
    mixture_std = math.sqrt(max(mixture_variance, 1.0e-12))
    lower = float(min(np.min(means - 10.0 * std), mixture_mean - 10.0 * mixture_std))
    upper = float(max(np.max(means + 10.0 * std), mixture_mean + 10.0 * mixture_std))
    for _ in range(12):
        if cdf_minus_alpha(lower) <= 0.0 and cdf_minus_alpha(upper) >= 0.0:
            return float(brentq(cdf_minus_alpha, lower, upper, xtol=1.0e-12, rtol=1.0e-12))
        width = upper - lower
        lower -= width
        upper += width
    raise ValueError("could not bracket Gaussian-mixture VaR quantile")


def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a probability vector after clipping negatives."""

    clipped = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    total = float(np.sum(clipped))
    return np.full_like(clipped, 1.0 / clipped.size) if total <= 0.0 else clipped / total


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of a transition matrix after clipping negatives."""

    rows = np.clip(np.asarray(matrix, dtype=float), 0.0, None)
    totals = rows.sum(axis=1)
    normalized = rows / np.where(totals <= 0.0, 1.0, totals)[:, None]
    normalized[totals <= 0.0] = 1.0 / rows.shape[1]
    return normalized


def normal_logpdf(x: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Vectorized normal log-density."""

    return -0.5 * (np.log(2.0 * np.pi * variance) + ((x - mean) ** 2) / variance)


def logsumexp(values: np.ndarray) -> float:
    """Small local log-sum-exp helper to avoid a scipy import."""

    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


class temporary_numpy_seed:
    """Temporarily seed NumPy global RNG for statsmodels multi-start search."""

    def __init__(self, seed: int) -> None:
        """Record the temporary seed while preserving the current RNG state."""

        self._seed = seed
        self._state: Any = None

    def __enter__(self) -> None:
        """Set deterministic temporary seed."""

        self._state = np.random.get_state()
        np.random.seed(self._seed)

    def __exit__(self, *args: object) -> None:
        """Restore prior RNG state."""

        if self._state is not None:
            np.random.set_state(self._state)
