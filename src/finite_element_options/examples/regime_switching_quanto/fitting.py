"""Statsmodels Markov-switching fit for the quanto research example."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from finite_element_options.examples.regime_switching_quanto._types import (
    MarkovSwitchingDiffusionResult,
    RegimeCandidateResult,
    RegimeSummary,
)
from finite_element_options.examples.regime_switching_quanto.generator import (
    TRADING_DAYS,
    discrete_to_continuous_generator,
)
from finite_element_options.examples.regime_switching_quanto.quality import FACTOR_COLUMNS


def fit_markov_switching_joint_diffusion(
    returns: pd.DataFrame,
    k_regimes: int = 3,
    autoregressive_order: int = 1,
    seed: int = 42,
    search_reps: int = 8,
    search_iter: int = 10,
    maxiter: int = 200,
    fit_attempts: int = 1,
) -> MarkovSwitchingDiffusionResult:
    """Fit a Markov-switching AR composite model and weighted bivariate summaries."""

    clean = _validate_returns(returns)
    if k_regimes < 1:
        raise ValueError("k_regimes must be at least 1")
    if fit_attempts < 1:
        raise ValueError("fit_attempts must be at least 1")
    if autoregressive_order < 0 or autoregressive_order > 5:
        raise ValueError("autoregressive_order must be between zero and five")
    if k_regimes == 1:
        return _fit_gaussian_baseline(clean, autoregressive_order)

    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "statsmodels":
            raise ImportError(
                "Regime calibration requires finite-element-options[calibration]."
            ) from exc
        raise

    composite = 100.0 * clean.loc[:, list(FACTOR_COLUMNS)].sum(axis=1).to_numpy(float)
    fitted_attempts: list[tuple[Any, dict[str, Any]]] = []
    for attempt in range(fit_attempts):
        attempt_seed = seed + attempt
        with _temporary_numpy_seed(attempt_seed):
            if autoregressive_order:
                model = MarkovAutoregression(
                    composite,
                    k_regimes=k_regimes,
                    order=autoregressive_order,
                    trend="c",
                    switching_ar=True,
                    switching_trend=True,
                    switching_variance=True,
                )
            else:
                model = MarkovRegression(
                    composite,
                    k_regimes=k_regimes,
                    trend="c",
                    switching_trend=True,
                    switching_variance=True,
                )
            candidate: Any = model.fit(
                method="bfgs",
                maxiter=maxiter,
                disp=False,
                search_reps=search_reps,
                search_iter=search_iter,
            )
        candidate_retvals = getattr(candidate, "mle_retvals", {}) or {}
        fitted_attempts.append(
            (
                candidate,
                {
                    "seed": attempt_seed,
                    "llf": float(candidate.llf),
                    "converged": bool(candidate_retvals.get("converged", False)),
                },
            )
        )
    fitted, _ = max(fitted_attempts, key=lambda item: float(item[0].llf))

    probabilities = _as_probability_array(fitted.smoothed_marginal_probabilities)
    transition = _normalize_rows(np.asarray(fitted.regime_transition[:, :, 0], float).T)
    aligned = clean.iloc[autoregressive_order:].reset_index(drop=True)
    order = _regime_order(aligned, probabilities)
    probabilities = probabilities[:, order]
    transition = _normalize_rows(transition[np.ix_(order, order)])

    regimes, occupancies = _regime_summaries(aligned, probabilities)
    generator, generator_residual = discrete_to_continuous_generator(transition)
    retvals = getattr(fitted, "mle_retvals", {}) or {}
    return MarkovSwitchingDiffusionResult(
        k_regimes=k_regimes,
        regimes=regimes,
        transition_matrix=transition.tolist(),
        continuous_time_generator=generator.tolist(),
        generator_residual=generator_residual,
        current_probabilities=_normalize_vector(probabilities[-1, :]).tolist(),
        occupancies=occupancies.tolist(),
        expected_durations=_expected_durations(transition),
        llf=float(fitted.llf),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        converged=bool(retvals.get("converged", False)),
        residual_diagnostics=_residual_diagnostics(
            np.asarray(fitted.resid, dtype=float), probabilities
        ),
        var1_diagnostics=_var1_diagnostics(clean),
        autoregressive_order=autoregressive_order,
        ar_coefficients=_ar_coefficients(fitted, order, autoregressive_order),
        fit_attempt_diagnostics=[diagnostic for _, diagnostic in fitted_attempts],
    )


def fit_regime_candidates(
    returns: pd.DataFrame,
    candidate_ks: list[int] | tuple[int, ...] | None = None,
    autoregressive_order: int = 1,
    seed: int = 42,
    search_reps: int = 4,
) -> list[RegimeCandidateResult]:
    """Fit candidate regime counts and return AIC/BIC/duration diagnostics."""

    candidates: list[RegimeCandidateResult] = []
    for k_regimes in list(candidate_ks or (1, 2, 3, 4)):
        fit = fit_markov_switching_joint_diffusion(
            returns,
            k_regimes=k_regimes,
            autoregressive_order=autoregressive_order,
            seed=seed,
            search_reps=search_reps,
        )
        candidates.append(
            RegimeCandidateResult(
                k_regimes=k_regimes,
                llf=fit.llf,
                aic=fit.aic,
                bic=fit.bic,
                converged=fit.converged,
                expected_durations=fit.expected_durations,
                occupancies=fit.occupancies,
                autoregressive_order=fit.autoregressive_order,
            )
        )
    return candidates


def _fit_gaussian_baseline(
    clean: pd.DataFrame, autoregressive_order: int
) -> MarkovSwitchingDiffusionResult:
    y = 100.0 * clean.loc[:, list(FACTOR_COLUMNS)].sum(axis=1).to_numpy(float)
    if autoregressive_order:
        try:
            from statsmodels.tsa.ar_model import AutoReg
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "statsmodels":
                raise ImportError(
                    "Regime calibration requires finite-element-options[calibration]."
                ) from exc
            raise

        fitted = AutoReg(y, lags=autoregressive_order, trend="c").fit()
        residuals = np.asarray(fitted.resid, dtype=float)
        ar_coefficients = [[float(value) for value in fitted.params[-autoregressive_order:]]]
        llf = float(fitted.llf)
        aic = float(fitted.aic)
        bic = float(fitted.bic)
    else:
        mu = float(np.mean(y))
        variance = max(float(np.var(y, ddof=0)), 1.0e-16)
        residuals = y - mu
        ar_coefficients = []
        llf = float(
            np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + (residuals * residuals) / variance))
        )
        n_params = 2
        aic = float(2 * n_params - 2 * llf)
        bic = float(np.log(len(clean)) * n_params - 2 * llf)

    aligned = clean.iloc[autoregressive_order:].reset_index(drop=True)
    probabilities = np.ones((len(aligned), 1), dtype=float)
    regimes, occupancies = _regime_summaries(aligned, probabilities)
    return MarkovSwitchingDiffusionResult(
        k_regimes=1,
        regimes=regimes,
        transition_matrix=[[1.0]],
        continuous_time_generator=[[0.0]],
        generator_residual=0.0,
        current_probabilities=[1.0],
        occupancies=occupancies.tolist(),
        expected_durations=[None],
        llf=llf,
        aic=aic,
        bic=bic,
        converged=True,
        residual_diagnostics=_residual_diagnostics(residuals, probabilities),
        var1_diagnostics=_var1_diagnostics(clean),
        autoregressive_order=autoregressive_order,
        ar_coefficients=ar_coefficients,
    )


def _validate_returns(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", *FACTOR_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required return columns: {missing}")
    clean = frame.loc[:, required].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean = clean.dropna(subset=["date", *FACTOR_COLUMNS]).sort_values("date")
    values = clean.loc[:, list(FACTOR_COLUMNS)].to_numpy(float)
    if len(clean) < 30 or not np.isfinite(values).all():
        raise ValueError("returns must contain at least 30 finite observations")
    return clean.reset_index(drop=True)


def _regime_order(clean: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    composite = clean.loc[:, list(FACTOR_COLUMNS)].sum(axis=1).to_numpy(float)
    variances = []
    for idx in range(probabilities.shape[1]):
        weights = probabilities[:, idx]
        mean = _weighted_mean(composite[:, None], weights)[0]
        variances.append(float(_weighted_cov(composite[:, None], weights, [mean])[0, 0]))
    return np.argsort(np.asarray(variances))


def _regime_summaries(
    clean: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[list[RegimeSummary], np.ndarray]:
    values = clean.loc[:, list(FACTOR_COLUMNS)].to_numpy(float)
    occupancies = probabilities.mean(axis=0)
    regimes: list[RegimeSummary] = []
    for idx in range(probabilities.shape[1]):
        weights = probabilities[:, idx]
        mean = _weighted_mean(values, weights)
        cov = _weighted_cov(values, weights, mean)
        annual_cov = cov * TRADING_DAYS
        comp_var = _weighted_cov(values.sum(axis=1)[:, None], weights, None)[0, 0]
        regimes.append(
            RegimeSummary(
                label=idx,
                occupancy=float(occupancies[idx]),
                daily_mean=mean.tolist(),
                annual_mean=(mean * TRADING_DAYS).tolist(),
                annual_covariance=annual_cov.tolist(),
                annual_volatility=np.sqrt(np.maximum(np.diag(annual_cov), 0)).tolist(),
                correlation=_correlation_from_cov(annual_cov).tolist(),
                composite_vol=float(np.sqrt(max(comp_var * TRADING_DAYS, 0.0))),
            )
        )
    return regimes, occupancies


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = max(float(np.sum(weights)), 1.0e-16)
    return np.sum(values * weights[:, None], axis=0) / total


def _weighted_cov(
    values: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray | list[float] | None,
) -> np.ndarray:
    mean_array = _weighted_mean(values, weights) if mean is None else np.asarray(mean)
    demeaned = values - mean_array
    total = max(float(np.sum(weights)), 1.0e-16)
    return (demeaned * weights[:, None]).T @ demeaned / total


def _correlation_from_cov(covariance: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denom = np.outer(vols, vols)
    corr = np.divide(covariance, denom, out=np.eye(len(vols)), where=denom > 0.0)
    return np.clip(corr, -1.0, 1.0)


def _residual_diagnostics(residuals: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.stats.stattools import jarque_bera
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "statsmodels":
            raise ImportError(
                "Regime calibration requires finite-element-options[calibration]."
            ) from exc
        raise

    values = np.asarray(residuals, dtype=float)
    variances = [
        _weighted_cov(values[:, None], probabilities[:, idx], None)[0, 0]
        for idx in range(probabilities.shape[1])
    ]
    fitted_var = np.maximum(probabilities @ np.maximum(variances, 1.0e-16), 1.0e-16)
    standardized = values / np.sqrt(fitted_var)
    lag = max(1, min(10, len(values) // 5))
    lb = acorr_ljungbox(standardized, lags=[lag], return_df=True)
    lb_sq = acorr_ljungbox(standardized * standardized, lags=[lag], return_df=True)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(standardized)
    return {
        "standardized_ljung_box_lag": lag,
        "standardized_ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "standardized_ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "squared_ljung_box_stat": float(lb_sq["lb_stat"].iloc[0]),
        "squared_ljung_box_pvalue": float(lb_sq["lb_pvalue"].iloc[0]),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
    }


def _ar_coefficients(
    fitted: Any, regime_order: np.ndarray, autoregressive_order: int
) -> list[list[float]]:
    """Extract switching AR coefficients and apply the volatility regime order."""

    if autoregressive_order == 0:
        return []
    names = list(fitted.model.param_names)
    params = np.asarray(fitted.params, dtype=float)
    coefficients = np.empty((len(regime_order), autoregressive_order), dtype=float)
    for regime in range(len(regime_order)):
        for lag in range(1, autoregressive_order + 1):
            name = f"ar.L{lag}[{regime}]"
            coefficients[regime, lag - 1] = params[names.index(name)]
    return coefficients[regime_order].tolist()


def _var1_diagnostics(clean: pd.DataFrame) -> dict[str, Any]:
    if len(clean) < 40:
        return {"available": False, "reason": "sample too small"}
    try:
        from statsmodels.tsa.api import VAR

        fitted = VAR(clean.loc[:, list(FACTOR_COLUMNS)].to_numpy(float)).fit(maxlags=1)
        nlags = max(2, min(10, len(clean) // 5))
        whiteness = fitted.test_whiteness(nlags=nlags, adjusted=False)
        return {
            "available": True,
            "lag_order": 1,
            "stable": bool(fitted.is_stable(verbose=False)),
            "whiteness_lags": nlags,
            "whiteness_stat": float(whiteness.test_statistic),
            "whiteness_pvalue": float(whiteness.pvalue),
        }
    except Exception as exc:  # pragma: no cover - statsmodels numeric fallback
        return {"available": False, "reason": str(exc)}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.clip(np.asarray(matrix, dtype=float), 0.0, 1.0)
    return matrix / matrix.sum(axis=1)[:, None]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    total = float(vector.sum())
    return np.full_like(vector, 1.0 / len(vector)) if total <= 0.0 else vector / total


def _as_probability_array(probabilities: Any) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2:
        raise ValueError("smoothed probabilities must be a 2D array")
    return np.apply_along_axis(_normalize_vector, 1, array)


def _expected_durations(transition: np.ndarray) -> list[float | None]:
    durations: list[float | None] = []
    for stay_probability in np.diag(transition):
        exit_probability = 1.0 - float(stay_probability)
        durations.append(None if exit_probability <= 1.0e-12 else 1.0 / exit_probability)
    return durations


class _temporary_numpy_seed:
    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._state: Any = None

    def __enter__(self) -> None:
        self._state = np.random.get_state()
        np.random.seed(self._seed)

    def __exit__(self, *args: object) -> None:
        if self._state is not None:
            np.random.set_state(self._state)
