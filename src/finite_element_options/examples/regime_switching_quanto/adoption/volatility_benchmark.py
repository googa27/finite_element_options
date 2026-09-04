"""Volatility challenger benchmark for immutable quanto adoption evidence."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from finite_element_options.examples.regime_switching_quanto.adoption.optional import (
    require_optional,
)
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_changepoints import (
    detect_volatility_changepoints,
    joint_response,
)
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
    SCHEMA_VERSION,
    RESPONSE_DEFINITION,
    CandidateBenchmarkResult,
    CandidateFailure,
    MarkovBaselineResult,
    PromotionDecision,
    RollingBoundary,
    VolatilityBenchmarkConfig,
    VolatilityBenchmarkResult,
)
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_markov import (
    fit_markov_baseline,
    full_markov_high_volatility_probability,
)
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_metrics import (
    calculate_var_diagnostics,
    candidate_result,
    canonical_json,
    canonical_json_sha256,
    file_sha256,
    params_dict,
    rolling_boundaries,
    write_atomic_json,
    qlike_loss,
)
from finite_element_options.examples.regime_switching_quanto.quality import (
    prepare_joint_log_returns,
)

__all__ = [
    "CandidateFailure",
    "VolatilityBenchmarkConfig",
    "VolatilityBenchmarkResult",
    "calculate_var_diagnostics",
    "canonical_json",
    "canonical_json_sha256",
    "detect_volatility_changepoints",
    "file_sha256",
    "qlike_loss",
    "rolling_boundaries",
    "run_volatility_benchmark",
    "write_atomic_json",
]


def run_volatility_benchmark(
    levels: Any,
    *,
    input_sha256: str,
    config: VolatilityBenchmarkConfig | None = None,
) -> VolatilityBenchmarkResult:
    """Run the immutable evidence volatility challenger benchmark."""

    cfg = config or VolatilityBenchmarkConfig()
    returns, quality = prepare_joint_log_returns(levels)
    dates = list(returns["date"].dt.date.astype(str))
    response = joint_response(returns)
    boundaries = rolling_boundaries(len(response), cfg)
    candidates = [
        _fit_arch_candidate(returns, response, boundaries, cfg, "GJR-GARCH", "student-t"),
        _fit_arch_candidate(returns, response, boundaries, cfg, "GJR-GARCH", "skewed-t"),
        _fit_arch_candidate(returns, response, boundaries, cfg, "EGARCH", "student-t"),
        _fit_arch_candidate(returns, response, boundaries, cfg, "EGARCH", "skewed-t"),
    ]
    markov = fit_markov_baseline(response, boundaries, cfg)
    high_prob = full_markov_high_volatility_probability(response, cfg)
    changepoints = detect_volatility_changepoints(
        returns,
        high_volatility_probability=high_prob.probability,
        regime_probability_failure=high_prob.failure,
        window=cfg.changepoint_window,
        penalty=cfg.changepoint_penalty,
        threshold=cfg.high_volatility_probability_threshold,
    )
    first = boundaries[0]
    return VolatilityBenchmarkResult(
        schema_version=SCHEMA_VERSION,
        seed=cfg.seed,
        immutable_input_sha256=input_sha256,
        observed_response=RESPONSE_DEFINITION,
        train_start=dates[first.train_start],
        train_end=dates[first.train_end - 1],
        holdout_start=dates[first.holdout_start],
        holdout_end=dates[boundaries[-1].holdout_end - 1],
        fit_count=sum(candidate.fit_count for candidate in candidates) + markov.fit_count,
        metric_definitions=_metric_definitions(),
        config=cfg.to_dict(),
        data_quality=_summarize_data_quality(quality),
        candidates=candidates,
        markov_baseline=markov,
        changepoints=changepoints,
        decision=_promotion_decision(candidates, markov),
        limitations=[
            "ARCH challenger densities and Markov Gaussian-mixture scores are hold-out "
            "predictive scores on the same scalar response but are not nested likelihoods.",
            "Markov block forecasts are sequential one-step-ahead Gaussian mixtures "
            "with observed training/earlier-holdout AR(2) lags and post-score "
            "Bayesian regime filtering; they do not feed forecast means back as "
            "pseudo-observations.",
            "Changepoint comparison is descriptive and does not enter the promotion gate.",
            "Data-quality diagnostics summarize quarantines and bridged gaps without "
            "serializing raw level rows.",
        ],
    )


def _fit_arch_candidate(
    returns: Any,
    response: np.ndarray,
    boundaries: list[RollingBoundary],
    cfg: VolatilityBenchmarkConfig,
    family: str,
    distribution: str,
) -> CandidateBenchmarkResult:
    arch = require_optional("arch")
    dist_name = "StudentsT" if distribution == "student-t" else "skewstudent"
    vol_name, asym = ("GARCH", 1) if family == "GJR-GARCH" else ("EGARCH", 0)
    observations: list[float] = []
    means: list[float] = []
    variances: list[float] = []
    log_scores: list[float] = []
    var_values: list[float] = []
    params: list[dict[str, float]] = []
    converged: list[bool] = []
    try:
        for boundary in boundaries:
            block = response[boundary.train_start : boundary.holdout_end]
            train_size = boundary.train_end - boundary.train_start
            horizon = boundary.holdout_end - boundary.holdout_start
            model = arch.arch_model(
                block,
                mean="AR",
                lags=2,
                vol=vol_name,
                p=1,
                o=asym,
                q=1,
                dist=dist_name,
                rescale=False,
            )
            fitted = model.fit(
                disp="off",
                show_warning=False,
                last_obs=train_size,
                options={"maxiter": cfg.arch_maxiter},
            )
            params.append(params_dict(fitted.params))
            converged.append(int(getattr(fitted, "convergence_flag", 1)) == 0)
            forecast = fitted.forecast(
                start=train_size - 1,
                horizon=1,
                align="target",
                method="analytic",
                reindex=False,
            )
            mean = np.asarray(forecast.mean["h.1"].iloc[1 : horizon + 1], dtype=float)
            variance = np.maximum(
                np.asarray(forecast.residual_variance["h.1"].iloc[1 : horizon + 1], dtype=float),
                1.0e-12,
            )
            holdout = response[boundary.holdout_start : boundary.holdout_end]
            dist_params = _distribution_params(fitted, distribution)
            loglike = fitted.model.distribution.loglikelihood(
                dist_params, holdout - mean, variance, individual=True
            )
            quantile = fitted.model.distribution.ppf(cfg.var_alpha, dist_params)
            observations.extend(float(item) for item in holdout)
            means.extend(float(item) for item in mean)
            variances.extend(float(item) for item in variance)
            log_scores.extend(float(item) for item in np.asarray(loglike, dtype=float))
            var_values.extend(float(item) for item in mean + np.sqrt(variance) * quantile)
    except Exception as exc:  # pragma: no cover - numeric optimizer fallback
        return CandidateBenchmarkResult(
            family,
            distribution,
            False,
            CandidateFailure("optimizer", str(exc), len(params)),
            len(params),
            None,
            None,
            None,
            {},
        )
    return candidate_result(
        family,
        distribution,
        all(converged),
        observations,
        means,
        variances,
        log_scores,
        var_values,
        cfg,
        params,
    )


def _distribution_params(fitted: Any, distribution: str) -> np.ndarray:
    names = ["nu"] if distribution == "student-t" else ["eta", "lambda"]
    params = params_dict(fitted.params)
    return np.asarray([params[name] for name in names], dtype=float)


def _promotion_decision(
    candidates: list[CandidateBenchmarkResult], markov: MarkovBaselineResult
) -> PromotionDecision:
    successful = [c for c in candidates if c.failure is None and c.qlike is not None]
    if not successful:
        return PromotionDecision("reject", None, ["all challenger candidates failed closed"])
    best = min(successful, key=lambda c: cast(float, c.qlike))
    label = f"{best.family}/{best.distribution}"
    if markov.failure is not None or not markov.converged:
        if markov.failure is None:
            detail = "nonconverged without detailed optimizer failure"
        else:
            detail = f"{markov.failure.kind} {markov.failure.message}"
        return PromotionDecision(
            "reject",
            label,
            [
                "invalid Markov AR(2) baseline: "
                f"{detail}; challenger promotion is disabled until the baseline converges"
            ],
        )
    reasons: list[str] = []
    if markov.qlike is None or markov.mean_predictive_log_score is None or markov.var is None:
        reasons.append("invalid Markov AR(2) baseline metrics; no promotion comparison is valid")
    else:
        if cast(float, best.qlike) >= 0.95 * markov.qlike:
            reasons.append(
                "best challenger QLIKE did not improve on Markov baseline by at least 5%"
            )
        if (
            best.mean_predictive_log_score is None
            or best.mean_predictive_log_score <= markov.mean_predictive_log_score + 0.05
        ):
            reasons.append("best challenger log score did not beat Markov baseline by 0.05")
        if best.var is None or best.var.coverage_error > markov.var.coverage_error:
            reasons.append("best challenger VaR coverage error exceeded Markov baseline")
    stability = best.parameter_stability.get("l1_relative_first_last")
    if stability is None or float(stability) > 2.0:
        reasons.append("best challenger parameter stability gate failed")
    if reasons:
        return PromotionDecision("reject", label, reasons)
    return PromotionDecision("promote", label, ["best challenger passed all hold-out gates"])


def _metric_definitions() -> dict[str, str]:
    return {
        "response": RESPONSE_DEFINITION,
        "QLIKE": "mean(log(h_t) + (y_t - mean_t)^2 / h_t) on hold-out forecasts",
        "mean_predictive_log_score": "mean one-step hold-out predictive density log score for observed y_t",
        "VaR": "left-tail predictive quantile at config.var_alpha; exceedance is y_t < VaR_t",
        "Kupiec": "likelihood-ratio unconditional coverage statistic with chi-square(1) p-value",
        "parameter_stability": "L1 relative difference between first and last rolling refit common parameters",
    }


def _summarize_data_quality(quality: dict[str, Any]) -> dict[str, Any]:
    """Return bounded data-quality diagnostics for public benchmark artifacts."""

    bridged = list(quality.get("bridged_return_gaps", []))
    max_gap = max((int(item.get("calendar_gap_days", 0)) for item in bridged), default=0)
    return {
        "input_rows": quality.get("input_rows"),
        "valid_level_rows": quality.get("valid_level_rows"),
        "quarantined_row_count": quality.get("quarantined_row_count"),
        "return_rows": quality.get("return_rows"),
        "bounds": quality.get("bounds"),
        "reason_counts": quality.get("reason_counts", {}),
        "bridged_return_gap_count": len(bridged),
        "max_bridged_calendar_gap_days": max_gap,
    }
