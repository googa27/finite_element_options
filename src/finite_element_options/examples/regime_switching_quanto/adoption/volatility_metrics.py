"""Metric and serialization helpers for volatility benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from finite_element_options.examples.regime_switching_quanto._types import json_safe
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
    CandidateBenchmarkResult,
    CandidateFailure,
    RollingBoundary,
    VarDiagnostics,
    VolatilityBenchmarkConfig,
)


def rolling_boundaries(
    n_observations: int, config: VolatilityBenchmarkConfig
) -> list[RollingBoundary]:
    """Return rolling no-leakage train/hold-out blocks."""

    if n_observations < config.rolling_window + config.holdout_size:
        raise ValueError("not enough observations for rolling window plus holdout")
    holdout_start = n_observations - config.holdout_size
    blocks: list[RollingBoundary] = []
    start = holdout_start
    while start < n_observations:
        end = min(start + config.refit_block, n_observations)
        blocks.append(RollingBoundary(start - config.rolling_window, start, start, end))
        start = end
    return blocks


def qlike_loss(realized_variance_proxy: np.ndarray, variance_forecast: np.ndarray) -> np.ndarray:
    """Compute QLIKE ``log(h) + rv / h`` with positive finite forecasts only."""

    rv = np.asarray(realized_variance_proxy, dtype=float)
    h = np.asarray(variance_forecast, dtype=float)
    if rv.shape != h.shape or np.any(rv < 0.0) or np.any(~np.isfinite(rv)):
        raise ValueError("realized variance proxy must match forecasts and be finite")
    if np.any(h <= 0.0) or np.any(~np.isfinite(h)):
        raise ValueError("variance forecasts must be positive and finite")
    return np.log(h) + rv / h


def calculate_var_diagnostics(
    observations: np.ndarray, var_forecasts: np.ndarray, alpha: float
) -> VarDiagnostics:
    """Compute exceedance and Kupiec unconditional coverage diagnostics."""

    y = np.asarray(observations, dtype=float)
    var_values = np.asarray(var_forecasts, dtype=float)
    if y.shape != var_values.shape or y.size == 0:
        raise ValueError("observations and VaR forecasts must have the same non-empty shape")
    exceed = y < var_values
    count = int(np.sum(exceed))
    rate = float(count / y.size)
    statistic, pvalue = _kupiec(count, int(y.size), alpha)
    return VarDiagnostics(
        alpha=float(alpha),
        exceedance_count=count,
        exceedance_rate=rate,
        coverage_error=abs(rate - alpha),
        kupiec_statistic=statistic,
        kupiec_pvalue=pvalue,
    )


def candidate_result(
    family: str,
    distribution: str,
    converged: bool,
    observations: list[float],
    means: list[float],
    variances: list[float],
    log_scores: list[float],
    var_values: list[float],
    cfg: VolatilityBenchmarkConfig,
    params: list[dict[str, float]],
) -> CandidateBenchmarkResult:
    """Build a fail-closed candidate result from forecast arrays."""

    y = np.asarray(observations, dtype=float)
    mu = np.asarray(means, dtype=float)
    variance = np.asarray(variances, dtype=float)
    logs = np.asarray(log_scores, dtype=float)
    try:
        qlike = float(np.mean(qlike_loss((y - mu) ** 2, variance)))
        log_score = float(np.mean(logs))
        var = calculate_var_diagnostics(y, np.asarray(var_values, dtype=float), cfg.var_alpha)
    except ValueError as exc:
        return CandidateBenchmarkResult(
            family,
            distribution,
            False,
            CandidateFailure("metric_nonfinite", str(exc), len(params)),
            len(params),
            None,
            None,
            None,
            parameter_stability(params),
        )
    if not all(math.isfinite(value) for value in [qlike, log_score, var.exceedance_rate]):
        return CandidateBenchmarkResult(
            family,
            distribution,
            False,
            CandidateFailure("metric_nonfinite", "nonfinite hold-out metric", len(params)),
            len(params),
            None,
            None,
            None,
            parameter_stability(params),
        )
    failure = (
        None
        if converged
        else CandidateFailure("optimizer", "one or more fits did not converge", len(params))
    )
    return CandidateBenchmarkResult(
        family,
        distribution,
        converged,
        failure,
        len(params),
        qlike,
        log_score,
        var,
        parameter_stability(params),
    )


def parameter_stability(params: list[dict[str, float]]) -> dict[str, Any]:
    """Compare first and last rolling refit parameters."""

    if len(params) < 2:
        return {"available": False, "reason": "fewer than two fits"}
    common = sorted(set(params[0]) & set(params[-1]))
    if not common:
        return {"available": False, "reason": "no common parameters"}
    first = np.asarray([params[0][key] for key in common], dtype=float)
    last = np.asarray([params[-1][key] for key in common], dtype=float)
    value = float(np.sum(np.abs(last - first)) / max(np.sum(np.abs(first)), 1.0e-12))
    return {"available": True, "l1_relative_first_last": value, "parameter_count": len(common)}


def params_dict(params: Any, names: list[str] | None = None) -> dict[str, float]:
    """Convert optimizer parameters to a string-keyed float dictionary."""

    if hasattr(params, "items"):
        return {str(key): float(value) for key, value in params.items()}
    labels = names or [str(i) for i in range(len(params))]
    return {label: float(value) for label, value in zip(labels, params, strict=True)}


def canonical_json(payload: Any) -> str:
    """Serialize payload as deterministic JSON with no local path dependence."""

    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def canonical_json_sha256(payload: Any) -> str:
    """Return SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 of a file without storing its path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_json(path: str | Path, payload: Any) -> str:
    """Atomically write canonical JSON and return the artifact hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _kupiec(count: int, nobs: int, alpha: float) -> tuple[float | None, float | None]:
    if nobs <= 0:
        return None, None
    rate = min(max(count / nobs, 1.0e-12), 1.0 - 1.0e-12)
    a = min(max(alpha, 1.0e-12), 1.0 - 1.0e-12)
    ll_null = count * math.log(a) + (nobs - count) * math.log(1.0 - a)
    ll_alt = count * math.log(rate) + (nobs - count) * math.log(1.0 - rate)
    stat = max(0.0, -2.0 * (ll_null - ll_alt))
    return stat, float(math.erfc(math.sqrt(stat / 2.0)))
