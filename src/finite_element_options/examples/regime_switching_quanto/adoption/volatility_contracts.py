"""Typed records for volatility challenger benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from finite_element_options.examples.regime_switching_quanto._types import json_safe

SCHEMA_VERSION = "regime_volatility_benchmark.v1"
RESPONSE_DEFINITION = "100 * (sp500_log_return + usdclp_log_return)"
MetricDecision = Literal["promote", "reject"]
FailureKind = Literal["dependency", "optimizer", "metric_nonfinite", "input", "unknown"]


@dataclass(frozen=True)
class VolatilityBenchmarkConfig:
    """Deterministic rolling hold-out design for volatility challengers."""

    seed: int = 131
    holdout_size: int = 126
    rolling_window: int = 756
    refit_block: int = 21
    var_alpha: float = 0.05
    markov_regimes: int = 3
    markov_order: int = 2
    markov_search_reps: int = 2
    markov_search_iter: int = 5
    markov_maxiter: int = 150
    arch_maxiter: int = 150
    forecast_simulations: int = 400
    changepoint_window: int = 63
    changepoint_penalty: float = 6.0
    high_volatility_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate benchmark dimensions and probabilities."""

        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("holdout_size", "rolling_window", "refit_block"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rolling_window <= self.markov_order + 2:
            raise ValueError("rolling_window leaves too few observations after AR lags")
        if not 0.0 < self.var_alpha < 1.0:
            raise ValueError("var_alpha must be in (0, 1)")
        if self.markov_regimes < 2:
            raise ValueError("markov_regimes must be at least two")
        if self.changepoint_window < 2:
            raise ValueError("changepoint_window must be at least two")
        if self.changepoint_penalty <= 0.0:
            raise ValueError("changepoint_penalty must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe configuration."""

        return cast(dict[str, Any], json_safe(self))


@dataclass(frozen=True)
class RollingBoundary:
    """Integer train/hold-out indices for one no-leakage refit block."""

    train_start: int
    train_end: int
    holdout_start: int
    holdout_end: int


@dataclass(frozen=True)
class CandidateFailure:
    """Typed fail-closed benchmark failure."""

    kind: FailureKind
    message: str
    fit_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe failure details."""

        return cast(dict[str, Any], json_safe(self))


@dataclass(frozen=True)
class VarDiagnostics:
    """Value-at-Risk coverage diagnostics."""

    alpha: float
    exceedance_count: int
    exceedance_rate: float
    coverage_error: float
    kupiec_statistic: float | None
    kupiec_pvalue: float | None


@dataclass(frozen=True)
class CandidateBenchmarkResult:
    """Hold-out metrics for one ARCH-family challenger."""

    family: str
    distribution: str
    converged: bool
    failure: CandidateFailure | None
    fit_count: int
    qlike: float | None
    mean_predictive_log_score: float | None
    var: VarDiagnostics | None
    parameter_stability: dict[str, Any]


@dataclass(frozen=True)
class MarkovBaselineResult:
    """Hold-out metrics for the statsmodels Markov AR baseline."""

    family: str
    distribution: str
    regimes: int
    autoregressive_order: int
    converged: bool
    failure: CandidateFailure | None
    fit_count: int
    qlike: float | None
    mean_predictive_log_score: float | None
    var: VarDiagnostics | None
    parameter_stability: dict[str, Any]
    forecast_note: str


@dataclass(frozen=True)
class MarkovHighVolatilityProbability:
    """Full-sample high-volatility regime probabilities or typed failure."""

    probability: list[float] | None
    failure: CandidateFailure | None


@dataclass(frozen=True)
class VolatilityChangepoint:
    """Ruptures volatility/covariance changepoint mapped to a return date."""

    index: int
    date: str
    nearest_regime_gap_days: int | None


@dataclass(frozen=True)
class ChangepointComparison:
    """Ruptures method settings and comparison to high-volatility regime dates."""

    method: str
    model: str
    penalty: float
    window: int
    breakpoint_count: int
    breakpoints: list[VolatilityChangepoint]
    breakpoints_truncated: bool
    high_volatility_threshold: float
    regime_probability_failure: CandidateFailure | None
    nearest_regime_gap_days: int | None
    overlap_count: int
    overlap_rate: float


@dataclass(frozen=True)
class PromotionDecision:
    """Explicit challenger promotion or rejection decision."""

    decision: MetricDecision
    selected_candidate: str | None
    reasons: list[str]


@dataclass(frozen=True)
class VolatilityBenchmarkResult:
    """Immutable JSON-safe volatility benchmark artifact."""

    schema_version: str
    seed: int
    immutable_input_sha256: str
    observed_response: str
    train_start: str
    train_end: str
    holdout_start: str
    holdout_end: str
    fit_count: int
    metric_definitions: dict[str, str]
    config: dict[str, Any]
    data_quality: dict[str, Any]
    candidates: list[CandidateBenchmarkResult]
    markov_baseline: MarkovBaselineResult
    changepoints: ChangepointComparison
    decision: PromotionDecision
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible dictionary."""

        return cast(dict[str, Any], json_safe(self))
