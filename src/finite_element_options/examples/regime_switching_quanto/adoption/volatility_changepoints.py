"""Ruptures changepoint helpers for volatility benchmark diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from finite_element_options.examples.regime_switching_quanto.adoption.optional import (
    require_optional,
)
from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
    CandidateFailure,
    ChangepointComparison,
    VolatilityChangepoint,
)

MAX_SERIALIZED_CHANGEPOINTS = 20


def detect_volatility_changepoints(
    returns: Any,
    *,
    high_volatility_probability: np.ndarray | list[float] | None,
    regime_probability_failure: CandidateFailure | None = None,
    window: int,
    penalty: float,
    threshold: float = 0.5,
) -> ChangepointComparison:
    """Detect volatility/covariance changepoints with ruptures PELT."""

    rpt = require_optional("ruptures")
    frame = returns.sort_values("date").reset_index(drop=True)
    y = joint_response(frame)
    sp = 100.0 * frame["sp500"].to_numpy(float)
    fx = 100.0 * frame["usdclp"].to_numpy(float)
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Volatility changepoints require finite-element-options[calibration]."
        ) from exc
    features = (
        pd.DataFrame(
            {
                "composite_abs": np.abs(y),
                "composite_sq": y * y,
                "sp500_sq": sp * sp,
                "usdclp_sq": fx * fx,
                "cross": sp * fx,
            }
        )
        .rolling(window=window, min_periods=window)
        .mean()
        .dropna()
    )
    if len(features) < max(2 * window, 10):
        raise ValueError("not enough rows for changepoint window")
    scaled = (features - features.mean()) / features.std(ddof=0).replace(0.0, 1.0)
    raw_breaks = (
        rpt.Pelt(model="rbf", min_size=max(2, window // 2))
        .fit(scaled.to_numpy(float))
        .predict(pen=float(penalty))
    )
    feature_start = window - 1
    regime_dates = regime_transition_dates(frame, high_volatility_probability, threshold)
    all_breakpoints: list[VolatilityChangepoint] = []
    for point in raw_breaks:
        if point >= len(features):
            continue
        index = int(feature_start + point - 1)
        date = frame["date"].iloc[index].date().isoformat()
        gap = nearest_gap_days(frame["date"].iloc[index], regime_dates)
        all_breakpoints.append(
            VolatilityChangepoint(index=index, date=date, nearest_regime_gap_days=gap)
        )
    gaps = [
        bp.nearest_regime_gap_days
        for bp in all_breakpoints
        if bp.nearest_regime_gap_days is not None
    ]
    overlap = sum(1 for gap in gaps if gap <= window)
    breakpoints = _bounded_breakpoints(all_breakpoints)
    return ChangepointComparison(
        method="ruptures.Pelt",
        model="rbf",
        penalty=float(penalty),
        window=int(window),
        breakpoint_count=len(all_breakpoints),
        breakpoints=breakpoints,
        breakpoints_truncated=len(breakpoints) < len(all_breakpoints),
        high_volatility_threshold=float(threshold),
        regime_probability_failure=regime_probability_failure,
        nearest_regime_gap_days=min(gaps) if gaps else None,
        overlap_count=int(overlap),
        overlap_rate=float(overlap / len(all_breakpoints)) if all_breakpoints else 0.0,
    )


def joint_response(frame: Any) -> np.ndarray:
    """Return the benchmark scalar response from joint equity/FX returns."""

    return 100.0 * frame.loc[:, ["sp500", "usdclp"]].sum(axis=1).to_numpy(float)


def regime_transition_dates(
    frame: Any, probability: np.ndarray | list[float] | None, threshold: float
) -> list[Any]:
    """Map high-volatility probability threshold transitions to dates."""

    if probability is None:
        return []
    probs = np.asarray(probability, dtype=float)
    if len(probs) != len(frame):
        return []
    mask = probs >= threshold
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    indices = changes if len(changes) else np.flatnonzero(mask)
    return [frame["date"].iloc[int(idx)] for idx in indices]


def _bounded_breakpoints(
    breakpoints: list[VolatilityChangepoint],
) -> list[VolatilityChangepoint]:
    if len(breakpoints) <= MAX_SERIALIZED_CHANGEPOINTS:
        return breakpoints
    head = MAX_SERIALIZED_CHANGEPOINTS // 2
    tail = MAX_SERIALIZED_CHANGEPOINTS - head
    return [*breakpoints[:head], *breakpoints[-tail:]]


def nearest_gap_days(date_value: Any, regime_dates: list[Any]) -> int | None:
    """Return nearest absolute calendar-day gap to regime transition/high-vol dates."""

    if not regime_dates:
        return None
    return min(abs((date_value - item).days) for item in regime_dates)
