"""Data quality preparation for PDP-style quanto level snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from finite_element_options.examples.regime_switching_quanto._types import (
    DataQualityConfig,
    json_safe,
)

FACTOR_COLUMNS = ("sp500", "usdclp")


def prepare_joint_log_returns(
    frame: pd.DataFrame,
    config: DataQualityConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean S&P 500 / USDCLP levels and return decimal log returns.

    Invalid level rows are quarantined before differencing, so each return uses
    the previous valid observation rather than a bad intermediate level.
    """

    cfg = config or DataQualityConfig()
    required = [cfg.date_column, *FACTOR_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if cfg.min_return_rows < 1:
        raise ValueError("min_return_rows must be at least 1")

    bounds = {
        "sp500": (cfg.sp500_min, cfg.sp500_max),
        "usdclp": (cfg.usdclp_min, cfg.usdclp_max),
    }
    working = frame.loc[:, required].copy()
    working[cfg.date_column] = pd.to_datetime(working[cfg.date_column], errors="coerce")
    for column in FACTOR_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.sort_values(cfg.date_column, kind="mergesort")

    quarantined: list[dict[str, Any]] = []
    valid_mask: list[bool] = []
    reason_counts: dict[str, int] = {}
    for index, row in working.iterrows():
        reasons = _row_reasons(row, cfg.date_column, bounds)
        valid_mask.append(not reasons)
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            quarantined.append(
                {
                    "index": json_safe(index),
                    "date": _date_to_str(row[cfg.date_column]),
                    "reasons": reasons,
                    "levels": {column: _finite_or_none(row[column]) for column in FACTOR_COLUMNS},
                }
            )

    valid_levels = working.loc[valid_mask, required].reset_index(drop=True)
    quarantined_dates = [
        value
        for value in working.loc[np.logical_not(valid_mask), cfg.date_column]
        if not pd.isna(value)
    ]
    needed = cfg.min_return_rows + 1
    if len(valid_levels) < needed:
        raise ValueError(
            "not enough valid adjacent observations after quarantine: "
            f"need {needed}, got {len(valid_levels)}"
        )

    levels = valid_levels.loc[:, list(FACTOR_COLUMNS)].astype(float)
    log_returns = np.log(levels / levels.shift(1)).iloc[1:]
    returns = pd.DataFrame(
        {
            "date": valid_levels.loc[1:, cfg.date_column].reset_index(drop=True),
            "sp500": log_returns["sp500"].to_numpy(dtype=float),
            "usdclp": log_returns["usdclp"].to_numpy(dtype=float),
        }
    )
    report = {
        "input_rows": int(len(frame)),
        "valid_level_rows": int(len(valid_levels)),
        "quarantined_row_count": int(len(quarantined)),
        "return_rows": int(len(returns)),
        "bridged_return_gaps": _bridged_gaps(valid_levels, cfg.date_column, quarantined_dates),
        "bounds": {
            "sp500": [cfg.sp500_min, cfg.sp500_max],
            "usdclp": [cfg.usdclp_min, cfg.usdclp_max],
        },
        "reason_counts": reason_counts,
        "quarantined_rows": quarantined,
    }
    return returns, report


def _bridged_gaps(
    valid_levels: pd.DataFrame,
    date_column: str,
    quarantined_dates: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    dates = valid_levels[date_column].reset_index(drop=True)
    gaps: list[dict[str, Any]] = []
    for idx in range(1, len(dates)):
        previous = dates.iloc[idx - 1]
        current = dates.iloc[idx]
        bridged = sorted(date for date in quarantined_dates if previous < date < current)
        if bridged:
            gaps.append(
                {
                    "previous_valid_date": _date_to_str(previous),
                    "return_date": _date_to_str(current),
                    "calendar_gap_days": int((current - previous).days),
                    "quarantined_dates": [_date_to_str(date) for date in bridged],
                }
            )
    return gaps


def _row_reasons(
    row: pd.Series,
    date_column: str,
    bounds: dict[str, tuple[float, float]],
) -> list[str]:
    reasons: list[str] = []
    if pd.isna(row[date_column]):
        reasons.append("date_nonfinite")
    for column, (lower, upper) in bounds.items():
        value = row[column]
        if not np.isfinite(value):
            reasons.append(f"{column}_nonfinite")
            continue
        if value <= 0.0:
            reasons.append(f"{column}_nonpositive")
            continue
        if value < lower or value > upper:
            reasons.append(f"{column}_out_of_bounds")
    return reasons


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _date_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)
