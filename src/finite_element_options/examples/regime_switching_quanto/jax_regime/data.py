"""Read-only loader for the immutable PDP regime-study export."""

from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

from .contracts import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_LEVELS_SHA256,
    PDPObservationBatch,
    PDPPreprocessingAudit,
)

_MEMBER_ROOT = "regime-switching-quanto-2026-09-03/"


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _load_pdp_snapshot(
    archive: str | Path,
    *,
    expected_archive_sha256: str = EXPECTED_ARCHIVE_SHA256,
    expected_levels_sha256: str = EXPECTED_LEVELS_SHA256,
) -> tuple[PDPObservationBatch, bytes]:
    """Load bivariate log returns after verifying content-addressed PDP inputs.

    The two historically malformed USDCLP levels (near 5 rather than near
    500--1000) are rejected by the documented `>=100` quality rule. Missing
    joint levels are dropped before differencing, matching the accepted study.
    """

    path = Path(archive).expanduser().resolve()
    raw_archive = path.read_bytes()
    archive_hash = _digest(raw_archive)
    if archive_hash != expected_archive_sha256:
        raise ValueError(
            f"evidence archive SHA-256 mismatch: expected {expected_archive_sha256}, "
            f"received {archive_hash}"
        )
    with ZipFile(BytesIO(raw_archive)) as bundle:
        levels_raw = bundle.read(f"{_MEMBER_ROOT}pdp_joint_levels.csv")
        provenance = json.loads(bundle.read(f"{_MEMBER_ROOT}input_provenance.json"))
    levels_hash = _digest(levels_raw)
    if levels_hash != expected_levels_sha256:
        raise ValueError(
            f"PDP level SHA-256 mismatch: expected {expected_levels_sha256}, received {levels_hash}"
        )

    rows = list(csv.DictReader(StringIO(levels_raw.decode("utf-8"))))
    missing_equity = 0
    missing_fx = 0
    invalid_equity = 0
    invalid_fx = 0
    quarantined: list[tuple[str, str]] = []
    levels: list[tuple[str, float, float]] = []
    for row in rows:
        try:
            equity = float(row["sp500"])
        except (TypeError, ValueError):
            missing_equity += 1
            continue
        try:
            fx = float(row["usdclp"])
        except (TypeError, ValueError):
            missing_fx += 1
            continue
        if not math.isfinite(equity) or equity <= 0.0:
            invalid_equity += 1
            quarantined.append((row["date"], "equity_not_finite_or_positive"))
            continue
        if not math.isfinite(fx) or fx < 100.0:
            invalid_fx += 1
            quarantined.append((row["date"], "fx_not_finite_or_below_100"))
            continue
        levels.append((row["date"], equity, fx))
    if len(levels) < 2:
        raise ValueError("PDP export contains fewer than two valid joint levels")

    dates: list[str] = []
    returns: list[tuple[float, float]] = []
    for previous, current in zip(levels, levels[1:], strict=False):
        dates.append(current[0])
        returns.append((math.log(current[1] / previous[1]), math.log(current[2] / previous[2])))

    window = provenance["requested_window"]
    preprocessing = PDPPreprocessingAudit(
        source_level_rows=len(rows),
        valid_level_rows=len(levels),
        missing_or_unparseable_equity_rows=missing_equity,
        missing_or_unparseable_fx_rows=missing_fx,
        invalid_equity_rows=invalid_equity,
        invalid_fx_rows=invalid_fx,
        quarantined_rows=tuple(quarantined),
    )
    return (
        PDPObservationBatch(
            dates=tuple(dates),
            returns=tuple(returns),
            levels_sha256=levels_hash,
            archive_sha256=archive_hash,
            requested_start=str(window["start"]),
            requested_end_exclusive=str(window["end_exclusive"]),
            equity_spot=levels[-1][1],
            fx_spot=levels[-1][2],
            preprocessing=preprocessing,
        ),
        raw_archive,
    )


def load_pdp_observations(
    archive: str | Path,
    *,
    expected_archive_sha256: str = EXPECTED_ARCHIVE_SHA256,
    expected_levels_sha256: str = EXPECTED_LEVELS_SHA256,
) -> PDPObservationBatch:
    """Return observations parsed from the exact archive bytes that passed SHA-256 validation."""

    batch, _raw_archive = _load_pdp_snapshot(
        archive,
        expected_archive_sha256=expected_archive_sha256,
        expected_levels_sha256=expected_levels_sha256,
    )
    return batch
