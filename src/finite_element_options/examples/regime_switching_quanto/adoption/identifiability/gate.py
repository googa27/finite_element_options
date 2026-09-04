"""Fail-closed identifiability gate predicates."""

from __future__ import annotations

import math
from typing import Any

from .contracts import PARAMETERS, CalibrationCase


def identification_decision(
    case: CalibrationCase,
    minimum: dict[str, Any],
    hesse: dict[str, Any],
    minos: dict[str, Any],
    boundary_contact: dict[str, Any],
    finite_difference: dict[str, Any],
    profiles: dict[str, Any],
    optimizer_failure: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a fail-closed identified flag with actionable reasons."""

    reasons: list[str] = []
    if optimizer_failure is not None:
        reasons.append(f"optimizer failure: {optimizer_failure['type']}")
    if not minimum.get("valid") or minimum.get("fval") is None:
        reasons.append("minimum is not finite and valid")
    edm = minimum.get("edm")
    if not isinstance(edm, (float, int)) or float(edm) > case.edm_threshold:
        reasons.append("EDM exceeds threshold")
    flags = minimum.get("fmin_flags", {})
    if flags.get("has_parameters_at_limit"):
        reasons.append("iminuit reports parameters at limit")
    if boundary_contact.get("any_near_bound"):
        reasons.append("parameter is at or near a configured bound")
    if not _hesse_identified(hesse):
        reasons.append(
            "HESSE covariance is missing, repaired, inaccurate, or not positive definite"
        )
    if not _standard_errors_identified(minimum):
        reasons.append("finite positive HESSE standard errors are missing")
    if not _minos_identified(minos):
        reasons.append("valid two-sided MINOS intervals are missing")
    if not _finite_difference_identified(finite_difference):
        reasons.append("finite-difference local curvature is unstable or non-positive")
    if not _profiles_identified(profiles):
        reasons.append("bounded mnprofile traces do not bracket Delta-chi2=1 on both sides")
    return {
        "identified": not reasons,
        "reasons": reasons,
        "gate_version": "fail_closed.v1",
        "scope": "public-synthetic instrument-target calibration only",
    }


def _hesse_identified(hesse: dict[str, Any]) -> bool:
    quality = hesse.get("covariance_quality", {})
    return (
        hesse.get("status") == "available"
        and quality.get("has_covariance")
        and quality.get("accurate")
        and quality.get("positive_definite")
        and not quality.get("forced_positive_definite_repair")
        and not quality.get("hesse_failed")
    )


def _standard_errors_identified(minimum: dict[str, Any]) -> bool:
    errors = minimum.get("errors", {})
    return all(
        isinstance(errors.get(parameter), (float, int))
        and math.isfinite(float(errors[parameter]))
        and float(errors[parameter]) > 0.0
        for parameter in PARAMETERS
    )


def _minos_identified(minos: dict[str, Any]) -> bool:
    records = minos.get("parameters", {})
    for parameter in PARAMETERS:
        record = records.get(parameter, {})
        if not (
            record.get("status") == "available"
            and record.get("is_valid")
            and record.get("lower_valid")
            and record.get("upper_valid")
            and not record.get("at_lower_limit")
            and not record.get("at_upper_limit")
            and not record.get("lower_new_min")
            and not record.get("upper_new_min")
            and isinstance(record.get("lower"), (float, int))
            and isinstance(record.get("upper"), (float, int))
            and float(record["lower"]) < 0.0
            and float(record["upper"]) > 0.0
        ):
            return False
    return True


def _finite_difference_identified(finite_difference: dict[str, Any]) -> bool:
    records = finite_difference.get("parameters", {})
    return all(
        records.get(parameter, {}).get("status") == "available"
        and records.get(parameter, {}).get("positive_local_curvature")
        for parameter in PARAMETERS
    )


def _profiles_identified(profiles: dict[str, Any]) -> bool:
    records = profiles.get("parameters", {})
    for parameter in PARAMETERS:
        evidence = records.get(parameter, {}).get("evidence", {})
        if not (
            records.get(parameter, {}).get("status") == "available"
            and evidence.get("finite_stable")
            and evidence.get("lower_crosses_delta_chi2_1")
            and evidence.get("upper_crosses_delta_chi2_1")
        ):
            return False
    return True


__all__ = ["identification_decision"]
