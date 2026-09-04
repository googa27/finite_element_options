"""Lazy iminuit adapter and identification gate for profile diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
    canonical_json_sha256,
)
from finite_element_options.examples.regime_switching_quanto.adoption.optional import (
    require_optional,
)

from .contracts import (
    OBJECTIVE_NAME,
    PARAMETERS,
    SCHEMA_VERSION,
    CalibrationCase,
    IdentifiabilityResult,
    WeightedQuantoCalibrationObjective,
)
from .gate import identification_decision

DELTA_CHI2_ONE = 1.0
MIN_CURVATURE = 1.0e-8
PROFILE_CENTER_DELTA_TOLERANCE = 1.0e-6
PROFILE_CENTER_SPACING_FRACTION = 0.51


def run_iminuit_identifiability(case: CalibrationCase) -> IdentifiabilityResult:
    """Run migrad, hesse, minos, mnprofile, and fail-closed gates."""

    case_input = case.to_input_dict()
    case_hash = canonical_json_sha256(case_input)
    objective = WeightedQuantoCalibrationObjective(case)
    try:
        iminuit = require_optional("iminuit")
        minuit = iminuit.Minuit(
            objective,
            equity_vol=case.initial_equity_vol,
            correlation=case.initial_correlation,
        )
    except ImportError as exc:
        failure = _optimizer_failure("missing_optional_dependency", str(exc))
        return _failed_result(case, case_hash, objective, failure)
    except Exception as exc:  # pragma: no cover - defensive serialization path
        failure = _optimizer_failure(type(exc).__name__, str(exc))
        return _failed_result(case, case_hash, objective, failure)

    minuit.errordef = 1.0
    for parameter in PARAMETERS:
        minuit.limits[parameter] = case.bounds.for_parameter(parameter)
    optimizer_failure: dict[str, Any] | None = None
    try:
        minuit.migrad()
        minuit.hesse()
        minuit.minos(*PARAMETERS, cl=case.minos_cl, ncall=case.minos_ncall)
    except Exception as exc:
        optimizer_failure = _optimizer_failure(type(exc).__name__, str(exc))

    minimum = _serialize_minimum(minuit)
    hesse = _serialize_hesse(minuit)
    minos = _serialize_minos(minuit, optimizer_failure)
    boundary_contact = _boundary_contact(case, minimum.get("values", {}))
    finite_difference = _finite_difference_diagnostics(case, objective, minimum)
    profiles = _profile_diagnostics(case, minuit, minimum)
    optimizer = {
        "library": "iminuit",
        "version": str(getattr(iminuit, "__version__", "unknown")),
        "errordef": 1.0,
        "failure": optimizer_failure,
    }
    identification = identification_decision(
        case,
        minimum,
        hesse,
        minos,
        boundary_contact,
        finite_difference,
        profiles,
        optimizer_failure,
    )
    return IdentifiabilityResult(
        schema_version=SCHEMA_VERSION,
        case_input_hash=case_hash,
        case=case_input,
        objective_name=OBJECTIVE_NAME,
        optimizer=optimizer,
        minimum=minimum,
        hesse=hesse,
        minos=minos,
        boundary_contact=boundary_contact,
        finite_difference=finite_difference,
        profiles=profiles,
        identification=identification,
        objective_diagnostics=objective.diagnostics(),
    )


def _failed_result(
    case: CalibrationCase,
    case_hash: str,
    objective: WeightedQuantoCalibrationObjective,
    failure: dict[str, Any],
) -> IdentifiabilityResult:
    return IdentifiabilityResult(
        schema_version=SCHEMA_VERSION,
        case_input_hash=case_hash,
        case=case.to_input_dict(),
        objective_name=OBJECTIVE_NAME,
        optimizer={"library": "iminuit", "version": None, "errordef": 1.0, "failure": failure},
        minimum={"status": "missing", "reason": failure["type"]},
        hesse={"status": "missing", "reason": failure["type"]},
        minos={"status": "missing", "reason": failure["type"]},
        boundary_contact={"status": "missing", "reason": failure["type"]},
        finite_difference={"status": "missing", "reason": failure["type"]},
        profiles={"status": "missing", "reason": failure["type"]},
        identification={
            "identified": False,
            "reasons": [f"optimizer failure: {failure['type']}"],
            "gate_version": "fail_closed.v1",
        },
        objective_diagnostics=objective.diagnostics(),
    )


def _optimizer_failure(kind: str, message: str) -> dict[str, Any]:
    del message
    return {
        "type": str(kind),
        "message": "optimizer exception captured; raw implementation details suppressed",
    }


def _serialize_minimum(minuit: Any) -> dict[str, Any]:
    fmin = getattr(minuit, "fmin", None)
    values = {name: _finite_or_none(minuit.values[name]) for name in PARAMETERS}
    errors = {name: _finite_or_none(minuit.errors[name]) for name in PARAMETERS}
    flags = _fmin_flags(fmin)
    return {
        "status": "available",
        "valid": bool(getattr(fmin, "is_valid", False)),
        "fval": _finite_or_none(getattr(fmin, "fval", math.inf)),
        "edm": _finite_or_none(getattr(fmin, "edm", math.inf)),
        "edm_goal": _finite_or_none(getattr(fmin, "edm_goal", math.inf)),
        "nfcn": int(getattr(fmin, "nfcn", 0)),
        "values": values,
        "errors": errors,
        "fmin_flags": flags,
    }


def _fmin_flags(fmin: Any) -> dict[str, Any]:
    names = (
        "has_accurate_covar",
        "has_covariance",
        "has_made_posdef_covar",
        "has_parameters_at_limit",
        "has_posdef_covar",
        "has_reached_call_limit",
        "has_valid_parameters",
        "hesse_failed",
        "is_above_max_edm",
        "is_valid",
    )
    return {name: bool(getattr(fmin, name, False)) for name in names}


def _serialize_hesse(minuit: Any) -> dict[str, Any]:
    fmin = getattr(minuit, "fmin", None)
    flags = _fmin_flags(fmin)
    covariance = getattr(minuit, "covariance", None)
    if covariance is None:
        matrix: dict[str, Any] | None = None
        status = "missing"
        reason = "not_available"
    else:
        array = np.asarray(covariance, dtype=float)
        if array.shape != (len(PARAMETERS), len(PARAMETERS)) or not np.all(np.isfinite(array)):
            matrix = None
            status = "missing"
            reason = "invalid"
        else:
            symmetric = 0.5 * (array + array.T)
            matrix = {
                row_name: {
                    col_name: float(symmetric[i, j]) for j, col_name in enumerate(PARAMETERS)
                }
                for i, row_name in enumerate(PARAMETERS)
            }
            status = "available"
            reason = None
    return {
        "status": status,
        "reason": reason,
        "covariance_quality": {
            "accurate": flags["has_accurate_covar"],
            "positive_definite": flags["has_posdef_covar"],
            "forced_positive_definite_repair": flags["has_made_posdef_covar"],
            "hesse_failed": flags["hesse_failed"],
            "has_covariance": flags["has_covariance"],
        },
        "covariance": matrix,
    }


def _serialize_minos(minuit: Any, optimizer_failure: dict[str, Any] | None) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for parameter in PARAMETERS:
        error = getattr(minuit, "merrors", {}).get(parameter)
        if error is None:
            records[parameter] = {
                "status": "missing",
                "reason": "optimizer_failure" if optimizer_failure else "not_available",
            }
            continue
        records[parameter] = {
            "status": "available",
            "lower": _finite_or_none(error.lower),
            "upper": _finite_or_none(error.upper),
            "lower_valid": bool(error.lower_valid),
            "upper_valid": bool(error.upper_valid),
            "is_valid": bool(error.is_valid),
            "at_lower_limit": bool(error.at_lower_limit),
            "at_upper_limit": bool(error.at_upper_limit),
            "at_lower_max_fcn": bool(error.at_lower_max_fcn),
            "at_upper_max_fcn": bool(error.at_upper_max_fcn),
            "lower_new_min": bool(error.lower_new_min),
            "upper_new_min": bool(error.upper_new_min),
            "nfcn": int(error.nfcn),
        }
    return {"status": "available", "parameters": records}


def _boundary_contact(case: CalibrationCase, values: dict[str, Any]) -> dict[str, Any]:
    contacts: dict[str, Any] = {}
    any_contact = False
    for parameter in PARAMETERS:
        value = values.get(parameter)
        lower, upper = case.bounds.for_parameter(parameter)
        numeric_value = (
            float(value)
            if isinstance(value, (float, int)) and math.isfinite(float(value))
            else None
        )
        lower_distance = numeric_value - lower if numeric_value is not None else None
        upper_distance = upper - numeric_value if numeric_value is not None else None
        near = (
            numeric_value is not None
            and lower_distance is not None
            and upper_distance is not None
            and min(lower_distance, upper_distance) <= case.bound_contact_tolerance
        )
        any_contact = any_contact or near
        contacts[parameter] = {
            "value": value,
            "bounds": [lower, upper],
            "lower_distance": lower_distance,
            "upper_distance": upper_distance,
            "near_bound": near,
        }
    return {"status": "available", "any_near_bound": any_contact, "parameters": contacts}


def _finite_difference_diagnostics(
    case: CalibrationCase,
    objective: WeightedQuantoCalibrationObjective,
    minimum: dict[str, Any],
) -> dict[str, Any]:
    values = minimum.get("values", {})
    if not _minimum_has_finite_values(values):
        return {"status": "missing", "reason": "nonfinite_minimum"}
    base = {name: float(values[name]) for name in PARAMETERS}
    base_value = objective(**base)
    records: dict[str, Any] = {}
    for parameter in PARAMETERS:
        h = float(case.finite_difference_step)
        lower, upper = case.bounds.for_parameter(parameter)
        if base[parameter] - h < lower or base[parameter] + h > upper:
            records[parameter] = {"status": "missing", "reason": "step_outside_bounds"}
            continue
        lower_point = dict(base)
        upper_point = dict(base)
        lower_point[parameter] -= h
        upper_point[parameter] += h
        f_minus = objective(**lower_point)
        f_plus = objective(**upper_point)
        finite = all(math.isfinite(value) for value in (base_value, f_minus, f_plus))
        gradient = (f_plus - f_minus) / (2.0 * h) if finite else None
        curvature = (f_plus - 2.0 * base_value + f_minus) / (h * h) if finite else None
        records[parameter] = {
            "status": "available" if finite else "missing",
            "step": h,
            "gradient": _finite_or_none(gradient),
            "curvature": _finite_or_none(curvature),
            "positive_local_curvature": bool(
                curvature is not None and math.isfinite(curvature) and curvature > MIN_CURVATURE
            ),
        }
    return {"status": "available", "parameters": records}


def _profile_diagnostics(
    case: CalibrationCase, minuit: Any, minimum: dict[str, Any]
) -> dict[str, Any]:
    values = minimum.get("values", {})
    if not _minimum_has_finite_values(values):
        return {"status": "missing", "reason": "nonfinite_minimum"}
    result: dict[str, Any] = {}
    for grid in case.profile_grids:
        parameter = grid.parameter
        try:
            points, deltas, valid = minuit.mnprofile(
                parameter,
                grid=grid.values(),
                subtract_min=True,
                ncall=case.minos_ncall,
                use_simplex=False,
            )
            trace = [
                {
                    "value": float(point),
                    "delta_chi2": float(delta),
                    "valid": bool(is_valid),
                }
                for point, delta, is_valid in zip(points, deltas, valid, strict=True)
            ]
            evidence = _profile_evidence(trace, float(values[parameter]))
            result[parameter] = {
                "status": "available",
                "grid": grid.to_dict(),
                "trace": trace,
                "evidence": evidence,
            }
        except Exception as exc:
            result[parameter] = {
                "status": "failed",
                "grid": grid.to_dict(),
                "failure": _optimizer_failure(type(exc).__name__, str(exc)),
                "trace": [],
                "evidence": {
                    "finite_stable": False,
                    "lower_crosses_delta_chi2_1": False,
                    "upper_crosses_delta_chi2_1": False,
                },
            }
    return {"status": "available", "parameters": result}


def _profile_evidence(trace: list[dict[str, Any]], best_value: float) -> dict[str, Any]:
    finite_stable = bool(trace) and all(
        row["valid"]
        and math.isfinite(float(row["value"]))
        and math.isfinite(float(row["delta_chi2"]))
        and float(row["delta_chi2"]) >= -1.0e-8
        for row in trace
    )
    ordered = sorted(trace, key=lambda row: float(row["value"]))
    center_supported = finite_stable and _profile_center_is_supported(ordered, best_value)
    lower_side = sorted(
        (row for row in ordered if float(row["value"]) < best_value),
        key=lambda row: abs(float(row["value"]) - best_value),
    )
    upper_side = sorted(
        (row for row in ordered if float(row["value"]) > best_value),
        key=lambda row: abs(float(row["value"]) - best_value),
    )
    min_delta = min((float(row["delta_chi2"]) for row in trace), default=math.inf)
    lower_cross = center_supported and _side_brackets_delta_one(lower_side)
    upper_cross = center_supported and _side_brackets_delta_one(upper_side)
    return {
        "finite_stable": finite_stable,
        "lower_crosses_delta_chi2_1": lower_cross,
        "upper_crosses_delta_chi2_1": upper_cross,
        "delta_chi2_threshold": DELTA_CHI2_ONE,
        "min_delta_chi2": min_delta if math.isfinite(min_delta) else None,
        "max_delta_chi2": max((float(row["delta_chi2"]) for row in trace), default=None),
    }


def _profile_center_is_supported(rows: list[dict[str, Any]], best_value: float) -> bool:
    if not rows:
        return False
    values = [float(row["value"]) for row in rows]
    positive_spacings = [
        right - left for left, right in zip(values, values[1:], strict=False) if right > left
    ]
    if not positive_spacings:
        value_tolerance = 1.0e-12
    else:
        value_tolerance = max(
            1.0e-12,
            PROFILE_CENTER_SPACING_FRACTION * min(positive_spacings),
        )
    nearest = min(rows, key=lambda row: abs(float(row["value"]) - best_value))
    return (
        abs(float(nearest["value"]) - best_value) <= value_tolerance
        and abs(float(nearest["delta_chi2"])) <= PROFILE_CENTER_DELTA_TOLERANCE
    )


def _side_brackets_delta_one(rows: list[dict[str, Any]]) -> bool:
    previous_delta = 0.0
    for row in rows:
        current_delta = float(row["delta_chi2"])
        if (
            min(previous_delta, current_delta)
            <= DELTA_CHI2_ONE
            <= max(previous_delta, current_delta)
        ):
            return True
        previous_delta = current_delta
    return False


def _minimum_has_finite_values(values: dict[str, Any]) -> bool:
    return all(
        isinstance(values.get(parameter), (float, int)) and math.isfinite(float(values[parameter]))
        for parameter in PARAMETERS
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = ["DELTA_CHI2_ONE", "MIN_CURVATURE", "run_iminuit_identifiability"]
