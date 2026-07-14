"""Fail-closed validation gates for versioned FEM numerical evidence."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isclose, isfinite, log
from typing import Any, Callable

from ..black_scholes_parity import (
    DEFAULT_DELTA_TOLERANCE_ABSOLUTE,
    DEFAULT_GAMMA_TOLERANCE_ABSOLUTE,
    DEFAULT_TOLERANCE_ABSOLUTE,
    DEFAULT_TOLERANCE_RELATIVE,
    run_public_black_scholes_parity_fixture,
)
from ..verification_gates import OptionSurfacePoint, evaluate_call_arbitrage
from .black_scholes_surface import solve_black_scholes_surface
from .manufactured import sympy_manufactured_problem

FEM_VERIFICATION_SCHEMA_VERSION = "fem-verification-evidence/v1"
_NUMERICAL_VALIDATION_REL_TOL = 1.0e-9
_NUMERICAL_VALIDATION_ABS_TOL = 1.0e-8


def convention_contract() -> dict[str, Any]:
    """Return the canonical manufactured-problem and time-orientation convention."""

    return {
        "manufactured": sympy_manufactured_problem(),
        "black_scholes": {
            "time": "tau=T-t",
            "measure": "risk_neutral",
            "numeraire": "money_market_account",
            "operator_sign": "forward tau Black-Scholes",
        },
    }


def canonical_hash(payload: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible evidence payloads."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def tolerance_taxonomy() -> dict[str, Any]:
    """Return immutable tolerance classes used by generation and validation."""

    return {
        "discretization": {
            "manufactured_l2": 5.0e-4,
            "manufactured_h1": 2.5e-2,
            "manufactured_payoff": 2.0e-4,
            "black_scholes_price": DEFAULT_TOLERANCE_ABSOLUTE,
            "black_scholes_delta": DEFAULT_DELTA_TOLERANCE_ABSOLUTE,
            "black_scholes_gamma": DEFAULT_GAMMA_TOLERANCE_ABSOLUTE,
        },
        "oracle": {
            "sympy_symbolic_residual": 1.0e-12,
            "black_scholes_analytic": 1.0e-12,
        },
        "algebraic": {
            "linear_solve_residual_inf": 1.0e-11,
            "boundary_residual_inf": 1.0e-12,
        },
        "floating_point": {
            "hashes": "sha256 canonical JSON",
            "dtype": "float64",
        },
    }


def validate_evidence(evidence: dict[str, Any]) -> None:
    """Recompute identities and numerical gates; never trust stored pass booleans."""

    if evidence.get("schema_version") != FEM_VERIFICATION_SCHEMA_VERSION:
        raise ValueError("unexpected verification evidence schema")
    if evidence.get("status") != "accepted" or evidence.get("accepted") is not True:
        raise ValueError("verification evidence did not accept")
    if evidence.get("tolerance_taxonomy") != tolerance_taxonomy():
        raise ValueError("tolerance taxonomy mismatch")
    if evidence.get("convention") != convention_contract():
        raise ValueError("manufactured convention mismatch")
    _validate_hashes(evidence)

    result = _required_mapping(evidence.get("result"), "result")
    h_rows = _required_rows(
        result.get("manufactured_h_refinement"), "manufactured_h_refinement"
    )
    time_rows = _required_rows(
        result.get("manufactured_time_refinement"), "manufactured_time_refinement"
    )
    if len(h_rows) < 3 or len(time_rows) < 3:
        raise ValueError(
            "verification requires at least three independent h and time levels"
        )
    for row in h_rows + time_rows:
        recomputed = _manufactured_row_accepts(row)
        if row.get("accepted") is not recomputed or not recomputed:
            raise ValueError(
                "manufactured refinement contains a numerically failed level"
            )

    black_scholes_rows = _required_rows(
        result.get("black_scholes_rows"), "black_scholes_rows"
    )
    if len(black_scholes_rows) < 3:
        raise ValueError(
            "Black-Scholes verification requires at least three refinement levels"
        )
    _validate_black_scholes_rows(black_scholes_rows)
    _validate_observed_orders(result, h_rows, time_rows, black_scholes_rows)
    _validate_perturbations(result)
    _validate_black_scholes_summary(result, black_scholes_rows[-1])
    _validate_no_arbitrage(result)


def _validate_hashes(evidence: dict[str, Any]) -> None:
    hashes = _required_mapping(evidence.get("hashes"), "hashes")
    for hash_key, section_key in (
        ("backend_hash", "backend"),
        ("mesh_time_hash", "mesh_time"),
        ("request_hash", "request"),
        ("convention_hash", "convention"),
        ("result_hash", "result"),
    ):
        if hashes.get(hash_key) != canonical_hash(evidence.get(section_key)):
            raise ValueError(f"immutable hash mismatch: {hash_key}")
    payload = {key: value for key, value in evidence.items() if key != "evidence_hash"}
    if evidence.get("evidence_hash") != canonical_hash(payload):
        raise ValueError("evidence_hash mismatch")


def _manufactured_row_accepts(row: dict[str, Any]) -> bool:
    limits = tolerance_taxonomy()
    discretization = _required_mapping(limits["discretization"], "discretization")
    algebraic = _required_mapping(limits["algebraic"], "algebraic")
    values = {
        name: _finite_float(row, name)
        for name in (
            "l2_error",
            "h1_error",
            "payoff_relevant_error",
            "algebraic_residual_inf",
            "boundary_residual_inf",
        )
    }
    return (
        values["l2_error"] < float(discretization["manufactured_l2"])
        and values["h1_error"] < float(discretization["manufactured_h1"])
        and values["payoff_relevant_error"]
        < float(discretization["manufactured_payoff"])
        and values["algebraic_residual_inf"]
        < float(algebraic["linear_solve_residual_inf"])
        and values["boundary_residual_inf"] < float(algebraic["boundary_residual_inf"])
    )


def _validate_observed_orders(
    result: dict[str, Any],
    h_rows: list[dict[str, Any]],
    time_rows: list[dict[str, Any]],
    black_scholes_rows: list[dict[str, Any]],
) -> None:
    reported = _required_mapping(result.get("observed_orders"), "observed_orders")
    recomputed = {
        "manufactured_l2_h": _orders_from_rows(
            h_rows, "l2_error", lambda row: _finite_float(row, "h")
        ),
        "manufactured_h1_h": _orders_from_rows(
            h_rows, "h1_error", lambda row: _finite_float(row, "h")
        ),
        "manufactured_payoff_h": _orders_from_rows(
            h_rows, "payoff_relevant_error", lambda row: _finite_float(row, "h")
        ),
        "manufactured_l2_time": _orders_from_rows(
            time_rows, "l2_error", lambda row: _finite_float(row, "dt")
        ),
        "black_scholes_price_h": _orders_from_rows(
            black_scholes_rows,
            "absolute_error",
            lambda row: 1.0 / (_finite_float(row, "degrees_of_freedom") - 1.0),
        ),
    }
    minimums = {
        "manufactured_l2_h": 1.8,
        "manufactured_h1_h": 0.8,
        "manufactured_payoff_h": 1.8,
        "manufactured_l2_time": 1.5,
        "black_scholes_price_h": 1.0,
    }
    for name, expected in recomputed.items():
        actual = reported.get(name)
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"observed order evidence mismatch: {name}")
        if any(
            not isclose(
                _finite_number(value, name), target, rel_tol=1e-10, abs_tol=1e-12
            )
            for value, target in zip(actual, expected)
        ):
            raise ValueError(f"observed order evidence mismatch: {name}")
        if any(value < minimums[name] for value in expected):
            raise ValueError(f"observed order gate failed: {name}")


def _orders_from_rows(
    rows: list[dict[str, Any]],
    error_key: str,
    step: Callable[[dict[str, Any]], float],
) -> list[float]:
    errors = [_finite_float(row, error_key) for row in rows]
    steps = [step(row) for row in rows]
    if any(value <= 0.0 for value in errors + steps):
        raise ValueError(f"positive errors and steps required for {error_key} orders")
    return [
        log(errors[index - 1] / errors[index]) / log(steps[index - 1] / steps[index])
        for index in range(1, len(rows))
    ]


def _validate_perturbations(result: dict[str, Any]) -> None:
    perturbations = _required_mapping(
        result.get("perturbation_failures"), "perturbation_failures"
    )
    required = {"operator_sign", "source", "reaction", "boundary"}
    if set(perturbations) != required:
        raise ValueError("perturbation evidence set is incomplete")
    for name in sorted(required):
        row = _required_mapping(perturbations[name], f"perturbation {name}")
        recomputed = _manufactured_row_accepts(row)
        if row.get("accepted") is not recomputed or recomputed:
            raise ValueError(f"perturbation did not fail numerical gates: {name}")
        if name == "boundary":
            if _finite_float(row, "boundary_residual_inf") <= 1.0e-12:
                raise ValueError(
                    "boundary perturbation did not fail the boundary metric"
                )
        elif (
            max(
                _finite_float(row, "l2_error") / 5.0e-4,
                _finite_float(row, "h1_error") / 2.5e-2,
                _finite_float(row, "payoff_relevant_error") / 2.0e-4,
            )
            <= 1.0
        ):
            raise ValueError(
                f"perturbation did not fail a discretization metric: {name}"
            )


def _validate_black_scholes_rows(rows: list[dict[str, Any]]) -> None:
    levels = [_integral_field(row, "refinement_level") for row in rows]
    time_steps = [_integral_field(row, "time_steps") for row in rows]
    if len(set(time_steps)) != 1:
        raise ValueError("Black-Scholes evidence must share one time-step count")
    recomputed_rows = run_public_black_scholes_parity_fixture(
        refinement_levels=tuple(levels),
        time_steps=time_steps[0],
    ).convergence_rows
    if len(recomputed_rows) != len(rows):
        raise ValueError("Black-Scholes route row count mismatch")
    for row, recomputed_row in zip(rows, recomputed_rows):
        recomputed = recomputed_row.to_public_dict()
        for name in (
            "refinement_level",
            "time_steps",
            "degrees_of_freedom",
        ):
            if _integral_field(row, name) != _integral_field(recomputed, name):
                raise ValueError(f"Black-Scholes route metadata mismatch: {name}")
        for name in (
            "observed_price",
            "expected_price",
            "absolute_error",
            "relative_error",
            "observed_delta",
            "expected_delta",
            "delta_absolute_error",
            "observed_gamma",
            "expected_gamma",
            "gamma_absolute_error",
        ):
            if not isclose(
                _finite_float(row, name),
                _finite_float(recomputed, name),
                rel_tol=_NUMERICAL_VALIDATION_REL_TOL,
                abs_tol=_NUMERICAL_VALIDATION_ABS_TOL,
            ):
                raise ValueError(
                    f"Black-Scholes row does not match the FEM route and analytical oracle: {name}"
                )
        expected_price = _finite_float(row, "expected_price")
        observed_price = _finite_float(row, "observed_price")
        expected_delta = _finite_float(row, "expected_delta")
        observed_delta = _finite_float(row, "observed_delta")
        expected_gamma = _finite_float(row, "expected_gamma")
        observed_gamma = _finite_float(row, "observed_gamma")
        checks = {
            "absolute_error": abs(observed_price - expected_price),
            "relative_error": abs(observed_price - expected_price)
            / max(abs(expected_price), 1.0),
            "delta_absolute_error": abs(observed_delta - expected_delta),
            "gamma_absolute_error": abs(observed_gamma - expected_gamma),
        }
        for name, expected in checks.items():
            if not isclose(
                _finite_float(row, name), expected, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(f"Black-Scholes row inconsistency: {name}")


def _validate_black_scholes_summary(
    result: dict[str, Any], final: dict[str, Any]
) -> None:
    summary = _required_mapping(
        result.get("black_scholes_summary"), "black_scholes_summary"
    )
    key_map = {
        "expected_price": "expected_price",
        "observed_price": "observed_price",
        "price_absolute_error": "absolute_error",
        "price_relative_error": "relative_error",
        "expected_delta": "expected_delta",
        "observed_delta": "observed_delta",
        "delta_absolute_error": "delta_absolute_error",
        "expected_gamma": "expected_gamma",
        "observed_gamma": "observed_gamma",
        "gamma_absolute_error": "gamma_absolute_error",
    }
    for summary_key, row_key in key_map.items():
        if not isclose(
            _finite_float(summary, summary_key),
            _finite_float(final, row_key),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Black-Scholes summary mismatch: {summary_key}")
    exact_tolerances = {
        "price_tolerance_absolute": DEFAULT_TOLERANCE_ABSOLUTE,
        "price_tolerance_relative": DEFAULT_TOLERANCE_RELATIVE,
        "delta_tolerance_absolute": DEFAULT_DELTA_TOLERANCE_ABSOLUTE,
        "gamma_tolerance_absolute": DEFAULT_GAMMA_TOLERANCE_ABSOLUTE,
    }
    for name, expected in exact_tolerances.items():
        if _finite_float(summary, name) != expected:
            raise ValueError(f"Black-Scholes tolerance mismatch: {name}")
    for metric in ("price", "delta", "gamma"):
        if _finite_float(summary, f"{metric}_absolute_error") > _finite_float(
            summary, f"{metric}_tolerance_absolute"
        ):
            raise ValueError(f"Black-Scholes {metric} oracle gate failed")


def _validate_no_arbitrage(result: dict[str, Any]) -> None:
    report = _required_mapping(result.get("no_arbitrage"), "no_arbitrage")
    rows = _required_rows(report.get("rows"), "no_arbitrage.rows")
    numerical = solve_black_scholes_surface((80.0, 100.0, 120.0))
    if len(rows) != len(numerical):
        raise ValueError("no-arbitrage surface point count mismatch")
    points: list[OptionSurfacePoint] = []
    for row, expected in zip(rows, numerical):
        for key, target in (
            ("spot", expected.spot),
            ("price", expected.price),
            ("delta", expected.delta),
            ("gamma", expected.gamma),
        ):
            if not isclose(
                _finite_float(row, key),
                target,
                rel_tol=_NUMERICAL_VALIDATION_REL_TOL,
                abs_tol=_NUMERICAL_VALIDATION_ABS_TOL,
            ):
                raise ValueError(f"no-arbitrage FEM surface mismatch: {key}")
        points.append(
            OptionSurfacePoint(
                spot=expected.spot,
                strike=100.0,
                rate=0.05,
                maturity=1.0,
                price=expected.price,
                delta=expected.delta,
                gamma=expected.gamma,
            )
        )
    recomputed = evaluate_call_arbitrage(tuple(points)).to_public_dict()
    if (
        report.get("accepted") is not recomputed["accepted"]
        or report.get("failures") != recomputed["failures"]
        or recomputed["accepted"] is not True
    ):
        raise ValueError("no-arbitrage evidence mismatch")
    recomputed_rows = _required_rows(
        recomputed.get("rows"), "recomputed no_arbitrage.rows"
    )
    for row, recomputed_row in zip(rows, recomputed_rows):
        for key in ("spot", "price", "lower_bound", "upper_bound", "delta", "gamma"):
            if not isclose(
                _finite_float(row, key),
                _finite_float(recomputed_row, key),
                rel_tol=_NUMERICAL_VALIDATION_REL_TOL,
                abs_tol=_NUMERICAL_VALIDATION_ABS_TOL,
            ):
                raise ValueError(f"no-arbitrage evidence mismatch: {key}")


def _required_rows(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integral_field(mapping: dict[str, Any], key: str) -> int:
    number = _finite_float(mapping, key)
    if not number.is_integer():
        raise ValueError(f"{key} must be integral")
    return int(number)


def _finite_float(mapping: dict[str, Any], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"missing numerical evidence field: {key}")
    return _finite_number(mapping[key], key)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


__all__ = [
    "FEM_VERIFICATION_SCHEMA_VERSION",
    "canonical_hash",
    "convention_contract",
    "tolerance_taxonomy",
    "validate_evidence",
]
