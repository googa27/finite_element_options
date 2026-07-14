"""Versioned FEM numerical verification evidence for VQPW issue #117."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
from math import isfinite, log
from typing import Any

from .black_scholes_parity import run_public_black_scholes_parity_fixture
from .fem_manufactured import (
    FAILURE_PERTURBATIONS,
    ManufacturedRunConfig,
    ManufacturedRunResult,
    VerificationPerturbation,
    run_manufactured_case,
    sympy_manufactured_problem,
)
from .verification_gates import OptionSurfacePoint, evaluate_call_arbitrage

FEM_VERIFICATION_SCHEMA_VERSION = "fem-verification-evidence/v1"
FEM_VERIFICATION_BENCHMARK_ID = "fem-bs-001"


def canonical_hash(payload: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible evidence payloads."""

    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_verification_benchmark() -> dict[str, Any]:
    """Run the deterministic FEM verification benchmark and return evidence JSON."""

    manufactured = sympy_manufactured_problem()
    h_rows = tuple(
        run_manufactured_case(ManufacturedRunConfig(elements=n, time_steps=2048))
        for n in (16, 32, 64)
    )
    time_rows = tuple(
        run_manufactured_case(ManufacturedRunConfig(elements=256, time_steps=n))
        for n in (4, 8, 16)
    )
    perturbations = {
        name: run_manufactured_case(
            ManufacturedRunConfig(elements=64, time_steps=512, perturbation=name)
        ).to_public_dict()
        for name in FAILURE_PERTURBATIONS
    }
    bs_report = run_public_black_scholes_parity_fixture(
        refinement_levels=(4, 5, 6), time_steps=80
    )
    bs_rows = [row.to_public_dict() for row in bs_report.convergence_rows]
    arbitrage = _black_scholes_arbitrage_report()
    request = _request_payload()
    convention = _convention_payload(manufactured)
    backend = _backend_payload()
    mesh_time = _mesh_time_payload(h_rows, time_rows)
    result = {
        "manufactured_h_refinement": [row.to_public_dict() for row in h_rows],
        "manufactured_time_refinement": [row.to_public_dict() for row in time_rows],
        "black_scholes_rows": bs_rows,
        "black_scholes_summary": bs_report.export_payload()["summary"],
        "perturbation_failures": perturbations,
        "no_arbitrage": arbitrage.to_public_dict(),
        "observed_orders": {
            "manufactured_l2_h": _orders(h_rows, "l2_error"),
            "manufactured_h1_h": _orders(h_rows, "h1_error"),
            "manufactured_payoff_h": _orders(h_rows, "payoff_relevant_error"),
            "manufactured_l2_time": _orders(time_rows, "l2_error"),
            "black_scholes_price_h": _orders_from_public_rows(
                bs_rows, "absolute_error"
            ),
        },
    }
    hashes = {
        "backend_hash": canonical_hash(backend),
        "mesh_time_hash": canonical_hash(mesh_time),
        "request_hash": canonical_hash(request),
        "convention_hash": canonical_hash(convention),
        "result_hash": canonical_hash(result),
    }
    tolerance_taxonomy = _tolerance_taxonomy()
    accepted = (
        all(row.accepted for row in h_rows)
        and all(row.accepted for row in time_rows[-1:])
        and all(not row["accepted"] for row in perturbations.values())
        and arbitrage.accepted
        and bs_report.status == "converged"
    )
    evidence = {
        "schema_version": FEM_VERIFICATION_SCHEMA_VERSION,
        "benchmark_id": FEM_VERIFICATION_BENCHMARK_ID,
        "capability_id": "VQPW-FEM-VERIFICATION-EVIDENCE-V0",
        "privacy_class": "public_synthetic",
        "status": "accepted" if accepted else "failed",
        "accepted": accepted,
        "request": request,
        "backend": backend,
        "convention": convention,
        "mesh_time": mesh_time,
        "tolerance_taxonomy": tolerance_taxonomy,
        "result": result,
        "hashes": hashes,
        "issues": [
            "googa27/finite_element_options#117",
            "googa27/finite_element_options#116",
        ],
    }
    evidence["evidence_hash"] = canonical_hash(
        {k: v for k, v in evidence.items() if k != "evidence_hash"}
    )
    return evidence


def _orders(rows: tuple[ManufacturedRunResult, ...], attr: str) -> list[float]:
    values = [float(getattr(row, attr)) for row in rows]
    steps = [row.h if rows[0].elements != rows[-1].elements else row.dt for row in rows]
    return [
        log(values[i - 1] / values[i]) / log(steps[i - 1] / steps[i])
        for i in range(1, len(rows))
        if values[i] > 0
    ]


def _orders_from_public_rows(rows: list[dict[str, Any]], error_key: str) -> list[float]:
    values = [float(row[error_key]) for row in rows]
    steps = [1.0 / float(row["degrees_of_freedom"] - 1) for row in rows]
    return [
        log(values[i - 1] / values[i]) / log(steps[i - 1] / steps[i])
        for i in range(1, len(rows))
        if values[i] > 0
    ]


def _black_scholes_arbitrage_report():
    bs = run_public_black_scholes_parity_fixture(refinement_levels=(6,), time_steps=80)
    summary = bs.export_payload()["summary"]
    points = (
        OptionSurfacePoint(
            spot=80.0,
            strike=100.0,
            rate=0.05,
            maturity=1.0,
            price=1.8594195728121814,
            delta=0.22192212952299497,
            gamma=0.018579456785272243,
        ),
        OptionSurfacePoint(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            maturity=1.0,
            price=float(summary["observed_price"]),
            delta=float(summary["observed_delta"]),
            gamma=float(summary["observed_gamma"]),
        ),
        OptionSurfacePoint(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            maturity=1.0,
            price=26.169043946847304,
            delta=0.8964550237252614,
            gamma=0.009268773121754995,
        ),
    )
    return evaluate_call_arbitrage(points)


def _request_payload() -> dict[str, Any]:
    return {
        "benchmark_id": FEM_VERIFICATION_BENCHMARK_ID,
        "outputs": ["value", "delta", "gamma", "L2", "H1", "residuals"],
        "public_fixture": "embedded://finite_element_options.validation.compiled_weak_form_golden/black_scholes_call_v0",
    }


def _backend_payload() -> dict[str, str]:
    try:
        package_version = metadata.version("finite-element-options")
    except metadata.PackageNotFoundError:
        package_version = "0.1.0"
    return {
        "package": "finite_element_options",
        "package_version": package_version,
        "backend": "scipy_sparse_direct",
        "fem_library": "scikit-fem maintained route plus scipy manufactured check",
        "oracle_library": "sympy",
    }


def _convention_payload(manufactured: dict[str, Any]) -> dict[str, Any]:
    return {
        "manufactured": manufactured,
        "black_scholes": {
            "time": "tau=T-t",
            "measure": "risk_neutral",
            "numeraire": "money_market_account",
            "operator_sign": "forward tau Black-Scholes",
        },
    }


def _mesh_time_payload(
    h_rows: tuple[ManufacturedRunResult, ...],
    time_rows: tuple[ManufacturedRunResult, ...],
) -> dict[str, Any]:
    return {
        "h_refinement": [
            {"elements": r.elements, "h": r.h, "time_steps": r.time_steps}
            for r in h_rows
        ],
        "time_refinement": [
            {"elements": r.elements, "dt": r.dt, "time_steps": r.time_steps}
            for r in time_rows
        ],
        "separation_policy": "h-study fixes dt=1/2048; time-study fixes h=1/256",
    }


def _tolerance_taxonomy() -> dict[str, Any]:
    return {
        "discretization": {
            "manufactured_l2": 5.0e-4,
            "manufactured_h1": 2.5e-2,
            "black_scholes_price": 2.0e-3,
        },
        "oracle": {
            "sympy_symbolic_residual": 1.0e-12,
            "black_scholes_analytic": 1.0e-12,
        },
        "algebraic": {
            "linear_solve_residual_inf": 1.0e-11,
            "boundary_residual_inf": 1.0e-12,
        },
        "floating_point": {"hashes": "sha256 canonical JSON", "dtype": "float64"},
    }


def validate_evidence(evidence: dict[str, Any]) -> None:
    """Raise when a generated evidence bundle does not meet issue #117 gates."""

    if evidence.get("schema_version") != FEM_VERIFICATION_SCHEMA_VERSION:
        raise ValueError("unexpected verification evidence schema")
    if evidence.get("status") != "accepted" or evidence.get("accepted") is not True:
        raise ValueError("verification evidence did not accept")
    hashes = _required_mapping(evidence.get("hashes"), "hashes")
    for hash_key, section_key in (
        ("backend_hash", "backend"),
        ("mesh_time_hash", "mesh_time"),
        ("request_hash", "request"),
        ("convention_hash", "convention"),
        ("result_hash", "result"),
    ):
        expected = canonical_hash(evidence.get(section_key))
        if hashes.get(hash_key) != expected:
            raise ValueError(f"immutable hash mismatch: {hash_key}")
    expected_evidence_hash = canonical_hash(
        {key: value for key, value in evidence.items() if key != "evidence_hash"}
    )
    if evidence.get("evidence_hash") != expected_evidence_hash:
        raise ValueError("evidence_hash mismatch")

    result = _required_mapping(evidence.get("result"), "result")
    h_rows = result.get("manufactured_h_refinement")
    time_rows = result.get("manufactured_time_refinement")
    if (
        not isinstance(h_rows, list)
        or len(h_rows) < 3
        or not isinstance(time_rows, list)
        or len(time_rows) < 3
    ):
        raise ValueError(
            "verification requires at least three independent h and time levels"
        )
    if not all(
        isinstance(row, dict) and row.get("accepted") is True
        for row in h_rows + time_rows
    ):
        raise ValueError("manufactured refinement contains a failed level")
    orders = _required_mapping(result.get("observed_orders"), "observed_orders")
    minimum_orders = {
        "manufactured_l2_h": 1.8,
        "manufactured_h1_h": 0.8,
        "manufactured_payoff_h": 1.8,
        "manufactured_l2_time": 1.5,
        "black_scholes_price_h": 1.0,
    }
    for name, minimum in minimum_orders.items():
        values = orders.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isfinite(float(value)) or float(value) < minimum for value in values
            )
        ):
            raise ValueError(f"observed order gate failed: {name}")
    perturbations = _required_mapping(
        result.get("perturbation_failures"), "perturbation_failures"
    )
    if set(perturbations) != {"operator_sign", "source", "reaction", "boundary"}:
        raise ValueError("perturbation evidence set is incomplete")
    if not all(
        isinstance(row, dict) and row.get("accepted") is False
        for row in perturbations.values()
    ):
        raise ValueError("perturbation failures must be independent failing checks")
    no_arbitrage = _required_mapping(result.get("no_arbitrage"), "no_arbitrage")
    if no_arbitrage.get("accepted") is not True:
        raise ValueError("no-arbitrage gate failed")
    summary = _required_mapping(
        result.get("black_scholes_summary"), "black_scholes_summary"
    )
    for metric in ("price", "delta", "gamma"):
        error = float(summary.get(f"{metric}_absolute_error", float("inf")))
        tolerance = float(summary.get(f"{metric}_tolerance_absolute", float("-inf")))
        if not isfinite(error) or not isfinite(tolerance) or error > tolerance:
            raise ValueError(f"Black-Scholes {metric} oracle gate failed")


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


__all__ = [
    "FEM_VERIFICATION_SCHEMA_VERSION",
    "FEM_VERIFICATION_BENCHMARK_ID",
    "ManufacturedRunConfig",
    "ManufacturedRunResult",
    "VerificationPerturbation",
    "canonical_hash",
    "run_manufactured_case",
    "run_verification_benchmark",
    "sympy_manufactured_problem",
    "validate_evidence",
]
