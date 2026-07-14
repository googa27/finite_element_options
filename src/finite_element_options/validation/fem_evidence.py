"""Versioned FEM numerical verification evidence for VQPW issue #117."""

from __future__ import annotations

from importlib import metadata
from math import log
from typing import Any

from .black_scholes_parity import run_public_black_scholes_parity_fixture
from .evidence.black_scholes_surface import solve_black_scholes_surface
from .evidence.gates import (
    FEM_VERIFICATION_SCHEMA_VERSION,
    canonical_hash,
    tolerance_taxonomy,
    validate_evidence,
)
from .evidence.manufactured import (
    FAILURE_PERTURBATIONS,
    ManufacturedRunConfig,
    ManufacturedRunResult,
    VerificationPerturbation,
    run_manufactured_case,
    sympy_manufactured_problem,
)
from .verification_gates import OptionSurfacePoint, evaluate_call_arbitrage

FEM_VERIFICATION_BENCHMARK_ID = "fem-bs-001"


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
    evidence_tolerances = tolerance_taxonomy()
    accepted = (
        all(row.accepted for row in h_rows)
        and all(row.accepted for row in time_rows)
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
        "tolerance_taxonomy": evidence_tolerances,
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
    validate_evidence(evidence)
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
    numerical = solve_black_scholes_surface(
        (80.0, 100.0, 120.0),
        refinement_level=6,
        time_steps=80,
        strike=100.0,
    )
    points = tuple(
        OptionSurfacePoint(
            spot=point.spot,
            strike=100.0,
            rate=0.05,
            maturity=1.0,
            price=point.price,
            delta=point.delta,
            gamma=point.gamma,
        )
        for point in numerical
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
