"""Fail-closed adapter for public-synthetic compiled PDE weak-form fixtures.

The adapter consumes only serialized JSON artifacts.  It deliberately does not
import ``financial_problem_formulations`` or any source-tree compiler module:
compiled ``pde_ir.v0`` evidence is treated as data and screened before the
existing scikit-fem Black--Scholes route is allowed to assemble.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .black_scholes_parity import run_public_black_scholes_parity_fixture
from .compiled_weak_form_contract import (
    ALLOWED_TOP_LEVEL_FIELDS,
    COMPILED_OPERATOR_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    PDE_IR_SCHEMA_VERSION,
    PUBLIC_BS_COMPILED_HASH,
    PUBLIC_BS_FIXTURE_ID,
    PUBLIC_BS_FORMULATION_ID,
    PUBLIC_BS_SOURCE_HASH,
    PUBLIC_BS_SOURCE_PROBLEM_ID,
    SUPPORTED_OUTPUTS,
    SUPPORTED_ROUTE,
    CompiledWeakFormDiagnostic,
    CompiledWeakFormScreen,
    CompiledWeakFormUnsupportedError,
    append_diagnostic,
    as_mapping,
    as_sequence,
    capability_status,
    expect_field,
    hash_json,
    json_roundtrip,
    optional_string,
    rejected,
    reject_private_markers,
    stringify,
)


def load_compiled_weak_form_json(path: str | Path) -> dict[str, Any]:
    """Load a strict JSON object from ``path`` without importing compiler code."""

    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompiledWeakFormUnsupportedError(
            rejected(
                "compiled_weak_form.json", "json", "<invalid>", f"Invalid JSON: {exc}"
            )
        ) from exc
    if not isinstance(loaded, dict):
        raise CompiledWeakFormUnsupportedError(
            rejected(
                "compiled_weak_form.json",
                "payload",
                type(loaded).__name__,
                "Input must be a JSON object.",
            )
        )
    return loaded


def screen_compiled_weak_form(payload: Mapping[str, Any]) -> CompiledWeakFormScreen:
    """Validate the exact public-synthetic compiled route before assembly."""

    diagnostics: list[CompiledWeakFormDiagnostic] = []
    reject_private_markers(payload, diagnostics)
    unknown = sorted(set(payload) - ALLOWED_TOP_LEVEL_FIELDS)
    for key in unknown:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.unknown_field",
            key,
            stringify(payload.get(key)),
            "Unknown top-level fixture field; refusing to infer semantics.",
        )

    pde_ir = as_mapping(payload.get("pde_ir"))
    compiled = as_mapping(payload.get("compiled_operator"))
    route = as_mapping(payload.get("fem_route"))
    operator = as_mapping(pde_ir.get("operator"))

    expect_field(
        payload.get("schema_version"),
        FIXTURE_SCHEMA_VERSION,
        "schema_version",
        "compiled_weak_form.schema",
        diagnostics,
    )
    expect_field(
        payload.get("fixture_id"),
        PUBLIC_BS_FIXTURE_ID,
        "fixture_id",
        "compiled_weak_form.fixture_id",
        diagnostics,
    )
    if stringify(payload.get("privacy_class")).replace("-", "_") != "public_synthetic":
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.privacy",
            "privacy_class",
            stringify(payload.get("privacy_class")),
            "Only public-synthetic compiled fixtures may be consumed.",
        )
    expect_field(
        pde_ir.get("schema_version"),
        PDE_IR_SCHEMA_VERSION,
        "pde_ir.schema_version",
        "pde_ir.schema",
        diagnostics,
    )
    expect_field(
        pde_ir.get("schema_id"),
        "financial_problem_formulations.pde_ir.v0",
        "pde_ir.schema_id",
        "pde_ir.schema",
        diagnostics,
    )
    expect_field(
        compiled.get("schema_version"),
        COMPILED_OPERATOR_SCHEMA_VERSION,
        "compiled_operator.schema_version",
        "compiled_operator.schema",
        diagnostics,
    )
    expect_field(
        compiled.get("schema_id"),
        "financial_problem_formulations.pde_ir.compiled_symbolic_operator.v0",
        "compiled_operator.schema_id",
        "compiled_operator.schema",
        diagnostics,
    )
    expect_field(
        pde_ir.get("problem_id"),
        PUBLIC_BS_SOURCE_PROBLEM_ID,
        "pde_ir.problem_id",
        "compiled_weak_form.problem",
        diagnostics,
    )
    expect_field(
        pde_ir.get("formulation_id"),
        PUBLIC_BS_FORMULATION_ID,
        "pde_ir.formulation_id",
        "compiled_weak_form.formulation",
        diagnostics,
    )
    expect_field(
        pde_ir.get("canonical_hash"),
        PUBLIC_BS_SOURCE_HASH,
        "pde_ir.canonical_hash",
        "compiled_weak_form.source_hash",
        diagnostics,
    )
    expect_field(
        compiled.get("source_ir_canonical_hash"),
        pde_ir.get("canonical_hash"),
        "compiled_operator.source_ir_canonical_hash",
        "compiled_weak_form.hash_link",
        diagnostics,
    )
    expect_field(
        compiled.get("compiled_hash"),
        PUBLIC_BS_COMPILED_HASH,
        "compiled_operator.compiled_hash",
        "compiled_weak_form.compiled_hash",
        diagnostics,
    )
    expect_field(
        compiled.get("source_problem_id"),
        pde_ir.get("problem_id"),
        "compiled_operator.source_problem_id",
        "compiled_weak_form.problem_link",
        diagnostics,
    )
    expect_field(
        operator.get("sign_convention"),
        "backward_generator_minus_discount",
        "pde_ir.operator.sign_convention",
        "compiled_weak_form.sign",
        diagnostics,
    )
    expect_field(
        pde_ir.get("time_orientation"),
        "backward",
        "pde_ir.time_orientation",
        "compiled_weak_form.time_orientation",
        diagnostics,
    )
    expect_field(
        pde_ir.get("measure"),
        "Q",
        "pde_ir.measure",
        "compiled_weak_form.measure",
        diagnostics,
    )

    state_variables = as_sequence(pde_ir.get("state_variables"))
    if len(state_variables) != 1:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.dimension",
            "pde_ir.state_variables",
            str(len(state_variables)),
            "Only the exact one-dimensional public fixture is supported.",
        )
    else:
        state = as_mapping(state_variables[0])
        expect_field(
            state.get("symbol"),
            "S",
            "pde_ir.state_variables[0].symbol",
            "compiled_weak_form.state",
            diagnostics,
        )
        expect_field(
            state.get("domain"),
            "positive_real",
            "pde_ir.state_variables[0].domain",
            "compiled_weak_form.domain",
            diagnostics,
        )

    terms = as_sequence(operator.get("terms"))
    term_kinds = tuple(stringify(as_mapping(term).get("kind")) for term in terms)
    if term_kinds != ("diffusion", "discount", "drift"):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.terms",
            "pde_ir.operator.terms",
            repr(term_kinds),
            "Only diffusion/discount/drift Black-Scholes terms in compiler order are supported.",
        )
    _check_boundary_split(pde_ir, route, diagnostics)
    _check_route(route, diagnostics)
    _check_compiled_expressions(compiled, diagnostics)

    request = _request_payload(payload, pde_ir, compiled, route)
    route_hash = hash_json(request) if not diagnostics else None
    return CompiledWeakFormScreen(
        accepted=not diagnostics,
        fixture_id=optional_string(payload.get("fixture_id")),
        problem_id=optional_string(pde_ir.get("problem_id")),
        source_ir_hash=optional_string(pde_ir.get("canonical_hash")),
        compiled_operator_hash=optional_string(compiled.get("compiled_hash")),
        route_hash=route_hash,
        diagnostics=tuple(diagnostics),
        request=request,
    )


def solve_compiled_weak_form(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the validated compiled route through the existing scikit-fem solver."""

    screen = screen_compiled_weak_form(payload)
    if not screen.accepted:
        raise CompiledWeakFormUnsupportedError(screen)
    route = as_mapping(payload.get("fem_route"))
    time = as_mapping(route.get("time"))
    report = run_public_black_scholes_parity_fixture(
        refinement_levels=(4, 5, 6), time_steps=int(time.get("steps", 80))
    )
    result = report.export_payload()
    result["format_version"] = "compiled-weak-form-fem-result-v0"
    result["fixture_id"] = screen.fixture_id
    result["source_ir_hash"] = screen.source_ir_hash
    result["compiled_operator_hash"] = screen.compiled_operator_hash
    result["route_hash"] = screen.route_hash
    result["problem_id"] = screen.problem_id
    result["measure"] = "Q"
    result["numeraire"] = as_mapping(payload["pde_ir"].get("numeraire"))
    result["units"] = _compiled_units(as_mapping(payload["pde_ir"]))
    result["screen"] = screen.to_public_dict()
    result["weak_form"]["source_sign_convention"] = "backward_generator_minus_discount"
    result["weak_form"]["source_equation"] = "dV_dt + L[V] = 0"
    result["boundary_partition"] = as_mapping(route.get("boundary_partition"))
    result["domain"] = as_mapping(route.get("domain"))
    result["time"] = as_mapping(route.get("time"))
    result["diagnostics"] = {
        **as_mapping(result.get("diagnostics")),
        "capability": "compiled_pde_ir_v0_black_scholes_call_line_uniform_lagrange_p2_theta_scipy_direct",
        "source_repo": stringify(payload.get("source_repo")),
        "compiler_issue": stringify(payload.get("compiler_issue")),
        "fem_issue": stringify(payload.get("fem_issue")),
    }
    return json_roundtrip(result)


def solve_compiled_weak_form_file(path: str | Path) -> dict[str, Any]:
    """Load, screen and solve a compiled weak-form fixture file."""

    return solve_compiled_weak_form(load_compiled_weak_form_json(path))


def evidence_for_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact machine-readable review evidence for a solve result."""

    return {
        "schema_version": "compiled-weak-form-fem-evidence/v0",
        "fixture_id": result.get("fixture_id"),
        "problem_id": result.get("problem_id"),
        "status": result.get("status"),
        "source_ir_hash": result.get("source_ir_hash"),
        "compiled_operator_hash": result.get("compiled_operator_hash"),
        "route_hash": result.get("route_hash"),
        "capability_status": capability_status(),
        "error_summary": result.get("summary", {}),
        "screen_accepted": as_mapping(result.get("screen")).get("accepted"),
    }


def _check_boundary_split(
    pde_ir: Mapping[str, Any],
    route: Mapping[str, Any],
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    boundaries = tuple(
        as_mapping(item) for item in as_sequence(pde_ir.get("boundary_conditions"))
    )
    kinds = tuple(stringify(item.get("kind")) for item in boundaries)
    if kinds != ("dirichlet", "asymptotic"):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.boundary",
            "pde_ir.boundary_conditions",
            repr(kinds),
            "Only lower Dirichlet plus upper asymptotic Black-Scholes boundaries are supported.",
        )
    partition = as_mapping(route.get("boundary_partition"))
    if (
        tuple(partition.get("essential", ())) != ("lower", "upper")
        or tuple(partition.get("natural", ())) != ()
    ):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.boundary_partition",
            "fem_route.boundary_partition",
            stringify(partition),
            "The v0 route must preserve essential lower/upper enforcement and an empty natural split.",
        )


def _check_route(
    route: Mapping[str, Any], diagnostics: list[CompiledWeakFormDiagnostic]
) -> None:
    for field, expected in SUPPORTED_ROUTE.items():
        expect_field(
            route.get(field),
            expected,
            f"fem_route.{field}",
            "compiled_weak_form.route",
            diagnostics,
        )
    requested_outputs = tuple(route.get("requested_outputs", ()))
    if requested_outputs != SUPPORTED_OUTPUTS:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.outputs",
            "fem_route.requested_outputs",
            repr(requested_outputs),
            "The exact v0 route exposes value, delta and gamma only.",
        )
    domain = as_mapping(route.get("domain"))
    if (
        domain.get("state"),
        float(domain.get("lower", -1.0)),
        float(domain.get("upper", -1.0)),
    ) != ("S", 0.0, 400.0):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.domain",
            "fem_route.domain",
            stringify(domain),
            "The exact v0 route uses the finite public spot domain [0, 400].",
        )


def _check_compiled_expressions(
    compiled: Mapping[str, Any], diagnostics: list[CompiledWeakFormDiagnostic]
) -> None:
    expressions = as_sequence(compiled.get("expressions"))
    paths = {
        stringify(as_mapping(item).get("path")): as_mapping(item) for item in expressions
    }
    required = {
        "operator.terms[0].expression": "(((0.5 * (sigma ^ 2)) * (S ^ 2)) * d2V_dS2)",
        "operator.terms[1].expression": "(-r * V)",
        "operator.terms[2].expression": "((r * S) * dV_dS)",
        "terminal_condition.expression": "max((S - K), 0)",
    }
    for path, normalized in required.items():
        if path not in paths:
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.expression_missing",
                path,
                "<missing>",
                "Compiled expression path is absent.",
            )
        elif paths[path].get("normalized") != normalized:
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.expression",
                path,
                stringify(paths[path].get("normalized")),
                "Compiled expression normalization changed; refusing route.",
            )


def _request_payload(
    payload: Mapping[str, Any],
    pde_ir: Mapping[str, Any],
    compiled: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "fixture_id": payload.get("fixture_id"),
        "source_problem_id": pde_ir.get("problem_id"),
        "source_formulation_id": pde_ir.get("formulation_id"),
        "source_ir_hash": pde_ir.get("canonical_hash"),
        "compiled_operator_hash": compiled.get("compiled_hash"),
        "measure": pde_ir.get("measure"),
        "numeraire": pde_ir.get("numeraire"),
        "units": _compiled_units(pde_ir),
        "time_orientation": pde_ir.get("time_orientation"),
        "operator_sign_convention": as_mapping(pde_ir.get("operator")).get(
            "sign_convention"
        ),
        "route": route,
    }


def _compiled_units(pde_ir: Mapping[str, Any]) -> dict[str, Any]:
    state_variables = as_sequence(pde_ir.get("state_variables"))
    state_unit = (
        as_mapping(as_mapping(state_variables[0]).get("unit")) if state_variables else {}
    )
    terminal_unit = as_mapping(as_mapping(pde_ir.get("terminal_condition")).get("unit"))
    return {"state": state_unit, "value": terminal_unit}



__all__ = [
    "COMPILED_OPERATOR_SCHEMA_VERSION",
    "FIXTURE_SCHEMA_VERSION",
    "PDE_IR_SCHEMA_VERSION",
    "PUBLIC_BS_COMPILED_HASH",
    "PUBLIC_BS_FIXTURE_ID",
    "PUBLIC_BS_SOURCE_HASH",
    "CompiledWeakFormDiagnostic",
    "CompiledWeakFormScreen",
    "CompiledWeakFormUnsupportedError",
    "evidence_for_result",
    "load_compiled_weak_form_json",
    "screen_compiled_weak_form",
    "solve_compiled_weak_form",
    "solve_compiled_weak_form_file",
]
