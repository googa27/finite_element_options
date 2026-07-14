"""Reusable screening primitives for compiled weak-form fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from .compiled_weak_form_contract import (
    SUPPORTED_OUTPUTS,
    SUPPORTED_ROUTE,
    CompiledWeakFormDiagnostic,
    append_diagnostic,
    as_mapping,
    as_sequence,
    expect_field,
    stringify,
)


def expect_mapping_field(
    value: Any, field: str, diagnostics: list[CompiledWeakFormDiagnostic]
) -> None:
    """Append a diagnostic if ``value`` is not a JSON object."""

    if not isinstance(value, Mapping):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.type",
            field,
            type(value).__name__,
            f"Expected {field} to be a JSON object.",
        )


def reject_nested_unknown_fields(
    value: Any,
    golden: Any,
    path: str,
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Reject mapping keys absent from the matching packaged golden fixture path."""

    if isinstance(value, Mapping) and isinstance(golden, Mapping):
        for key in sorted(set(value) - set(golden)):
            child_path = f"{path}.{key}"
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.unknown_field",
                child_path,
                stringify(value.get(key)),
                "Unknown nested fixture field; refusing to infer semantics.",
            )
        for key in sorted(set(value) & set(golden)):
            reject_nested_unknown_fields(
                value.get(key), golden.get(key), f"{path}.{key}", diagnostics
            )
        return
    if isinstance(value, list) and isinstance(golden, list):
        for index, child in enumerate(value[: len(golden)]):
            reject_nested_unknown_fields(
                child, golden[index], f"{path}[{index}]", diagnostics
            )


def canonical_json_hash(value: Any) -> str | None:
    """Return a deterministic JSON SHA-256 hash, or ``None`` on conversion failure."""

    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return "sha256:" + sha256(encoded).hexdigest()


def check_exact_json_subobject(
    actual: Any,
    expected: Any,
    field: str,
    code: str,
    diagnostics: list[CompiledWeakFormDiagnostic],
    message: str,
) -> None:
    """Require exact canonical JSON equality against a golden fixture subobject."""

    actual_hash = canonical_json_hash(actual)
    expected_hash = canonical_json_hash(expected)
    if actual_hash is None:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.conversion",
            field,
            type(actual).__name__,
            f"{field} must be JSON-canonicalizable before FEM assembly.",
        )
        return
    if actual_hash != expected_hash:
        append_diagnostic(
            diagnostics,
            code,
            field,
            actual_hash,
            message,
        )


def expect_exact_int(
    value: Any,
    expected: int,
    field: str,
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Require a non-bool integer exactly equal to ``expected`` and positive."""

    if type(value) is not int:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.conversion",
            field,
            stringify(value),
            f"Expected {field} to be integer {expected}; got {type(value).__name__}.",
        )
        return
    if value != expected:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.time",
            field,
            str(value),
            f"Expected {field}={expected}; got {value}.",
        )
    if value <= 0:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.time",
            field,
            str(value),
            f"Expected {field} to be positive.",
        )


def expect_exact_number(
    value: Any,
    expected: float,
    field: str,
    code: str,
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Require a numeric JSON value exactly equal to ``expected``."""

    if type(value) not in (int, float):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.conversion",
            field,
            stringify(value),
            f"Expected {field} to be numeric {expected}; got {type(value).__name__}.",
        )
        return
    numeric_value = float(value)
    if not isfinite(numeric_value):
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.conversion",
            field,
            stringify(value),
            f"Expected {field} to be finite numeric {expected}; got {value!r}.",
        )
        return
    if numeric_value != expected:
        append_diagnostic(
            diagnostics,
            code,
            field,
            str(value),
            f"Expected {field}={expected}; got {value}.",
        )


def check_terminal_and_boundary(
    pde_ir: Mapping[str, Any],
    route: Mapping[str, Any],
    golden_pde_ir: Mapping[str, Any],
    golden_route: Mapping[str, Any],
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Check terminal payoff, PDE boundaries and FEM boundary map exactly."""

    check_exact_json_subobject(
        pde_ir.get("terminal_condition"),
        golden_pde_ir.get("terminal_condition"),
        "pde_ir.terminal_condition",
        "compiled_weak_form.terminal",
        diagnostics,
        "Terminal payoff block must exactly match the packaged Black-Scholes call fixture.",
    )
    check_exact_json_subobject(
        pde_ir.get("boundary_conditions"),
        golden_pde_ir.get("boundary_conditions"),
        "pde_ir.boundary_conditions",
        "compiled_weak_form.boundary",
        diagnostics,
        "Boundary-condition block must exactly match lower Dirichlet and upper asymptotic fixture semantics.",
    )
    check_exact_json_subobject(
        route.get("boundary_map"),
        golden_route.get("boundary_map"),
        "fem_route.boundary_map",
        "compiled_weak_form.boundary",
        diagnostics,
        "FEM boundary map must exactly preserve lower/upper essential enforcement.",
    )


def check_boundary_split(
    pde_ir: Mapping[str, Any],
    route: Mapping[str, Any],
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Validate released lower/upper boundaries by boundary_id, not list order."""

    boundaries_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for index, item in enumerate(as_sequence(pde_ir.get("boundary_conditions"))):
        boundary = as_mapping(item)
        boundary_id = stringify(boundary.get("boundary_id"))
        if boundary_id in boundaries_by_id:
            duplicate_ids.add(boundary_id)
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.boundary",
                f"pde_ir.boundary_conditions[{index}].boundary_id",
                boundary_id,
                "Duplicate boundary_id values are not allowed.",
            )
            continue
        boundaries_by_id[boundary_id] = boundary
    expected_kinds = {"lower": "dirichlet", "upper": "asymptotic"}
    actual_kinds = {
        boundary_id: stringify(boundaries_by_id.get(boundary_id, {}).get("kind"))
        for boundary_id in expected_kinds
    }
    if actual_kinds != expected_kinds or duplicate_ids:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.boundary",
            "pde_ir.boundary_conditions",
            stringify(actual_kinds),
            "Only lower Dirichlet plus upper asymptotic Black-Scholes boundaries keyed by boundary_id are supported.",
        )
    unexpected_ids = sorted(set(boundaries_by_id) - set(expected_kinds))
    for boundary_id in unexpected_ids:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.boundary",
            "pde_ir.boundary_conditions.boundary_id",
            boundary_id,
            "Unsupported boundary_id in public Black-Scholes fixture.",
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


def check_route(
    route: Mapping[str, Any], diagnostics: list[CompiledWeakFormDiagnostic]
) -> None:
    """Validate route family, outputs, domain and time block."""

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
    if domain.get("state") != "S" or domain.get("truncated_coordinate") != "spot":
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.domain",
            "fem_route.domain",
            stringify(domain),
            "The exact v0 route uses spot state S on the finite public domain [0, 400].",
        )
    expect_exact_number(
        domain.get("lower"),
        0.0,
        "fem_route.domain.lower",
        "compiled_weak_form.domain",
        diagnostics,
    )
    expect_exact_number(
        domain.get("upper"),
        400.0,
        "fem_route.domain.upper",
        "compiled_weak_form.domain",
        diagnostics,
    )
    check_route_time(as_mapping(route.get("time")), diagnostics)


def check_route_time(
    time: Mapping[str, Any], diagnostics: list[CompiledWeakFormDiagnostic]
) -> None:
    """Validate the exact backward/80-step/tau/theta time contract."""

    if not time:
        append_diagnostic(
            diagnostics,
            "compiled_weak_form.time",
            "fem_route.time",
            "<missing>",
            "The v0 route requires a complete backward theta time block.",
        )
    expect_field(
        time.get("orientation"),
        "backward",
        "fem_route.time.orientation",
        "compiled_weak_form.time",
        diagnostics,
    )
    expect_exact_int(time.get("steps"), 80, "fem_route.time.steps", diagnostics)
    expect_exact_number(
        time.get("tau_start"),
        0.0,
        "fem_route.time.tau_start",
        "compiled_weak_form.time",
        diagnostics,
    )
    expect_exact_number(
        time.get("tau_end"),
        1.0,
        "fem_route.time.tau_end",
        "compiled_weak_form.time",
        diagnostics,
    )
    expect_exact_number(
        time.get("theta"),
        0.5,
        "fem_route.time.theta",
        "compiled_weak_form.time",
        diagnostics,
    )


def check_compiled_expressions(
    compiled: Mapping[str, Any],
    golden_compiled: Mapping[str, Any],
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Validate required compiled expression semantics, hashes and unit evidence."""

    expressions = as_sequence(compiled.get("expressions"))
    paths: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(expressions):
        expression = as_mapping(item)
        path = stringify(expression.get("path"))
        if path in paths:
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.expression_duplicate",
                f"compiled_operator.expressions[{index}].path",
                path,
                "Duplicate compiled expression paths are not allowed.",
            )
            continue
        paths[path] = expression
    expected_paths = {
        stringify(as_mapping(item).get("path")): as_mapping(item)
        for item in as_sequence(golden_compiled.get("expressions"))
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
            continue
        actual = paths[path]
        expected = expected_paths.get(path, {})
        if actual.get("normalized") != normalized:
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.expression",
                path,
                stringify(actual.get("normalized")),
                "Compiled expression normalization changed; refusing route.",
            )
        if actual.get("expression_hash") != expected.get("expression_hash"):
            append_diagnostic(
                diagnostics,
                "compiled_weak_form.expression_hash",
                path,
                stringify(actual.get("expression_hash")),
                "Compiled expression hash changed; refusing route.",
            )
    expect_field(
        as_mapping(compiled.get("semantic_evidence")).get(
            "operator_term_units_match_terminal"
        ),
        True,
        "compiled_operator.semantic_evidence.operator_term_units_match_terminal",
        "compiled_weak_form.units",
        diagnostics,
    )
    expect_field(
        as_mapping(compiled.get("semantic_evidence")).get(
            "boundary_units_match_terminal"
        ),
        True,
        "compiled_operator.semantic_evidence.boundary_units_match_terminal",
        "compiled_weak_form.units",
        diagnostics,
    )


def request_payload(
    payload: Mapping[str, Any],
    pde_ir: Mapping[str, Any],
    compiled: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the accepted request payload used for public route hashing."""

    return {
        "fixture_id": payload.get("fixture_id"),
        "source_problem_id": pde_ir.get("problem_id"),
        "source_formulation_id": pde_ir.get("formulation_id"),
        "source_ir_hash": pde_ir.get("canonical_hash"),
        "compiled_operator_hash": compiled.get("compiled_hash"),
        "measure": pde_ir.get("measure"),
        "numeraire": pde_ir.get("numeraire"),
        "units": compiled_units(pde_ir),
        "time_orientation": pde_ir.get("time_orientation"),
        "operator_sign_convention": as_mapping(pde_ir.get("operator")).get(
            "sign_convention"
        ),
        "route": route,
    }


def compiled_units(pde_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Extract state/value unit evidence from a screened pde_ir block."""

    state_variables = as_sequence(pde_ir.get("state_variables"))
    state_unit = (
        as_mapping(as_mapping(state_variables[0]).get("unit")) if state_variables else {}
    )
    terminal_unit = as_mapping(as_mapping(pde_ir.get("terminal_condition")).get("unit"))
    return {"state": state_unit, "value": terminal_unit}

