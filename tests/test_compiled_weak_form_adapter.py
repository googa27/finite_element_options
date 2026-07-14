"""Compiled pde_ir.v0 weak-form adapter tests for VQPW FEM #116."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from finite_element_options.validation import compiled_weak_form_adapter as adapter
from finite_element_options.validation import compiled_weak_form_screening as screening

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "compiled_weak_form" / "black_scholes_call_v0.json"
)


def _payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _codes(screen: adapter.CompiledWeakFormScreen) -> set[str]:
    return {item.code for item in screen.diagnostics}


def test_exact_compiled_black_scholes_fixture_screens_with_hashes_and_conventions() -> (
    None
):
    payload = _payload()

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is True
    assert screen.diagnostics == ()
    assert screen.fixture_id == adapter.PUBLIC_BS_FIXTURE_ID
    assert screen.source_ir_hash == adapter.PUBLIC_BS_SOURCE_HASH
    assert screen.compiled_operator_hash == adapter.PUBLIC_BS_COMPILED_HASH
    assert screen.route_hash and screen.route_hash.startswith("sha256:")
    assert screen.request["measure"] == "Q"
    assert screen.request["numeraire"] == {
        "currency": "USD",
        "kind": "money_market_account",
    }
    assert screen.request["time_orientation"] == "backward"
    assert (
        screen.request["operator_sign_convention"]
        == "backward_generator_minus_discount"
    )
    assert screen.request["route"]["boundary_partition"] == {
        "essential": ["lower", "upper"],
        "natural": [],
    }


def test_compiled_black_scholes_solve_reuses_existing_fem_route_and_matches_analytic_oracle() -> (
    None
):
    first = adapter.solve_compiled_weak_form(_payload())
    second = adapter.solve_compiled_weak_form(_payload())

    assert first == second
    assert first["format_version"] == "compiled-weak-form-fem-result-v0"
    assert first["status"] == "converged"
    assert first["problem_id"] == "black_scholes_call_public_synthetic"
    assert first["source_ir_hash"] == adapter.PUBLIC_BS_SOURCE_HASH
    assert first["compiled_operator_hash"] == adapter.PUBLIC_BS_COMPILED_HASH
    assert first["measure"] == "Q"
    assert first["numeraire"] == {"currency": "USD", "kind": "money_market_account"}
    assert first["boundary_partition"] == {
        "essential": ["lower", "upper"],
        "natural": [],
    }
    assert first["domain"] == {
        "lower": 0.0,
        "state": "S",
        "truncated_coordinate": "spot",
        "upper": 400.0,
    }
    assert first["time"]["orientation"] == "backward"
    assert (
        first["weak_form"]["source_sign_convention"]
        == "backward_generator_minus_discount"
    )
    assert (
        first["summary"]["price_absolute_error"]
        <= first["comparison_policy"]["metric_tolerances"]["price_absolute"]
    )
    assert (
        first["summary"]["delta_absolute_error"]
        <= first["comparison_policy"]["metric_tolerances"]["delta_absolute"]
    )
    assert (
        first["summary"]["gamma_absolute_error"]
        <= first["comparison_policy"]["metric_tolerances"]["gamma_absolute"]
    )


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (
            lambda p: p.__setitem__("privacy_class", "private"),
            "compiled_weak_form.privacy",
        ),
        (
            lambda p: p["compiled_operator"].__setitem__(
                "compiled_hash", "sha256:mutated"
            ),
            "compiled_weak_form.compiled_hash",
        ),
        (
            lambda p: p["pde_ir"].__setitem__("canonical_hash", "sha256:mutated"),
            "compiled_weak_form.source_hash",
        ),
        (
            lambda p: p["fem_route"].__setitem__(
                "mesh_family", "adaptive_unstructured"
            ),
            "compiled_weak_form.route",
        ),
        (
            lambda p: p["fem_route"].__setitem__("element_family", "lagrange_p1"),
            "compiled_weak_form.route",
        ),
        (
            lambda p: p["fem_route"].__setitem__(
                "requested_outputs", ["value", "vega"]
            ),
            "compiled_weak_form.outputs",
        ),
        (
            lambda p: p["pde_ir"].__setitem__(
                "state_variables", [p["pde_ir"]["state_variables"][0], {"symbol": "v"}]
            ),
            "compiled_weak_form.dimension",
        ),
        (
            lambda p: p["pde_ir"].__setitem__(
                "boundary_conditions", [{"kind": "neumann"}]
            ),
            "compiled_weak_form.boundary",
        ),
        (
            lambda p: p["pde_ir"].__setitem__("time_orientation", "forward"),
            "compiled_weak_form.time_orientation",
        ),
    ],
)
def test_private_mutated_or_unsupported_compiled_fixtures_fail_before_assembly(
    mutator, expected_code: str
) -> None:
    payload = _payload()
    mutator(payload)

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert expected_code in _codes(screen)
    with pytest.raises(adapter.CompiledWeakFormUnsupportedError) as excinfo:
        adapter.solve_compiled_weak_form(payload)
    assert excinfo.value.screen.accepted is False


@pytest.mark.parametrize(
    ("label", "mutator", "expected_code"),
    [
        (
            "numeraire_currency",
            lambda p: p["pde_ir"]["numeraire"].__setitem__("currency", "EUR"),
            "compiled_weak_form.numeraire",
        ),
        (
            "state_unit_currency",
            lambda p: p["pde_ir"]["state_variables"][0]["unit"].__setitem__(
                "currency", "EUR"
            ),
            "compiled_weak_form.units",
        ),
        (
            "terminal_payoff",
            lambda p: p["pde_ir"]["terminal_condition"].__setitem__(
                "expression", "max(K - S, 0)"
            ),
            "compiled_weak_form.terminal",
        ),
        (
            "boundary_expression",
            lambda p: p["pde_ir"]["boundary_conditions"][1].__setitem__(
                "expression", "V = S"
            ),
            "compiled_weak_form.boundary",
        ),
        (
            "boundary_map",
            lambda p: p["fem_route"]["boundary_map"]["upper"].__setitem__(
                "enforced_as", "natural"
            ),
            "compiled_weak_form.boundary",
        ),
        (
            "route_parameters",
            lambda p: p["fem_route"]["parameters"].__setitem__("rate", 0.06),
            "compiled_weak_form.route_exact",
        ),
        (
            "expression_hash",
            lambda p: p["compiled_operator"]["expressions"][5].__setitem__(
                "expression_hash", "sha256:mutated"
            ),
            "compiled_weak_form.expression_hash",
        ),
        (
            "expression_semantics",
            lambda p: p["compiled_operator"]["expressions"][5].__setitem__(
                "normalized", "0"
            ),
            "compiled_weak_form.expression",
        ),
        (
            "semantic_unit_evidence",
            lambda p: p["compiled_operator"]["semantic_evidence"].__setitem__(
                "operator_term_units_match_terminal", False
            ),
            "compiled_weak_form.units",
        ),
        (
            "nested_unknown",
            lambda p: p["pde_ir"]["operator"].__setitem__("extra", "x"),
            "compiled_weak_form.unknown_field",
        ),
    ],
)
def test_exact_screening_rejects_review_probe_mutations(
    label: str, mutator, expected_code: str
) -> None:
    payload = _payload()
    mutator(payload)

    screen = adapter.screen_compiled_weak_form(payload)

    assert label
    assert screen.accepted is False
    assert expected_code in _codes(screen)
    with pytest.raises(adapter.CompiledWeakFormUnsupportedError) as excinfo:
        adapter.solve_compiled_weak_form(payload)
    assert expected_code in _codes(excinfo.value.screen)


@pytest.mark.parametrize(
    ("label", "mutator", "expected_code"),
    [
        (
            "time_orientation",
            lambda p: p["fem_route"]["time"].__setitem__("orientation", "forward"),
            "compiled_weak_form.time",
        ),
        (
            "time_steps_not_80",
            lambda p: p["fem_route"]["time"].__setitem__("steps", 79),
            "compiled_weak_form.time",
        ),
        (
            "time_steps_non_positive",
            lambda p: p["fem_route"]["time"].__setitem__("steps", -1),
            "compiled_weak_form.time",
        ),
        (
            "time_steps_string_conversion",
            lambda p: p["fem_route"]["time"].__setitem__("steps", "80"),
            "compiled_weak_form.conversion",
        ),
        (
            "tau_start",
            lambda p: p["fem_route"]["time"].__setitem__("tau_start", 0.1),
            "compiled_weak_form.time",
        ),
        (
            "tau_end",
            lambda p: p["fem_route"]["time"].__setitem__("tau_end", 0.9),
            "compiled_weak_form.time",
        ),
        (
            "theta",
            lambda p: p["fem_route"]["time"].__setitem__("theta", 1.0),
            "compiled_weak_form.time",
        ),
        (
            "theta_string_conversion",
            lambda p: p["fem_route"]["time"].__setitem__("theta", "0.5"),
            "compiled_weak_form.conversion",
        ),
    ],
)
def test_time_block_is_exact_and_conversion_diagnostics_are_fail_closed(
    label: str, mutator, expected_code: str
) -> None:
    payload = _payload()
    mutator(payload)

    screen = adapter.screen_compiled_weak_form(payload)

    assert label
    assert screen.accepted is False
    assert expected_code in _codes(screen)
    with pytest.raises(adapter.CompiledWeakFormUnsupportedError) as excinfo:
        adapter.solve_compiled_weak_form(payload)
    assert excinfo.value.screen.accepted is False


def test_non_json_conversion_payload_returns_diagnostic_not_raw_value_error() -> None:
    payload = _payload()
    payload["fem_route"]["time"]["theta"] = object()

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert "compiled_weak_form.conversion" in _codes(screen)


def test_public_screen_input_requires_mapping_diagnostic() -> None:
    screen = adapter.screen_compiled_weak_form(["not", "a", "mapping"])

    assert screen.accepted is False
    assert screen.request == {}
    assert "compiled_weak_form.type" in _codes(screen)


def test_domain_number_conversions_are_typed_and_fail_closed() -> None:
    payload = _payload()
    payload["fem_route"]["domain"]["upper"] = "400.0"

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert "compiled_weak_form.conversion" in _codes(screen)


def test_non_finite_route_numbers_are_rejected_with_conversion_diagnostic() -> None:
    payload = _payload()
    payload["fem_route"]["domain"]["lower"] = float("nan")

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert "compiled_weak_form.conversion" in _codes(screen)


def test_boundary_split_is_keyed_by_boundary_id_not_list_order() -> None:
    payload = _payload()
    diagnostics: list[adapter.CompiledWeakFormDiagnostic] = []
    pde_ir = {
        "boundary_conditions": list(reversed(payload["pde_ir"]["boundary_conditions"]))
    }

    screening.check_boundary_split(pde_ir, payload["fem_route"], diagnostics)

    assert diagnostics == []


def test_duplicate_boundary_ids_are_rejected() -> None:
    payload = _payload()
    payload["pde_ir"]["boundary_conditions"][0]["boundary_id"] = "upper"

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert "compiled_weak_form.boundary" in _codes(screen)


def test_duplicate_compiled_expression_paths_are_rejected() -> None:
    payload = _payload()
    payload["compiled_operator"]["expressions"].append(
        dict(payload["compiled_operator"]["expressions"][-1])
    )

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert "compiled_weak_form.expression_duplicate" in _codes(screen)


def test_adapter_uses_safe_payload_access_after_screening() -> None:
    source = (
        ROOT
        / "src"
        / "finite_element_options"
        / "validation"
        / "compiled_weak_form_adapter.py"
    ).read_text(encoding="utf-8")

    assert 'payload["pde_ir"]' not in source
    assert "int(time.get" not in source


def test_unknown_private_field_fails_closed_without_source_tree_imports() -> None:
    payload = _payload()
    payload["private_statement_id"] = "secret-123"

    screen = adapter.screen_compiled_weak_form(payload)

    assert screen.accepted is False
    assert {
        "compiled_weak_form.private_field",
        "compiled_weak_form.unknown_field",
    } <= _codes(screen)
    source = (
        ROOT
        / "src"
        / "finite_element_options"
        / "validation"
        / "compiled_weak_form_adapter.py"
    ).read_text(encoding="utf-8")
    assert "from financial_problem_formulations" not in source
    assert "import financial_problem_formulations" not in source
    assert "importlib" not in source


def test_cli_screen_and_solve_emit_deterministic_json(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    evidence_path = tmp_path / "evidence.json"

    screen_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "finite_element_options.cli",
            "qps",
            "screen",
            str(FIXTURE),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    screen_payload = json.loads(screen_run.stdout)
    assert screen_payload["accepted"] is True
    assert screen_payload["compiled_operator_hash"] == adapter.PUBLIC_BS_COMPILED_HASH

    solve_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "finite_element_options.cli",
            "qps",
            "solve",
            str(FIXTURE),
            "--out",
            str(result_path),
            "--evidence",
            str(evidence_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    solve_stdout = json.loads(solve_run.stdout)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert solve_stdout["status"] == "converged"
    assert result["status"] == "converged"
    assert evidence == adapter.evidence_for_result(result)
    assert evidence["compiled_operator_hash"] == adapter.PUBLIC_BS_COMPILED_HASH
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_cli_qps_rejects_legacy_heston_flags_instead_of_ignoring_them() -> None:
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "finite_element_options.cli",
            "--k",
            "2.0",
            "qps",
            "screen",
            str(FIXTURE),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert run.returncode == 2
    assert "legacy Heston flags cannot be used with qps" in run.stderr
