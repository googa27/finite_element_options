"""Compiled pde_ir.v0 weak-form adapter tests for VQPW FEM #116."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from finite_element_options.validation import compiled_weak_form_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "compiled_weak_form" / "black_scholes_call_v0.json"
)


def _payload() -> dict[str, object]:
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
