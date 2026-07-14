"""Black-Scholes FEM verification evidence and CLI coverage for #117."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from finite_element_options import cli
from finite_element_options.validation import (
    FEM_VERIFICATION_BENCHMARK_ID,
    canonical_hash,
    run_verification_benchmark,
    validate_evidence,
)


@pytest.fixture(scope="module")
def accepted_evidence() -> dict[str, Any]:
    evidence = run_verification_benchmark()
    validate_evidence(evidence)
    return evidence


def _rebind_hashes(evidence: dict[str, Any]) -> None:
    for hash_key, section_key in (
        ("backend_hash", "backend"),
        ("mesh_time_hash", "mesh_time"),
        ("request_hash", "request"),
        ("convention_hash", "convention"),
        ("result_hash", "result"),
    ):
        evidence["hashes"][hash_key] = canonical_hash(evidence[section_key])
    evidence["evidence_hash"] = canonical_hash(
        {key: value for key, value in evidence.items() if key != "evidence_hash"}
    )


def test_black_scholes_convergence_greeks_and_no_arbitrage(
    accepted_evidence: dict[str, Any],
) -> None:
    result = accepted_evidence["result"]
    summary = result["black_scholes_summary"]

    assert accepted_evidence["benchmark_id"] == FEM_VERIFICATION_BENCHMARK_ID
    assert accepted_evidence["status"] == "accepted"
    assert summary["price_absolute_error"] <= summary["price_tolerance_absolute"]
    assert summary["delta_absolute_error"] <= summary["delta_tolerance_absolute"]
    assert summary["gamma_absolute_error"] <= summary["gamma_tolerance_absolute"]
    assert result["no_arbitrage"]["accepted"] is True
    assert result["no_arbitrage"]["failures"] == []
    assert len(result["black_scholes_rows"]) >= 3
    assert all(row["absolute_error"] >= 0.0 for row in result["black_scholes_rows"])
    assert all(
        order >= 1.0 for order in result["observed_orders"]["black_scholes_price_h"]
    )


def test_verification_evidence_is_deterministic_and_self_hashing(
    accepted_evidence: dict[str, Any],
) -> None:
    second = run_verification_benchmark()
    validate_evidence(second)

    assert second == accepted_evidence
    assert second["evidence_hash"] == accepted_evidence["evidence_hash"]
    assert accepted_evidence["evidence_hash"] == canonical_hash(
        {
            key: value
            for key, value in accepted_evidence.items()
            if key != "evidence_hash"
        }
    )


def test_validate_evidence_rejects_tampered_sections_hashes_and_bundle(
    accepted_evidence: dict[str, Any],
) -> None:
    tampered_section = deepcopy(accepted_evidence)
    tampered_section["backend"]["backend"] = "mutated"
    with pytest.raises(ValueError, match="backend_hash"):
        validate_evidence(tampered_section)

    tampered_hash = deepcopy(accepted_evidence)
    tampered_hash["hashes"]["result_hash"] = "0" * 64
    with pytest.raises(ValueError, match="result_hash"):
        validate_evidence(tampered_hash)

    tampered_bundle = deepcopy(accepted_evidence)
    tampered_bundle["evidence_hash"] = "0" * 64
    with pytest.raises(ValueError, match="evidence_hash"):
        validate_evidence(tampered_bundle)


def test_validate_evidence_rejects_rehashed_numerically_false_rows(
    accepted_evidence: dict[str, Any],
) -> None:
    manufactured = deepcopy(accepted_evidence)
    manufactured["result"]["manufactured_h_refinement"][0]["l2_error"] = 99.0
    _rebind_hashes(manufactured)
    with pytest.raises(ValueError, match="manufactured refinement"):
        validate_evidence(manufactured)

    black_scholes = deepcopy(accepted_evidence)
    black_scholes["result"]["black_scholes_rows"][-1]["observed_price"] = 99.0
    _rebind_hashes(black_scholes)
    with pytest.raises(ValueError, match="Black-Scholes row inconsistency"):
        validate_evidence(black_scholes)

    no_arbitrage = deepcopy(accepted_evidence)
    no_arbitrage["result"]["no_arbitrage"]["rows"][0]["price"] = 79.0
    _rebind_hashes(no_arbitrage)
    with pytest.raises(ValueError, match="no-arbitrage FEM surface mismatch"):
        validate_evidence(no_arbitrage)

    perturbation = deepcopy(accepted_evidence)
    row = perturbation["result"]["perturbation_failures"]["source"]
    for key in (
        "l2_error",
        "h1_error",
        "payoff_relevant_error",
        "algebraic_residual_inf",
        "boundary_residual_inf",
    ):
        row[key] = 0.0
    row["accepted"] = False
    _rebind_hashes(perturbation)
    with pytest.raises(ValueError, match="perturbation did not fail numerical gates"):
        validate_evidence(perturbation)


def test_validation_cli_writes_only_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import finite_element_options.validation.fem_evidence as fem_evidence

    calls: list[str] = []
    payload = {
        "status": "accepted",
        "accepted": True,
        "benchmark_id": FEM_VERIFICATION_BENCHMARK_ID,
        "evidence_hash": "abc123",
    }

    def fake_run() -> dict[str, Any]:
        calls.append("run")
        return payload

    def fake_validate(evidence: dict[str, Any]) -> None:
        calls.append("validate")
        assert evidence is payload

    monkeypatch.setattr(fem_evidence, "run_verification_benchmark", fake_run)
    monkeypatch.setattr(fem_evidence, "validate_evidence", fake_validate)
    out = tmp_path / "fem_evidence.json"

    assert (
        cli.main(["validation", "run-benchmark", "fem-bs-001", "--out", str(out)]) == 0
    )

    assert calls == ["run", "validate"]
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    status = json.loads(capsys.readouterr().out)
    assert status == {
        "benchmark_id": FEM_VERIFICATION_BENCHMARK_ID,
        "evidence_hash": "abc123",
        "out": str(out),
        "status": "accepted",
    }


def test_validation_cli_emits_json_stdout_without_out(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import finite_element_options.validation.fem_evidence as fem_evidence

    payload = {
        "status": "accepted",
        "accepted": True,
        "benchmark_id": FEM_VERIFICATION_BENCHMARK_ID,
        "evidence_hash": "stdout-hash",
    }
    monkeypatch.setattr(fem_evidence, "run_verification_benchmark", lambda: payload)
    monkeypatch.setattr(fem_evidence, "validate_evidence", lambda evidence: None)

    assert cli.main(["validation", "run-benchmark", "fem-bs-001"]) == 0

    assert json.loads(capsys.readouterr().out) == payload
