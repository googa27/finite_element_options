"""Static contract tests for committed pyMOR benchmark evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from finite_element_options.validation.evidence.reduced_order import (
    PymorBenchmarkReport,
    PymorBlackScholesConfig,
    verify_pymor_benchmark,
)
from finite_element_options.validation.evidence.serialization import (
    canonical_json_sha256,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs" / "evidence" / "black_scholes_pymor_rom_2026-09-05.json"
EXPECTED_SHA256 = "f30d712e054937ac7e17ea452fc2bcbc0a874087b1ec180caa5e63dc190ea4b7"
EXPECTED_INPUT_SHA256 = "d56805683c07bd8ef5bd7a54b39c3faca3bcd48fd01366ef0de3f7e7a97a0044"


def test_committed_pymor_evidence_is_hash_bound_and_promoted() -> None:
    """Keep docs, scientific gates, timings, and evidence bytes synchronized."""

    assert file_sha256(ARTIFACT) == EXPECTED_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["study_input_hash"] == EXPECTED_INPUT_SHA256
    assert canonical_json_sha256(payload["study_input"]) == EXPECTED_INPUT_SHA256
    assert payload["privacy_class"] == "public_synthetic"
    assert payload["predecessor"]["verified"] is True
    assert payload["decomposition"]["passed"] is True
    assert payload["decision"]["status"] == "promote_optional_adapter"
    assert payload["decision"]["capability_matrix_upgrade"] is False
    assert all(payload["decision"]["checks"].values())
    assert payload["timing"]["median_online_speedup"] >= 10.0
    assert payload["timing"]["ten_x_amortization_solve_count"] <= 1000
    assert len(payload["timing"]["full_order"]["samples_seconds"]) == 66
    assert len(payload["timing"]["full_order_setup"]["samples_seconds"]) == 6
    assert len(payload["timing"]["reduced_order"]["samples_seconds"]) == 66
    assert payload["timing"]["full_order_policy"].startswith("parameter-specific")
    assert payload["timing"]["full_order_factorization_reuses_per_holdout"] == 13
    assert payload["memory"]["online_rom_numerical_payload_bytes"] > 0
    assert payload["environment"]["pymor_cache_policy"].startswith("scoped and RLock-serialized")
    assert all(row["passed"] for row in payload["holdouts"])
    assert (
        max(row["solver_diagnostics"]["full_order_residual_linf"] for row in payload["holdouts"])
        <= payload["study_input"]["linear_residual_tolerance"]
    )
    assert (
        max(row["solver_diagnostics"]["reduced_order_residual_linf"] for row in payload["holdouts"])
        <= payload["study_input"]["linear_residual_tolerance"]
    )
    assert all(row["passed"] for row in payload["envelope_refusals"])
    capability_matrix = (ROOT / "docs" / "CAPABILITY_MATRIX.md").read_text(encoding="utf-8")
    normalized_matrix = capability_matrix.lower()
    assert all(token not in normalized_matrix for token in ("pymor", "reduced-order", "rom-"))
    assert "/home/" not in ARTIFACT.read_text(encoding="utf-8")


def test_semantic_verify_rejects_tampered_reference_input() -> None:
    """The payload itself, not merely its copied digest field, is hash-bound."""

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    tampered = deepcopy(payload)
    tampered["study_input"]["time_steps"] += 1
    fresh_payload = deepcopy(payload)
    fresh_payload["decision"]["checks"]["solver_residuals"] = True

    class ArtifactReport:
        def to_dict(self) -> dict[str, object]:
            return fresh_payload

    result = verify_pymor_benchmark(tampered, ArtifactReport())  # type: ignore[arg-type]
    assert result["exact"]["reference_study_input_hash"] is False
    assert result["passed"] is False


def test_serialized_high_precision_study_input_remains_hash_bound() -> None:
    """Evidence normalization must not rewrite hash-bound caller inputs."""

    base = PymorBlackScholesConfig()
    precise_minimum = 0.10000000001
    config = replace(
        base,
        volatility_min=precise_minimum,
        training_volatilities=(precise_minimum, *base.training_volatilities[1:]),
    )
    report = PymorBenchmarkReport(
        config=config,
        predecessor={},
        environment={},
        decomposition={},
        offline={},
        holdouts=(),
        timing={},
        memory={},
        envelope_refusals=(),
        decision={},
    ).to_dict()
    assert report["study_input"]["volatility_min"] == precise_minimum
    assert canonical_json_sha256(report["study_input"]) == report["study_input_hash"]
