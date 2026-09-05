"""Static contract tests for committed pyMOR benchmark evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from finite_element_options.validation.evidence.reduced_order import verify_pymor_benchmark
from finite_element_options.validation.evidence.serialization import (
    canonical_json_sha256,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs" / "evidence" / "black_scholes_pymor_rom_2026-09-05.json"
EXPECTED_SHA256 = "3de04c89a854b9b8751c9a24be38581c2b7a56ba5a882a212428857fb6de1f55"
EXPECTED_INPUT_SHA256 = "5360959855b4e573914e5e18ead47c3b55cfbc2f419cf62f6939445f29ab17be"


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
    assert payload["memory"]["online_rom_numerical_payload_bytes"] > 0
    assert payload["environment"]["pymor_cache_policy"].startswith("disabled")
    assert all(row["passed"] for row in payload["holdouts"])
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

    class ArtifactReport:
        def to_dict(self) -> dict[str, object]:
            return payload

    result = verify_pymor_benchmark(tampered, ArtifactReport())  # type: ignore[arg-type]
    assert result["exact"]["reference_study_input_hash"] is False
    assert result["passed"] is False
