"""Static contract for the real external PETSc VI assessment artifact."""

from __future__ import annotations

import json
from pathlib import Path

from finite_element_options.validation.evidence.serialization import file_sha256
from finite_element_options.validation.evidence.petsc_vi import adapter


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/evidence/petsc_vi_assessment_2026-09-05.json"
EXPECTED_SHA256 = "b0ebd55b748c2c36382854ad6624f3f983b8a1ee25cdb1e34419df7fa9da5b35"


def test_petsc_version_matching_rejects_mismatched_runtime() -> None:
    assert adapter._petsc_versions_match("3.24.6", (3, 24, 6)) is True
    assert adapter._petsc_versions_match("3.24.6", (3, 24, 7)) is False


def test_petsc_runtime_pass_requires_version_match() -> None:
    assert (
        adapter._petsc_runtime_passed(
            version_match=False,
            ksp_converged=True,
            vi_converged=True,
            ts_converged=True,
        )
        is False
    )
    assert (
        adapter._petsc_runtime_passed(
            version_match=True,
            ksp_converged=True,
            vi_converged=True,
            ts_converged=True,
        )
        is True
    )


def test_petsc_external_evidence_is_real_bounded_and_fail_closed() -> None:
    """Keep trigger, runtime, parity, residual, and scope evidence synchronized."""

    assert file_sha256(ARTIFACT) == EXPECTED_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["privacy_class"] == "public_synthetic"
    assert payload["predecessor"]["verified"] is True
    assert payload["trigger"]["triggered"] is True
    assert payload["trigger"]["capability_status"] == "validated"
    assert payload["runtime_doctor"]["passed"] is True
    assert (
        payload["runtime_doctor"]["petsc4py_version"] == payload["runtime_doctor"]["petsc_version"]
    )
    assert payload["environment"]["finite_element_options_install_mode"] == "wheel"
    assert payload["runtime_doctor"]["ksp"]["converged"] is True
    assert payload["runtime_doctor"]["snes_vi"]["converged"] is True
    assert payload["runtime_doctor"]["ts"]["converged"] is True
    assert payload["projected_sor"]["solve_count"] == 80
    assert payload["petsc_snes_vi"]["solve_count"] == 80
    assert payload["projected_sor"]["projected_residual_max"] <= 1.0e-9
    assert payload["petsc_snes_vi"]["projected_residual_max"] <= 1.0e-9
    assert payload["decision"]["status"] == "promote_external_single_rank_vi_adapter"
    assert all(payload["decision"]["checks"].values())
    assert payload["decision"]["scipy_remains_canonical"] is True
    assert payload["decision"]["distributed_assembly_claim"] is False
    assert payload["failure_evidence"]["passed"] is True
    assert payload["failure_evidence"]["typed_exception_caught"] is True
    assert payload["timing"]["backend_order"] == "alternated by repeat"
    assert payload["parity_errors"]["projected_sor_early_exercise_premium"] > 0.0
    assert payload["parity_errors"]["petsc_early_exercise_premium"] > 0.0
    assert "/home/" not in ARTIFACT.read_text(encoding="utf-8")
