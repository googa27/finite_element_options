"""Static and replay gates for the adoption-sequence closeout artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/evidence/adoption_sequence_closeout_2026-09-05.json"
VISUAL = ROOT / "docs/images/adoption_sequence_closeout_2026-09-05.png"
REPORT = ROOT / "docs/ADOPTION_SEQUENCE_CLOSEOUT.md"
SCRIPT = ROOT / "scripts/generate_adoption_sequence_closeout.py"
EXPECTED_MATRIX_SHA256 = "6bd088cd664f2dd6c69fa7a6188d177da031548226c7412bae67156be7914ca0"
EXPECTED_VISUAL_SHA256 = "8d9b8d8ac59c1da3e8c51e4e78c0ad597ab02a08b989e11affb37d4cd902b360"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def test_closeout_matrix_is_hash_bound_and_fail_closed() -> None:
    """All eight decisions and their source hashes remain immutable."""

    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert _sha256(MATRIX) == EXPECTED_MATRIX_SHA256
    assert payload["schema_version"] == "adoption-sequence-closeout/v1"
    assert payload["privacy_class"] == "public_synthetic"
    assert payload["baseline"]["issue"] == 129
    assert payload["baseline"]["status"] == "complete"
    assert [step["step"] for step in payload["steps"]] == list(range(1, 9))
    assert [step["issue"] for step in payload["steps"]] == list(range(130, 138))
    assert sum(step["class"] == "REJECT" for step in payload["steps"]) == 1
    assert payload["steps"][0]["evidence"] == [
        {
            "path": "docs/evidence/dependency_boundaries_2026-09-04.json",
            "sha256": _sha256(ROOT / "docs/evidence/dependency_boundaries_2026-09-04.json"),
        }
    ]
    assert all(step["decision_complete"] for step in payload["steps"])
    assert [step["route_action"] for step in payload["steps"]] == [
        "adopt",
        "reject",
        "retain",
        "retain",
        "retain",
        "promote",
        "promote_external",
        "adopt",
    ]
    assert payload["portfolio_decision"] == {
        "baseline_complete": True,
        "base_wheel_optional_stack_leaks": 0,
        "bounded_non_rejections": 7,
        "capability_matrix_maturity_upgrades": 0,
        "epic_ready_to_close": True,
        "evidence_backed_rejections": 1,
        "route_action_counts": {
            "adopt": 2,
            "promote": 1,
            "promote_external": 1,
            "reject": 1,
            "retain": 3,
        },
        "status": "close_adoption_sequence",
        "steps_complete": 8,
    }
    assert set(payload["uncertainty_decomposition"]) == {
        "statistical",
        "model_form",
        "numerical",
        "sampling",
    }
    for item in [payload["baseline"], *payload["steps"]]:
        for source in item["evidence"]:
            assert _sha256(ROOT / source["path"]) == source["sha256"]


def test_closeout_visual_and_report_match_canonical_matrix() -> None:
    """The mobile-first visual and prose report name the canonical hashes."""

    assert _sha256(VISUAL) == EXPECTED_VISUAL_SHA256
    assert _png_dimensions(VISUAL) == (2400, 3000)
    report = REPORT.read_text(encoding="utf-8")
    assert EXPECTED_MATRIX_SHA256 in report
    assert EXPECTED_VISUAL_SHA256 in report
    assert "NO PRODUCTION MATURITY UPGRADE" not in report
    assert "Capability-matrix production maturity remains unchanged" in report


def test_closeout_generator_replays_canonical_json(tmp_path: Path) -> None:
    """Regeneration must reproduce matrix bytes from the seven source artifacts."""

    generated_json = tmp_path / "matrix.json"
    generated_png = tmp_path / "matrix.png"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json",
            str(generated_json),
            "--png",
            str(generated_png),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated_json.read_bytes() == MATRIX.read_bytes()
    assert _png_dimensions(generated_png) == (2400, 3000)
