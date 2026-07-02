"""Executable CI-contract tests for issue #59."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CI_CONTRACT = ROOT / "scripts" / "check_ci_contract.py"

spec = importlib.util.spec_from_file_location("check_ci_contract", CI_CONTRACT)
assert spec is not None and spec.loader is not None
check_ci_contract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_ci_contract_module)
check_ci_contract = check_ci_contract_module.check_ci_contract


def test_ci_contract_script_passes() -> None:
    assert check_ci_contract() == []


def test_actions_are_pinned_to_full_commit_shas() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    mutable_refs = re.findall(r"uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s#]+)", text)
    assert mutable_refs
    for action, ref in mutable_refs:
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{action}@{ref} is mutable"


def test_ci_profiles_are_required_and_named() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for job in ("package:", "test:", "optional_imports:", "supply_chain:"):
        assert f"  {job}" in text
    for profile in ("fd", "jax", "calibration", "viz", "ui"):
        assert f"profile: {profile}" in text


def test_supply_chain_and_artifact_gates_are_present() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for snippet in (
        "ruff check src tests scripts",
        "mypy --ignore-missing-imports",
        "python -m pip_audit",
        "cyclonedx-py environment",
        "python -m twine check dist/*",
        "--benchmark-json=benchmark.json",
        "coverage.xml",
        "backend_capabilities",
        "python scripts/check_ci_contract.py",
    ):
        assert snippet in text
