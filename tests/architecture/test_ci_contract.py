"""Executable CI-contract tests for issue #59."""

from __future__ import annotations

import importlib.util
import re
import tomllib
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
    for profile in (
        "fd",
        "jax",
        "calibration",
        "viz",
        "ui",
        "volatility",
        "changepoints",
        "quantlib",
        "identifiability",
    ):
        assert f"profile: {profile}" in text


def test_new_optional_profiles_import_actual_dependency_and_cover_supported_pythons() -> None:
    """Issue #130 CI must prove each new extra on Python 3.11 and 3.12."""

    text = WORKFLOW.read_text(encoding="utf-8")
    required = {
        "volatility": "arch",
        "changepoints": "ruptures",
        "quantlib": "QuantLib",
        "identifiability": "iminuit",
    }
    for profile, dependency in required.items():
        for python_version in ("3.11", "3.12"):
            pattern = re.compile(
                rf"profile:\s*{profile}\b(?:(?!\n\s*-\s*profile:).)*"
                rf"python-version:\s*['\"]?{re.escape(python_version)}['\"]?(?:(?!\n\s*-\s*profile:).)*"
                rf"dependency:\s*{re.escape(dependency)}\b",
                re.DOTALL,
            )
            assert pattern.search(text), f"missing {profile} {python_version} dependency proof"
    assert "DEPENDENCY: ${{ matrix.dependency }}" in text
    assert "importlib.import_module(dependency)" in text or "require_optional(dependency)" in text
    assert "quantlib_evaluation_date" in text
    assert "forced QuantLib failure" in text


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
        "python scripts/generate_capability_docs.py --check",
        "python scripts/check_readme_examples.py README.md",
    ):
        assert snippet in text


def test_static_analysis_toolchain_is_bounded_for_reproducible_ci() -> None:
    """Avoid silent linter-major drift breaking the only full test job."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]
    for extra in ("validation", "dev"):
        ruff_specs = [dep for dep in optional_deps[extra] if dep.startswith("ruff")]
        assert ruff_specs == ["ruff>=0.8,<0.13"]

    constraints = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    assert "ruff>=0.8,<0.13" in constraints
