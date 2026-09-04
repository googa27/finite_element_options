"""Executable CI-contract tests for issue #59."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CI_CONTRACT = ROOT / "scripts" / "check_ci_contract.py"

spec = importlib.util.spec_from_file_location("check_ci_contract", CI_CONTRACT)
assert spec is not None and spec.loader is not None
check_ci_contract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_ci_contract_module)
check_ci_contract = check_ci_contract_module.check_ci_contract

AUDITED_OPTIONAL_EXTRAS = (
    "volatility",
    "changepoints",
    "quantlib",
    "identifiability",
)


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
    assert "test_regime_switching_quanto_quantlib_oracle.py" in text
    assert "-k 'quantlib or canonical_artifact_scope'" in text


def test_quantlib_optional_pytest_uses_workspace_paths_after_tmp_cd() -> None:
    """Installed-wheel QuantLib tests must remain executable after CI cd's to /tmp."""

    text = WORKFLOW.read_text(encoding="utf-8")
    after_tmp_cd = text.split("cd /tmp", 1)[1]
    expected_paths = [
        '"${GITHUB_WORKSPACE}/tests/examples/test_regime_switching_quanto_adoption_boundaries.py"',
        '"${GITHUB_WORKSPACE}/tests/examples/test_regime_switching_quanto_quantlib_oracle.py"',
    ]
    pytest_block = after_tmp_cd.split("-k 'quantlib or canonical_artifact_scope'", 1)[0]
    for path in expected_paths:
        assert path in pytest_block
    assert '-c "${GITHUB_WORKSPACE}/constraints.txt"' in after_tmp_cd
    assert "\n              tests/examples/" not in pytest_block


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


def test_supply_chain_audits_new_optional_dependency_extras() -> None:
    """Supply-chain evidence must include the issue #130 optional stacks."""

    text = WORKFLOW.read_text(encoding="utf-8")
    supply_chain = text.split("  supply_chain:", 1)[1]
    install_commands = re.findall(r"python -m pip install(?:[^\n]*)", supply_chain)
    audited_project_installs = [command for command in install_commands if ".[" in command]
    assert audited_project_installs, "supply_chain must install this project for audit"
    for extra in AUDITED_OPTIONAL_EXTRAS:
        assert any(extra in command for command in audited_project_installs), (
            f"supply_chain audit install must include [{extra}]"
        )


@pytest.mark.parametrize("removed_extra", AUDITED_OPTIONAL_EXTRAS)
def test_ci_contract_rejects_supply_chain_audit_missing_new_optional_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, removed_extra: str
) -> None:
    """The executable CI contract must reject dropped audit coverage."""

    text = WORKFLOW.read_text(encoding="utf-8")
    mutated = text.replace(f",{removed_extra}", "", 1)
    workflow = tmp_path / "ci.yml"
    workflow.write_text(mutated, encoding="utf-8")
    monkeypatch.setattr(check_ci_contract_module, "WORKFLOW", workflow)

    errors = check_ci_contract()

    assert any("supply_chain" in error and removed_extra in error for error in errors)


def test_static_analysis_toolchain_is_bounded_for_reproducible_ci() -> None:
    """Avoid silent linter-major drift breaking the only full test job."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]
    for extra in ("validation", "dev"):
        ruff_specs = [dep for dep in optional_deps[extra] if dep.startswith("ruff")]
        assert ruff_specs == ["ruff>=0.8,<0.13"]

    constraints = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    assert "ruff>=0.8,<0.13" in constraints
