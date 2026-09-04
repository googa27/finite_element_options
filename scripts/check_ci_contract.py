#!/usr/bin/env python3
"""CI workflow contract checks for finite_element_options.

The workflow is part of the repository's supply-chain surface.  This script keeps
issue #59's non-negotiables executable without depending on PyYAML in the base
runtime: Actions must be pinned to immutable SHAs, jobs must declare explicit
permissions/timeouts, and required CI profiles must remain present.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PINNED_ACTION = re.compile(r"uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9a-f]{40})\b")
MUTABLE_ACTION = re.compile(r"uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s#]+)")
JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")

REQUIRED_JOBS = {
    "package",
    "test",
    "fenicsx_contract",
    "optional_imports",
    "supply_chain",
}

REQUIRED_SNIPPETS = {
    "least privilege permissions": "permissions:\n  contents: read",
    "workflow concurrency": "concurrency:",
    "python 3.11 support": "'3.11'",
    "python 3.12 support": "'3.12'",
    "wheel build": "python -m build --sdist --wheel",
    "twine check": "python -m twine check dist/*",
    "installed wheel import contract": "installed import contract OK",
    "installed wheel README examples": "scripts/check_readme_examples.py README.md",
    "capability doc staleness check": "scripts/generate_capability_docs.py --check",
    "pydocstyle gate": "pydocstyle src/finite_element_options",
    "ruff gate": "ruff check src tests scripts",
    "type gate": "mypy --ignore-missing-imports",
    "architecture contract": "scripts/check_architecture_contract.py",
    "packaging contract": "tests/test_packaging_contract.py",
    "coverage gate": "--cov=finite_element_options",
    "benchmark artifact": "--benchmark-json=benchmark.json",
    "FEniCSx contract job": "fenicsx_contract",
    "FEniCSx backend contract tests": "tests/test_fenics_solver.py",
    "pip audit": "python -m pip_audit",
    "cyclonedx sbom": "cyclonedx-py environment",
    "optional fd profile": "profile: fd",
    "optional jax profile": "profile: jax",
    "optional calibration profile": "profile: calibration",
    "optional viz profile": "profile: viz",
    "optional ui profile": "profile: ui",
    "optional volatility profile": "profile: volatility",
    "optional changepoints profile": "profile: changepoints",
    "optional quantlib profile": "profile: quantlib",
    "optional identifiability profile": "profile: identifiability",
    "optional dependency matrix field": "DEPENDENCY: ${{ matrix.dependency }}",
    "optional dependency import proof": "importlib.import_module(dependency)",
    "QuantLib evaluation-date restoration proof": "quantlib_evaluation_date",
    "QuantLib failure restoration proof": "forced QuantLib failure",
    "iminuit focused installed-wheel tests": "test_regime_switching_quanto_iminuit_identifiability.py",
    "iminuit canonical artifact verification": "run_iminuit_identifiability.py",
}

OPTIONAL_PROFILE_DEPENDENCIES = {
    "fd": "findiff",
    "jax": "jax",
    "calibration": "pymc",
    "viz": "matplotlib",
    "ui": "streamlit",
    "volatility": "arch",
    "changepoints": "ruptures",
    "quantlib": "QuantLib",
    "identifiability": "iminuit",
}

NEW_OPTIONAL_PROFILES = {
    "volatility",
    "changepoints",
    "quantlib",
    "identifiability",
}

NEW_OPTIONAL_PROFILE_PYTHONS = {"3.11", "3.12"}
SUPPLY_CHAIN_AUDITED_EXTRAS = (
    "volatility",
    "changepoints",
    "quantlib",
    "identifiability",
)
PROJECT_EXTRA_INSTALL = re.compile(r"python -m pip install\b[^\n]*(?:-e\s+)?['\"]?\.\[([^\]\s]+)\]")


def _workflow_text() -> str:
    if not WORKFLOW.exists():
        raise AssertionError(f"missing workflow: {WORKFLOW}")
    return WORKFLOW.read_text(encoding="utf-8")


def _job_blocks(text: str) -> dict[str, str]:
    jobs_start = text.find("jobs:\n")
    if jobs_start < 0:
        raise AssertionError("workflow must contain a jobs block")
    lines = text[jobs_start:].splitlines()
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:]:
        match = JOB_HEADER.match(line)
        if match:
            current = match.group(1)
            blocks[current] = [line]
            continue
        if current is not None:
            assert current is not None
            blocks[current].append(line)
    return {name: "\n".join(block) for name, block in blocks.items()}


def _yaml_scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _matrix_include_entries(job_block: str) -> list[dict[str, str]]:
    if "include:" not in job_block:
        return []
    include_block = job_block.split("include:", 1)[1].split("steps:", 1)[0]
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    item = re.compile(r"^\s*-\s+([A-Za-z0-9_-]+):\s*(.+?)\s*$")
    field = re.compile(r"^\s+([A-Za-z0-9_-]+):\s*(.+?)\s*$")

    for line in include_block.splitlines():
        item_match = item.match(line)
        if item_match:
            current = {item_match.group(1): _yaml_scalar(item_match.group(2))}
            entries.append(current)
            continue
        field_match = field.match(line)
        if current is not None and field_match:
            current[field_match.group(1)] = _yaml_scalar(field_match.group(2))
    return entries


def _check_optional_import_matrix(blocks: dict[str, str]) -> list[str]:
    block = blocks.get("optional_imports")
    if block is None:
        return ["optional_imports job is required for optional dependency proofs"]

    errors: list[str] = []
    entries = _matrix_include_entries(block)
    if not entries:
        return ["optional_imports job must use an explicit matrix.include list"]

    for entry in entries:
        profile = entry.get("profile")
        dependency = entry.get("dependency")
        if not profile:
            errors.append(f"optional_imports matrix entry lacks profile: {entry}")
            continue
        if not dependency:
            errors.append(f"optional profile {profile} must declare dependency")
            continue
        expected_dependency = OPTIONAL_PROFILE_DEPENDENCIES.get(profile)
        if expected_dependency is None:
            errors.append(f"unexpected optional profile in CI matrix: {profile}")
        elif dependency != expected_dependency:
            errors.append(
                f"optional profile {profile} must import dependency "
                f"{expected_dependency!r}, got {dependency!r}"
            )

    for profile in NEW_OPTIONAL_PROFILES:
        covered = {
            entry.get("python-version")
            for entry in entries
            if entry.get("profile") == profile
            and entry.get("dependency") == OPTIONAL_PROFILE_DEPENDENCIES[profile]
        }
        missing = sorted(NEW_OPTIONAL_PROFILE_PYTHONS - covered)
        if missing:
            errors.append(
                f"optional profile {profile} must cover Python "
                f"{sorted(NEW_OPTIONAL_PROFILE_PYTHONS)}, missing {missing}"
            )

    steps_block = block.split("steps:", 1)[1] if "steps:" in block else ""
    setup_is_matrixed = "${{ matrix.python-version }}" in steps_block
    name_is_matrixed = "${{ matrix.python-version }}" in block.split("steps:", 1)[0]
    if not setup_is_matrixed:
        errors.append("optional_imports must set up matrix.python-version")
    if "${{ matrix.profile }}" not in block:
        errors.append("optional_imports job name must include matrix.profile")
    if not name_is_matrixed:
        errors.append("optional_imports job name must include matrix.python-version")

    return errors


def _project_extras_in_pip_installs(job_block: str) -> set[str]:
    extras: set[str] = set()
    for match in PROJECT_EXTRA_INSTALL.finditer(job_block):
        extras.update(extra.strip() for extra in match.group(1).split(",") if extra.strip())
    return extras


def _check_supply_chain_audit(blocks: dict[str, str]) -> list[str]:
    block = blocks.get("supply_chain")
    if block is None:
        return ["supply_chain job is required for vulnerability audit and SBOM"]

    extras = _project_extras_in_pip_installs(block)
    if not extras:
        return ["supply_chain must install this project with audited extras"]

    missing = sorted(set(SUPPLY_CHAIN_AUDITED_EXTRAS) - extras)
    if missing:
        return [f"supply_chain audited install missing optional extras: {missing}"]
    return []


def check_ci_contract() -> list[str]:
    """Return CI workflow contract violations, or an empty list when valid."""

    text = _workflow_text()
    errors: list[str] = []

    for label, snippet in REQUIRED_SNIPPETS.items():
        if snippet not in text:
            errors.append(f"missing {label}: {snippet!r}")

    actions = MUTABLE_ACTION.findall(text)
    if not actions:
        errors.append("workflow must use pinned third-party actions")
    for action, ref in actions:
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            errors.append(f"action {action}@{ref} is not pinned to a full commit SHA")
    pinned = {action for action, _ in PINNED_ACTION.findall(text)}
    for expected in {"actions/checkout", "actions/setup-python", "actions/upload-artifact"}:
        if expected not in pinned:
            errors.append(f"missing pinned {expected} usage")

    blocks = _job_blocks(text)
    missing_jobs = sorted(REQUIRED_JOBS - set(blocks))
    if missing_jobs:
        errors.append(f"missing required jobs: {missing_jobs}")
    for name, block in blocks.items():
        if "timeout-minutes:" not in block:
            errors.append(f"job {name} must declare timeout-minutes")
        if "runs-on:" not in block:
            errors.append(f"job {name} must declare runs-on")

    errors.extend(_check_optional_import_matrix(blocks))
    errors.extend(_check_supply_chain_audit(blocks))

    return errors


def main() -> int:
    """Run CI workflow contract checks as a command-line gate."""

    errors = check_ci_contract()
    if errors:
        print("CI contract violations:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"CI contract passed: {WORKFLOW.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
