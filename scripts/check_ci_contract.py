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
}


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


def check_ci_contract() -> list[str]:
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

    return errors


def main() -> int:
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
