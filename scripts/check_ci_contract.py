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
    "supply_chain_bayesian",
    "supply_chain_jax_regime",
    "visual_reproducibility",
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
    "bounded diskcache advisory exception": "--ignore-vuln PYSEC-2026-2447",
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
    "optional uncertainty profile": "profile: uncertainty",
    "optional reduction profile": "profile: reduction",
    "optional bayesian profile": "profile: bayesian",
    "optional bayesian-jax profile": "profile: bayesian-jax",
    "optional jax-regime profile": "profile: jax-regime",
    "optional dependency matrix field": "DEPENDENCY: ${{ matrix.dependency }}",
    "optional dependency import proof": "importlib.import_module(dependency)",
    "QuantLib evaluation-date restoration proof": "quantlib_evaluation_date",
    "QuantLib failure restoration proof": "forced QuantLib failure",
    "iminuit focused installed-wheel tests": "test_regime_switching_quanto_iminuit_identifiability.py",
    "OpenTURNS focused installed-wheel tests": "test_regime_switching_quanto_openturns_uq.py",
    "pyMOR focused installed-wheel tests": "test_pymor_black_scholes_rom.py",
    "PyMC focused installed-wheel tests": "test_pymc_profile.py",
    "NumPyro focused installed-wheel tests": "test_numpyro_profile.py",
    "Bayesian/JAX semantic replay": "scripts/run_bayesian_jax_profile.py --verify",
    "Bayesian/JAX hash-pinned lock install": "environments/bayesian-jax-py312/requirements.lock",
    "Bayesian hash-pinned lock install": "environments/bayesian-py312/requirements.lock",
    "Bayesian/JAX require hashes": "--require-hashes",
    "Bayesian/JAX supply-chain artifact": "supply-chain-bayesian-jax-evidence",
    "JAX regime focused installed-wheel tests": "external_tests/jax_regime/test_profile.py",
    "JAX regime hash-pinned wheel profile": "environments/jax-regime-py312/requirements.lock",
    "JAX regime synthetic CI verification": "--synthetic --warmup 75 --samples 75",
    "JAX regime supply-chain artifact": "supply-chain-jax-regime-evidence",
    "JAX regime visual hash-pinned lock": (
        "environments/jax-regime-visual-py312/requirements.lock"
    ),
    "JAX regime visual layout QA": "--qa-layout",
    "JAX regime visual exact byte comparison": (
        'sha256sum -c "${GITHUB_WORKSPACE}/docs/images/jax_regime_study_2026-09-07.sha256"'
    ),
    "audited release wheel install": (
        "python -m pip install --no-deps dist/finite_element_options-*.whl"
    ),
}

OPTIONAL_PROFILE_DEPENDENCIES = {
    "fd": "findiff",
    "jax": "jax",
    "calibration": "statsmodels",
    "viz": "matplotlib",
    "ui": "streamlit",
    "volatility": "arch",
    "changepoints": "ruptures",
    "quantlib": "QuantLib",
    "identifiability": "iminuit",
    "uncertainty": "openturns",
    "reduction": "pymor",
    "bayesian": "pymc",
    "bayesian-jax": "numpyro",
    "jax-regime": "dynamax",
}

NEW_OPTIONAL_PROFILES = {
    "volatility",
    "changepoints",
    "quantlib",
    "identifiability",
    "uncertainty",
    "reduction",
    "bayesian",
    "bayesian-jax",
    "jax-regime",
}

NEW_OPTIONAL_PROFILE_PYTHONS = {"3.11", "3.12"}
PY312_ONLY_OPTIONAL_PROFILES = {"bayesian", "bayesian-jax", "jax-regime"}
SUPPLY_CHAIN_AUDITED_EXTRAS = (
    "build",
    "calibration",
    "fd",
    "io",
    "jax",
    "viz",
    "ui",
    "volatility",
    "changepoints",
    "quantlib",
    "identifiability",
    "uncertainty",
    "reduction",
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
            str(entry.get("python-version"))
            for entry in entries
            if entry.get("profile") == profile
            and entry.get("dependency") == OPTIONAL_PROFILE_DEPENDENCIES[profile]
            and entry.get("python-version") is not None
        }
        expected_versions = (
            {"3.12"} if profile in PY312_ONLY_OPTIONAL_PROFILES else NEW_OPTIONAL_PROFILE_PYTHONS
        )
        missing = sorted(expected_versions - covered)
        unexpected = sorted(covered - expected_versions)
        if missing:
            errors.append(
                f"optional profile {profile} must cover Python "
                f"{sorted(expected_versions)}, missing {missing}"
            )
        if unexpected:
            errors.append(
                f"optional profile {profile} has unsupported Python coverage {unexpected}"
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

    jax_marker = 'if [ "${PROFILE}" = "jax-regime" ]; then'
    install_marker = 'elif [ "${PROFILE}" = "jax-regime" ]; then'
    ci_lock = "environments/jax-regime-py312/ci-requirements.lock"
    if jax_marker not in steps_block or install_marker not in steps_block:
        errors.append("optional_imports must define all jax-regime build/install/test branches")
    else:
        build_branch = steps_block.split(jax_marker, 1)[1].split("\n          else", 1)[0]
        if (
            ci_lock not in build_branch
            or "--require-hashes" not in build_branch
            or "python -m build --wheel --no-isolation" not in build_branch
        ):
            errors.append("jax-regime wheel build must use its hash-pinned CI-tool lock")
        venv_section = steps_block.split("python -m venv /tmp/feo-${PROFILE}-check", 1)[1].split(
            "WHEEL=", 1
        )[0]
        bootstrap_branch = venv_section.split(jax_marker, 1)[1].split("\n          else", 1)[0]
        if (
            ci_lock not in bootstrap_branch
            or "--require-hashes" not in bootstrap_branch
            or "pip install --upgrade" in bootstrap_branch
        ):
            errors.append("jax-regime venv bootstrap must install locked CI tools first")
        install_branch = steps_block.split(install_marker, 1)[1].split("\n          else", 1)[0]
        if ci_lock not in install_branch or "--require-hashes" not in install_branch:
            errors.append("jax-regime venv must install its hash-pinned CI-tool lock")
        jax_branch = steps_block.rsplit(jax_marker, 1)[1].split("\n          fi", 1)[0]
        test_lock = "environments/jax-regime-py312/test-requirements.lock"
        if test_lock not in jax_branch or "--require-hashes" not in jax_branch:
            errors.append("jax-regime tests must install their hash-pinned test-tool lock")
        if "pip install pytest" in jax_branch:
            errors.append("jax-regime tests must not install unpinned pytest tooling")

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

    errors: list[str] = []
    for job_name, label, lock_path in (
        (
            "supply_chain_bayesian",
            "supply_chain_bayesian",
            "environments/bayesian-jax-py312/requirements.lock",
        ),
        (
            "supply_chain_jax_regime",
            "supply_chain_jax_regime",
            "environments/jax-regime-py312/requirements.lock",
        ),
    ):
        audited = blocks.get(job_name, "")
        if "python-version: '3.12'" not in audited:
            errors.append(f"{label} must use Python 3.12")
        if lock_path not in audited:
            errors.append(f"{label} must install its hash-pinned lock")
        if job_name == "supply_chain_jax_regime":
            for tool_lock in (
                "environments/jax-regime-py312/test-requirements.lock",
                "environments/jax-regime-py312/ci-requirements.lock",
            ):
                if tool_lock not in audited:
                    errors.append(f"{label} must audit {tool_lock}")
            for floating_install in (
                "pip install --upgrade pip",
                "pip install build pip-audit cyclonedx-bom",
            ):
                if floating_install in audited:
                    errors.append(
                        f"{label} must not use floating tool install {floating_install!r}"
                    )
            build_command = "python -m build --wheel --no-isolation --outdir dist"
        else:
            build_command = "python -m build --wheel --outdir dist"
        if "--require-hashes" not in audited:
            errors.append(f"{label} must enforce lock hashes")
        if build_command not in audited:
            errors.append(f"{label} must build the release wheel reproducibly")
        if "python -m pip install --no-deps dist/finite_element_options-*.whl" not in audited:
            errors.append(f"{label} must install the release wheel before its SBOM")
        if "python -m pip_audit" not in audited or "cyclonedx-py environment" not in audited:
            errors.append(f"{label} must run vulnerability and SBOM gates")
    return errors


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
