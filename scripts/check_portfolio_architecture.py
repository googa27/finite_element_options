#!/usr/bin/env python3
"""Validate the portfolio architecture contract without third-party dependencies.

`docs/ARCHITECTURE.yaml` is deliberately written in the JSON subset of YAML 1.2,
so the standard-library JSON parser is sufficient for the bootstrap gate. Repos
may use PyYAML/check-jsonschema in richer gates; this checker never attempts to
reimplement a YAML parser.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "ARCHITECTURE.yaml"
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "repository",
    "architecture",
    "source_layout",
    "limits",
    "libraries",
    "interfaces",
    "tests",
    "data",
    "governance",
    "exceptions",
}
REQUIRED_EXCEPTION_FIELDS = {
    "rule",
    "path",
    "reason",
    "owner",
    "risk",
    "accepted_ceiling",
    "refactoring_trigger",
}
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}
DEFAULT_METADATA = {
    "__init__.py",
    "README.md",
    "ARCHITECTURE.md",
    "ARCHITECTURE.yaml",
    "py.typed",
}


def ignored_name(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


def ignored_path(path: Path) -> bool:
    return any(ignored_name(part) for part in path.parts)


def load_contract() -> dict[str, Any]:
    try:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {CONTRACT.relative_to(ROOT)}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            "docs/ARCHITECTURE.yaml must remain in the JSON-compatible YAML 1.2 subset "
            f"for the dependency-free bootstrap checker: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("architecture contract root must be an object")
    return payload


def exception_map(
    contract: dict[str, Any], errors: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(contract.get("exceptions", [])):
        if not isinstance(item, dict):
            errors.append(f"exceptions[{index}] must be an object")
            continue
        missing = REQUIRED_EXCEPTION_FIELDS - set(item)
        if missing:
            errors.append(f"exceptions[{index}] missing metadata: {sorted(missing)}")
            continue
        key = (str(item["rule"]), str(item["path"]))
        if key in result:
            errors.append(f"duplicate exception for {key[0]}:{key[1]}")
        result[key] = item
    return result


def require_exception(
    exceptions: dict[tuple[str, str], dict[str, Any]],
    rule: str,
    path: str,
    actual: int,
    errors: list[str],
) -> None:
    item = exceptions.get((rule, path))
    if item is None:
        errors.append(f"{rule} violation at {path}: {actual}; no documented exception")
        return
    ceiling = item.get("accepted_ceiling")
    if not isinstance(ceiling, int):
        errors.append(f"{rule} exception at {path} must have integer accepted_ceiling")
    elif actual > ceiling:
        errors.append(f"{rule} no-growth ratchet exceeded at {path}: {actual}>{ceiling}")


def runtime_dir(path: Path) -> bool:
    try:
        return any(p.suffix == ".py" for p in path.rglob("*.py") if not ignored_path(p))
    except OSError:
        return False


def validate_source(
    contract: dict[str, Any], exceptions: dict[tuple[str, str], dict[str, Any]], errors: list[str]
) -> None:
    layout = contract["source_layout"]
    if not layout.get("python_rules_applicable", True):
        return
    max_entries = int(contract["limits"]["max_immediate_runtime_entries"])
    max_lines = int(contract["limits"]["max_python_module_lines"])
    allowed_non_python = set(layout.get("allowed_non_python_files", []))
    metadata = DEFAULT_METADATA | set(layout.get("metadata_names", []))
    roots = [ROOT / p for p in layout.get("python_source_roots", [])]
    for source_root in roots:
        rel_root = source_root.relative_to(ROOT).as_posix()
        if not source_root.is_dir():
            errors.append(f"declared Python source root is missing: {rel_root}")
            continue
        for current, dirs, files in os.walk(source_root):
            dirs[:] = sorted(d for d in dirs if not ignored_name(d) and not d.startswith("."))
            current_path = Path(current)
            rel_dir = current_path.relative_to(ROOT).as_posix()
            runtime_dirs = [d for d in dirs if runtime_dir(current_path / d)]
            runtime_files = [f for f in files if f.endswith(".py") and f != "__init__.py"]
            count = len(runtime_dirs) + len(runtime_files)
            if count > max_entries:
                require_exception(exceptions, "source_fanout", rel_dir, count, errors)
            for filename in files:
                rel = (current_path / filename).relative_to(ROOT).as_posix()
                if (
                    filename.endswith((".py", ".pyi"))
                    or filename in metadata
                    or rel in allowed_non_python
                ):
                    continue
                require_exception(exceptions, "source_entry_type", rel, 1, errors)
        for module in sorted(source_root.rglob("*.py")):
            if ignored_path(module):
                continue
            try:
                lines = len(module.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                errors.append(f"Python module is not UTF-8 text: {module.relative_to(ROOT)}")
                continue
            if lines > max_lines:
                require_exception(
                    exceptions,
                    "python_module_max_lines",
                    module.relative_to(ROOT).as_posix(),
                    lines,
                    errors,
                )


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(contract)
    if missing:
        errors.append(f"contract missing top-level keys: {sorted(missing)}")
        return errors
    exceptions = exception_map(contract, errors)
    repo = contract["repository"]
    for key in ("owner", "name", "profile", "status"):
        if not repo.get(key):
            errors.append(f"repository.{key} is required")
    if contract["limits"].get("max_immediate_runtime_entries") != 10:
        errors.append(
            "default max_immediate_runtime_entries must be 10; "
            "repo override belongs in a documented exception"
        )
    if contract["limits"].get("max_python_module_lines") != 500:
        errors.append(
            "default max_python_module_lines must be 500; "
            "repo override belongs in a documented exception"
        )
    required_docs = contract["governance"].get("required_documents", [])
    for rel in required_docs:
        path = ROOT / rel
        if not path.is_file() or not path.read_text(encoding="utf-8", errors="ignore").strip():
            errors.append(f"required document missing or empty: {rel}")
    for suite in contract["tests"].get("required_suites", []):
        path = ROOT / "tests" / suite
        if not path.is_dir():
            errors.append(f"required test suite directory missing: tests/{suite}")
    ai = contract["interfaces"].get("ai", {})
    human = contract["interfaces"].get("human", {})
    if ai.get("context_file") != "AGENTS.md":
        errors.append("interfaces.ai.context_file must be AGENTS.md")
    if not ai.get("interaction") or not ai.get("capability_discovery"):
        errors.append("AI interaction and capability discovery decisions are required")
    if not human.get("interaction") or not human.get("dunder_policy"):
        errors.append("human interaction and dunder policy decisions are required")
    if not contract["libraries"].get("selection_policy"):
        errors.append("maintained-library selection policy is required")
    if not isinstance(contract["libraries"].get("decisions"), list):
        errors.append("libraries.decisions must be a list")
    core = contract["data"].get("core_repositories", {})
    for name in ("PDP", "financial_problem_formulations", "ui_and_artifacts"):
        if name not in core:
            errors.append(f"data.core_repositories must decide {name} posture")
    validate_source(contract, exceptions, errors)
    return errors


def main() -> int:
    try:
        contract = load_contract()
    except ValueError as exc:
        print(f"architecture contract FAILED\n- {exc}")
        return 1
    errors = validate_contract(contract)
    if errors:
        print("architecture contract FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "architecture contract OK: "
        f"{contract['repository']['owner']}/{contract['repository']['name']} "
        f"profile={contract['repository']['profile']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
