#!/usr/bin/env python3
"""Validate the portfolio architecture contract without third-party dependencies.

`docs/ARCHITECTURE.yaml` is deliberately written in the JSON subset of YAML 1.2,
so the standard-library JSON parser is sufficient for the bootstrap gate. Richer
repository gates may add PyYAML or check-jsonschema; this checker never attempts
to reimplement a YAML parser.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "ARCHITECTURE.yaml"
DEFAULT_MAX_ENTRIES = 10
DEFAULT_MAX_LINES = 500
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


class ContractLoadError(ValueError):
    """Architecture contract could not be loaded."""

    @classmethod
    def missing(cls, relative_path: Path) -> ContractLoadError:
        return cls(f"missing {relative_path}")

    @classmethod
    def invalid_json_subset(cls, error: json.JSONDecodeError) -> ContractLoadError:
        return cls(
            "docs/ARCHITECTURE.yaml must remain in the JSON-compatible YAML 1.2 "
            f"subset for the dependency-free bootstrap checker: {error}"
        )

    @classmethod
    def unreadable(cls, relative_path: Path, error: OSError) -> ContractLoadError:
        return cls(f"could not read {relative_path}: {error}")


class ContractShapeError(TypeError):
    """Architecture contract has an invalid root shape."""

    def __init__(self) -> None:
        super().__init__("architecture contract root must be an object")


def ignored_name(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


def ignored_path(path: Path) -> bool:
    return any(ignored_name(part) for part in path.parts)


def load_contract() -> dict[str, Any]:
    try:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractLoadError.missing(CONTRACT.relative_to(ROOT)) from exc
    except OSError as exc:
        raise ContractLoadError.unreadable(CONTRACT.relative_to(ROOT), exc) from exc
    except json.JSONDecodeError as exc:
        raise ContractLoadError.invalid_json_subset(exc) from exc
    if not isinstance(payload, dict):
        raise ContractShapeError
    return payload


def exception_map(
    contract: dict[str, Any], errors: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    exceptions = contract.get("exceptions", [])
    if not isinstance(exceptions, list):
        errors.append("exceptions must be a list")
        return result
    for index, item in enumerate(exceptions):
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
        errors.append(
            f"{rule} no-growth ratchet exceeded at {path}: {actual}>{ceiling}"
        )


def object_section(
    contract: dict[str, Any], key: str, errors: list[str]
) -> dict[str, Any] | None:
    section = contract.get(key)
    if not isinstance(section, dict):
        errors.append(f"{key} must be an object")
        return None
    return section


def add_error_once(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def integer_limit(
    limits: dict[str, Any], key: str, expected_default: int, errors: list[str]
) -> int | None:
    value = limits.get(key)
    if value is None:
        add_error_once(errors, f"limits.{key} is required")
        return None
    if not isinstance(value, int):
        add_error_once(errors, f"limits.{key} must be an integer")
        return None
    if value != expected_default:
        add_error_once(
            errors,
            f"default {key} must be {expected_default}; "
            "repo override belongs in a documented exception",
        )
    return value


def validate_limit_defaults(contract: dict[str, Any], errors: list[str]) -> None:
    limits = object_section(contract, "limits", errors)
    if limits is None:
        return
    integer_limit(limits, "max_immediate_runtime_entries", DEFAULT_MAX_ENTRIES, errors)
    integer_limit(limits, "max_python_module_lines", DEFAULT_MAX_LINES, errors)


def collect_source_tree(
    source_root: Path,
) -> tuple[dict[Path, tuple[list[str], list[str]]], list[Path]]:
    tree: dict[Path, tuple[list[str], list[str]]] = {}
    modules: list[Path] = []
    for current, dirs, files in os.walk(source_root):
        dirs[:] = sorted(
            name for name in dirs if not ignored_name(name) and not name.startswith(".")
        )
        current_path = Path(current)
        tree[current_path] = (list(dirs), sorted(files))
        modules.extend(
            current_path / filename for filename in files if filename.endswith(".py")
        )
    return tree, sorted(modules)


def runtime_package_dirs(source_root: Path, modules: list[Path]) -> set[Path]:
    runtime_dirs: set[Path] = set()
    for module in modules:
        if module.name == "__init__.py" or ignored_path(module):
            continue
        current = module.parent
        while current == source_root or source_root in current.parents:
            runtime_dirs.add(current)
            if current == source_root:
                break
            current = current.parent
    return runtime_dirs


def validate_source(
    contract: dict[str, Any],
    exceptions: dict[tuple[str, str], dict[str, Any]],
    errors: list[str],
) -> None:
    layout = object_section(contract, "source_layout", errors)
    limits = object_section(contract, "limits", errors)
    if layout is None or limits is None:
        return
    if not layout.get("python_rules_applicable", True):
        return
    max_entries = integer_limit(
        limits, "max_immediate_runtime_entries", DEFAULT_MAX_ENTRIES, errors
    )
    max_lines = integer_limit(
        limits, "max_python_module_lines", DEFAULT_MAX_LINES, errors
    )
    if max_entries is None or max_lines is None:
        return
    allowed_non_python = set(layout.get("allowed_non_python_files", []))
    metadata = DEFAULT_METADATA | set(layout.get("metadata_names", []))
    roots = [ROOT / path for path in layout.get("python_source_roots", [])]
    for source_root in roots:
        rel_root = source_root.relative_to(ROOT).as_posix()
        if not source_root.is_dir():
            errors.append(f"declared Python source root is missing: {rel_root}")
            continue
        source_tree, modules = collect_source_tree(source_root)
        runtime_dirs = runtime_package_dirs(source_root, modules)
        for current_path, (dirs, files) in source_tree.items():
            rel_dir = current_path.relative_to(ROOT).as_posix()
            runtime_child_dirs = [
                name for name in dirs if current_path / name in runtime_dirs
            ]
            runtime_files = [
                name for name in files if name.endswith(".py") and name != "__init__.py"
            ]
            count = len(runtime_child_dirs) + len(runtime_files)
            if count > max_entries:
                require_exception(exceptions, "source_fanout", rel_dir, count, errors)
            for filename in files:
                rel = (current_path / filename).relative_to(ROOT).as_posix()
                allowed = (
                    filename.endswith((".py", ".pyi"))
                    or filename in metadata
                    or rel in allowed_non_python
                )
                if not allowed:
                    require_exception(exceptions, "source_entry_type", rel, 1, errors)
        for module in modules:
            if ignored_path(module):
                continue
            try:
                lines = len(module.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                errors.append(
                    f"Python module is not UTF-8 text: {module.relative_to(ROOT)}"
                )
                continue
            except OSError as exc:
                errors.append(
                    f"could not read Python module {module.relative_to(ROOT)}: {exc}"
                )
                continue
            if lines > max_lines:
                require_exception(
                    exceptions,
                    "python_module_max_lines",
                    module.relative_to(ROOT).as_posix(),
                    lines,
                    errors,
                )


def validate_repository(contract: dict[str, Any], errors: list[str]) -> None:
    repository = object_section(contract, "repository", errors)
    if repository is None:
        return
    for key in ("owner", "name", "profile", "status"):
        if not repository.get(key):
            errors.append(f"repository.{key} is required")


def validate_documents_and_tests(contract: dict[str, Any], errors: list[str]) -> None:
    governance = object_section(contract, "governance", errors)
    tests = object_section(contract, "tests", errors)
    if governance is not None:
        for rel in governance.get("required_documents", []):
            path = ROOT / rel
            try:
                exists_with_content = path.is_file() and bool(
                    path.read_text(encoding="utf-8", errors="ignore").strip()
                )
            except OSError as exc:
                errors.append(f"required document could not be read: {rel}: {exc}")
                continue
            if not exists_with_content:
                errors.append(f"required document missing or empty: {rel}")
    if tests is not None:
        for suite in tests.get("required_suites", []):
            if not (ROOT / "tests" / suite).is_dir():
                errors.append(f"required test suite directory missing: tests/{suite}")


def validate_interfaces(contract: dict[str, Any], errors: list[str]) -> None:
    interfaces = object_section(contract, "interfaces", errors)
    if interfaces is None:
        return
    ai = interfaces.get("ai", {})
    human = interfaces.get("human", {})
    if not isinstance(ai, dict):
        errors.append("interfaces.ai must be an object")
        ai = {}
    if not isinstance(human, dict):
        errors.append("interfaces.human must be an object")
        human = {}
    if ai.get("context_file") != "AGENTS.md":
        errors.append("interfaces.ai.context_file must be AGENTS.md")
    if not ai.get("interaction") or not ai.get("capability_discovery"):
        errors.append("AI interaction and capability discovery decisions are required")
    if not human.get("interaction") or not human.get("dunder_policy"):
        errors.append("human interaction and dunder policy decisions are required")


def validate_libraries_and_data(contract: dict[str, Any], errors: list[str]) -> None:
    libraries = object_section(contract, "libraries", errors)
    data = object_section(contract, "data", errors)
    if libraries is not None:
        if not libraries.get("selection_policy"):
            errors.append("maintained-library selection policy is required")
        if not isinstance(libraries.get("decisions"), list):
            errors.append("libraries.decisions must be a list")
    if data is not None:
        core = data.get("core_repositories", {})
        if not isinstance(core, dict):
            errors.append("data.core_repositories must be an object")
            core = {}
        for name in ("PDP", "financial_problem_formulations", "ui_and_artifacts"):
            if name not in core:
                errors.append(f"data.core_repositories must decide {name} posture")


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(contract)
    if missing:
        errors.append(f"contract missing top-level keys: {sorted(missing)}")
    exceptions = exception_map(contract, errors)
    validate_repository(contract, errors)
    validate_limit_defaults(contract, errors)
    validate_documents_and_tests(contract, errors)
    validate_interfaces(contract, errors)
    validate_libraries_and_data(contract, errors)
    validate_source(contract, exceptions, errors)
    return errors


def write_line(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def main() -> int:
    try:
        contract = load_contract()
    except (ContractLoadError, ContractShapeError) as exc:
        write_line(f"architecture contract FAILED\n- {exc}")
        return 1
    errors = validate_contract(contract)
    if errors:
        write_line("architecture contract FAILED")
        for error in errors:
            write_line(f"- {error}")
        return 1
    repository = contract["repository"]
    write_line(
        "architecture contract OK: "
        f"{repository['owner']}/{repository['name']} profile={repository['profile']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
