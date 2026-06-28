#!/usr/bin/env python3
# ruff: noqa: E501, PLR0912, PLR0915, SIM102, T201, TRY003
"""Validate the repository-local architecture contract.

The contract lives in ``docs/architecture_contract.toml`` and intentionally uses
TOML so the checker can run with the Python standard library (``tomllib``) in CI.
It is a small architecture-fitness function: keep package topology and CI/docs
policy explicit, versioned, and reviewed instead of hidden in ad hoc tests.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path


DEFAULT_CONTRACT = Path("docs/architecture_contract.toml")


def _read_list(section: dict[str, object], key: str) -> list[str]:
    value = section.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a TOML array of strings")
    return list(value)


def _has_python(path: Path) -> bool:
    return any(child.suffix == ".py" and "__pycache__" not in child.parts for child in path.rglob("*.py"))


def _module_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - compileall should catch this first, message aids local use.
        raise AssertionError(f"Cannot parse {path}: {exc}") from exc

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                imports.add(node.module.split(".")[0])
    return imports


def _matches_prefix(import_name: str, prefixes: Iterable[str]) -> bool:
    return any(import_name == prefix or import_name.startswith(f"{prefix}.") for prefix in prefixes)


def validate_contract(repo_root: Path, contract_path: Path) -> list[str]:
    failures: list[str] = []
    contract = tomllib.loads(contract_path.read_text(encoding="utf-8"))

    paths = contract.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("[paths] must be a TOML table")
    package_root_value = paths.get("package_root")
    if not isinstance(package_root_value, str):
        raise TypeError("[paths].package_root is required")
    package_root = repo_root / package_root_value
    if not package_root.is_dir():
        failures.append(f"Package root does not exist: {package_root_value}")
        return failures

    topology = contract.get("topology", {})
    if not isinstance(topology, dict):
        raise TypeError("[topology] must be a TOML table")
    ignored_dirs = set(_read_list(topology, "ignore_top_level_dirs")) | {"__pycache__"}
    allowed_modules = set(_read_list(topology, "allowed_root_modules"))
    allowed_packages = set(_read_list(topology, "allowed_top_level_packages"))
    required_packages = set(_read_list(topology, "required_top_level_packages")) or allowed_packages

    root_modules = sorted(path.name for path in package_root.iterdir() if path.is_file() and path.suffix == ".py")
    unexpected_modules = sorted(set(root_modules) - allowed_modules)
    if unexpected_modules:
        failures.append(
            f"Unexpected root Python modules under {package_root_value}: {unexpected_modules}. "
            "Move implementation into an architecture package or update the contract in the same PR."
        )

    top_level_packages = sorted(
        path.name
        for path in package_root.iterdir()
        if path.is_dir() and path.name not in ignored_dirs and not path.name.endswith(".egg-info") and _has_python(path)
    )
    unexpected_packages = sorted(set(top_level_packages) - allowed_packages)
    missing_packages = sorted(required_packages - set(top_level_packages))
    if unexpected_packages:
        failures.append(
            f"Unexpected top-level packages under {package_root_value}: {unexpected_packages}. "
            "Add a reviewed boundary to docs/architecture_contract.toml or move the code."
        )
    if missing_packages:
        failures.append(
            f"Required top-level packages missing under {package_root_value}: {missing_packages}. "
            "Shrink the contract in the same PR if the boundary was intentionally removed."
        )

    max_count = topology.get("max_top_level_package_count")
    if max_count is not None:
        if not isinstance(max_count, int):
            raise TypeError("[topology].max_top_level_package_count must be an integer")
        if len(top_level_packages) > max_count:
            failures.append(
                f"Top-level package count {len(top_level_packages)} exceeds contract maximum {max_count}: "
                f"{top_level_packages}"
            )

    repo_root_policy = contract.get("repository_root", {})
    if repo_root_policy:
        if not isinstance(repo_root_policy, dict):
            raise TypeError("[repository_root] must be a TOML table")
        allowed_repo_py = set(_read_list(repo_root_policy, "allowed_python_files"))
        repo_py = sorted(path.name for path in repo_root.glob("*.py") if path.is_file())
        unexpected_repo_py = sorted(set(repo_py) - allowed_repo_py)
        if unexpected_repo_py:
            failures.append(
                f"Unexpected repository-root Python files: {unexpected_repo_py}. "
                "Use scripts/, examples/, or package entry points instead."
            )

    documentation = contract.get("documentation", {})
    if documentation:
        if not isinstance(documentation, dict):
            raise TypeError("[documentation] must be a TOML table")
        for doc_path in _read_list(documentation, "required_docs"):
            full = repo_root / doc_path
            if not full.is_file():
                failures.append(f"Required architecture documentation file missing: {doc_path}")
                continue
            text = full.read_text(encoding="utf-8")
            for fragment in _read_list(documentation, "required_fragments"):
                if fragment not in text:
                    failures.append(f"{doc_path} must mention architecture-contract fragment: {fragment!r}")

    ci = contract.get("ci", {})
    if ci:
        if not isinstance(ci, dict):
            raise TypeError("[ci] must be a TOML table")
        for workflow in _read_list(ci, "workflow_paths"):
            full = repo_root / workflow
            if not full.is_file():
                failures.append(f"Required CI workflow missing: {workflow}")
                continue
            text = full.read_text(encoding="utf-8")
            for fragment in _read_list(ci, "required_fragments"):
                if fragment not in text:
                    failures.append(f"{workflow} must contain CI architecture-contract fragment: {fragment!r}")

    import_rules = contract.get("import_rules", [])
    if import_rules:
        if not isinstance(import_rules, list):
            raise TypeError("[[import_rules]] must be an array of TOML tables")
        for index, rule in enumerate(import_rules):
            if not isinstance(rule, dict):
                raise TypeError(f"import_rules[{index}] must be a TOML table")
            source = rule.get("source")
            if not isinstance(source, str):
                raise TypeError(f"import_rules[{index}].source is required")
            forbidden = _read_list(rule, "forbidden_prefixes")
            source_path = repo_root / source
            if not source_path.exists():
                continue
            files = [source_path] if source_path.is_file() else sorted(source_path.rglob("*.py"))
            violations: dict[str, list[str]] = {}
            for path in files:
                if "__pycache__" in path.parts:
                    continue
                hits = sorted(imported for imported in _module_imports(path) if _matches_prefix(imported, forbidden))
                if hits:
                    violations[str(path.relative_to(repo_root))] = hits
            if violations:
                failures.append(f"Import rule violation for {source}: {violations}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    contract_path = args.contract if args.contract.is_absolute() else repo_root / args.contract
    if not contract_path.is_file():
        print(f"Architecture contract missing: {contract_path.relative_to(repo_root)}", file=sys.stderr)
        return 2

    failures = validate_contract(repo_root, contract_path)
    if failures:
        print("Architecture contract violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Architecture contract passed: {contract_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
