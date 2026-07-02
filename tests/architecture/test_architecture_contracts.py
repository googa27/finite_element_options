"""Executable architecture gates for the FEM package namespace.

These checks keep the issue #44 packaging migration honest: the project now uses
``src/finite_element_options`` as a real package namespace while ``src`` remains
only the source-layout container.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = Path(__file__).resolve().parents[2]
SRC_LAYOUT_ROOT = ROOT / "src"
PACKAGE_ROOT = SRC_LAYOUT_ROOT / "finite_element_options"
PACKAGE = "finite_element_options"
ARCHITECTURE_DOC = ROOT / "docs" / "ARCHITECTURE.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

PACKAGE_ROOT_ENTRIES = {
    "__init__.py",
    "cli.py",
    "contracts",
    "core",
    "data_utils.py",
    "estimation",
    "examples",
    "fdsolver.py",
    "jax_greeks.py",
    "plots.py",
    "problems",
    "sidebar.py",
    "space",
    "time_integration",
    "transform.py",
    "validation",
}

FEM_CORE_PACKAGES_AND_MODULES = {
    "contracts",
    "core",
    "fdsolver.py",
    "problems",
    "space",
    "time_integration",
    "transform.py",
    "validation",
}

OUTER_LAYER_PACKAGES_AND_MODULES = {
    "cli.py",
    "data_utils.py",
    "estimation",
    "examples",
    "jax_greeks.py",
    "plots.py",
    "sidebar.py",
}

FORBIDDEN_CORE_IMPORT_PREFIXES = {
    "streamlit",
    "matplotlib",
    "plotly",
    "pandas",
    "pymc",
    "jax",
    "numba",
    "xarray",
    "pyarrow",
    "statsmodels",
    "findiff",
}

FORBIDDEN_INTERNAL_LAYER_IMPORTS = {
    "cli",
    "data_utils",
    "estimation",
    "examples",
    "jax_greeks",
    "plots",
    "sidebar",
}

# Baseline-aware exception for the pre-existing FD compatibility module. This
# allows #44 to fix distribution topology without pretending the #50 ownership
# cleanup has already happened.
KNOWN_CORE_IMPORT_EXCEPTIONS = {
    "src/finite_element_options/fdsolver.py": {
        "data_utils",
        "data_utils.snapshot",
        f"{PACKAGE}.data_utils",
        f"{PACKAGE}.data_utils.snapshot",
        "findiff",
        "findiff.FinDiff",
        "xarray",
        "xarray.DataArray",
    },
}

REQUIRED_ARCHITECTURE_PHRASES = {
    "Target package topology",
    "Dependency direction",
    "Architecture tests enforce these rules",
    "Optional capability, optional dependency",
    "No module imports the distribution as `src`",
    "Compatibility and deprecation policy",
    "Architecture fitness gates",
    "haircut-engine",
    "executable FEM adapter evidence",
    "finite_element_options",
}


def test_architecture_contract_file_is_executable_source_of_truth() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_architecture_contract.py"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        child
        for child in path.rglob("*.py")
        if "__pycache__" not in child.parts and "egg-info" not in child.parts
    )


def _package_parts(path: Path) -> list[str]:
    relative_module = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative_module.parts)
    if parts and parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _resolve_import_from_base(path: Path, node: ast.ImportFrom) -> str:
    module_parts = node.module.split(".") if node.module else []
    if node.level == 0:
        return ".".join(module_parts)

    package_parts = _package_parts(path)
    keep = max(0, len(package_parts) - node.level + 1)
    return ".".join([*package_parts[:keep], *module_parts])


def _imports_from_tree(tree: ast.AST, path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(path, node)
            if base:
                imports.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                imports.add(f"{base}.{alias.name}" if base else alias.name)
    return imports


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _imports_from_tree(tree, path)


def _top_level_import(import_name: str) -> str:
    return import_name.split(".")[0]


def _without_package_prefix(import_name: str) -> str:
    if import_name == PACKAGE:
        return ""
    return import_name.removeprefix(f"{PACKAGE}.")


def _is_forbidden_core_import(import_name: str) -> bool:
    if import_name == "src" or import_name.startswith("src."):
        return True

    if _top_level_import(import_name) in FORBIDDEN_CORE_IMPORT_PREFIXES:
        return True

    normalized = _without_package_prefix(import_name)
    return any(
        normalized == layer or normalized.startswith(f"{layer}.")
        for layer in FORBIDDEN_INTERNAL_LAYER_IMPORTS
    )


def test_architecture_document_exists_and_covers_required_transition_rules() -> None:
    assert ARCHITECTURE_DOC.is_file(), (
        "docs/ARCHITECTURE.md is the architecture source of truth."
    )
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    missing = sorted(phrase for phrase in REQUIRED_ARCHITECTURE_PHRASES if phrase not in text)
    assert not missing, (
        "docs/ARCHITECTURE.md is missing required package-transition language: "
        + ", ".join(missing)
    )


def test_source_layout_exports_only_finite_element_options_package() -> None:
    assert not (SRC_LAYOUT_ROOT / "__init__.py").exists(), (
        "The source-layout container must not itself be importable as package `src`."
    )
    assert PACKAGE_ROOT.is_dir(), "The real package root must be src/finite_element_options."
    actual = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if (path.is_file() and path.suffix == ".py")
        or (path.is_dir() and not path.name.startswith("__") and any(path.rglob("*.py")))
    }
    unexpected = actual - PACKAGE_ROOT_ENTRIES
    missing = PACKAGE_ROOT_ENTRIES - actual
    assert not unexpected, (
        "New package-root modules/packages must update docs/architecture_contract.toml "
        f"in the same PR. Unexpected entries: {sorted(unexpected)}"
    )
    assert not missing, (
        "Removed package-root modules/packages must shrink docs/architecture_contract.toml "
        f"in the same PR. Missing entries: {sorted(missing)}"
    )


def test_package_root_entries_are_classified_as_core_or_outer_layer() -> None:
    classified = FEM_CORE_PACKAGES_AND_MODULES | OUTER_LAYER_PACKAGES_AND_MODULES | {
        "__init__.py"
    }
    duplicated = FEM_CORE_PACKAGES_AND_MODULES & OUTER_LAYER_PACKAGES_AND_MODULES
    unclassified = PACKAGE_ROOT_ENTRIES - classified
    stale_classifications = classified - PACKAGE_ROOT_ENTRIES
    assert not duplicated, "Entries cannot be both FEM core and outer-layer modules: " + repr(
        sorted(duplicated)
    )
    assert not unclassified, "Every package root must be classified before it is accepted: " + repr(
        sorted(unclassified)
    )
    assert not stale_classifications, "Removed roots must also be removed from layer classifications: " + repr(
        sorted(stale_classifications)
    )


def test_import_parser_detects_package_prefixed_and_relative_application_imports() -> None:
    tree = ast.parse(
        "from finite_element_options import plots\n"
        "from finite_element_options.sidebar import render\n"
        "from finite_element_options import estimation\n"
        "from finite_element_options.jax_greeks import compute_greeks\n"
        "from ..examples import demo\n"
        "from .solver import LinearSolver\n"
        "from numba import njit\n"
        "import pyarrow\n"
        "import pandas\n"
    )
    imports = _imports_from_tree(tree, PACKAGE_ROOT / "space" / "solver.py")
    assert {
        "finite_element_options",
        "finite_element_options.plots",
        "finite_element_options.sidebar",
        "finite_element_options.sidebar.render",
        "finite_element_options.estimation",
        "finite_element_options.jax_greeks",
        "finite_element_options.jax_greeks.compute_greeks",
        "examples",
        "examples.demo",
    } <= imports
    assert all(
        _is_forbidden_core_import(name)
        for name in [
            "src",
            "src.core",
            "finite_element_options.plots",
            "finite_element_options.sidebar.render",
            "finite_element_options.estimation",
            "finite_element_options.jax_greeks.compute_greeks",
            "examples",
            "examples.demo",
            "numba.njit",
            "pyarrow",
            "pandas",
        ]
    )
    assert not _is_forbidden_core_import("space.solver")
    assert not _is_forbidden_core_import("finite_element_options.core.market")


def test_fem_core_does_not_import_application_or_research_stacks() -> None:
    violations: dict[str, list[str]] = {}
    for entry in sorted(FEM_CORE_PACKAGES_AND_MODULES):
        for path in _python_files(PACKAGE_ROOT / entry):
            relative_path = str(path.relative_to(ROOT))
            allowed = KNOWN_CORE_IMPORT_EXCEPTIONS.get(relative_path, set())
            forbidden = sorted(
                name
                for name in _imports(path)
                if _is_forbidden_core_import(name) and name not in allowed
            )
            if forbidden:
                violations[relative_path] = forbidden
    assert not violations, "FEM core imported application/research-only layers: " + repr(
        violations
    )


def test_base_package_import_surface_stays_lightweight() -> None:
    imported = _imports(PACKAGE_ROOT / "__init__.py")
    forbidden = sorted(
        name
        for name in imported
        if _is_forbidden_core_import(name) or name.startswith(f"{PACKAGE}.")
    )
    assert not forbidden, "Base package import must remain lightweight: " + repr(forbidden)


def test_ci_exposes_architecture_and_packaging_gates() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/architecture" in workflow, "CI must run the architecture gate."
    assert "tests/test_packaging_contract.py" in workflow, "CI must run packaging import gates."
