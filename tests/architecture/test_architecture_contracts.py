"""Executable architecture gates for the FEM backend transition.

These checks implement the M0 architecture gate from issue #57. They are
baseline-aware: the repository still exposes the transitional literal ``src``
package until #44/#50 complete, but package growth, app-layer leakage, and a
missing CI architecture gate now fail fast.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ARCHITECTURE_DOC = ROOT / "docs" / "ARCHITECTURE.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

TRANSITIONAL_SRC_PACKAGES_AND_MODULES = {
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
    "time",
    "transform.py",
    "validation",
}

FEM_CORE_PACKAGES_AND_MODULES = {
    "contracts",
    "core",
    "fdsolver.py",
    "problems",
    "space",
    "time",
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

# Baseline-aware exception for pre-existing ``fdsolver.py`` dependencies. This
# keeps the gate red for any new core->optional/research imports while avoiding a
# big-bang rewrite inside the M0 architecture-fitness slice.
KNOWN_TRANSITIONAL_CORE_IMPORT_EXCEPTIONS = {
    "src/fdsolver.py": {
        "data_utils",
        "data_utils.snapshot",
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
    "Issue #64 adds",
}


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        child
        for child in path.rglob("*.py")
        if "__pycache__" not in child.parts and "egg-info" not in child.parts
    )


def _package_parts(path: Path) -> list[str]:
    relative_module = path.relative_to(SRC_ROOT).with_suffix("")
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


def _without_src_prefix(import_name: str) -> str:
    return import_name.removeprefix("src.")


def _is_forbidden_core_import(import_name: str) -> bool:
    if import_name == "src":
        return True

    if _top_level_import(import_name) in FORBIDDEN_CORE_IMPORT_PREFIXES:
        return True

    normalized = _without_src_prefix(import_name)
    return any(
        normalized == layer or normalized.startswith(f"{layer}.")
        for layer in FORBIDDEN_INTERNAL_LAYER_IMPORTS
    )


def test_architecture_document_exists_and_covers_required_transition_rules() -> None:
    assert ARCHITECTURE_DOC.is_file(), "docs/ARCHITECTURE.md is the M0 architecture source of truth."
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    missing = sorted(phrase for phrase in REQUIRED_ARCHITECTURE_PHRASES if phrase not in text)
    assert not missing, "docs/ARCHITECTURE.md is missing required M0 transition language: " + ", ".join(missing)


def test_current_src_surface_is_declared_transition_baseline() -> None:
    actual = {
        path.name
        for path in SRC_ROOT.iterdir()
        if (path.is_file() and path.suffix == ".py" and path.name != "__init__.py")
        or (path.is_dir() and not path.name.startswith("__") and any(path.rglob("*.py")))
    }
    unexpected = actual - TRANSITIONAL_SRC_PACKAGES_AND_MODULES
    missing = TRANSITIONAL_SRC_PACKAGES_AND_MODULES - actual
    assert not unexpected, (
        "New transitional src modules/packages must be added to docs/ARCHITECTURE.md and the architecture "
        f"baseline before code lands. Unexpected entries: {sorted(unexpected)}"
    )
    assert not missing, (
        "Removed transitional src modules/packages must shrink docs/ARCHITECTURE.md and the architecture "
        f"baseline in the same PR. Missing entries: {sorted(missing)}"
    )


def test_transitional_src_entries_are_classified_as_core_or_outer_layer() -> None:
    classified = FEM_CORE_PACKAGES_AND_MODULES | OUTER_LAYER_PACKAGES_AND_MODULES
    duplicated = FEM_CORE_PACKAGES_AND_MODULES & OUTER_LAYER_PACKAGES_AND_MODULES
    unclassified = TRANSITIONAL_SRC_PACKAGES_AND_MODULES - classified
    stale_classifications = classified - TRANSITIONAL_SRC_PACKAGES_AND_MODULES
    assert not duplicated, "Entries cannot be both FEM core and outer-layer modules: " + repr(sorted(duplicated))
    assert not unclassified, "Every transitional src root must be classified before it is accepted: " + repr(sorted(unclassified))
    assert not stale_classifications, "Removed transitional roots must also be removed from layer classifications: " + repr(sorted(stale_classifications))


def test_import_parser_detects_src_prefixed_and_relative_application_imports() -> None:
    tree = ast.parse(
        "from src import plots\n"
        "from src.sidebar import render\n"
        "from src import estimation\n"
        "from src.jax_greeks import compute_greeks\n"
        "from ..examples import demo\n"
        "from .solver import LinearSolver\n"
        "from numba import njit\n"
        "import pyarrow\n"
        "import pandas\n"
    )
    imports = _imports_from_tree(tree, SRC_ROOT / "space" / "solver.py")
    assert {
        "src",
        "src.plots",
        "src.sidebar",
        "src.sidebar.render",
        "src.estimation",
        "src.jax_greeks",
        "src.jax_greeks.compute_greeks",
        "examples",
        "examples.demo",
    } <= imports
    assert all(
        _is_forbidden_core_import(name)
        for name in [
            "src",
            "src.plots",
            "src.sidebar.render",
            "src.estimation",
            "src.jax_greeks.compute_greeks",
            "examples",
            "examples.demo",
            "numba.njit",
            "pyarrow",
            "pandas",
        ]
    )
    assert not _is_forbidden_core_import("space.solver")


def test_fem_core_does_not_import_application_or_research_stacks() -> None:
    violations: dict[str, list[str]] = {}
    for entry in sorted(FEM_CORE_PACKAGES_AND_MODULES):
        for path in _python_files(SRC_ROOT / entry):
            relative_path = str(path.relative_to(ROOT))
            allowed = KNOWN_TRANSITIONAL_CORE_IMPORT_EXCEPTIONS.get(relative_path, set())
            forbidden = sorted(name for name in _imports(path) if _is_forbidden_core_import(name) and name not in allowed)
            if forbidden:
                violations[relative_path] = forbidden
    assert not violations, "FEM core imported application/research-only layers: " + repr(violations)


def test_base_package_import_surface_stays_lightweight() -> None:
    imported = _imports(SRC_ROOT / "__init__.py")
    forbidden = sorted(name for name in imported if _is_forbidden_core_import(name) or name.startswith("src."))
    assert not forbidden, "Transitional src package import must remain a lightweight shim: " + repr(forbidden)


def test_ci_exposes_architecture_gate() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/architecture" in workflow, "CI must run the architecture gate from tests/architecture."
