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
    "acceleration.py",
    "cli.py",
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
}

FEM_CORE_PACKAGES_AND_MODULES = {
    "core",
    "fdsolver.py",
    "problems",
    "space",
    "time",
    "transform.py",
}

FORBIDDEN_CORE_IMPORT_PREFIXES = {
    "streamlit",
    "matplotlib",
    "plotly",
    "pandas",
    "pymc",
    "jax",
}

FORBIDDEN_INTERNAL_LAYER_IMPORTS = {
    "src.examples",
    "src.plots",
    "src.sidebar",
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
}


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        child
        for child in path.rglob("*.py")
        if "__pycache__" not in child.parts and "egg-info" not in child.parts
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _top_level_import(import_name: str) -> str:
    return import_name.split(".")[0]


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
    assert not unexpected, (
        "New transitional src modules/packages must be added to docs/ARCHITECTURE.md and the architecture "
        f"baseline before code lands. Unexpected entries: {sorted(unexpected)}"
    )


def test_fem_core_does_not_import_application_or_research_stacks() -> None:
    violations: dict[str, list[str]] = {}
    for entry in sorted(FEM_CORE_PACKAGES_AND_MODULES):
        for path in _python_files(SRC_ROOT / entry):
            imported = _imports(path)
            forbidden = sorted(
                name
                for name in imported
                if _top_level_import(name) in FORBIDDEN_CORE_IMPORT_PREFIXES
                or any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_INTERNAL_LAYER_IMPORTS)
            )
            if forbidden:
                violations[str(path.relative_to(ROOT))] = forbidden
    assert not violations, "FEM core imported application/research-only layers: " + repr(violations)


def test_base_package_import_surface_stays_lightweight() -> None:
    imported = _imports(SRC_ROOT / "__init__.py")
    forbidden = sorted(
        name for name in imported if _top_level_import(name) in FORBIDDEN_CORE_IMPORT_PREFIXES or name.startswith("src.")
    )
    assert not forbidden, "Transitional src package import must remain a lightweight shim: " + repr(forbidden)


def test_ci_exposes_architecture_gate() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/architecture" in workflow, "CI must run the architecture gate from tests/architecture."
