"""Architecture fitness tests for the isolated Bayesian/JAX profile."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from finite_element_options.estimation.bayesian_profile import contracts


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "src/finite_element_options/estimation/bayesian_profile"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_numpyro_route_has_no_numpy_scipy_or_fem_imports() -> None:
    """JAX-native inference must stay structurally separate from the FEM stack."""

    imports = _imported_modules(PROFILE / "numpyro_smoke.py")
    banned_roots = {"numpy", "scipy", "skfem"}
    assert not {name.split(".")[0] for name in imports} & banned_roots
    assert not {
        name
        for name in imports
        if name.startswith("finite_element_options.space")
        or name.startswith("finite_element_options.time_integration")
    }
    assert "jax.numpy" in imports


def test_bayesian_engines_have_separate_adapter_modules() -> None:
    """PyMC and NumPyro must not collapse into one mixed numerical adapter."""

    pymc_imports = _imported_modules(PROFILE / "pymc_smoke.py")
    numpyro_imports = _imported_modules(PROFILE / "numpyro_smoke.py")
    assert "pymc" in pymc_imports
    assert "numpyro" not in pymc_imports
    assert "numpyro" in numpyro_imports
    assert "pymc" not in numpyro_imports


def test_bayesian_profile_runtime_guard_rejects_python_311(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata markers and runtime behavior must agree on Python 3.12 only."""

    class Python311:
        major = 3
        minor = 11

        def __getitem__(self, key: slice) -> tuple[int, ...]:
            return (self.major, self.minor)[key]

    monkeypatch.setattr(contracts.sys, "version_info", Python311())
    with pytest.raises(RuntimeError, match="require Python 3.12"):
        contracts.require_supported_python()
