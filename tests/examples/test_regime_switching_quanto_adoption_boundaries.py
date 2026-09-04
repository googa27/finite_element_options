"""Optional dependency boundary tests for regime-switching quanto adoption."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADOPTION = "finite_element_options.examples.regime_switching_quanto.adoption"
OPTIONAL = f"{ADOPTION}.optional"
QUANTLIB_STATE = f"{ADOPTION}.quantlib_state"
OPTIONAL_DEPENDENCIES = {
    "arch": ("volatility", "arch>=8,<9"),
    "ruptures": ("changepoints", "ruptures>=1.1.10,<2"),
    "QuantLib": ("quantlib", "QuantLib>=1.43,<2"),
    "iminuit": ("identifiability", "iminuit>=2.32,<3"),
}


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_base_and_adoption_modules_import_with_adoption_dependencies_blocked() -> None:
    """Base imports and adoption modules must not eagerly import new libraries."""

    probe = _run_import_probe(
        f"""
        import builtins
        import importlib
        import sys

        blocked = {set(OPTIONAL_DEPENDENCIES)!r}
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in blocked:
                raise ModuleNotFoundError(f'blocked optional dependency: {{name}}', name=name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        for module_name in (
            'finite_element_options',
            'finite_element_options.examples.regime_switching_quanto',
            {ADOPTION!r},
            {OPTIONAL!r},
            {QUANTLIB_STATE!r},
        ):
            importlib.import_module(module_name)
        leaked = sorted(name for name in sys.modules if name.split('.')[0] in blocked)
        assert leaked == [], leaked
        """
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.parametrize("module_name,expected", OPTIONAL_DEPENDENCIES.items())
def test_missing_extra_errors_are_actionable(module_name: str, expected: tuple[str, str]) -> None:
    """Each optional dependency error must name the exact extra to install."""

    extra, _dependency = expected
    probe = _run_import_probe(
        f"""
        import importlib.abc
        import sys
        from {OPTIONAL} import require_optional

        target = {module_name!r}

        class BlockedOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target_spec=None):
                if fullname == target or fullname.startswith(target + '.'):
                    raise ModuleNotFoundError(
                        f'blocked optional dependency: {{fullname}}', name=target
                    )
                return None

        sys.modules.pop(target, None)
        sys.meta_path.insert(0, BlockedOptional())
        try:
            require_optional(target)
        except ImportError as exc:
            message = str(exc)
            assert target in message, message
            assert 'finite-element-options[{extra}]' in message, message
        else:
            raise AssertionError('expected actionable missing-extra ImportError')
        """
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.parametrize(
    "module_name,extra",
    [("arch", "volatility"), ("QuantLib", "quantlib")],
)
def test_exact_missing_top_level_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch, module_name: str, extra: str
) -> None:
    """Only an exact missing optional top-level module gets an install hint."""

    from finite_element_options.examples.regime_switching_quanto.adoption import optional

    def missing_top_level_import(name: str) -> ModuleType:
        raise ModuleNotFoundError(f"simulated missing optional dependency: {name}", name=name)

    monkeypatch.setattr(optional, "import_module", missing_top_level_import)

    with pytest.raises(ImportError) as exc_info:
        optional.require_optional(module_name)

    message = str(exc_info.value)
    assert module_name in message
    assert f"finite-element-options[{extra}]" in message
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert exc_info.value.__cause__.name == module_name


@pytest.mark.parametrize(
    "module_name,missing_name",
    [("arch", "arch._cext"), ("QuantLib", "QuantLib._QuantLib")],
)
def test_same_top_level_missing_submodule_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch, module_name: str, missing_name: str
) -> None:
    """Missing extensions under installed packages are not missing-extra errors."""

    from finite_element_options.examples.regime_switching_quanto.adoption import optional

    original = ModuleNotFoundError(
        f"simulated missing imported extension: {missing_name}", name=missing_name
    )

    def missing_submodule_import(name: str) -> ModuleType:
        assert name == module_name
        raise original

    monkeypatch.setattr(optional, "import_module", missing_submodule_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        optional.require_optional(module_name)

    assert exc_info.value is original


def test_registry_is_json_safe_immutable_and_contracts_do_not_expose_quantlib_types() -> None:
    """The public adoption registry and result/domain contracts expose plain data only."""

    probe = _run_import_probe(
        f"""
        import dataclasses
        import json
        from {OPTIONAL} import OPTIONAL_DEPENDENCIES, optional_dependency_registry
        from finite_element_options.examples.regime_switching_quanto import (
            ContractSpec,
            FEMGridSpec,
            FEMPriceResult,
            TwoFactorRegimeModel,
        )

        registry = optional_dependency_registry()
        assert registry == OPTIONAL_DEPENDENCIES
        assert registry is not OPTIONAL_DEPENDENCIES
        json.dumps([dataclasses.asdict(item) for item in registry], sort_keys=True)
        assert all(item.maturity in {{'boundary_only', 'experimental'}} for item in registry)
        assert all(type(dataclasses.asdict(item)[key]) is str for item in registry for key in dataclasses.asdict(item))
        try:
            registry[0].extra = 'mutated'
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError('optional dependency registry entries must be immutable')
        for contract in (ContractSpec, FEMGridSpec, FEMPriceResult, TwoFactorRegimeModel):
            assert 'QuantLib' not in repr(contract), contract
            annotations = getattr(contract, '__annotations__', {{}})
            assert 'QuantLib' not in repr(annotations), annotations
        """
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_advertised_install_hints_map_to_real_pyproject_extras() -> None:
    """Every registry install hint must resolve to the pyproject extra dependency."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    from finite_element_options.examples.regime_switching_quanto.adoption.optional import (
        optional_dependency_registry,
    )

    for item in optional_dependency_registry():
        assert item.extra in optional_deps
        assert item.install_hint == f"finite-element-options[{item.extra}]"
        assert OPTIONAL_DEPENDENCIES[item.module_name] == (item.extra, item.dependency)
        assert item.dependency in optional_deps[item.extra]


def test_adoption_facade_is_lazy() -> None:
    """Importing the adoption facade must not import implementation submodules."""

    probe = _run_import_probe(
        f"""
        import importlib
        import sys

        facade = importlib.import_module({ADOPTION!r})
        assert {OPTIONAL!r} not in sys.modules
        assert {QUANTLIB_STATE!r} not in sys.modules
        assert 'OPTIONAL_DEPENDENCIES' in dir(facade)
        _ = facade.OPTIONAL_DEPENDENCIES
        assert {OPTIONAL!r} in sys.modules
        assert {QUANTLIB_STATE!r} not in sys.modules
        _ = facade.quantlib_evaluation_date
        assert {QUANTLIB_STATE!r} in sys.modules
        """
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_optional_profile_import_smoke() -> None:
    """The adoption registry and QuantLib adapter modules are importable without eager deps."""

    probe = _run_import_probe(
        f"""
        import importlib
        for module_name in ({ADOPTION!r}, {OPTIONAL!r}, {QUANTLIB_STATE!r}):
            module = importlib.import_module(module_name)
            print('optional adoption import OK:', module.__name__)
        """
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "optional adoption import OK" in probe.stdout


def test_quantlib_evaluation_date_restores_on_success_and_failure() -> None:
    """The QuantLib state adapter must serialize mutation and always restore."""

    QuantLib = pytest.importorskip("QuantLib")

    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_state import (
        quantlib_evaluation_date,
    )

    settings = QuantLib.Settings.instance()
    original = settings.evaluationDate
    first = QuantLib.Date(3, QuantLib.January, 2026)
    second = QuantLib.Date(4, QuantLib.January, 2026)

    with quantlib_evaluation_date(first) as ql:
        assert ql.__name__ == "QuantLib"
        assert settings.evaluationDate == first
    assert settings.evaluationDate == original

    with pytest.raises(RuntimeError, match="forced QuantLib failure"):
        with quantlib_evaluation_date(second):
            assert settings.evaluationDate == second
            raise RuntimeError("forced QuantLib failure")
    assert settings.evaluationDate == original
