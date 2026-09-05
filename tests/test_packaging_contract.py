"""Packaging and installed-wheel contract tests for issue #44."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

pytestmark = pytest.mark.packaging

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_wheel_exports_namespaced_package_and_no_src_package(tmp_path: Path) -> None:
    outdir = tmp_path / "dist"
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=ROOT)
    wheels = sorted(outdir.glob("finite_element_options-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    assert any(name.startswith("finite_element_options/") for name in names)
    assert "finite_element_options/py.typed" in names
    assert not any(name == "src/__init__.py" or name.startswith("src/") for name in names)
    assert "finite_element_options/time_integration/stepper.py" in names
    assert "finite_element_options/time/stepper.py" not in names


def test_sdist_contains_profile_replay_and_evidence_contracts(tmp_path: Path) -> None:
    """Source releases must not contain README links to omitted verification assets."""

    outdir = tmp_path / "dist"
    _run([sys.executable, "-m", "build", "--sdist", "--outdir", str(outdir)], cwd=ROOT)
    sdist = next(outdir.glob("finite_element_options-*.tar.gz"))
    required = {
        ".github/workflows/ci.yml",
        "MANIFEST.in",
        "docs/BAYESIAN_JAX_PROFILE.md",
        "docs/architecture_contract.toml",
        "docs/evidence/bayesian_jax_profile_2026-09-05.json",
        "environments/bayesian-py312/requirements.lock",
        "environments/bayesian-jax-py312/requirements.lock",
        "external_tests/bayesian_profile/test_pymc_profile.py",
        "external_tests/bayesian_profile/test_numpyro_profile.py",
        "scripts/run_bayesian_jax_profile.py",
        "tests/validation/test_bayesian_jax_profile_evidence.py",
    }
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    assert required <= members


def test_wheel_registers_exactly_one_canonical_haircut_backend_entry_point(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "dist"
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=ROOT)
    wheel = next(outdir.glob("finite_element_options-*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        entry_point_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_point_files) == 1
        entry_points = archive.read(entry_point_files[0]).decode("utf-8")

    assert "[haircut.solver_backends]" in entry_points
    assert "[console_scripts]" in entry_points
    assert "fem-options = finite_element_options.cli:main" in entry_points
    assert entry_points.count("finite_element_options =") == 1
    assert (
        "finite_element_options = finite_element_options.integrations.haircut_backend:create_backend"
        in entry_points
    )
    assert "haircut_engine.solver_backends" not in entry_points


def _requires_dist() -> list[str]:
    return metadata.metadata("finite-element-options").get_all("Requires-Dist") or []


def _has_extra_dependency(requires_dist: list[str], extra: str, dependency: str) -> bool:
    expected = canonicalize_name(dependency)
    for item in requires_dist:
        requirement = Requirement(item)
        applies = requirement.marker is None or requirement.marker.evaluate(
            {"extra": extra, "python_version": "3.12"}
        )
        if canonicalize_name(requirement.name) == expected and applies:
            return True
    return False


def test_base_metadata_keeps_optional_stacks_out_of_core_dependencies() -> None:
    requires_dist = _requires_dist()
    forbidden_core = [
        "aleatory",
        "arch",
        "arviz",
        "findiff",
        "iminuit",
        "jax",
        "fenics",
        "dolfin",
        "matplotlib",
        "pandas",
        "pymc",
        "pymor",
        "petsc4py",
        "numpyro",
        "quantlib",
        "ruptures",
        "statsmodels",
        "streamlit",
        "xarray",
    ]
    forbidden_names = {canonicalize_name(name) for name in forbidden_core}
    offenders = []
    for item in requires_dist:
        requirement = Requirement(item)
        applies_to_base = requirement.marker is None or requirement.marker.evaluate({"extra": ""})
        if canonicalize_name(requirement.name) in forbidden_names and applies_to_base:
            offenders.append(item)
    assert not offenders, (
        f"Optional stacks leaked into core dependencies: {offenders}\n" + "\n".join(requires_dist)
    )


def test_advertised_extras_cover_eager_import_dependencies() -> None:
    requires_dist = _requires_dist()
    assert _has_extra_dependency(requires_dist, "fd", "pandas"), (
        "The advertised FD extra must install pandas because fdsolver imports "
        "data_utils.snapshot at module import time."
    )
    assert _has_extra_dependency(requires_dist, "viz", "streamlit"), (
        "The advertised viz extra must install streamlit because plots imports "
        "streamlit at module import time."
    )
    assert _has_extra_dependency(requires_dist, "ui", "streamlit"), (
        "The advertised UI extra must install streamlit because sidebar imports "
        "it lazily when widgets are constructed."
    )
    assert not _has_extra_dependency(requires_dist, "ui", "aleatory"), (
        "The UI domain policy should not depend on the auxiliary aleatory package."
    )


def test_purpose_specific_adoption_extras_are_advertised() -> None:
    """Issue #130 optional adoption libraries stay purpose-specific extras."""

    requires_dist = _requires_dist()
    expected = {
        "volatility": "arch",
        "changepoints": "ruptures",
        "quantlib": "QuantLib",
        "identifiability": "iminuit",
        "uncertainty": "openturns",
        "reduction": "pymor",
        "calibration": "statsmodels",
        "bayesian": "pymc",
        "bayesian-jax": "numpyro",
    }
    for extra, dependency in expected.items():
        assert _has_extra_dependency(requires_dist, extra, dependency), (
            f"The {extra!r} extra must install {dependency!r}; requires-dist was:\n"
            + "\n".join(requires_dist)
        )
    assert not _has_extra_dependency(requires_dist, "calibration", "pymc")
    assert not _has_extra_dependency(requires_dist, "calibration", "arviz")
    assert _has_extra_dependency(requires_dist, "bayesian", "arviz")
    for dependency in ("arviz", "jax", "pymc", "numpyro"):
        assert _has_extra_dependency(requires_dist, "bayesian-jax", dependency)
    assert not _has_extra_dependency(requires_dist, "petsc", "petsc4py"), (
        "PETSc must remain an explicit matched external environment, not a portable extra"
    )
    provides_extras = metadata.metadata("finite-element-options").get_all("Provides-Extra") or []
    assert "petsc" not in {extra.lower() for extra in provides_extras}


def test_bayesian_split_has_breaking_version_and_migration_evidence() -> None:
    """Removing PyMC from calibration must be an explicit versioned migration."""

    assert metadata.version("finite-element-options") == "0.2.0"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Breaking in 0.2.0" in changelog
    assert "finite-element-options[bayesian]" in changelog


def test_bayesian_extras_are_bounded_to_python_312() -> None:
    """Bayesian/JAX extras must fail closed outside the evidenced Python minor."""

    requirements = [Requirement(item) for item in _requires_dist()]
    for extra, names in {
        "bayesian": {"arviz", "pymc"},
        "bayesian-jax": {"arviz", "jax", "numpyro", "pymc"},
    }.items():
        canonical_names = {canonicalize_name(name) for name in names}
        selected = [
            requirement
            for requirement in requirements
            if canonicalize_name(requirement.name) in canonical_names
            and requirement.marker is not None
            and requirement.marker.evaluate({"extra": extra, "python_version": "3.12"})
        ]
        assert {canonicalize_name(item.name) for item in selected} == canonical_names
        for requirement in selected:
            assert requirement.marker is not None
            assert not requirement.marker.evaluate({"extra": extra, "python_version": "3.11"})
            assert not requirement.marker.evaluate({"extra": extra, "python_version": "3.13"})


def test_installed_wheel_import_contract_has_no_checkout_path_hack(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "dist"
    venv = tmp_path / "venv"
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=ROOT)
    wheel = next(outdir.glob("finite_element_options-*.whl"))

    _run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        cwd=tmp_path,
    )
    python = venv / "bin" / "python"
    _run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], cwd=tmp_path)

    env = {**os.environ, "PYTHONPATH": ""}
    code = textwrap.dedent(
        """
        import importlib.metadata as md
        import importlib.util
        import pathlib
        import sys

        assert pathlib.Path.cwd().name != 'finite_element_options'
        import finite_element_options

        assert importlib.util.find_spec('finite_element_options.core.market') is not None
        assert importlib.util.find_spec('finite_element_options.core.config') is not None

        dist = md.distribution('finite-element-options')
        files = {str(item) for item in (dist.files or [])}
        assert any(item.startswith('finite_element_options/') for item in files)
        assert not any(item == 'src/__init__.py' or item.startswith('src/') for item in files)
        assert importlib.util.find_spec('src') is None
        assert finite_element_options.__name__ == 'finite_element_options'
        print('installed wheel import contract OK')
        """
    )
    _run([str(python), "-c", code], cwd=tmp_path, env=env)


def test_installed_wheel_base_imports_do_not_load_adoption_optional_dependencies(
    tmp_path: Path,
) -> None:
    """The base installed wheel must not import adoption-only optional libraries."""

    outdir = tmp_path / "dist"
    venv = tmp_path / "base-wheel-venv"
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=ROOT)
    wheel = next(outdir.glob("finite_element_options-*.whl"))

    _run([sys.executable, "-m", "venv", str(venv)], cwd=tmp_path)
    python = venv / "bin" / "python"
    _run([str(python), "-m", "pip", "install", str(wheel)], cwd=tmp_path)

    env = {**os.environ, "PYTHONPATH": ""}
    code = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import pathlib
        import sys

        blocked = {"arch", "arviz", "jax", "pandas", "pymc", "ruptures", "QuantLib", "iminuit", "openturns", "pymor", "petsc4py", "numpyro", "statsmodels"}
        checkout = pathlib.Path({str(ROOT)!r}).resolve()

        preloaded = sorted(name for name in sys.modules if name.split('.')[0] in blocked)
        assert preloaded == [], preloaded

        class BlockAdoptionOptionals(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError(
                        f'blocked adoption optional dependency: {{fullname}}',
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockAdoptionOptionals())
        for module_name in (
            'finite_element_options',
            'finite_element_options.examples.regime_switching_quanto',
            'finite_element_options.examples.regime_switching_quanto.adoption',
            'finite_element_options.examples.regime_switching_quanto.adoption.optional',
            'finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle.state',
            'finite_element_options.validation.evidence.reduced_order',
            'finite_element_options.validation.evidence.reduced_order.pymor_adapter',
            'finite_element_options.validation.evidence.petsc_vi',
            'finite_element_options.validation.evidence.petsc_vi.adapter',
            'finite_element_options.estimation.bayesian_profile',
            'finite_element_options.estimation.bayesian_profile.pymc_smoke',
            'finite_element_options.estimation.bayesian_profile.numpyro_smoke',
        ):
            module = importlib.import_module(module_name)
            module_file = pathlib.Path(module.__file__).resolve()
            assert checkout not in module_file.parents, module_file

        leaked = sorted(name for name in sys.modules if name.split('.')[0] in blocked)
        assert leaked == [], leaked
        print('installed wheel blocked adoption optionals OK')
        """
    )
    _run([str(python), "-c", code], cwd=tmp_path, env=env)
