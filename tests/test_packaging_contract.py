"""Packaging and installed-wheel contract tests for issue #44."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

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


def test_base_metadata_keeps_optional_stacks_out_of_core_dependencies() -> None:
    project = metadata.metadata("finite-element-options")
    requires_dist = project.get_all("Requires-Dist") or []
    requires_text = "\n".join(requires_dist)
    forbidden_core = ["jax", "fenics", "dolfin", "streamlit", "pymc", "statsmodels"]
    offenders = [
        name
        for name in forbidden_core
        if name in requires_text.lower() and "extra ==" not in requires_text.lower()
    ]
    assert not offenders, (
        f"Optional stacks leaked into core dependencies: {offenders}\n{requires_text}"
    )
    assert any(
        item.lower().startswith("aleatory") and 'extra == "ui"' in item.lower()
        for item in requires_dist
    ), "The advertised UI extra must install aleatory for sidebar imports."


def test_installed_wheel_import_contract_has_no_checkout_path_hack(tmp_path: Path) -> None:
    outdir = tmp_path / "dist"
    venv = tmp_path / "venv"
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=ROOT)
    wheel = next(outdir.glob("finite_element_options-*.whl"))

    _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=tmp_path)
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
        import finite_element_options.core.market
        import finite_element_options.core.config
        import finite_element_options.time_integration.stepper

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
