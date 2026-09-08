"""Architecture gates for the isolated JAX regime research profile."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

import pytest

pytestmark = pytest.mark.architecture
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "finite_element_options.examples.regime_switching_quanto.jax_regime"
HEAVY = {
    "jax",
    "jaxlib",
    "numpyro",
    "dynamax",
    "diffrax",
    "tensorflow_probability",
    "pandas",
    "pymc",
    "skfem",
    "statsmodels",
}


def test_jax_regime_facade_and_contracts_import_without_optional_stack() -> None:
    code = f"""
    import builtins
    import importlib
    import sys
    blocked = {HEAVY!r}
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in blocked:
            raise ModuleNotFoundError(f'blocked optional dependency: {{name}}', name=name)
        return original(name, *args, **kwargs)
    builtins.__import__ = guarded
    for name in ({PACKAGE!r}, {f"{PACKAGE}.contracts"!r}):
        importlib.import_module(name)
    leaked = sorted(name for name in sys.modules if name.split('.')[0] in blocked)
    assert leaked == [], leaked
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_jax_regime_public_facade_exposes_typed_boundaries_not_dict_runners() -> None:
    import finite_element_options.examples.regime_switching_quanto.jax_regime as facade

    assert "run_jax_regime_study" not in facade.__all__
    assert "run_synthetic_verification" not in facade.__all__
    assert "JaxRegimeStudyConfig" in facade.__all__
    assert "PDPObservationBatch" in facade.__all__
    assert "PDPPreprocessingAudit" in facade.__all__


def test_jax_regime_contracts_are_dependency_light() -> None:
    path = (
        ROOT / "src/finite_element_options/examples/regime_switching_quanto/jax_regime/contracts.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint(HEAVY), sorted(imports & HEAVY)


def test_jax_regime_extra_is_python312_only_and_complete() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = project["optional-dependencies"]["jax-regime"]
    joined = "\n".join(requirements)
    for dependency in (
        "jax==0.11.1",
        "jaxlib==0.11.1",
        "numpyro==0.21.0",
        "dynamax==1.0.2",
        "diffrax==0.7.2",
        "fastprogress==1.0.3",
        "statsmodels==0.14.6",
        "tfp-nightly==0.26.0.dev20260907",
    ):
        assert dependency in joined
    assert all("python_version >= '3.12'" in item for item in requirements)
    assert all("python_version < '3.13'" in item for item in requirements)


def test_jax_regime_cli_requires_explicit_input_and_canonical_publication() -> None:
    runner = ROOT / "scripts/run_jax_regime_study.py"
    cases = (
        ((), "--input is required"),
        (
            (
                "--input",
                "/tmp/unread.zip",
                "--output",
                "docs/evidence/jax_regime_study_2026-09-07.json",
            ),
            "canonical evidence requires --publish-canonical",
        ),
        (
            (
                "--input",
                "/tmp/unread.zip",
                "--output",
                str(ROOT / "docs/evidence/jax_regime_study_2026-09-07.json.sha256"),
            ),
            "canonical evidence requires --publish-canonical",
        ),
        (
            ("--synthetic", "--output", "/tmp/not-json.sha256"),
            "--output must name a JSON path",
        ),
        (
            ("--input", "/tmp/unread.zip", "--publish-canonical", "--samples", "299"),
            "exact canonical configuration",
        ),
    )
    for arguments, message in cases:
        result = subprocess.run(
            [sys.executable, str(runner), *arguments],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert message in result.stderr


@pytest.mark.parametrize(
    ("output", "cwd"),
    (
        ("docs/images/jax_regime_study_2026-09-07.png", ROOT),
        (str(ROOT / "docs/images/jax_regime_study_2026-09-07.pdf"), Path("/tmp")),
    ),
)
def test_jax_regime_plot_refuses_implicit_canonical_write(output: str, cwd: Path) -> None:
    generator = ROOT / "scripts/generate_jax_regime_plot.py"
    result = subprocess.run(
        [sys.executable, str(generator), "--output", output],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "canonical visual output requires --publish-canonical" in result.stderr


def test_jax_regime_plot_rejects_non_png_output_before_rendering() -> None:
    generator = ROOT / "scripts/generate_jax_regime_plot.py"
    result = subprocess.run(
        [sys.executable, str(generator), "--output", "/tmp/not-a-png.pdf"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--output must name a PNG path" in result.stderr
