"""Architecture gates for the isolated JAX regime research profile."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import runpy
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


def test_canonical_visual_publication_requires_bound_evidence(
    tmp_path: Path,
) -> None:
    generator = ROOT / "scripts/generate_jax_regime_plot.py"
    namespace = runpy.run_path(str(generator))
    load_payload = namespace["_load_payload"]
    validate_payload = namespace["_validate_canonical_payload"]
    regime_labels = namespace["_regime_labels"]
    canonical = ROOT / "docs/evidence/jax_regime_study_2026-09-07.json"

    assert regime_labels(2) == ["Low", "High"]
    assert len(regime_labels(3)) == 3
    assert len(regime_labels(4)) == 4
    with pytest.raises(ValueError, match="unsupported visual state count"):
        regime_labels(5)

    payload = load_payload(canonical, publish_canonical=True)
    assert payload["status"] == "passed"

    wrong_config = json.loads(json.dumps(payload))
    wrong_config["config"]["seed"] += 1
    with pytest.raises(ValueError, match="exact evidence configuration"):
        validate_payload(wrong_config)

    failed_gate = json.loads(json.dumps(payload))
    failed_gate["verification"]["gates"]["synthetic_recovery"] = False
    with pytest.raises(ValueError, match="passed evidence and named gates"):
        validate_payload(failed_gate)

    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(canonical.read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--input",
            str(alternate),
            "--publish-canonical",
        ],
        cwd=Path("/tmp"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires the canonical evidence path" in result.stderr


def test_jax_regime_clis_reject_hard_links_to_every_canonical_artifact(
    tmp_path: Path,
) -> None:
    runner = ROOT / "scripts/run_jax_regime_study.py"
    generator = ROOT / "scripts/generate_jax_regime_plot.py"
    cases = (
        (
            ROOT / "docs/evidence/jax_regime_study_2026-09-07.json",
            runner,
            ".json",
            ("--synthetic",),
        ),
        (
            ROOT / "docs/evidence/jax_regime_study_2026-09-07.json.sha256",
            runner,
            ".json",
            ("--synthetic",),
        ),
        (
            ROOT / "docs/images/jax_regime_study_2026-09-07.png",
            generator,
            ".png",
            (),
        ),
        (
            ROOT / "docs/images/jax_regime_study_2026-09-07.pdf",
            generator,
            ".png",
            (),
        ),
    )
    for index, (canonical, script, suffix, prefix) in enumerate(cases):
        before = canonical.read_bytes()
        alias = tmp_path / f"alias-{index}{suffix}"
        os.link(canonical, alias)
        result = subprocess.run(
            [sys.executable, str(script), *prefix, "--output", str(alias)],
            cwd=Path("/tmp"),
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "requires --publish-canonical" in result.stderr
        assert canonical.read_bytes() == before
