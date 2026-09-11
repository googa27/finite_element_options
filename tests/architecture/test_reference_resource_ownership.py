"""Packaged and checkout reference snapshots have one explicit maintainer owner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIRECTORY = (
    "src/finite_element_options/validation/evidence/reference_data/fem_bs_001"
)


def test_packaged_reference_snapshots_equal_checkout_mirrors() -> None:
    contract = json.loads((ROOT / "docs/ARCHITECTURE.yaml").read_text())
    ownership = contract["public_reference_resources"]
    assert ownership["resource_directory"] == RESOURCE_DIRECTORY
    assert ownership["library_writes"] == "explicit_caller_owned_paths_only"
    assert ownership["paired_result_export_uri"] == "result_export.json"
    assert ownership["configuration_hash_owner"] == (
        "src/finite_element_options/validation/evidence/public_fixture.py"
    )
    for name in ("problem_spec.json", "result_export.json"):
        assert (ROOT / RESOURCE_DIRECTORY / name).read_bytes() == (
            ROOT / "tests/fixtures/fem_bs_001" / name
        ).read_bytes()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["tool"]["setuptools"]["package-data"][
        "finite_element_options.validation.evidence"
    ] == ["reference_data/fem_bs_001/*.json"]


def _script():
    spec = importlib.util.spec_from_file_location(
        "fem_reference_maintainer",
        ROOT / "scripts/export_arxiv_lab_black_scholes_fixture.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maintainer_script_requires_explicit_output_policy_before_solve(
    monkeypatch,
) -> None:
    script = _script()

    def unexpected(*args, **kwargs):
        pytest.fail("maintainer output policy must be explicit before solving")

    monkeypatch.setattr(script, "run_public_black_scholes_parity_fixture", unexpected)
    with pytest.raises(SystemExit) as error:
        script.main([])
    assert error.value.code == 2


def test_maintainer_canonical_publish_updates_both_owned_mirrors(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    script = _script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    script.main(["--publish-canonical"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["generated"]) == 4
    for name in ("problem_spec.json", "result_export.json"):
        assert (tmp_path / RESOURCE_DIRECTORY / name).read_bytes() == (
            tmp_path / "tests/fixtures/fem_bs_001" / name
        ).read_bytes()
