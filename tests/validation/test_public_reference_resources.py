"""Read-only packaged references and explicit caller-owned export destinations."""

from __future__ import annotations

from importlib.resources import files
import json
import os
from pathlib import Path
import pickle

import pytest

from finite_element_options.validation import black_scholes_parity as parity


@pytest.mark.parametrize("filename", ["problem_spec.json", "result_export.json"])
def test_reference_bytes_are_available_as_package_data(filename: str) -> None:
    resource = files("finite_element_options.validation.evidence").joinpath(
        "reference_data", "fem_bs_001", filename
    )
    public = (
        parity.FEM_BS_001_PROBLEM_SPEC_PATH
        if filename == "problem_spec.json"
        else parity.FEM_BS_001_RESULT_EXPORT_PATH
    )
    assert resource.is_file()
    assert resource.read_bytes() == public.read_bytes()


@pytest.mark.parametrize(
    "writer", ["write_public_fem_bs_oracle_spec", "write_public_fem_bs_result_export"]
)
def test_missing_export_path_refuses_before_build_or_solve(
    writer: str, monkeypatch
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("output destination must be checked before generation")

    monkeypatch.setattr(parity, "build_public_fem_bs_oracle_problem_spec", unexpected)
    monkeypatch.setattr(parity, "run_public_black_scholes_parity_fixture", unexpected)
    with pytest.raises(ValueError, match="explicit.*path"):
        getattr(parity, writer)()


def test_refresh_requires_directory_before_numerical_work(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("output directory must be checked before numerical work")

    monkeypatch.setattr(parity, "_run_row", unexpected)
    with pytest.raises(ValueError, match="export_directory"):
        parity.run_public_black_scholes_parity_fixture(refresh_exports=True)


def test_explicit_refresh_directory_contains_current_report(tmp_path: Path) -> None:
    report = parity.run_public_black_scholes_parity_fixture(
        refinement_levels=(4, 5),
        time_steps=40,
        refresh_exports=True,
        export_directory=tmp_path,
    )
    result = json.loads((tmp_path / "result_export.json").read_text())
    spec = json.loads((tmp_path / "problem_spec.json").read_text())
    assert result == report.export_payload()
    assert spec["mesh_metadata"]["mesh_refinement_levels"] == [4, 5]
    assert spec["mesh_metadata"]["default_time_steps"] == 40


@pytest.mark.parametrize("filename", ["problem_spec.json", "result_export.json"])
def test_library_refuses_packaged_reference_destination(
    filename: str, monkeypatch
) -> None:
    public = (
        parity.FEM_BS_001_PROBLEM_SPEC_PATH
        if filename == "problem_spec.json"
        else parity.FEM_BS_001_RESULT_EXPORT_PATH
    )
    writer = (
        parity.write_public_fem_bs_oracle_spec
        if filename == "problem_spec.json"
        else parity.write_public_fem_bs_result_export
    )

    def unexpected(*args, **kwargs):
        pytest.fail("reference destination must be checked before generation")

    before = public.read_bytes()
    monkeypatch.setattr(parity, "build_public_fem_bs_oracle_problem_spec", unexpected)
    monkeypatch.setattr(parity, "run_public_black_scholes_parity_fixture", unexpected)
    with pytest.raises(ValueError, match="reference|package"):
        writer(public)
    assert public.read_bytes() == before


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_reference_alias_refuses_without_changing_resource(
    tmp_path: Path, alias_kind: str
) -> None:
    resource = parity.FEM_BS_001_PROBLEM_SPEC_PATH
    before = resource.read_bytes()
    target = tmp_path / "alias.json"
    if alias_kind == "symlink":
        target.symlink_to(resource)
    else:
        os.link(resource, target)
    with pytest.raises(ValueError, match="reference|package"):
        parity.write_public_fem_bs_oracle_spec(target)
    assert resource.read_bytes() == before


def test_refresh_validates_both_files_before_computing(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "result_export.json"
    target.symlink_to(parity.FEM_BS_001_RESULT_EXPORT_PATH)

    def unexpected(*args, **kwargs):
        pytest.fail("both output files must be checked before numerical work")

    monkeypatch.setattr(parity, "_run_row", unexpected)
    with pytest.raises(ValueError, match="reference|package"):
        parity.run_public_black_scholes_parity_fixture(
            refresh_exports=True, export_directory=tmp_path
        )
    assert not (tmp_path / "problem_spec.json").exists()


def test_existing_result_is_retained_without_refresh_or_numerical_work(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "caller.json"
    target.write_bytes(b"caller-owned existing result\n")

    def unexpected(*args, **kwargs):
        pytest.fail("an existing result is retained unless refresh is explicit")

    monkeypatch.setattr(parity, "run_public_black_scholes_parity_fixture", unexpected)
    assert parity.write_public_fem_bs_result_export(target) == target
    assert target.read_bytes() == b"caller-owned existing result\n"


def test_explicit_exports_are_deterministic_and_report_identity_is_preserved(
    tmp_path: Path,
) -> None:
    report = parity.run_public_black_scholes_parity_fixture()
    assert type(pickle.loads(pickle.dumps(report))) is parity.FEMParityReport
    assert (
        type(report).__module__
        == "finite_element_options.validation.black_scholes_parity"
    )
    assert (
        parity.write_public_fem_bs_result_export.__module__ == type(report).__module__
    )
    for name, writer in [
        ("problem_spec.json", parity.write_public_fem_bs_oracle_spec),
        ("result_export.json", parity.write_public_fem_bs_result_export),
    ]:
        first, second = tmp_path / "first" / name, tmp_path / "second" / name
        assert writer(first, report=report) == first
        assert writer(str(second), report=report) == second
        assert first.read_bytes() == second.read_bytes()


def test_unused_export_directory_is_not_silently_accepted(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("invalid export policy must fail before numerical work")

    monkeypatch.setattr(parity, "_run_row", unexpected)
    with pytest.raises(ValueError, match="refresh_exports"):
        parity.run_public_black_scholes_parity_fixture(export_directory=tmp_path)
