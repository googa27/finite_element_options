"""Packaged Black--Scholes references and explicit caller-owned artifact I/O."""

from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files
from importlib.resources.abc import Traversable
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .public_fixture import build_fixture_config_hash, finalize_public_result_payload

if TYPE_CHECKING:
    from ..black_scholes_parity import FEMParityReport


FIXTURE_ROOT: Traversable = files(
    "finite_element_options.validation.evidence"
).joinpath("reference_data", "fem_bs_001")
FEM_BS_001_PROBLEM_SPEC_PATH: Traversable = FIXTURE_ROOT.joinpath("problem_spec.json")
FEM_BS_001_RESULT_EXPORT_PATH: Traversable = FIXTURE_ROOT.joinpath("result_export.json")


def artifact_destination(path: Path | str | None) -> Path:
    """Validate an explicit output file without creating directories or computing."""
    if path is None:
        raise ValueError("an explicit caller-owned output path is required")
    if not isinstance(path, (str, Path)):
        raise ValueError(
            "output path must be a filesystem path, not a package reference"
        )
    target = Path(path)
    package_root = files("finite_element_options")
    if isinstance(package_root, Path) and target.resolve().is_relative_to(
        package_root.resolve()
    ):
        raise ValueError(
            "output path must not target the installed package or its references"
        )
    if target.exists():
        if not target.is_file():
            raise ValueError("output path must identify a file")
        for reference in (FEM_BS_001_PROBLEM_SPEC_PATH, FEM_BS_001_RESULT_EXPORT_PATH):
            if isinstance(reference, Path) and target.samefile(reference):
                raise ValueError("output path must not alias a packaged reference")
    return target


def export_destinations(directory: Path | str | None) -> tuple[Path, Path]:
    """Validate both refresh destinations before any numerical work or writes."""
    if directory is None:
        raise ValueError("refresh_exports requires an explicit export_directory")
    root = Path(directory)
    if root.exists() and not root.is_dir():
        raise ValueError("export_directory must identify a directory")
    return (
        artifact_destination(root / "problem_spec.json"),
        artifact_destination(root / "result_export.json"),
    )


def write_oracle_spec(
    path: Path | str | None,
    *,
    report: FEMParityReport | None,
    result_export_uri: str,
    build_spec: Callable[..., dict[str, Any]],
) -> Path:
    """Serialize the unchanged problem-spec payload to a validated output path."""
    target = artifact_destination(path)
    if report is None:
        payload = build_spec(result_export_uri=result_export_uri)
    else:
        payload = build_spec(
            refinement_levels=report.mesh_metadata.refinement_levels,
            time_steps=report.time_metadata.time_steps,
            result_export_uri=result_export_uri,
        )
    payload["contract_id"] = build_fixture_config_hash(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def write_result_export(
    path: Path | str | None,
    *,
    refresh: bool,
    report: FEMParityReport | None,
    run_fixture: Callable[[], FEMParityReport],
) -> Path:
    """Retain existing-file behavior and write an unchanged deterministic result."""
    target = artifact_destination(path)
    if (not target.exists()) or refresh:
        if report is None:
            report = run_fixture()
        payload = report.export_payload()
        payload["config_id"] = report.config_hash
        payload = finalize_public_result_payload(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return target
