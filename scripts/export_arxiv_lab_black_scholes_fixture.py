"""Regenerate the public-synthetic arXiv-Lab Black--Scholes FEM fixtures.

The script writes static JSON artifacts for downstream consumers such as
arxiv-implementation-lab. Consumers should read the artifacts rather than
importing FEM internals or relying on checkout-relative solver imports.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finite_element_options.validation.black_scholes_parity import (  # noqa: E402
    PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
    run_public_black_scholes_parity_fixture,
    write_public_fem_bs_oracle_spec,
    write_public_fem_bs_result_export,
)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a generated fixture file."""

    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path, *, repo_root: Path) -> str:
    """Return a stable path string for script stdout."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate deterministic fem-bs-001 public fixture JSON."
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--publish-canonical", action="store_true",
        help="Deliberately regenerate and mirror the checkout and packaged reference snapshots.",
    )
    destination.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Caller-owned directory for generated files; leaves reference snapshots unchanged."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Regenerate checked-in public-synthetic FEM oracle fixture JSON files."""

    args = _parse_args(argv)
    repo_root = REPO_ROOT
    report = run_public_black_scholes_parity_fixture()

    if args.publish_canonical:
        fixture_root = repo_root / "tests/fixtures/fem_bs_001"
        spec_path = write_public_fem_bs_oracle_spec(
            path=fixture_root / "problem_spec.json", report=report
        )
        result_path = write_public_fem_bs_result_export(
            path=fixture_root / "result_export.json", refresh=True, report=report
        )
    else:
        fixture_root = args.output_dir / "fem_bs_001"
        spec_path = write_public_fem_bs_oracle_spec(
            path=fixture_root / "problem_spec.json",
            report=report,
            result_export_uri="result_export.json",
        )
        result_path = write_public_fem_bs_result_export(
            path=fixture_root / "result_export.json", refresh=True, report=report
        )

    generated = [spec_path, result_path]
    if args.publish_canonical:
        # This maintainer script, never the library writer, owns package snapshots.
        packaged = repo_root / "src/finite_element_options/validation/evidence/reference_data/fem_bs_001"
        packaged.mkdir(parents=True, exist_ok=True)
        for source in (spec_path, result_path):
            target = packaged / source.name
            target.write_bytes(source.read_bytes())
            generated.append(target)

    stdout_payload = {
        "benchmark_id": PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
        "config_hash": report.config_hash,
        "generated": [
            {"path": _display_path(path, repo_root=repo_root), "sha256": _file_sha256(path)}
            for path in generated
        ],
        "status": report.status,
    }
    print(json.dumps(stdout_payload, sort_keys=True))


if __name__ == "__main__":
    main()
