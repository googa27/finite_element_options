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
    FEM_BS_001_PROBLEM_SPEC_PATH,
    FEM_BS_001_RESULT_EXPORT_PATH,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for generated files. Defaults to the checked-in "
            "tests/fixtures/fem_bs_001 paths."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Regenerate checked-in public-synthetic FEM oracle fixture JSON files."""

    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    report = run_public_black_scholes_parity_fixture()

    if args.output_dir is None:
        spec_path = write_public_fem_bs_oracle_spec(
            path=FEM_BS_001_PROBLEM_SPEC_PATH, report=report
        )
        result_path = write_public_fem_bs_result_export(
            path=FEM_BS_001_RESULT_EXPORT_PATH, refresh=True, report=report
        )
    else:
        fixture_root = args.output_dir / "fem_bs_001"
        spec_path = write_public_fem_bs_oracle_spec(
            path=fixture_root / "problem_spec.json", report=report
        )
        result_path = write_public_fem_bs_result_export(
            path=fixture_root / "result_export.json", refresh=True, report=report
        )

    stdout_payload = {
        "benchmark_id": PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
        "config_hash": report.config_hash,
        "generated": [
            {
                "path": _display_path(spec_path, repo_root=repo_root),
                "sha256": _file_sha256(spec_path),
            },
            {
                "path": _display_path(result_path, repo_root=repo_root),
                "sha256": _file_sha256(result_path),
            },
        ],
        "status": report.status,
    }
    print(json.dumps(stdout_payload, sort_keys=True))


if __name__ == "__main__":
    main()
