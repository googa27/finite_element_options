#!/usr/bin/env python3
"""Run or semantically verify the public-synthetic pyMOR adoption benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finite_element_options.validation.evidence.serialization import write_atomic_json
from finite_element_options.validation.evidence.reduced_order import (
    PymorBlackScholesConfig,
    run_pymor_benchmark,
    verify_pymor_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "evidence" / "black_scholes_pymor_rom_2026-09-05.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rerun scientific/performance gates and compare stable identities, not timing bytes",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the benchmark and write or verify bounded canonical evidence."""

    args = _arguments()
    report = run_pymor_benchmark(config=PymorBlackScholesConfig(), root=ROOT)
    payload = report.to_dict()
    if args.verify:
        if not args.output.is_file():
            raise SystemExit(f"reference artifact not found: {args.output}")
        with args.output.open(encoding="utf-8") as handle:
            reference = json.load(handle)
        verification = verify_pymor_benchmark(reference, report)
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["passed"] else 1
    artifact_hash = write_atomic_json(args.output, payload)
    print(f"wrote {args.output} sha256={artifact_hash}")
    print(f"study_input_hash={payload['study_input_hash']}")
    print(f"decision={payload['decision']['status']}")
    print(f"median_online_speedup={payload['timing']['median_online_speedup']}")
    print(f"ten_x_amortization_solve_count={payload['timing']['ten_x_amortization_solve_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
