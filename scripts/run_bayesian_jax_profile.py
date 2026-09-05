#!/usr/bin/env python3
"""Run or semantically verify the isolated Bayesian/JAX profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finite_element_options.estimation.bayesian_profile import (
    run_bayesian_jax_profile,
    stable_environment_checks,
)
from finite_element_options.validation.evidence.serialization import file_sha256, write_atomic_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/evidence/bayesian_jax_profile_2026-09-05.json"


def main() -> int:
    """Run both posterior engines and write bounded public-synthetic evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = run_bayesian_jax_profile(root=ROOT)
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        exact = {
            "schema_version": report["schema_version"] == expected.get("schema_version"),
            "input_hash": report["input_hash"] == expected.get("input_hash"),
            "lock_hash": report["environment_lock"]["sha256"]
            == expected.get("environment_lock", {}).get("sha256"),
            "decision": report["decision"]["status"] == expected.get("decision", {}).get("status"),
            **stable_environment_checks(report["environment"], expected.get("environment", {})),
        }
        gates = report["decision"]["checks"]
        result = {
            "mode": "semantic_gate_replay",
            "reference_sha256": file_sha256(args.output),
            "passed": all(exact.values()) and all(gates.values()),
            "exact": exact,
            "gates": gates,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 1
    write_atomic_json(args.output, report)
    print(f"wrote {args.output} sha256={file_sha256(args.output)}")
    print(f"decision={report['decision']['status']}")
    return 0 if report["decision"]["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
