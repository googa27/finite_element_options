#!/usr/bin/env python3
"""Run the real external PETSc American-VI assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finite_element_options.validation.evidence.petsc_vi import run_petsc_vi_assessment
from finite_element_options.validation.evidence.serialization import file_sha256, write_atomic_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/evidence/petsc_vi_assessment_2026-09-05.json"


def main() -> int:
    """Run the external profile and write bounded canonical evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = run_petsc_vi_assessment(root=ROOT)
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        exact = {
            "schema_version": report["schema_version"] == expected.get("schema_version"),
            "input": report["input"] == expected.get("input"),
            "petsc": report["environment"]["petsc"] == expected.get("environment", {}).get("petsc"),
            "petsc4py": report["environment"]["petsc4py"]
            == expected.get("environment", {}).get("petsc4py"),
            "decision": report["decision"]["status"] == expected.get("decision", {}).get("status"),
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
    print(f"petsc_over_psor_runtime_ratio={report['timing']['petsc_over_psor_runtime_ratio']}")
    return 0 if report["decision"]["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
