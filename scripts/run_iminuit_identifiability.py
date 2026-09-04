"""CLI runner for deterministic iminuit identifiability artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or verify the canonical iminuit identifiability artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="canonical JSON artifact path")
    parser.add_argument("--verify", action="store_true", help="regenerate and compare output")
    args = parser.parse_args(argv)

    try:
        from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (  # noqa: PLC0415
            canonical_json,
            write_atomic_json,
        )
        from finite_element_options.examples.regime_switching_quanto.adoption.optional import (  # noqa: PLC0415
            require_optional,
        )
        from finite_element_options.examples.regime_switching_quanto.adoption.identifiability import (  # noqa: PLC0415
            run_iminuit_identifiability_study,
        )

        require_optional("iminuit")
    except ImportError as exc:
        print(
            "iminuit identifiability execution requires "
            "finite-element-options[identifiability] with iminuit>=2.32,<3: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    result = run_iminuit_identifiability_study()
    output = Path(args.output)
    if args.verify:
        if not output.exists():
            print(f"missing artifact for verification: {output}", file=sys.stderr)
            return 3
        expected = canonical_json(result.to_dict()) + "\n"
        current = output.read_text(encoding="utf-8")
        if current != expected:
            print(
                "artifact verification failed: regenerated canonical JSON differs",
                file=sys.stderr,
            )
            return 4
        print("artifact verification OK")
        print(f"study_input_hash={result.study_input_hash}")
        print(f"all_expected_decisions_passed={result.summary['all_expected_decisions_passed']}")
        return 0
    artifact_hash = write_atomic_json(output, result.to_dict())
    print(f"wrote {output} sha256={artifact_hash}")
    print(f"study_input_hash={result.study_input_hash}")
    print(f"all_expected_decisions_passed={result.summary['all_expected_decisions_passed']}")
    return 0 if result.summary["all_expected_decisions_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
