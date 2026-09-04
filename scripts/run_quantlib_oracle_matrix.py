"""CLI runner for deterministic QuantLib vanilla/quanto oracle matrix artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or verify the canonical QuantLib oracle matrix artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="canonical JSON artifact path")
    parser.add_argument("--verify", action="store_true", help="regenerate and compare output")
    args = parser.parse_args(argv)

    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (  # noqa: PLC0415
        run_quantlib_oracle_matrix,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (  # noqa: PLC0415
        canonical_json,
        write_atomic_json,
    )

    result = run_quantlib_oracle_matrix()
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
        print(f"matrix_spec_hash={result.matrix_spec_hash}")
        print(f"all_passed={result.summary['all_passed']}")
        return 0
    artifact_hash = write_atomic_json(output, result.to_dict())
    print(f"wrote {output} sha256={artifact_hash}")
    print(f"matrix_spec_hash={result.matrix_spec_hash}")
    print(f"all_passed={result.summary['all_passed']}")
    return 0 if result.summary["all_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
