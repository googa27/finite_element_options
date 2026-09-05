"""CLI runner for canonical OpenTURNS FEM UQ pilot artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or verify the canonical OpenTURNS uncertainty artifact."""

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
        from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (  # noqa: PLC0415
            run_openturns_uq_pilot,
        )

        require_optional("openturns")
    except ImportError as exc:
        print(
            "OpenTURNS UQ pilot execution requires finite-element-options[uncertainty] "
            f"with openturns>=1.27,<2: {exc}",
            file=sys.stderr,
        )
        return 2

    started = perf_counter()
    result = run_openturns_uq_pilot(root=Path.cwd())
    runtime_seconds = perf_counter() - started
    output = Path(args.output)
    if args.verify:
        if not output.exists():
            print(f"missing artifact for verification: {output}", file=sys.stderr)
            return 3
        expected = canonical_json(result.to_dict()) + "\n"
        current = output.read_text(encoding="utf-8")
        if current != expected:
            print(
                "artifact verification failed: regenerated canonical JSON differs", file=sys.stderr
            )
            return 4
        print("artifact verification OK")
        print(f"study_input_hash={result.study_input_hash}")
        print(f"decision={result.decision['status']}")
        print(f"direct_reference_passed={result.direct_reference.passed}")
        print(f"additive_sobol_passed={result.additive_sobol_recovery.passed}")
        print(f"runtime_seconds={runtime_seconds:.6f}")
        return 0
    artifact_hash = write_atomic_json(output, result.to_dict())
    print(f"wrote {output} sha256={artifact_hash}")
    print(f"study_input_hash={result.study_input_hash}")
    print(f"decision={result.decision['status']}")
    print(f"direct_reference_passed={result.direct_reference.passed}")
    print(f"additive_sobol_passed={result.additive_sobol_recovery.passed}")
    print(f"runtime_seconds={runtime_seconds:.6f}")
    return 0 if result.decision["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
