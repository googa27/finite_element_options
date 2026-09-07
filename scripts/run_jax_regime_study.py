#!/usr/bin/env python3
"""Run the isolated JAX regime study without implicitly mutating canonical evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from finite_element_options.examples.regime_switching_quanto.jax_regime.contracts import (
    EvidenceHash,
    JaxRegimeStudyConfig,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.study import (
    run_jax_regime_study,
)

CANONICAL_OUTPUT = Path("docs/evidence/jax_regime_study_2026-09-07.json")
DEFAULT_REPLAY_OUTPUT = Path("/tmp/feo_jax_regime_study.json")
DEFAULT_SYNTHETIC_OUTPUT = Path("/tmp/feo_jax_regime_synthetic_smoke.json")
FAILED_PUBLICATION_OUTPUT = Path("/tmp/feo_jax_regime_failed_publication.json")


def _canonical(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def main() -> int:
    """Execute a bounded study; reserve canonical writes for an exact explicit publish."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="run archive-independent CI smoke")
    parser.add_argument(
        "--input",
        type=Path,
        help="required content-addressed PDP archive for every non-synthetic run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path; defaults to /tmp and cannot target canonical evidence without publication",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--num-states", type=int, default=4)
    parser.add_argument("--em-iters", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--pricing-paths", type=int, default=4096)
    parser.add_argument(
        "--verify", action="store_true", help="return nonzero if fresh-run gates fail"
    )
    parser.add_argument(
        "--publish-canonical",
        action="store_true",
        help="write canonical evidence only with the exact reference configuration",
    )
    args = parser.parse_args()

    config = JaxRegimeStudyConfig(
        seed=args.seed,
        num_states=args.num_states,
        em_iters=args.em_iters,
        warmup=args.warmup,
        posterior_samples=args.samples,
        chains=args.chains,
        pricing_paths=args.pricing_paths,
    )
    if args.synthetic:
        if args.input is not None:
            parser.error("--input is not accepted with --synthetic")
        if args.publish_canonical:
            parser.error("synthetic runs cannot publish canonical evidence")
        output = args.output or DEFAULT_SYNTHETIC_OUTPUT
        from finite_element_options.examples.regime_switching_quanto.jax_regime.synthetic import (
            run_synthetic_verification,
        )

        evidence = run_synthetic_verification(config)
        diagnostics_passed = all(evidence["gates"].values())
    else:
        if args.input is None:
            parser.error("--input is required unless --synthetic is selected")
        requested_output = args.output or (
            CANONICAL_OUTPUT if args.publish_canonical else DEFAULT_REPLAY_OUTPUT
        )
        if requested_output.resolve() == CANONICAL_OUTPUT.resolve() and not args.publish_canonical:
            parser.error("canonical evidence requires --publish-canonical")
        if args.publish_canonical:
            if requested_output.resolve() != CANONICAL_OUTPUT.resolve():
                parser.error("--publish-canonical requires the canonical output path")
            if config != JaxRegimeStudyConfig():
                parser.error("--publish-canonical requires the exact canonical configuration")
        output = requested_output
        evidence = run_jax_regime_study(args.input, config=config)
        diagnostics_passed = bool(evidence["verification"]["diagnostics_passed"])
        if args.publish_canonical and not diagnostics_passed:
            output = FAILED_PUBLICATION_OUTPUT

    serialized = _canonical(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    identity = EvidenceHash(digest=sha256(serialized).hexdigest(), filename=output.name)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{identity.digest}  {identity.filename}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "canonical_publish_refused": bool(
                    args.publish_canonical and not diagnostics_passed
                ),
                "diagnostics_passed": diagnostics_passed,
                "output": str(output),
                "sha256": identity.digest,
                "status": evidence["status"],
            },
            sort_keys=True,
        )
    )
    if (args.verify or args.publish_canonical) and not diagnostics_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
