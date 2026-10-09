"""Regenerate public-synthetic Pinares references with explicit output ownership."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from finite_element_options.validation.evidence.reference_artifacts import (
    pinares_export_destinations,
)
from finite_element_options.validation.pinares_fixed_price_proxy import (
    run_public_pinares_fixed_price_proxy_fixture,
    write_public_pinares_fixed_price_problem_spec,
    write_public_pinares_fixed_price_result_export,
    write_public_pinares_provider_evidence_manifest,
    write_public_pinares_unsupported_problem_spec,
    write_public_pinares_quant_problem_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> None:
    """Export an independent bundle or deliberately publish both canonical mirrors."""
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--publish-canonical", action="store_true")
    choice.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = REPO_ROOT if args.publish_canonical else args.output_dir
    destinations = pinares_export_destinations(root)
    report = run_public_pinares_fixed_price_proxy_fixture()
    if not report.converged:
        raise SystemExit("Pinares FEM fixed-price proxy fixture failed tolerance gates")
    generated = [
        write_public_pinares_fixed_price_problem_spec(destinations[0], report=report),
        write_public_pinares_fixed_price_result_export(
            destinations[1], report=report, refresh=True
        ),
        write_public_pinares_provider_evidence_manifest(
            destinations[2], report=report, refresh=True
        ),
        write_public_pinares_unsupported_problem_spec(destinations[3], refresh=True),
        write_public_pinares_quant_problem_spec(destinations[4], report=report),
    ]
    if args.publish_canonical:
        resource_root = (
            REPO_ROOT / "src/finite_element_options/validation/evidence/reference_data"
        )
        for source in tuple(generated):
            target = resource_root / source.relative_to(REPO_ROOT / "tests/fixtures")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            generated.append(target)
    print(
        json.dumps(
            {
                "config_hash": report.config_hash,
                "generated": [
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in generated
                ],
                "converged": report.converged,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
