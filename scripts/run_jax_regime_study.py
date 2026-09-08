#!/usr/bin/env python3
"""Run the isolated JAX regime study without implicitly mutating canonical evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from finite_element_options.examples.regime_switching_quanto.jax_regime.contracts import (
    EvidenceHash,
    JaxRegimeStudyConfig,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.study import (
    run_jax_regime_study,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUT = ROOT / "docs/evidence/jax_regime_study_2026-09-07.json"
CANONICAL_SIDECAR = CANONICAL_OUTPUT.with_suffix(CANONICAL_OUTPUT.suffix + ".sha256")
DEFAULT_REPLAY_OUTPUT = Path("/tmp/feo_jax_regime_study.json")
DEFAULT_SYNTHETIC_OUTPUT = Path("/tmp/feo_jax_regime_synthetic_smoke.json")
FAILED_PUBLICATION_OUTPUT = Path("/tmp/feo_jax_regime_failed_publication.json")


def _aliases_existing_protected_file(target: Path, protected: Path) -> bool:
    """Return whether two existing paths identify the same inode, failing closed on IO errors."""

    if not target.exists() or not protected.exists():
        return False
    try:
        return target.samefile(protected)
    except OSError as error:
        raise ValueError(f"cannot validate output identity: {error}") from error


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Replace one path atomically so a post-check hard-link swap cannot mutate its peer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validated_output_paths(output: Path, *, publish_canonical: bool) -> tuple[Path, Path]:
    """Return JSON/sidecar targets after protecting the complete canonical pair."""

    output = output.expanduser()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    resolved_targets = {output.resolve(), sidecar.resolve()}
    canonical_targets = {CANONICAL_OUTPUT.resolve(), CANONICAL_SIDECAR.resolve()}
    aliases_canonical = bool(resolved_targets & canonical_targets) or any(
        _aliases_existing_protected_file(target, canonical)
        for target in (output, sidecar)
        for canonical in (CANONICAL_OUTPUT, CANONICAL_SIDECAR)
    )
    if aliases_canonical and not publish_canonical:
        raise ValueError("canonical evidence requires --publish-canonical")
    if publish_canonical and resolved_targets != canonical_targets:
        raise ValueError("--publish-canonical requires the canonical JSON/sidecar output paths")
    if output.suffix.lower() != ".json":
        raise ValueError("--output must name a JSON path; the SHA-256 sidecar path is derived")
    return output, sidecar


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
        requested_output = args.output or DEFAULT_SYNTHETIC_OUTPUT
        try:
            output, sidecar = _validated_output_paths(requested_output, publish_canonical=False)
        except ValueError as error:
            parser.error(str(error))
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
        try:
            output, sidecar = _validated_output_paths(
                requested_output, publish_canonical=args.publish_canonical
            )
        except ValueError as error:
            parser.error(str(error))
        if args.publish_canonical:
            if config != JaxRegimeStudyConfig():
                parser.error("--publish-canonical requires the exact canonical configuration")
        evidence = run_jax_regime_study(args.input, config=config)
        diagnostics_passed = bool(evidence["verification"]["diagnostics_passed"])
        if args.publish_canonical and not diagnostics_passed:
            output, sidecar = _validated_output_paths(
                FAILED_PUBLICATION_OUTPUT, publish_canonical=False
            )

    serialized = _canonical(evidence)
    _atomic_write_bytes(output, serialized)
    identity = EvidenceHash(digest=sha256(serialized).hexdigest(), filename=output.name)
    _atomic_write_bytes(
        sidecar,
        f"{identity.digest}  {identity.filename}\n".encode(),
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
