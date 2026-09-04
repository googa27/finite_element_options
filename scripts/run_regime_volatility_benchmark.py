"""CLI runner for immutable regime-switching quanto volatility benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run or verify the canonical volatility benchmark artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PDP-exported joint level CSV")
    parser.add_argument("--expected-sha256", required=True, help="expected immutable CSV SHA-256")
    parser.add_argument("--output", required=True, help="canonical JSON artifact path")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--holdout-size", type=int, default=126)
    parser.add_argument("--rolling-window", type=int, default=756)
    parser.add_argument("--refit-block", type=int, default=21)
    parser.add_argument("--arch-maxiter", type=int, default=150)
    parser.add_argument("--markov-maxiter", type=int, default=150)
    parser.add_argument("--markov-search-reps", type=int, default=2)
    parser.add_argument("--forecast-simulations", type=int, default=400)
    parser.add_argument("--changepoint-window", type=int, default=63)
    parser.add_argument("--changepoint-penalty", type=float, default=6.0)
    parser.add_argument("--verify", action="store_true", help="regenerate and compare output")
    args = parser.parse_args(argv)

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (  # noqa: PLC0415
        VolatilityBenchmarkConfig,
        canonical_json,
        file_sha256,
        run_volatility_benchmark,
        write_atomic_json,
    )

    actual_sha = file_sha256(args.input)
    if actual_sha != args.expected_sha256:
        print(
            f"hash mismatch: expected {args.expected_sha256}, got {actual_sha}",
            file=sys.stderr,
        )
        return 2

    try:
        import pandas as pd  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ImportError("CSV input requires finite-element-options[calibration].") from exc

    config = VolatilityBenchmarkConfig(
        seed=args.seed,
        holdout_size=args.holdout_size,
        rolling_window=args.rolling_window,
        refit_block=args.refit_block,
        arch_maxiter=args.arch_maxiter,
        markov_maxiter=args.markov_maxiter,
        markov_search_reps=args.markov_search_reps,
        forecast_simulations=args.forecast_simulations,
        changepoint_window=args.changepoint_window,
        changepoint_penalty=args.changepoint_penalty,
    )
    result = run_volatility_benchmark(
        pd.read_csv(args.input), input_sha256=actual_sha, config=config
    )
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
        return 0
    artifact_hash = write_atomic_json(output, result.to_dict())
    print(f"wrote {output} sha256={artifact_hash}")
    print(f"decision={result.decision.decision} selected={result.decision.selected_candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
