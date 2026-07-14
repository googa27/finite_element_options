"""Command line interface for FEM option pricing and public fixture routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .core.dynamics_heston import DynamicsParametersHeston
from .core.market import Market
from .core.vanilla_bs import EuropeanOptionBs
from .space.mesh import create_mesh
from .space.solver import SpaceSolver
from .time_integration.stepper import ThetaScheme
from .validation.compiled_weak_form_adapter import (
    CompiledWeakFormUnsupportedError,
    evidence_for_result,
    load_compiled_weak_form_json,
    screen_compiled_weak_form,
    solve_compiled_weak_form,
)


def main(args: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run the requested public route."""

    parser = _build_parser()
    ns = parser.parse_args(args=args)
    if ns.command == "qps":
        return _run_qps(ns)
    return _run_legacy_heston(ns)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finite Element Options CLI")
    subparsers = parser.add_subparsers(dest="command")

    qps = subparsers.add_parser("qps", help="Screen/solve public Quant PDE fixtures")
    qps_sub = qps.add_subparsers(dest="qps_command", required=True)
    screen = qps_sub.add_parser(
        "screen", help="Fail-closed screen a compiled weak-form fixture"
    )
    screen.add_argument("payload", help="Path to compiled weak-form JSON")
    screen.add_argument("--json", action="store_true", help="Emit deterministic JSON")
    solve = qps_sub.add_parser(
        "solve", help="Solve an accepted compiled weak-form fixture"
    )
    solve.add_argument("payload", help="Path to compiled weak-form JSON")
    solve.add_argument("--out", required=True, help="Result JSON path")
    solve.add_argument("--evidence", required=True, help="Evidence JSON path")

    parser.add_argument("--k", type=float, default=0.4, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Maturity")
    parser.add_argument("--r", type=float, default=0.03, help="Risk free rate")
    parser.add_argument("--q", type=float, default=0.03, help="Dividend yield")
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--s-max", type=float, default=1.0, dest="s_max")
    parser.add_argument("--v-max", type=float, default=1.0, dest="v_max")
    parser.add_argument("--nt", type=int, default=10)
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--call", action="store_true", help="Price a call option")
    parser.add_argument("--american", action="store_true")
    return parser


def _run_qps(ns: argparse.Namespace) -> int:
    payload = load_compiled_weak_form_json(ns.payload)
    if ns.qps_command == "screen":
        screen = screen_compiled_weak_form(payload)
        if ns.json:
            print(json.dumps(screen.to_public_dict(), sort_keys=True))
        else:
            print("accepted" if screen.accepted else "rejected")
        return 0 if screen.accepted else 2
    if ns.qps_command == "solve":
        try:
            result = solve_compiled_weak_form(payload)
        except CompiledWeakFormUnsupportedError as exc:
            print(json.dumps(exc.screen.to_public_dict(), sort_keys=True))
            return 2
        _write_json(ns.out, result)
        _write_json(ns.evidence, evidence_for_result(result))
        print(
            json.dumps(
                {"status": result["status"], "out": ns.out, "evidence": ns.evidence},
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unknown qps command: {ns.qps_command}")


def _run_legacy_heston(ns: argparse.Namespace) -> int:
    """Run the legacy Heston demonstration path kept for CLI compatibility."""

    dh = DynamicsParametersHeston(
        r=ns.r, q=ns.q, kappa=ns.kappa, theta=ns.theta, sig=ns.sig, rho=ns.rho
    )
    market = Market(r=ns.r)
    bsopt = EuropeanOptionBs(ns.k, dh.q, market)

    times = np.linspace(0, ns.T, ns.nt)
    mesh, cfg = create_mesh([ns.s_max, ns.v_max], ns.refine)
    space = SpaceSolver(mesh, dh, bsopt, is_call=ns.call, config=cfg)
    stepper = ThetaScheme(theta=ns.lam)

    values = stepper.solve(
        times, space, boundary_condition=None, is_american=ns.american
    )
    print(values[-1])
    return 0


def _write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
