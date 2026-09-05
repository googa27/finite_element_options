#!/usr/bin/env python3
"""Generate the public-synthetic adoption-sequence closeout matrix and visual."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/evidence/adoption_sequence_closeout_2026-09-05.json"
DEFAULT_PNG = ROOT / "docs/images/adoption_sequence_closeout_2026-09-05.png"
EVIDENCE = ROOT / "docs/evidence"


def _load(name: str) -> dict[str, Any]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: str) -> dict[str, str]:
    return {"path": path, "sha256": _sha256(ROOT / path)}


def build_matrix() -> dict[str, Any]:
    """Build a bounded decision matrix from immutable canonical evidence artifacts."""

    boundaries = _load("dependency_boundaries_2026-09-04.json")
    volatility = _load("regime_switching_quanto_volatility_benchmark_2026-09-03.json")
    quantlib = _load("regime_switching_quanto_quantlib_oracle_2026-09-04.json")
    iminuit = _load("regime_switching_quanto_iminuit_identifiability_2026-09-04.json")
    openturns = _load("regime_switching_quanto_openturns_uq_2026-09-04.json")
    pymor = _load("black_scholes_pymor_rom_2026-09-05.json")
    petsc = _load("petsc_vi_assessment_2026-09-05.json")
    bayesian = _load("bayesian_jax_profile_2026-09-05.json")

    holdouts = pymor["holdouts"]
    steps = [
        {
            "step": 1,
            "issue": 130,
            "title": "Dependency boundaries",
            "decision": boundaries["decision"]["status"],
            "class": "ADOPT",
            "decision_complete": True,
            "route_action": "adopt",
            "metric": (
                f"{len(boundaries['isolated_profiles'])} research extras isolated; zero base leaks"
            ),
            "boundary": "Optional adapters only; base FEM imports remain lightweight.",
            "evidence": [_source("docs/evidence/dependency_boundaries_2026-09-04.json")],
        },
        {
            "step": 2,
            "issue": 131,
            "title": "ARCH volatility challengers",
            "decision": volatility["decision"]["decision"],
            "class": "REJECT",
            "decision_complete": True,
            "route_action": "reject",
            "metric": (
                f"best challenger {volatility['decision']['selected_candidate']}; "
                "promotion disabled"
            ),
            "boundary": "Invalid Markov AR(2) baseline blocks challenger promotion.",
            "evidence": [
                _source(
                    "docs/evidence/regime_switching_quanto_volatility_benchmark_2026-09-03.json"
                )
            ],
        },
        {
            "step": 3,
            "issue": 132,
            "title": "QuantLib oracle",
            "decision": "retain_optional_oracle",
            "class": "RETAIN",
            "decision_complete": True,
            "route_action": "retain",
            "metric": (
                f"{quantlib['summary']['case_count']} cases; max oracle error "
                f"{quantlib['summary']['max_quantlib_vs_analytical_abs']:.2e}"
            ),
            "boundary": "One-regime vanilla/fixed-FX reductions only.",
            "evidence": [
                _source("docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json")
            ],
        },
        {
            "step": 4,
            "issue": 133,
            "title": "iminuit identifiability",
            "decision": "retain_optional_profile_likelihood",
            "class": "RETAIN",
            "decision_complete": True,
            "route_action": "retain",
            "metric": (
                f"{iminuit['summary']['identified_case_count']}/"
                f"{iminuit['summary']['case_count']} cases identified"
            ),
            "boundary": "Weak rho/FX-vol case fails closed as intended.",
            "evidence": [
                _source(
                    "docs/evidence/regime_switching_quanto_iminuit_identifiability_2026-09-04.json"
                )
            ],
        },
        {
            "step": 5,
            "issue": 134,
            "title": "OpenTURNS uncertainty",
            "decision": openturns["decision"]["status"],
            "class": "RETAIN",
            "decision_complete": True,
            "route_action": "retain",
            "metric": (
                "Sobol additive max errors "
                f"{openturns['additive_sobol_recovery']['max_abs_error_first']:.3f}/"
                f"{openturns['additive_sobol_recovery']['max_abs_error_total']:.3f}"
            ),
            "boundary": "Optional non-production pilot; NumPy remains baseline.",
            "evidence": [
                _source("docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json")
            ],
        },
        {
            "step": 6,
            "issue": 135,
            "title": "pyMOR reduced-order model",
            "decision": pymor["decision"]["status"],
            "class": "PROMOTE",
            "decision_complete": True,
            "route_action": "promote",
            "metric": (
                f"{pymor['timing']['median_online_speedup']:.1f}x online; "
                f"break-even {pymor['timing']['break_even_solve_count']} solves"
            ),
            "boundary": (
                "Optional adapter; max ROM/FOM price error "
                f"{max(row['rom_fom_errors']['price'] for row in holdouts):.2e}."
            ),
            "evidence": [_source("docs/evidence/black_scholes_pymor_rom_2026-09-05.json")],
        },
        {
            "step": 7,
            "issue": 136,
            "title": "PETSc variational inequality",
            "decision": petsc["decision"]["status"],
            "class": "PROMOTE EXT.",
            "decision_complete": True,
            "route_action": "promote_external",
            "metric": (
                f"single-rank runtime ratio {petsc['timing']['petsc_over_psor_runtime_ratio']:.3f}; "
                f"price gap {petsc['parity_errors']['price_abs']:.2e}"
            ),
            "boundary": "External single-rank adapter; SciPy PSOR remains canonical.",
            "evidence": [_source("docs/evidence/petsc_vi_assessment_2026-09-05.json")],
        },
        {
            "step": 8,
            "issue": 137,
            "title": "Bayesian / JAX profile",
            "decision": bayesian["decision"]["status"],
            "class": "ADOPT",
            "decision_complete": True,
            "route_action": "adopt",
            "metric": (
                f"R-hat {bayesian['pymc']['rhat']:.2f}/{bayesian['numpyro']['rhat']:.2f}; "
                f"mean gap {bayesian['cross_engine']['posterior_mean_abs_difference']:.3f}"
            ),
            "boundary": "Python 3.12 wheel profile; FEM autodiff remains fail-closed.",
            "evidence": [_source("docs/evidence/bayesian_jax_profile_2026-09-05.json")],
        },
    ]
    action_counts = dict(sorted(Counter(step["route_action"] for step in steps).items()))
    return {
        "schema_version": "adoption-sequence-closeout/v1",
        "privacy_class": "public_synthetic",
        "parent_issue": 128,
        "scope": (
            "Evidence-gated maintained-library adoption after the public-synthetic "
            "regime-switching FEM baseline; not production model validation."
        ),
        "baseline": {
            "issue": 129,
            "title": "Regime-switching quanto research baseline",
            "status": "complete",
            "evidence": [_source("docs/REGIME_SWITCHING_QUANTO_RESEARCH.md")],
        },
        "steps": steps,
        "uncertainty_decomposition": {
            "statistical": {
                "status": "bounded",
                "evidence": (
                    "ARCH promotion rejected because the Markov baseline did not converge; "
                    "iminuit separates one identified and one deliberately weak case."
                ),
            },
            "model_form": {
                "status": "bounded",
                "evidence": (
                    "QuantLib covers only one-regime vanilla/fixed-FX reductions; OpenTURNS "
                    "reports model-form sensitivity without production maturity."
                ),
            },
            "numerical": {
                "status": "bounded",
                "evidence": (
                    f"pyMOR achieved {pymor['timing']['median_online_speedup']:.1f}x online "
                    f"speedup with holdout gates; PETSc price parity was "
                    f"{petsc['parity_errors']['price_abs']:.2e} at single rank."
                ),
            },
            "sampling": {
                "status": "bounded",
                "evidence": (
                    f"OpenTURNS uses n={openturns['propagation']['sample_size']} propagation "
                    f"and n={openturns['propagation']['sobol_base_size']} Sobol base samples; "
                    f"PyMC/NumPyro R-hat={bayesian['pymc']['rhat']:.2f}/"
                    f"{bayesian['numpyro']['rhat']:.2f} with zero divergences."
                ),
            },
        },
        "portfolio_decision": {
            "baseline_complete": True,
            "steps_complete": 8,
            "evidence_backed_rejections": 1,
            "bounded_non_rejections": 7,
            "route_action_counts": action_counts,
            "base_wheel_optional_stack_leaks": 0,
            "capability_matrix_maturity_upgrades": 0,
            "epic_ready_to_close": True,
            "status": "close_adoption_sequence",
        },
    }


def write_matrix(path: Path, payload: dict[str, Any]) -> str:
    """Write canonical JSON and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    path.write_text(data, encoding="utf-8")
    return _sha256(path)


def render_matrix(path: Path, payload: dict[str, Any], digest: str) -> None:
    """Render a mobile-first dark closeout card from the matrix."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    bg = "#0B0F14"
    surface = "#111820"
    border = "#2B3440"
    text = "#F1F5F9"
    muted = "#95A0AF"
    colors = {
        "ADOPT": "#16D89A",
        "RETAIN": "#45CFF4",
        "PROMOTE": "#A78BFA",
        "PROMOTE EXT.": "#C084FC",
        "REJECT": "#FF6B72",
    }

    fig = plt.figure(figsize=(8, 10), dpi=300, facecolor=bg)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.06,
        0.955,
        "BASELINE + ADOPTION SEQUENCE 1→8",
        color="#4DD5F7",
        fontsize=15.5,
        weight="bold",
        va="top",
    )
    ax.text(
        0.06,
        0.905,
        "CLOSED — EVIDENCE BEFORE DEPENDENCIES",
        color=text,
        fontsize=17.5,
        weight="bold",
        va="top",
    )
    ax.text(
        0.06,
        0.858,
        "7 bounded non-rejections  •  1 rejection  •  0 base leaks",
        color=muted,
        fontsize=10.5,
        va="top",
    )

    card_w, card_h = 0.415, 0.135
    xs = (0.06, 0.525)
    ys = (0.69, 0.535, 0.38, 0.225)
    for index, step in enumerate(payload["steps"]):
        col = index // 4
        row = index % 4
        x, y = xs[col], ys[row]
        color = colors[step["class"]]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                card_w,
                card_h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.5,
                edgecolor=border,
                facecolor=surface,
            )
        )
        ax.text(
            x + 0.018,
            y + card_h - 0.025,
            f"{step['step']}",
            color=color,
            fontsize=18,
            weight="bold",
            va="top",
        )
        ax.text(
            x + 0.068,
            y + card_h - 0.026,
            step["class"],
            color=color,
            fontsize=10.5,
            weight="bold",
            va="top",
        )
        ax.text(
            x + 0.018,
            y + 0.074,
            step["title"],
            color=text,
            fontsize=11.5,
            weight="bold",
            va="center",
        )
        ax.text(
            x + 0.018, y + 0.043, step["metric"], color=muted, fontsize=7.7, va="center", wrap=True
        )
        ax.text(
            x + 0.018,
            y + 0.017,
            step["boundary"],
            color=muted,
            fontsize=7.0,
            va="center",
            wrap=True,
        )

    ax.text(0.06, 0.19, "UNCERTAINTY OWNERSHIP", color=text, fontsize=15, weight="bold", va="top")
    labels = [
        ("STATISTICAL", "baseline + identifiability", "#FFB86B"),
        ("MODEL FORM", "scoped reductions + sensitivity", "#45CFF4"),
        ("NUMERICAL", "ROM accuracy + VI parity", "#A78BFA"),
        ("SAMPLING", "Sobol + MCMC diagnostics", "#16D89A"),
    ]
    for idx, (label, detail, color) in enumerate(labels):
        x = 0.06 + idx * 0.2325
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.105),
                0.205,
                0.058,
                boxstyle="round,pad=0.009,rounding_size=0.012",
                linewidth=1.2,
                edgecolor=color,
                facecolor=surface,
            )
        )
        ax.text(x + 0.012, 0.143, label, color=color, fontsize=8.3, weight="bold", va="center")
        ax.text(x + 0.012, 0.119, detail, color=muted, fontsize=6.4, va="center")

    ax.text(
        0.06, 0.063, "NO PRODUCTION MATURITY UPGRADE", color="#FFB86B", fontsize=9.5, weight="bold"
    )
    ax.text(
        0.06,
        0.035,
        f"public-synthetic  •  parent #128  •  matrix {digest[:8]}…{digest[-4:]}",
        color=muted,
        fontsize=8.2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=bg, bbox_inches=None)
    plt.close(fig)


def main() -> int:
    """Generate canonical JSON and its image-first closeout artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    payload = build_matrix()
    digest = write_matrix(args.json, payload)
    render_matrix(args.png, payload, digest)
    print(f"json={args.json} sha256={digest}")
    print(f"png={args.png} sha256={_sha256(args.png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
