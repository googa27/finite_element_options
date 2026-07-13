"""Generate README validation visuals from deterministic FEM fixtures.

The figures and SVG use checked-in public-synthetic fixtures plus executable
validation runners. They do not use FEniCSx-only paths or private Pinares data.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from finite_element_options.validation.black_scholes_parity import (
    run_public_black_scholes_parity_fixture,
)
from finite_element_options.validation.pinares_fixed_price_proxy import (
    run_public_pinares_fixed_price_proxy_fixture,
)

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#2dd4ff"
MAGENTA = "#ff4db8"
GREEN = "#3fb950"
ORANGE = "#f0883e"
PURPLE = "#a371f7"


def _dark_pixel_ratio(path: Path) -> float:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(np.mean(luminance < 64.0))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.8)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.14, facecolor=BG)
    plt.close(fig)


def _plot_black_scholes(path: Path) -> dict[str, Any]:
    report = run_public_black_scholes_parity_fixture()
    rows = [row.to_public_dict() for row in report.convergence_rows]
    dofs = np.array([row["degrees_of_freedom"] for row in rows], dtype=float)
    errors = np.array([row["absolute_error"] for row in rows], dtype=float)
    prices = np.array([row["observed_price"] for row in rows], dtype=float)
    oracle = float(report.expected_price)

    fig, (ax_price, ax_error) = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.78, wspace=0.18)
    fig.suptitle(
        "Finite-element Black-Scholes validation fixture",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    _style_axes(ax_price)
    ax_price.plot(dofs, prices, color=PURPLE, marker="o", linewidth=2.4)
    ax_price.axhline(oracle, color=CYAN, linestyle="--", linewidth=2.2)
    ax_price.set_xlabel("degrees of freedom")
    ax_price.set_ylabel("call value at S=K=100")
    ax_price.set_title("Lagrange-P2 FEM value approaches oracle")
    ax_price.text(dofs[-1], prices[-1], "  FEM", color=PURPLE, va="center", fontsize=9)
    ax_price.text(dofs[0], oracle, "  analytical", color=CYAN, va="bottom", fontsize=9)

    _style_axes(ax_error)
    ax_error.semilogy(dofs, errors, color=MAGENTA, marker="o", linewidth=2.4)
    ax_error.axhline(report.tolerance_absolute, color=GREEN, linestyle="--", linewidth=2.0)
    ax_error.fill_between(dofs, report.tolerance_absolute, errors.max() * 1.15, color=ORANGE, alpha=0.08)
    ax_error.set_xlabel("degrees of freedom")
    ax_error.set_ylabel("absolute price error")
    ax_error.set_title("Validated route satisfies price tolerance")
    ax_error.text(dofs[-1], errors[-1], f"  {errors[-1]:.2e}", color=MAGENTA, va="center", fontsize=9)
    ax_error.text(dofs[0], report.tolerance_absolute, "  tolerance", color=GREEN, va="bottom", fontsize=9)

    caption = (
        "Source: run_public_black_scholes_parity_fixture(), benchmark fem-bs-001; "
        "line-uniform Lagrange-P2 theta route with analytical Black-Scholes oracle."
    )
    fig.text(0.02, 0.055, caption, color=MUTED, fontsize=7.5)
    _save(fig, path)
    return {
        "benchmark_id": report.benchmark_id,
        "problem_id": report.problem_id,
        "oracle_price": report.expected_price,
        "final_price": report.observed_price,
        "final_abs_error": report.price_absolute_error,
        "tolerance_absolute": report.tolerance_absolute,
        "delta_abs_error": report.delta_absolute_error,
        "gamma_abs_error": report.gamma_absolute_error,
        "rows": rows,
    }


def _plot_pinares(path: Path) -> dict[str, Any]:
    report = run_public_pinares_fixed_price_proxy_fixture()
    rows = [row.to_public_dict() for row in report.rows]
    dofs = np.array([row["degrees_of_freedom"] for row in rows], dtype=float)
    errors = np.array([row["absolute_error_uf"] for row in rows], dtype=float)
    prices = np.array([row["observed_price_uf"] for row in rows], dtype=float)
    oracle = float(report.expected_price_uf)

    fig, (ax_price, ax_error) = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.78, wspace=0.18)
    fig.suptitle(
        "Pinares fixed-price proxy: public-synthetic FEM evidence",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    _style_axes(ax_price)
    ax_price.plot(dofs, prices, color=PURPLE, marker="o", linewidth=2.4)
    ax_price.axhline(oracle, color=CYAN, linestyle="--", linewidth=2.2)
    ax_price.set_xlabel("degrees of freedom")
    ax_price.set_ylabel("survival-scaled value (UF)")
    ax_price.set_title("Proxy weak-form solve against oracle")
    ax_price.text(dofs[-1], prices[-1], "  FEM proxy", color=PURPLE, va="center", fontsize=9)
    ax_price.text(dofs[0], oracle, "  oracle", color=CYAN, va="bottom", fontsize=9)

    _style_axes(ax_error)
    ax_error.semilogy(dofs, errors, color=MAGENTA, marker="o", linewidth=2.4)
    ax_error.axhline(report.case.price_abs_tolerance_uf, color=GREEN, linestyle="--", linewidth=2.0)
    ax_error.fill_between(
        dofs,
        report.case.price_abs_tolerance_uf,
        max(errors.max(), report.case.price_abs_tolerance_uf) * 1.25,
        color=ORANGE,
        alpha=0.08,
    )
    ax_error.set_xlabel("degrees of freedom")
    ax_error.set_ylabel("absolute price error (UF)")
    ax_error.set_title("Finest refinement inside 1.0 UF budget")
    ax_error.text(dofs[-1], errors[-1], f"  {errors[-1]:.3f} UF", color=MAGENTA, va="center", fontsize=9)
    ax_error.text(dofs[0], report.case.price_abs_tolerance_uf, "  budget", color=GREEN, va="bottom", fontsize=9)

    caption = (
        "Source: run_public_pinares_fixed_price_proxy_fixture(); public-synthetic Q* weak-form proxy only. "
        "Full ROFR/family-contract/legal/tax/HJB routes intentionally fail closed."
    )
    fig.text(0.02, 0.055, caption, color=MUTED, fontsize=7.5)
    _save(fig, path)
    return {
        "benchmark_id": "PINARES-FEM-FIXED-PRICE-PROXY-V0",
        "problem_id": report.case.problem_id,
        "converged": report.converged,
        "oracle_price_uf": report.expected_price_uf,
        "final_price_uf": report.observed_price_uf,
        "final_abs_error_uf": report.price_absolute_error_uf,
        "price_abs_tolerance_uf": report.case.price_abs_tolerance_uf,
        "delta_abs_error": report.delta_absolute_error,
        "gamma_abs_error": report.gamma_absolute_error,
        "rows": rows,
        "omission_note": None,
    }


def _write_pipeline_svg(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = [
        (40, 70, "explicit PDE", "domain, weak form, BCs", CYAN),
        (285, 70, "FEM route gate", "capability manifest", GREEN),
        (530, 70, "mesh + theta", "P2 line, SciPy direct", PURPLE),
        (775, 70, "diagnostics", "errors, residuals, fail-closed", MAGENTA),
    ]
    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="260" viewBox="0 0 1060 260" role="img" aria-labelledby="title desc">',
        '<title id="title">Finite Element Options capability pipeline</title>',
        '<desc id="desc">Dark-palette diagram showing explicit PDE inputs, route gates, mesh and theta solve, then diagnostics.</desc>',
        f'<rect width="1060" height="260" fill="{BG}"/>',
        f'<rect x="20" y="25" width="1020" height="185" rx="18" fill="{PANEL}" stroke="{GRID}"/>',
    ]
    for idx, (x, y, title, subtitle, color) in enumerate(boxes):
        svg_parts.append(f'<rect x="{x}" y="{y}" width="205" height="82" rx="13" fill="{BG}" stroke="{color}" stroke-width="2"/>')
        svg_parts.append(f'<text x="{x + 18}" y="{y + 34}" fill="{color}" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>')
        svg_parts.append(f'<text x="{x + 18}" y="{y + 61}" fill="{TEXT}" font-family="Inter,Arial,sans-serif" font-size="14">{html.escape(subtitle)}</text>')
        if idx < len(boxes) - 1:
            x2 = x + 226
            svg_parts.append(f'<path d="M{x + 205} {y + 41} L{x2} {y + 41}" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)"/>')
    svg_parts.insert(
        5,
        f'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{MUTED}"/></marker></defs>',
    )
    svg_parts.append(f'<text x="40" y="228" fill="{MUTED}" font-family="Inter,Arial,sans-serif" font-size="13">Evidence sources: docs/CAPABILITY_MATRIX.md, tests/fixtures/fem_bs_001, tests/fixtures/fem_pinares_fixed_price_proxy_v1. Optional FEniCSx/PETSc remain outside base validation.</text>')
    svg_parts.append("</svg>\n")
    path.write_text("\n".join(svg_parts), encoding="utf-8")
    return {"path": str(path), "visual_qa": "pass", "palette_background": BG, "palette_panel": PANEL}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/images")
    parser.add_argument(
        "--report",
        default="docs/images/readme_visual_provenance.json",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    assets = {
        "black_scholes": output_dir / "fem_black_scholes_validation_convergence.png",
        "pinares_proxy": output_dir / "fem_pinares_proxy_validation_convergence.png",
        "capability_pipeline": output_dir / "fem_capability_pipeline.svg",
    }
    report = {
        "repository": "finite_element_options",
        "generator": "scripts/generate_readme_validation_visuals.py",
        "palette": {
            "background": BG,
            "panel": PANEL,
            "analytical_reference": CYAN,
            "numerical_error": MAGENTA,
            "validated_pass": GREEN,
            "boundary_stability_caveat": ORANGE,
            "secondary_route": PURPLE,
        },
        "assets": {},
        "evidence": {
            "black_scholes": _plot_black_scholes(assets["black_scholes"]),
            "pinares_proxy": _plot_pinares(assets["pinares_proxy"]),
            "capability_pipeline": _write_pipeline_svg(assets["capability_pipeline"]),
        },
    }
    for key, path in assets.items():
        if path.suffix == ".png":
            image = Image.open(path)
            dark_ratio = _dark_pixel_ratio(path)
            report["assets"][key] = {
                "path": str(path),
                "dpi": image.info.get("dpi"),
                "size_px": image.size,
                "dark_pixel_ratio": dark_ratio,
                "visual_qa": "pass" if dark_ratio >= 0.65 else "review",
            }
        else:
            report["assets"][key] = {"path": str(path), "visual_qa": "pass"}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assets": {k: str(v) for k, v in assets.items()}, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
