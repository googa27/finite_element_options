#!/usr/bin/env python3
"""Generate the dark-theme JAX regime study result dashboard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

EXPECTED_VISUAL_STACK = {
    "matplotlib": "3.10.3",
    "numpy": "2.4.6",
    "pillow": "12.3.0",
}
ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUT = ROOT / "docs/images/jax_regime_study_2026-09-07.png"
CANONICAL_PDF = CANONICAL_OUTPUT.with_suffix(".pdf")
DEFAULT_OUTPUT = Path("/tmp/jax_regime_study_2026-09-07.png")


def _aliases_existing_protected_file(target: Path, protected: Path) -> bool:
    """Return whether two existing paths identify the same inode, failing closed on IO errors."""

    if not target.exists() or not protected.exists():
        return False
    try:
        return target.samefile(protected)
    except OSError as error:
        raise ValueError(f"cannot validate output identity: {error}") from error


def _temporary_sibling(target: Path) -> Path:
    """Create a closed temporary sibling suitable for atomic replacement."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=target.suffix, dir=target.parent, delete=False
    ) as stream:
        return Path(stream.name)


def _validated_output_paths(output: Path, *, publish_canonical: bool) -> tuple[Path, Path]:
    """Return PNG/PDF targets after protecting the complete canonical pair."""

    output = output.expanduser()
    pdf = output.with_suffix(".pdf")
    resolved_targets = {output.resolve(), pdf.resolve()}
    canonical_targets = {CANONICAL_OUTPUT.resolve(), CANONICAL_PDF.resolve()}
    aliases_canonical = bool(resolved_targets & canonical_targets) or any(
        _aliases_existing_protected_file(target, canonical)
        for target in (output, pdf)
        for canonical in (CANONICAL_OUTPUT, CANONICAL_PDF)
    )
    if aliases_canonical and not publish_canonical:
        raise ValueError("canonical visual output requires --publish-canonical")
    if publish_canonical and resolved_targets != canonical_targets:
        raise ValueError("--publish-canonical requires the canonical PNG/PDF output paths")
    if output.suffix.lower() != ".png":
        raise ValueError("--output must name a PNG path; the PDF path is derived")
    return output, pdf


def _validate_layout(figure: Any, axes: tuple[Any, ...]) -> None:
    """Fail when publication text escapes the canvas or same-role labels overlap."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox

    def bbox(artist: Any) -> Any:
        return artist.get_window_extent(renderer=renderer)

    def assert_contained(artist: Any, label: str) -> None:
        bounds = bbox(artist)
        tolerance = 2.0
        if (
            bounds.x0 < canvas.x0 - tolerance
            or bounds.y0 < canvas.y0 - tolerance
            or bounds.x1 > canvas.x1 + tolerance
            or bounds.y1 > canvas.y1 + tolerance
        ):
            raise RuntimeError(f"layout QA failed: {label} escapes the figure canvas")

    def assert_disjoint(artists: list[Any], label: str) -> None:
        boxes = [(artist, bbox(artist)) for artist in artists if artist.get_visible()]
        for left_index, (_, left) in enumerate(boxes):
            for _, right in boxes[left_index + 1 :]:
                overlap_width = min(left.x1, right.x1) - max(left.x0, right.x0)
                overlap_height = min(left.y1, right.y1) - max(left.y0, right.y0)
                if overlap_width > 2.0 and overlap_height > 2.0:
                    raise RuntimeError(f"layout QA failed: overlapping {label}")

    figure_text = [artist for artist in figure.texts if artist.get_text().strip()]
    assert_disjoint(figure_text, "figure header/footer text")
    for index, artist in enumerate(figure_text):
        assert_contained(artist, f"figure text {index}")

    for axis_index, axis in enumerate(axes):
        same_role_groups = (
            [label for label in axis.get_xticklabels() if label.get_text().strip()],
            [label for label in axis.get_yticklabels() if label.get_text().strip()],
            [label for label in axis.texts if label.get_text().strip()],
        )
        for group_index, group in enumerate(same_role_groups):
            assert_disjoint(group, f"axis {axis_index} label group {group_index}")
            for artist_index, artist in enumerate(group):
                assert_contained(
                    artist,
                    f"axis {axis_index} label group {group_index} item {artist_index}",
                )
        for label, artist in (
            ("title", axis.title),
            ("x label", axis.xaxis.label),
            ("y label", axis.yaxis.label),
        ):
            if artist.get_text().strip():
                assert_contained(artist, f"axis {axis_index} {label}")
        legend = axis.get_legend()
        if legend is not None:
            assert_contained(legend, f"axis {axis_index} legend")


def main() -> int:
    """Render a 16:9, 300-dpi evidence dashboard to PNG and vector PDF."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "docs/evidence/jax_regime_study_2026-09-07.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG path; defaults to /tmp",
    )
    parser.add_argument(
        "--publish-canonical",
        action="store_true",
        help="replace the canonical PNG/PDF explicitly",
    )
    parser.add_argument(
        "--qa-layout",
        action="store_true",
        help="fail on canvas escapes or overlaps among same-role labels",
    )
    args = parser.parse_args()
    args.output = args.output or (CANONICAL_OUTPUT if args.publish_canonical else DEFAULT_OUTPUT)
    try:
        args.output, pdf = _validated_output_paths(
            args.output, publish_canonical=args.publish_canonical
        )
    except ValueError as error:
        parser.error(str(error))

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import PIL

    actual_stack = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }
    if actual_stack != EXPECTED_VISUAL_STACK:
        raise RuntimeError(
            f"visual stack drift: expected {EXPECTED_VISUAL_STACK}, received {actual_stack}; "
            "install environments/jax-regime-visual-py312/requirements.lock"
        )

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = payload["hmm"]["candidate_comparison"]
    fit = payload["hmm"]["dynamax_full_fit"]
    posterior = payload["hmm"]["numpyro"]
    prices = payload["pricing"]["diffrax"]["point_prices"]
    intervals = payload["pricing"]["posterior_parameter_price_intervals"]
    historical = {
        row["contract"]: row for row in payload["pricing"]["historical_scikit_fem_and_numpy_mc"]
    }

    background = "#0d1117"
    panel = "#161b22"
    foreground = "#c9d1d9"
    muted = "#8b949e"
    cyan = "#22D3EE"
    violet = "#A78BFA"
    amber = "#FBBF24"
    green = "#34D399"
    red = "#FB7185"
    plt.rcParams.update(
        {
            "figure.facecolor": background,
            "axes.facecolor": panel,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": foreground,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": foreground,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
        }
    )
    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.45],
        left=0.065,
        right=0.97,
        bottom=0.13,
        top=0.80,
        hspace=0.52,
        wspace=0.18,
    )
    score_axis = figure.add_subplot(grid[0, 0])
    regime_axis = figure.add_subplot(grid[0, 1])
    price_axis = figure.add_subplot(grid[1, :])

    states = [row["states"] for row in candidates]
    scores = [row["heldout_mean_log_score"] for row in candidates]
    colors = ["#64748B", violet, cyan, amber]
    lower = min(scores) - 0.04
    score_axis.bar(
        states,
        np.asarray(scores) - lower,
        bottom=lower,
        color=colors,
        width=0.68,
    )
    score_axis.set_title("Chronological holdout density — higher is better")
    score_axis.set_xlabel("Gaussian states")
    score_axis.set_ylabel("Mean log score")
    score_axis.set_xticks(states)
    score_axis.set_ylim(lower, max(scores) + 0.02)
    selection = payload["hmm"]["selection"]
    selected_states = selection["selected_by_heldout_score"]
    selected_score = next(
        score for state, score in zip(states, scores, strict=True) if state == selected_states
    )
    score_axis.axhline(selected_score, color=cyan, lw=1, ls="--", alpha=0.6)
    for state, score in zip(states, scores, strict=True):
        score_axis.text(state, score + 0.006, f"{score:.4f}", ha="center", fontsize=9)
    score_axis.text(
        0.02,
        0.05,
        f"{selected_states} states selected: 4-state gain over 3 = "
        f"{selection['four_over_three_mean_log_score_gain']:.4f}/obs",
        transform=score_axis.transAxes,
        color=muted,
        fontsize=10,
    )

    volatilities = np.asarray(fit["annualized_composite_volatility_percent"])
    occupancy = np.asarray(fit["occupancy"])
    x = np.arange(len(volatilities))
    regime_colors = [green, cyan, violet, red][: len(volatilities)]
    regime_labels = (
        ["Low", "Middle", "High"]
        if len(volatilities) == 3
        else ["Low", "Middle-low", "Middle-high", "High"]
    )
    regime_axis.bar(x, volatilities, color=regime_colors, width=0.62)
    regime_axis.set_title("Four volatility-ordered DYNAMAX HMM states")
    regime_axis.set_ylabel("Annualized composite volatility (%)")
    regime_axis.set_xticks(x, regime_labels)
    for index, (volatility, weight) in enumerate(zip(volatilities, occupancy, strict=True)):
        regime_axis.text(
            index,
            volatility + max(volatilities) * 0.035,
            f"{volatility:.1f}%\nocc. {weight:.1%}",
            ha="center",
            fontsize=9,
        )
    regime_axis.set_ylim(0.0, max(volatilities) * 1.24)

    names = list(prices)
    positions = np.arange(len(names))
    point = np.asarray([prices[name]["price_clp"] for name in names]) / 1_000.0
    point_se = np.asarray([prices[name]["standard_error_clp"] for name in names]) / 1_000.0
    q05 = np.asarray([intervals[name]["posterior_parameter_q05_clp"] for name in names]) / 1_000.0
    posterior_median = (
        np.asarray([intervals[name]["posterior_parameter_median_clp"] for name in names]) / 1_000.0
    )
    q95 = np.asarray([intervals[name]["posterior_parameter_q95_clp"] for name in names]) / 1_000.0
    old_mc = np.asarray([historical[name]["numpy_exact_step_mc_clp"] for name in names]) / 1_000.0
    old_fem = np.asarray([historical[name]["fine_fem_clp"] for name in names]) / 1_000.0
    price_axis.errorbar(
        positions,
        posterior_median,
        yerr=[posterior_median - q05, q95 - posterior_median],
        fmt="o",
        color=cyan,
        ecolor=cyan,
        capsize=5,
        lw=2,
        label="Posterior parameter median and 90% interval",
    )
    price_axis.errorbar(
        positions,
        point,
        yerr=2.0 * point_se,
        fmt="o",
        color=foreground,
        ecolor=foreground,
        capsize=3,
        lw=1,
        label="Diffrax posterior-mean price ±2 MC SE",
    )
    price_axis.scatter(
        positions - 0.12,
        old_mc,
        color=green,
        marker="D",
        s=45,
        label="Prior 3-state NumPy exact-step MC",
    )
    price_axis.scatter(
        positions + 0.12,
        old_fem,
        color=violet,
        marker="s",
        s=45,
        label="Prior 3-state fine scikit-fem",
    )
    price_axis.set_title("Six-month constrained Qd research prices")
    price_axis.set_ylabel("Price (thousand CLP)")
    display_labels = {
        "ATM composite call": "Composite call",
        "ATM composite put": "Composite put",
        "ATM fixed-FX quanto call": "Fixed-FX quanto call",
        "Composite digital": "Composite digital",
        "Dual-trigger protection": "Dual-trigger\nprotection",
    }
    price_axis.set_xticks(
        positions,
        [display_labels[name] for name in names],
    )
    maximum_visible_price = max(
        float(np.max(q95)),
        float(np.max(point + 2.0 * point_se)),
        float(np.max(old_mc)),
        float(np.max(old_fem)),
    )
    price_axis.set_ylim(0.0, maximum_visible_price * 1.25)
    price_axis.grid(axis="y", color="#334155", alpha=0.45, lw=0.7)
    price_axis.legend(loc="upper right", frameon=False, ncols=2, fontsize=9)

    diagnostics = payload["verification"]["gates"]
    gate_color = green if all(diagnostics.values()) else red
    figure.suptitle(
        "JAX-native regime inference & quanto pricing",
        fontsize=23,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    figure.text(
        0.04,
        0.875,
        "DYNAMAX EM/filter/smoother  •  NumPyro marginalized-state NUTS  •  Diffrax aligned Itô Euler",
        color=muted,
        fontsize=11,
        ha="left",
    )
    figure.text(
        0.96,
        0.885,
        (
            f"GATES {'PASS' if all(diagnostics.values()) else 'FAIL'}   "
            f"R̂ {posterior['maximum_rhat']:.3f}   "
            f"ESS {posterior['minimum_ess']:.0f}   "
            f"div {posterior['divergences']}"
        ),
        color=gate_color,
        fontsize=11,
        fontweight="bold",
        ha="right",
    )
    figure.text(
        0.04,
        0.012,
        "Research-only • P transition reused under constrained Qd • prior 3-state markers are context, not same-model parity",
        color=amber,
        fontsize=10,
        ha="left",
    )

    layout_qa = args.qa_layout or args.publish_canonical
    if layout_qa:
        _validate_layout(figure, (score_axis, regime_axis, price_axis))

    png_temporary: Path | None = None
    pdf_temporary: Path | None = None
    try:
        png_temporary = _temporary_sibling(args.output)
        pdf_temporary = _temporary_sibling(pdf)
        figure.savefig(
            png_temporary,
            format="png",
            dpi=300,
            facecolor=background,
            metadata={"Software": "finite_element_options deterministic JAX regime visual"},
        )
        figure.savefig(
            pdf_temporary,
            format="pdf",
            facecolor=background,
            metadata={
                "Creator": "finite_element_options deterministic JAX regime visual",
                "Producer": "Matplotlib 3.10.3",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        png_temporary.chmod(0o644)
        pdf_temporary.chmod(0o644)
        os.replace(png_temporary, args.output)
        os.replace(pdf_temporary, pdf)
    finally:
        plt.close(figure)
        if png_temporary is not None:
            png_temporary.unlink(missing_ok=True)
        if pdf_temporary is not None:
            pdf_temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "png": str(args.output),
                "pdf": str(pdf),
                "stack": actual_stack,
                "layout_qa": layout_qa,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
