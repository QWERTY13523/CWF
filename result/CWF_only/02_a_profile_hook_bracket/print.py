#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOG / SIGGRAPH-style convergence figure.

Design principles
-----------------
- restrained and publication-oriented
- no decorative shadows / no overly colorful callouts
- compact two-panel layout
- zoom-in shown as a clean inset in the active-quads panel
- annotate only iter=50 (not the final value)
- suitable for TOG / SIGGRAPH paper figures

Required CSV columns
--------------------
    iteration, E_na, active_quads

Outputs
-------
    <csv_stem>_tog_siggraph.png
    <csv_stem>_tog_siggraph.pdf
    <csv_stem>_tog_siggraph.svg
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter, AutoMinorLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_CSV = "CWF_8000_02_a_profile_hook_bracket.obj_metrics.csv"
DEFAULT_ANNOT_ITER = 50
DEFAULT_ZOOM_MIN = 40
DEFAULT_ZOOM_MAX = 60


# =============================================================================
# Style: restrained TOG / SIGGRAPH aesthetics
# =============================================================================
COLOR_ENA = "#2B6CB0"          # controlled blue
COLOR_ENA_LIGHT = "#BFD3EA"
COLOR_QUADS = "#C05621"        # muted orange
COLOR_QUADS_LIGHT = "#F2C2A8"
COLOR_HIGHLIGHT = "#F6EEE8"    # subtle zoom source region
COLOR_TEXT = "#1A1A1A"
COLOR_AXIS = "#333333"
COLOR_GRID = "#E7E7E7"
COLOR_CONNECT = "#7A7A7A"

mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    "font.size": 8.8,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.9,
    "ytick.labelsize": 7.9,
    "legend.fontsize": 7.8,

    "axes.linewidth": 0.75,
    "lines.linewidth": 1.8,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # keep text editable in vector exports
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =============================================================================
# Utilities
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--annot-iter", type=float, default=DEFAULT_ANNOT_ITER)
    parser.add_argument("--zoom-min", type=float, default=DEFAULT_ZOOM_MIN)
    parser.add_argument("--zoom-max", type=float, default=DEFAULT_ZOOM_MAX)
    parser.add_argument(
        "--out-stem",
        type=str,
        default=None,
        help="Output stem. If omitted, use <csv_stem>_tog_siggraph.",
    )
    return parser.parse_args()


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"iteration", "E_na", "active_quads"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[["iteration", "E_na", "active_quads"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["iteration"] = pd.to_numeric(df["iteration"])
    df["E_na"] = pd.to_numeric(df["E_na"])
    df["active_quads"] = pd.to_numeric(df["active_quads"])
    df = df.sort_values("iteration")
    return df


def nearest_index(x: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(x - target)))


def padded_range(y: np.ndarray, pad_ratio: float = 0.12, min_pad: float = 1.0) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    if np.isclose(y0, y1):
        pad = max(min_pad, 0.08 * max(abs(y0), 1.0))
        return y0 - pad, y1 + pad
    pad = max(min_pad, (y1 - y0) * pad_ratio)
    return y0 - pad, y1 + pad


def style_axis(ax: plt.Axes, *, add_minor_y: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.grid(True, which="major", color=COLOR_GRID, linewidth=0.5)
    if add_minor_y:
        ax.grid(True, which="minor", axis="y", color="#F1F1F1", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", top=False, right=False, color=COLOR_AXIS, labelcolor=COLOR_TEXT)


def style_inset(ax: plt.Axes) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_linewidth(0.65)
        ax.spines[side].set_color(COLOR_AXIS)
    ax.grid(True, which="major", color="#ECECEC", linewidth=0.45)
    ax.tick_params(which="both", direction="in", top=False, right=False, labelsize=7.0,
                   color=COLOR_AXIS, labelcolor=COLOR_TEXT, pad=1.6)


def setup_energy_scale(ax: plt.Axes, y: np.ndarray) -> None:
    if np.all(y > 0):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        positive = y[y > 0]
        linthresh = max(float(np.min(positive)) * 0.5, 1e-12) if len(positive) else 1e-12
        ax.set_yscale("symlog", linthresh=linthresh)


def annotate_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    xytext: tuple[float, float] = (8, 7),
    ha: str = "left",
    va: str = "bottom",
) -> None:
    ax.scatter([x], [y], s=18, color=color, edgecolor="white", linewidth=0.45, zorder=8)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=7.7,
        color=COLOR_TEXT,
        bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.04", fc="white", ec="#B9B9B9", lw=0.55),
        arrowprops=dict(arrowstyle="-", lw=0.6, color="#5F5F5F", shrinkA=2.0, shrinkB=2.5),
        zorder=9,
    )


def add_panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.09, 1.02, tag,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9.2, fontweight="semibold", color=COLOR_TEXT,
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if args.out_stem is None:
        out_stem = f"{csv_path.with_suffix('').name}_tog_siggraph"
    else:
        out_stem = args.out_stem

    df = load_metrics(csv_path)
    x = df["iteration"].to_numpy(dtype=float)
    e_na = df["E_na"].to_numpy(dtype=float)
    active = df["active_quads"].to_numpy(dtype=float)

    idx = nearest_index(x, args.annot_iter)
    x_annot = float(x[idx])
    e_annot = float(e_na[idx])
    q_annot = float(active[idx])

    zoom_mask = (x >= args.zoom_min) & (x <= args.zoom_max)
    if np.count_nonzero(zoom_mask) < 2:
        raise ValueError(
            f"Not enough data points in zoom range [{args.zoom_min}, {args.zoom_max}]. "
            "Adjust --zoom-min and --zoom-max."
        )

    x_zoom = x[zoom_mask]
    q_zoom = active[zoom_mask]
    qz0, qz1 = padded_range(q_zoom, pad_ratio=0.16, min_pad=1.0)

    fig = plt.figure(figsize=(7.05, 4.15))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.16], hspace=0.16)
    ax_e = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[1, 0])

    # -------------------------------------------------------------------------
    # (a) E_NA panel
    # -------------------------------------------------------------------------
    ax_e.plot(x, e_na, color=COLOR_ENA, solid_capstyle="round", zorder=3)
    setup_energy_scale(ax_e, e_na)

    if np.all(e_na > 0):
        baseline = max(np.min(e_na[e_na > 0]) * 0.95, 1e-12)
        ax_e.fill_between(x, e_na, baseline, color=COLOR_ENA_LIGHT, alpha=0.14, zorder=1)

    ax_e.axvline(x_annot, color=COLOR_ENA, linestyle=(0, (3.5, 2.5)), lw=0.85, alpha=0.62, zorder=2)
    annotate_value(ax_e, x_annot, e_annot, rf"iter={int(round(x_annot))}: {e_annot:.2e}", COLOR_ENA, xytext=(8, 6))

    style_axis(ax_e, add_minor_y=True)
    ax_e.set_ylabel(r"$E_{\mathrm{NA}}$")
    ax_e.tick_params(labelbottom=False)
    add_panel_tag(ax_e, "(a)")

    # -------------------------------------------------------------------------
    # (b) Active quads + inset zoom
    # -------------------------------------------------------------------------
    ax_q.axvspan(args.zoom_min, args.zoom_max, color=COLOR_HIGHLIGHT, alpha=0.95, zorder=0)
    ax_q.plot(x, active, color=COLOR_QUADS, solid_capstyle="round", zorder=3)
    ax_q.fill_between(x, active, 0, color=COLOR_QUADS_LIGHT, alpha=0.16, zorder=1)
    ax_q.axvline(x_annot, color=COLOR_QUADS, linestyle=(0, (3.5, 2.5)), lw=0.85, alpha=0.58, zorder=2)
    ax_q.scatter([x_annot], [q_annot], s=16, color=COLOR_QUADS, edgecolor="white", linewidth=0.45, zorder=6)

    style_axis(ax_q, add_minor_y=False)
    ax_q.set_xlabel("Iteration")
    ax_q.set_ylabel("Bad quads")
    ax_q.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax_q.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_q.set_xlim(float(np.min(x)), float(np.max(x)))
    add_panel_tag(ax_q, "(b)")

    # Draw a clean source-region rectangle only around the local vertical extent.
    source_rect = Rectangle(
        (args.zoom_min, qz0),
        args.zoom_max - args.zoom_min,
        qz1 - qz0,
        facecolor="none",
        edgecolor="#B88A6B",
        linewidth=0.75,
        zorder=4,
    )
    ax_q.add_patch(source_rect)

    # Floating inset: compact, clean, TOG-like.
    axins = inset_axes(
        ax_q,
        width="42%",
        height="55%",
        loc="upper right",
        borderpad=1.0,
    )
    axins.set_facecolor("white")
    axins.plot(x_zoom, q_zoom, color=COLOR_QUADS, lw=1.45, zorder=3)
    axins.fill_between(x_zoom, q_zoom, qz0, color=COLOR_QUADS_LIGHT, alpha=0.20, zorder=1)
    axins.scatter(x_zoom, q_zoom, s=8, color=COLOR_QUADS, edgecolor="white", linewidth=0.25, zorder=4)
    axins.axvline(x_annot, color=COLOR_QUADS, linestyle=(0, (3.0, 2.0)), lw=0.75, alpha=0.7, zorder=2)

    if args.zoom_min <= x_annot <= args.zoom_max:
        annotate_value(axins, x_annot, q_annot, rf"iter={int(round(x_annot))}: {q_annot:.0f}", COLOR_QUADS, xytext=(7, 5))

    axins.set_xlim(args.zoom_min, args.zoom_max)
    axins.set_ylim(qz0, qz1)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axins.xaxis.set_minor_locator(AutoMinorLocator(2))
    style_inset(axins)
    axins.set_title(rf"Zoom-in: iterations {int(args.zoom_min)}–{int(args.zoom_max)}", fontsize=7.4, pad=3.0)

    # Minimal connectors from source rectangle to inset bottom corners.
    ax_q.annotate(
        "",
        xy=(args.zoom_min, qz1), xycoords=ax_q.transData,
        xytext=(0.03, 0.02), textcoords=axins.transAxes,
        arrowprops=dict(arrowstyle="-", lw=0.65, color=COLOR_CONNECT),
        zorder=4,
    )
    ax_q.annotate(
        "",
        xy=(args.zoom_max, qz1), xycoords=ax_q.transData,
        xytext=(0.97, 0.02), textcoords=axins.transAxes,
        arrowprops=dict(arrowstyle="-", lw=0.65, color=COLOR_CONNECT),
        zorder=4,
    )

    fig.align_ylabels([ax_e, ax_q])

    for ext in ("png", "pdf", "svg"):
        out_path = Path(f"{out_stem}.{ext}")
        fig.savefig(out_path)
        print(out_path)

    plt.close(fig)


if __name__ == "__main__":
    main()
