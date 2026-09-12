"""Shared chart theme and palette.

The palette is a validated categorical set (checked for colour-vision
deficiency separation, chroma and lightness band before use), applied in a
fixed slot order and never cycled. Three of the hues sit below 3:1 contrast on
a light surface, so every chart built here either carries direct value labels
or is published next to the summary table the same function returns -- that is
the documented relief for the contrast warning.

Rules held throughout:
  * one y-axis per chart, never a second scale
  * sequential magnitude uses one hue, light -> dark
  * diverging uses blue <-> red with a neutral grey midpoint
  * grid and axes are recessive; the data is the darkest ink on the plot
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in fixed assignment order.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue sequential ramp (blue), light -> dark.
SEQUENTIAL = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#2a78d6", "#256abf", "#184f95", "#0d366b",
]

# Diverging poles with a neutral midpoint that reads as "nothing".
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_theme() -> None:
    """Install the project chart theme into matplotlib's rcParams."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            # Recessive chrome: hairline grid, no top/right spines.
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            # Ink hierarchy: titles primary, labels secondary, ticks muted.
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            # Thin marks, per the mark spec.
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "figure.titlesize": 13,
            "figure.titleweight": "600",
        }
    )


def label_bars(ax, fmt: str = "{:,.0f}", fontsize: int = 8, padding: float = 3) -> None:
    """Direct-label every bar -- the relief for low-contrast fills."""
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=lambda v: fmt.format(v),
            fontsize=fontsize,
            color=INK_SECONDARY,
            padding=padding,
        )


def save(fig, name: str, figures_dir=None) -> str:
    """Save a figure into reports/figures and return its path."""
    from src.config import FIGURES_DIR

    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return str(path)


def finish(ax, title: str = "", subtitle: str = "", ylabel: str = "", xlabel: str = ""):
    """Apply the standard title/subtitle/axis-label treatment.

    The title is pushed clear of the plot when a subtitle is present, so the
    two never overlap; the subtitle sits just above the axes in muted ink.
    """
    if title:
        ax.set_title(title, pad=26 if subtitle else 12)
    if subtitle:
        ax.text(
            0, 1.015, subtitle,
            transform=ax.transAxes, fontsize=8.5,
            color=INK_MUTED, va="bottom", ha="left",
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return ax
