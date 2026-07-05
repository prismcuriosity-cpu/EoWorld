"""Shared matplotlib styling for journal-ready EoWorld figures.

One place to control the look so every figure in the paper is consistent:
serif-free clean type, 300 dpi raster + vector PDF, recessive spines/grid, and
the official CholecSeg8k class palette reused everywhere (a class has ONE colour
in every figure — overlays, bars, heatmaps).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from eoworld.data.class_info import CHOLECSEG8K_CLASSES, COLORS_01, CLASS_NAMES

# Ink tokens — text never wears a series colour.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"


def set_journal_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def class_cmap() -> ListedColormap:
    """A ListedColormap indexed by contiguous train id."""
    return ListedColormap(COLORS_01, name="cholecseg8k")


def class_legend_handles(include: list[int] | None = None) -> list[Patch]:
    ids = include if include is not None else [c.train_id for c in CHOLECSEG8K_CLASSES]
    return [
        Patch(facecolor=COLORS_01[i], edgecolor="#00000022", label=CLASS_NAMES[i])
        for i in ids
    ]


def save_figure(fig, out_dir: Path, name: str, *, pdf: bool = True) -> list[Path]:
    """Save a figure as high-res PNG (+ vector PDF for the manuscript)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    if pdf:
        paths.append(out_dir / f"{name}.pdf")
        fig.savefig(paths[1])
    plt.close(fig)
    return paths
