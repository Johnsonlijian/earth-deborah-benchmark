from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#2b6cb0",
    "sky": "#5aa6d6",
    "green": "#4c956c",
    "orange": "#d9903d",
    "red": "#bf3b3b",
    "purple": "#6b5b95",
    "gray": "#6b7280",
    "dark": "#1f2937",
    "light": "#e5e7eb",
}


ARCHIVE_COLORS = {
    "CAMELS-US": PALETTE["blue"],
    "CAMELS-GB v2": PALETTE["green"],
    "CAMELS-BR v1.2": PALETTE["orange"],
    "CAMELS-AUS v2": PALETTE["red"],
    "CAMELS-DK lowland": PALETTE["purple"],
    "CAMELS-GB": PALETTE["green"],
}


MODEL_COLORS = {
    "rrmpg_gr4j": "#4c956c",
    "rrmpg_hbvedu": "#2b6cb0",
    "global_lstm": "#6b5b95",
    "seasonal_climatology": "#d9903d",
    "seasonal_ar1_null": "#bf3b3b",
    "lagged_q_lower_bound": "#6b7280",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    text = label if label.startswith("(") else f"({label})"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=PALETTE["dark"],
    )


def clean_axis(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="x", color="#d9dee7", lw=0.5, alpha=0.8)
        ax.set_axisbelow(True)


def save_figure(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
