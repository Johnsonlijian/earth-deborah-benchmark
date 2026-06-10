"""Create the R30 mechanism-evidence mother figure.

The figure is a schematic plus quantitative source-data summary. It is not
used as scientific evidence beyond the values already reported in derived
tables; it organizes the claim boundary for the main paper.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
LATEX_FIG_DIR = ROOT / "figures"
LATEX_SOURCE_DIR = ROOT / "source_data"

COLORS = {
    "ink": "#202124",
    "muted": "#667085",
    "line": "#D0D5DD",
    "blue": "#2E6F9E",
    "green": "#2F8F5B",
    "gold": "#C58A1F",
    "red": "#B54747",
    "purple": "#756BB1",
    "teal": "#2C7A7B",
    "paper": "#F7F9FB",
}


def style_axes(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=7, colors=COLORS["muted"])


def add_panel_label(ax, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )
    ax.text(
        0.07,
        1.03,
        title,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )


def rounded_box(ax, xy, width, height, text, fc, ec=None, fontsize=7.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.2):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color or COLORS["muted"],
        zorder=4,
    )
    ax.add_patch(arr)


def panel_object(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", "Study object and boundary")

    rounded_box(ax, (0.04, 0.62), 0.23, 0.20, "Climate\nforcing", "#E9F2FA", COLORS["blue"])
    rounded_box(
        ax,
        (0.39, 0.55),
        0.27,
        0.34,
        "Catchment\nstorage + routing\nfilter",
        "#EEF7F1",
        COLORS["green"],
    )
    rounded_box(ax, (0.78, 0.62), 0.18, 0.20, "Daily\nQ(t)", "#FFF4DF", COLORS["gold"])
    arrow(ax, (0.28, 0.72), (0.38, 0.72), COLORS["blue"])
    arrow(ax, (0.67, 0.72), (0.77, 0.72), COLORS["green"])

    x = np.linspace(0.06, 0.95, 320)
    y = 0.31 + 0.07 * np.exp(-3.6 * (x - 0.06)) * np.cos(36 * x)
    ax.plot(x, y, color=COLORS["teal"], lw=1.5)
    ax.axhline(0.31, color=COLORS["line"], lw=0.8)
    ax.text(0.04, 0.42, r"Measured output memory: $\tau_{\rm acf}$", fontsize=7.6, color=COLORS["ink"])
    ax.text(
        0.04,
        0.10,
        "Not groundwater residence time\nNot tracer transit time\nNot a storage distribution",
        fontsize=7.1,
        color=COLORS["red"],
        linespacing=1.35,
    )
    ax.text(
        0.53,
        0.11,
        "Use as a falsifiable\nspectral benchmark",
        fontsize=7.4,
        color=COLORS["green"],
        linespacing=1.35,
    )


def panel_coordinate(ax):
    add_panel_label(ax, "b", "Coordinate and tested curve")
    style_axes(ax)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.9, 0.15)
    x = np.linspace(-2, 2, 200)
    shifts = [-0.55, 0.0, 0.45]
    colors = [COLORS["purple"], COLORS["blue"], COLORS["gold"]]
    for shift, color in zip(shifts, colors):
        raw = -0.55 - 1.85 / (1 + np.exp(-(x - shift) * 2.2))
        de = -0.55 - 1.85 / (1 + np.exp(-x * 2.2)) + 0.06 * shift
        ax.plot(x, raw, color=color, alpha=0.28, lw=1.0)
        ax.plot(x, de, color=color, alpha=0.95, lw=1.7)
    ax.axvline(0, color=COLORS["line"], lw=0.9, ls="--")
    ax.text(0.02, -0.17, r"$\mathrm{De}=1$", fontsize=7, color=COLORS["muted"])
    ax.set_xlabel(r"$\log_{10}(\mathrm{De})=\log_{10}(\tau_{\rm acf} f)$", fontsize=7.2)
    ax.set_ylabel(r"Local slope $\beta$", fontsize=7.2)
    ax.text(
        -1.95,
        -2.72,
        "faint = raw f, bold = memory coordinate\nObject: beta(De) curve, not one exponent",
        fontsize=6.8,
        color=COLORS["muted"],
    )


def panel_archives(ax):
    add_panel_label(ax, "c", "Portability domain")
    style_axes(ax)
    archives = ["US", "GB", "BR", "AUS", "DK"]
    values = [23.7, 14.7, 12.2, 6.4, -5.2]
    colors = [COLORS["green"], COLORS["green"], COLORS["green"], COLORS["gold"], COLORS["red"]]
    ypos = np.arange(len(archives))
    ax.barh(ypos, values, color=colors, alpha=0.92)
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_yticks(ypos, archives)
    ax.invert_yaxis()
    ax.set_xlim(-10, 28)
    ax.set_xlabel("Weighted beta-variance reduction (%)", fontsize=7.2)
    ax.grid(axis="x", color=COLORS["line"], lw=0.6, alpha=0.7)
    for y, v in zip(ypos, values):
        ha = "left" if v >= 0 else "right"
        x = v + 0.8 if v >= 0 else v - 0.8
        ax.text(x, y, f"{v:+.1f}", va="center", ha=ha, fontsize=7.1, color=COLORS["ink"])
    ax.text(
        1.2,
        4.42,
        "DK boundary:\nfailed fifth replication",
        fontsize=6.6,
        color=COLORS["muted"],
        va="center",
    )


def panel_ladder(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "d", "Evidence ladder and claim level")

    rows = [
        ("Benchmark support", "", ""),
        ("Coordinate support", "pass", "cross-fitting + controls"),
        ("Portability", "bounded", "4 positive; DK edge"),
        ("Model benchmark", "use case", "NSE/KGE differ"),
        ("Mechanism boundaries", "", ""),
        ("Artifact floor", "limit", "AR(1) > observed"),
        ("Storage evidence", "mixed", "proxy/state mixed"),
        ("GW / tracer", "unclosed", "causality open"),
    ]
    status_color = {
        "pass": COLORS["green"],
        "bounded": COLORS["gold"],
        "limit": COLORS["red"],
        "use case": COLORS["blue"],
        "mixed": COLORS["gold"],
        "unclosed": COLORS["red"],
    }
    y0 = 0.84
    row_h = 0.091
    for i, (claim, status, note) in enumerate(rows):
        y = y0 - i * row_h
        if status == "":
            ax.text(0.02, y, claim, va="center", ha="left", fontsize=6.8, color=COLORS["muted"], fontweight="bold")
            ax.plot([0.02, 0.98], [y - 0.03, y - 0.03], color=COLORS["line"], lw=0.5)
            continue
        ax.add_patch(Rectangle((0.02, y - 0.055), 0.96, 0.088, facecolor=COLORS["paper"], edgecolor="none"))
        ax.add_patch(
            Rectangle((0.02, y - 0.055), 0.017, 0.088, facecolor=status_color[status], edgecolor="none")
        )
        ax.text(0.055, y, claim, va="center", ha="left", fontsize=6.8, color=COLORS["ink"], fontweight="bold")
        ax.text(
            0.50,
            y,
            status,
            va="center",
            ha="center",
            fontsize=6.3,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=status_color[status], edgecolor="none"),
        )
        ax.text(0.63, y, note, va="center", ha="left", fontsize=6.2, color=COLORS["muted"])

    ax.text(
        0.02,
        0.04,
        "Main claim: bounded diagnostic benchmark\nfor catchment timescale structure.",
        fontsize=7.2,
        color=COLORS["ink"],
        fontweight="bold",
    )


def write_source_data() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("CAMELS-US", "primary positive archive", 23.7, "reports/tables/camels_673_collapse_metrics.csv"),
        ("CAMELS-GB v2", "external replication", 14.7, "reports/tables/camels_gb_v2_replication_collapse_metrics.csv"),
        ("CAMELS-BR v1.2", "third archive", 12.2, "reports/tables/r23_camels_br_third_archive_archive_metrics.csv"),
        ("CAMELS-AUS v2", "dry-continent fourth archive", 6.4, "reports/tables/r25_camels_aus_fourth_archive_archive_metrics.csv"),
        ("CAMELS-DK", "lowland boundary archive", -5.2, "reports/tables/r27_camels_dk_groundwater_storage_validation_archive_metrics.csv"),
    ]
    out = TABLE_DIR / "r30_mechanism_evidence_mother_figure_source.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["archive", "role", "beta_variance_reduction_percent", "source_table"])
        writer.writerows(rows)
    (LATEX_SOURCE_DIR / out.name).write_bytes(out.read_bytes())


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
        }
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.1, 5.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.90, bottom=0.075, wspace=0.23, hspace=0.36)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_object(ax_a)
    panel_coordinate(ax_b)
    panel_archives(ax_c)
    panel_ladder(ax_d)

    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"r30_mechanism_evidence_mother_figure.{ext}"
        fig.savefig(path, dpi=450 if ext == "png" else None)
        (LATEX_FIG_DIR / path.name).write_bytes(path.read_bytes())
    write_source_data()
    plt.close(fig)


if __name__ == "__main__":
    main()
