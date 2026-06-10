"""Build an R08 synthesis figure for the storage-memory manuscript.

This candidate figure is a visual argument map, not a replacement for the
existing main figures. It combines a code-drawn mechanism schematic with the
strongest quantitative gates: cross-archive collapse, alternative
normalizations, storage proxy controls and boundary tests. No generative image
tools are used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    }
)


COL = {
    "ink": "#222831",
    "muted": "#667085",
    "grid": "#E7EAF0",
    "water": "#2B7BB9",
    "water_light": "#D9ECF7",
    "storage": "#2F855A",
    "storage_light": "#D8EEE0",
    "signal": "#C93434",
    "signal_light": "#F7D6D6",
    "gb": "#6C5CE7",
    "event": "#9B4762",
    "gold": "#D79A20",
    "noise": "#7C8794",
}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
        color=COL["ink"],
    )


def save_pub(fig: plt.Figure, stem: str) -> None:
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(FIGURES / f"{stem}{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/svg/pdf")


def load_cross_archive() -> pd.DataFrame:
    comparison = pd.read_csv(TABLES / "camels_us_gb_replication_comparison.csv")
    us_metrics = pd.read_csv(TABLES / "camels_673_collapse_metrics.csv")
    gb_boot = pd.read_csv(TABLES / "camels_gb_v2_collapse_bootstrap_summary.csv")

    us_ci = us_metrics[us_metrics["axis"] == "variance_reduction"].iloc[0]
    rows = []
    for row in comparison.to_dict("records"):
        if row["archive"] == "CAMELS-US":
            rows.append(
                {
                    **row,
                    "ci95_low": float(us_ci["ci95_low"]),
                    "ci95_high": float(us_ci["ci95_high"]),
                    "ci_note": "station bootstrap, n=150",
                }
            )
        else:
            boot = gb_boot.iloc[0]
            rows.append(
                {
                    **row,
                    "ci95_low": float(boot["ci95_low"]),
                    "ci95_high": float(boot["ci95_high"]),
                    "ci_note": "station bootstrap, n=150",
                }
            )
    return pd.DataFrame(rows)


def draw_mechanism(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a")
    ax.text(
        0.04,
        0.92,
        "Storage-memory mechanism",
        fontsize=8.5,
        fontweight="bold",
        color=COL["ink"],
        va="top",
    )
    x = np.linspace(0.04, 0.27, 220)
    y = 0.58 + 0.055 * np.sin(65 * x) + 0.020 * np.sin(180 * x)
    ax.plot(x, y, lw=1.6, color=COL["water"])
    ax.text(0.055, 0.73, "precipitation\nforcing", color=COL["water"], fontsize=6.3, ha="left")

    box = patches.FancyBboxPatch(
        (0.36, 0.50),
        0.24,
        0.22,
        boxstyle="round,pad=0.03,rounding_size=0.035",
        fc=COL["storage_light"],
        ec=COL["storage"],
        lw=1.4,
    )
    ax.add_patch(box)
    ax.text(0.48, 0.64, "storage\nfilter", ha="center", va="center", color=COL["storage"], fontsize=8.0, fontweight="bold")
    ax.text(0.48, 0.535, "memory $\\tau_{acf}$", ha="center", va="center", color=COL["storage"], fontsize=6.2)
    ax.annotate("", xy=(0.36, 0.58), xytext=(0.28, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5, color=COL["water"]))

    x2 = np.linspace(0.68, 0.93, 220)
    y2 = 0.58 + 0.033 * np.sin(23 * x2) + 0.014 * np.sin(55 * x2)
    ax.plot(x2, y2, lw=1.8, color=COL["signal"])
    ax.annotate("", xy=(0.68, 0.58), xytext=(0.60, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5, color=COL["water"]))
    ax.text(0.80, 0.72, "discharge\noutput $Q(t)$", color=COL["signal"], fontsize=6.3, ha="center")

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.36, 0.18),
            0.24,
            0.15,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            fc=COL["signal_light"],
            ec=COL["signal"],
            lw=1.1,
        )
    )
    ax.text(0.48, 0.255, "$De = \\tau_{acf} f$", ha="center", va="center", color=COL["signal"], fontsize=9.5, fontweight="bold")
    ax.annotate("", xy=(0.48, 0.50), xytext=(0.48, 0.33), arrowprops=dict(arrowstyle="->", lw=1.3, color=COL["signal"]))

    ax.text(0.04, 0.08, "Claim boundary:", fontsize=6.4, fontweight="bold", color=COL["ink"])
    ax.text(
        0.04,
        0.025,
        "output-memory coordinate; not a unique aquifer constant",
        fontsize=6.2,
        color=COL["muted"],
    )


def draw_cross_archive(ax: plt.Axes, cross: pd.DataFrame) -> None:
    panel_label(ax, "b")
    rows = cross.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows))
    colors = [COL["gb"] if "GB" in name else COL["water"] for name in rows["archive"]]
    vals = rows["de_variance_reduction_vs_raw"].to_numpy(dtype=float) * 100
    lows = rows["ci95_low"].to_numpy(dtype=float) * 100
    highs = rows["ci95_high"].to_numpy(dtype=float) * 100
    ax.hlines(y, lows, highs, color=colors, lw=4, alpha=0.22)
    ax.scatter(vals, y, s=68, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for value, yy in zip(vals, y):
        ax.text(value + 1.0, yy, f"{value:.1f}%", va="center", fontsize=7, color=COL["ink"])
    ax.axvline(0, color="#B6BDC7", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["archive"])
    ax.set_xlabel("weighted beta-variance reduction vs raw frequency")
    ax.set_title("Effect direction replicates; magnitude differs", loc="left", fontsize=8)
    ax.set_xlim(-2, max(highs) + 8)
    ax.grid(True, axis="x", color=COL["grid"], lw=0.6)


def draw_alternatives(ax: plt.Axes) -> None:
    panel_label(ax, "c")
    alt = pd.read_csv(TABLES / "camels_673_alternative_normalization_metrics.csv")
    keep_order = [
        "tau_acf",
        "tau_acf_recession_subset",
        "tau_recession_matched",
        "area_rank_tau",
        "constant_tau",
        "bfi_rank_tau",
    ]
    labels = {
        "tau_acf": "tau_acf",
        "tau_acf_recession_subset": "tau_acf\nmatched",
        "tau_recession_matched": "recession\ntime",
        "area_rank_tau": "area\nrank",
        "constant_tau": "constant\ntau",
        "bfi_rank_tau": "BFI\nrank",
    }
    work = alt.set_index("method").loc[keep_order].reset_index()
    x = np.arange(len(work))
    vals = work["variance_reduction_vs_matched_raw"].to_numpy(dtype=float) * 100
    colors = [COL["signal"] if m.startswith("tau_acf") else COL["noise"] for m in work["method"]]
    ax.axhline(0, color="#9AA0A6", lw=0.8)
    for xx, vv, cc in zip(x, vals, colors):
        ax.vlines(xx, 0, vv, color=cc, lw=2.2, alpha=0.65)
        ax.scatter(xx, vv, s=42, color=cc, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(xx, vv + (1.3 if vv >= 0 else -2.5), f"{vv:.1f}", ha="center", va="center", fontsize=6, color=COL["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in work["method"]])
    ax.set_ylabel("variance reduction [%]")
    ax.set_title("The coordinate is not arbitrary rescaling", loc="left", fontsize=8)
    ax.set_ylim(-7, 28)
    ax.grid(True, axis="y", color=COL["grid"], lw=0.6)


def draw_mechanism_controls(ax: plt.Axes, cross: pd.DataFrame) -> None:
    panel_label(ax, "d")
    names = ["CAMELS-US", "CAMELS-GB v2"]
    x = np.arange(len(names))
    sub = cross.set_index("archive").loc[names]
    bfi = sub["rho_tau_bfi"].to_numpy(dtype=float)
    area = sub["rho_tau_log_area"].to_numpy(dtype=float)
    ax.plot(x, bfi, color=COL["storage"], lw=1.5, marker="o", ms=6, label="baseflow index")
    ax.plot(x, area, color=COL["noise"], lw=1.5, marker="o", ms=6, label="log area")
    for xx, yy in zip(x, bfi):
        ax.text(xx, yy + 0.045, f"{yy:.2f}", ha="center", fontsize=6, color=COL["storage"])
    for xx, yy in zip(x, area):
        ax.text(xx, yy + 0.045, f"{yy:.2f}", ha="center", fontsize=6, color=COL["noise"])
    ax.axhline(0, color="#B6BDC7", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["US", "GB"])
    ax.set_ylabel("Spearman rho with $\\tau_{acf}$")
    ax.set_title("Storage proxy, not basin area, tracks memory", loc="left", fontsize=8)
    ax.set_ylim(-0.08, 0.92)
    ax.legend(loc="upper left", handlelength=1.5)
    ax.grid(True, axis="y", color=COL["grid"], lw=0.6)


def draw_boundary(ax: plt.Axes) -> None:
    panel_label(ax, "e")
    rng = np.random.default_rng(20260601)
    river = pd.read_csv(TABLES / "camels_673_attributed_diagnostics.csv")["mean_beta_de_window"].dropna()
    eq = pd.read_csv(TABLES / "earthquake_regions_2000_20260528_summary.csv")["mean_beta_de_window"].dropna()
    land = pd.read_csv(TABLES / "landslide_glc_trigger_nulls_summary.csv")["observed_mean_beta_de_window"].dropna()
    noise = pd.read_csv(TABLES / "control_random_psd_summary.csv")
    noise = noise.loc[noise["kind"] == "colored_noise", "mean_beta_de_window"].dropna()
    groups = [river, eq, land, noise]
    labels = ["river", "earthquake", "landslide", "colored\nnoise"]
    colors = [COL["water"], COL["event"], "#C9897B", COL["gold"]]
    for i, (series, color) in enumerate(zip(groups, colors)):
        vals = series.to_numpy(dtype=float)
        if len(vals) > 220:
            vals = rng.choice(vals, size=220, replace=False)
        jitter = rng.normal(i, 0.045, size=len(vals))
        ax.scatter(jitter, vals, s=9, color=color, alpha=0.28, edgecolor="none")
        q1, med, q3 = np.nanpercentile(series, [25, 50, 75])
        lo, hi = np.nanpercentile(series, [5, 95])
        ax.vlines(i, lo, hi, color=COL["ink"], lw=1.0)
        ax.add_patch(patches.Rectangle((i - 0.16, q1), 0.32, q3 - q1, fc=color, ec=COL["ink"], alpha=0.35, lw=0.8))
        ax.hlines(med, i - 0.16, i + 0.16, color=COL["ink"], lw=1.4)
    ax.axhline(1.0, color="#AAB1BB", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean beta in 0.5 <= De <= 2")
    ax.set_title("Boundary tests prevent overgeneralization", loc="left", fontsize=8)
    ax.set_ylim(-0.2, 3.8)
    ax.grid(True, axis="y", color=COL["grid"], lw=0.6)


def build_summary_table(cross: pd.DataFrame) -> None:
    rows = []
    for row in cross.to_dict("records"):
        rows.append(
            {
                "component": row["archive"],
                "metric": "de_variance_reduction_vs_raw",
                "value": row["de_variance_reduction_vs_raw"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "note": row["ci_note"],
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "r08_synthesis_figure_metrics.csv", index=False)


def main() -> None:
    cross = load_cross_archive()
    build_summary_table(cross)
    fig = plt.figure(figsize=(7.2, 6.25))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.20, 0.95, 1.05], wspace=0.72, hspace=0.55)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:])
    ax_e = fig.add_subplot(gs[2, :])

    draw_mechanism(ax_a)
    draw_cross_archive(ax_b, cross)
    draw_alternatives(ax_c)
    draw_mechanism_controls(ax_d, cross)
    draw_boundary(ax_e)

    fig.suptitle(
        "Storage-memory coordinates organize river-discharge spectra with bounded portability",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=COL["ink"],
    )
    save_pub(fig, "nature_fig1_storage_memory_argument_v3")


if __name__ == "__main__":
    main()
