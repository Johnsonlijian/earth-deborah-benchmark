"""Build the R37 main artifact-floor figure.

The figure replaces the older Fig. 5 with a tighter decision graphic:
coordinate support passes, matched stochastic AR(1) fails the excess claim, and
full-archive analytic multiscale/null families define the process boundary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
LATEX_FIGURES = ROOT / "figures"
PREFIX = "r37_artifact_floor_boundary"
SUBMISSION_NAME = "fig5_artifact_floor_boundary"


def pct(x: float) -> float:
    return 100.0 * float(x)


def load_support() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "r20_support_matched_coordinate_sensitivity.csv")
    keep = data[data["support_contract"].isin(["same_point_central_support"])].copy()
    keep["label"] = keep["archive"] + "\n" + keep["binning"].str.replace("_", "\n")
    return keep


def load_ar1() -> pd.DataFrame:
    return pd.read_csv(TABLES / "r33_matched_ar1_ensemble_summary.csv")


def load_r37() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "r37_gauge_matched_analytic_nulls_summary.csv")
    return data[data["family"] != "observed"].copy()


def plot() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    support = load_support()
    ar1 = load_ar1()
    r37 = load_r37()

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel a: support-matched coordinate sensitivity.
    order = [
        ("CAMELS-US", "log_spaced"),
        ("CAMELS-US", "equal_count"),
        ("CAMELS-GB", "log_spaced"),
        ("CAMELS-GB", "equal_count"),
    ]
    support_rows = [
        support[(support["archive"] == archive) & (support["binning"] == binning)].iloc[0]
        for archive, binning in order
    ]
    x = np.arange(len(support_rows))
    vals = [pct(row["reduction_vs_raw"]) for row in support_rows]
    colors = ["#2E6FAD", "#4F91C9", "#4E9F74", "#79B88D"]
    ax0.bar(x, vals, color=colors, alpha=0.86, width=0.64)
    ax0.axhline(0, color="#777777", lw=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(["US\nlog", "US\nequal", "GB\nlog", "GB\nequal"], fontsize=6.5)
    ax0.set_ylabel("De reduction [%]")
    ax0.set_title("a  Same-point support checks remain positive", loc="left", fontweight="bold")
    ax0.grid(True, axis="y", color="#E4E9EF", lw=0.55)
    for xx, yy in zip(x, vals):
        ax0.text(xx, yy + 1.1, f"{yy:.1f}%", ha="center", va="bottom", fontsize=6)

    # Panel b: stochastic matched AR(1) ensemble.
    ar1 = ar1.sort_values("archive", ascending=False)
    x = np.arange(len(ar1))
    obs = 100.0 * ar1["observed_reduction"].to_numpy(dtype=float)
    null = 100.0 * ar1["null_median_reduction"].to_numpy(dtype=float)
    lo = 100.0 * ar1["null_p05_reduction"].to_numpy(dtype=float)
    hi = 100.0 * ar1["null_p95_reduction"].to_numpy(dtype=float)
    width = 0.34
    ax1.bar(x - width / 2, obs, width=width, color="#2E6FAD", alpha=0.85, label="observed")
    ax1.bar(x + width / 2, null, width=width, color="#C4513F", alpha=0.78, label="matched AR(1)")
    ax1.errorbar(x + width / 2, null, yerr=[null - lo, hi - null], fmt="none", ecolor="#1F2328", lw=0.8, capsize=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ar1["archive"].str.replace("CAMELS-", ""), fontsize=6.5)
    ax1.set_ylabel("De reduction [%]")
    ax1.set_title("b  Stochastic AR(1) exceeds observed", loc="left", fontweight="bold")
    ax1.grid(True, axis="y", color="#E4E9EF", lw=0.55)
    ax1.legend(fontsize=6, loc="upper left")
    for xx, row in zip(x, ar1.itertuples(index=False)):
        ax1.text(xx, 4, f"excess\n{100 * row.observed_minus_null_median:+.1f} pp", ha="center", va="bottom", fontsize=5.8)

    # Panel c: R37 analytic family boundary.
    family_order = [
        "analytic_ar1",
        "analytic_two_timescale_ar",
        "analytic_arfima_d0p15",
        "analytic_arfima_d0p28",
        "analytic_arfima_d0p40",
    ]
    family_labels = ["AR(1)", "two-scale", "ARFIMA\n0.15", "ARFIMA\n0.28", "ARFIMA\n0.40"]
    colors_fam = {
        "analytic_ar1": "#C4513F",
        "analytic_two_timescale_ar": "#2F855A",
        "analytic_arfima_d0p15": "#8E75D8",
        "analytic_arfima_d0p28": "#6C5CE7",
        "analytic_arfima_d0p40": "#4C3FB5",
    }
    width = 0.34
    xpos = np.arange(len(family_order))
    for offset, archive in [(-width / 2, "CAMELS-US"), (width / 2, "CAMELS-GB")]:
        data = r37[r37["archive"] == archive].set_index("family").loc[family_order]
        ax2.bar(
            xpos + offset,
            100.0 * data["null_reduction"].to_numpy(dtype=float),
            width=width,
            color=[colors_fam[f] for f in family_order],
            alpha=0.82 if archive == "CAMELS-US" else 0.48,
            label=archive.replace("CAMELS-", ""),
        )
        obs_val = float(100.0 * data["observed_reduction"].iloc[0])
        ax2.axhline(obs_val, color="#2E6FAD" if archive == "CAMELS-US" else "#4E9F74", lw=0.9, ls="--" if archive == "CAMELS-US" else ":")
    ax2.axhline(0, color="#777777", lw=0.75)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(family_labels, fontsize=6.3)
    ax2.set_ylabel("analytic null reduction [%]")
    ax2.set_title("c  Full-archive analytic nulls define the process boundary", loc="left", fontweight="bold")
    ax2.grid(True, axis="y", color="#E4E9EF", lw=0.55)
    ax2.legend(fontsize=6, loc="upper right")

    # Panel d: interpretation boundary.
    ax3.axis("off")
    rows = [
        ("Coordinate utility", "Supported", "support-matched and shuffled-tau checks"),
        ("Matched AR(1) excess", "Not supported", "US -45.3 pp; GB -51.0 pp"),
        ("Analytic two-scale null", "Boundary", "exceeds observed in US and GB"),
        ("ARFIMA-like relaxation", "Boundary", "d-dependent; can reverse De gain"),
        ("Use case", "Retained", "model timescale diagnostic, not intercomparison"),
    ]
    y = 0.92
    ax3.text(0.00, y, "d  Interpretation boundary", fontsize=8.5, fontweight="bold")
    y -= 0.11
    for label, status, detail in rows:
        color = {
            "Supported": "#2F855A",
            "Not supported": "#C4513F",
            "Boundary": "#D99A24",
            "Retained": "#2F855A",
        }[status]
        ax3.text(0.00, y, label, fontsize=7, fontweight="bold")
        ax3.text(0.55, y, status, fontsize=6.4, fontweight="bold", color=color)
        ax3.text(0.00, y - 0.043, detail, fontsize=6.3, color="#344054")
        y -= 0.155
    ax3.text(
        0.00,
        0.06,
        "Conclusion: the benchmark is useful because its artifact floor is explicit.",
        fontsize=7,
        color="#1F2328",
    )

    for ext in [".pdf", ".svg", ".png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == ".png":
            kwargs["dpi"] = 600
        out = FIGURES / f"{PREFIX}{ext}"
        fig.savefig(out, **kwargs)
        shutil.copy2(out, LATEX_FIGURES / f"{SUBMISSION_NAME}{ext}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
