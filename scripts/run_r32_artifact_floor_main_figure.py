"""Main artifact-floor figure with R33 ensemble support.

The figure promotes the strict matched-AR(1) boundary from supplementary
context to the main figure system. It combines support-matched coordinate
checks, full-archive stochastic-null scores, gauge-level residual distributions
and a compact decision ledger. If an R33 multi-draw ensemble summary is present,
that table is used; otherwise the historical R32 single-draw summary is used.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
LATEX_FIGURES = ROOT / "figures"
OUT = "fig5_artifact_floor_boundary"


def _archive_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {"CAMELS-US": 0, "CAMELS-GB": 1}
    return df.assign(_order=df["archive"].map(order)).sort_values(["_order"]).drop(columns="_order")


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=8)


def draw_decision_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        0.92,
        0.16,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.8,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.10),
    )
    ax.add_patch(box)
    ax.text(x + 0.035, y + 0.08, text, va="center", ha="left", fontsize=8.5, color="#202020")


def load_null_summary() -> tuple[pd.DataFrame, str]:
    r33 = TABLES / "r33_null_calibrated_ensemble_summary.csv"
    if r33.exists():
        return _archive_order(pd.read_csv(r33)), "multi-draw gauge-matched AR(1) ensemble"
    return _archive_order(pd.read_csv(TABLES / "r32_null_calibrated_benchmark_summary.csv")), "single-draw gauge-matched AR(1) diagnostic"


def main() -> None:
    support = pd.read_csv(TABLES / "r20_support_matched_coordinate_sensitivity.csv")
    r32, null_label = load_null_summary()
    scores = _archive_order(pd.read_csv(TABLES / "r31_gauge_level_ar1_boundary_gauge_scores.csv"))

    colors = {
        "CAMELS-US": "#31688e",
        "CAMELS-GB": "#35b779",
        "support": "#7ad151",
        "null": "#cc4778",
        "fail": "#b33641",
        "pass": "#2a788e",
        "use": "#5ec962",
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(11.2, 7.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.92, bottom=0.10, wspace=0.27, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel a: same-point support contract remains positive.
    keep = support[support["support_contract"].eq("same_point_central_support")].copy()
    keep["label"] = keep["archive"].str.replace("CAMELS-", "", regex=False) + "\n" + keep["binning"].str.replace("_", "-")
    x = np.arange(len(keep))
    bar_colors = [colors.get(a, "#33658a") for a in keep["archive"]]
    ax_a.bar(x, keep["reduction_vs_raw"] * 100.0, color=bar_colors, width=0.72)
    ax_a.axhline(0, color="#222222", lw=0.8)
    ax_a.set_xticks(x, keep["label"], fontsize=8)
    ax_a.set_ylabel("De reduction in beta-variance (%)")
    ax_a.set_ylim(0, max(34, float((keep["reduction_vs_raw"] * 100).max()) + 5))
    for xi, val in zip(x, keep["reduction_vs_raw"] * 100.0):
        ax_a.text(xi, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    panel_label(ax_a, "a", "Coordinate-support gate passes on the same point set")

    # Panel b: observed score is positive but lower than the matched AR(1) null.
    width = 0.32
    xb = np.arange(len(r32))
    obs = r32["observed_de_reduction_percent"].to_numpy()
    null = r32["matched_ar1_null_reduction_percent"].to_numpy()
    ax_b.bar(xb - width / 2, obs, width=width, color=colors["support"], label="Observed archive")
    ax_b.bar(xb + width / 2, null, width=width, color=colors["null"], label="Gauge-matched AR(1)")
    if {"matched_ar1_null_p05_percent", "matched_ar1_null_p95_percent"}.issubset(r32.columns):
        yerr = np.vstack(
            [
                null - r32["matched_ar1_null_p05_percent"].to_numpy(),
                r32["matched_ar1_null_p95_percent"].to_numpy() - null,
            ]
        )
        ax_b.errorbar(
            xb + width / 2,
            null,
            yerr=yerr,
            fmt="none",
            ecolor="#5b1234",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
    ax_b.axhline(0, color="#222222", lw=0.8)
    for xi, o, n, ex in zip(xb, obs, null, r32["null_adjusted_excess_percent"]):
        ax_b.plot([xi - width / 2, xi + width / 2], [o, n], color="#555555", lw=0.9, alpha=0.7)
        ax_b.text(xi, max(o, n) + 3.0, f"null-adjusted {ex:.1f}%", ha="center", fontsize=8, color=colors["fail"])
    ax_b.set_xticks(xb, r32["archive"].str.replace("CAMELS-", "", regex=False))
    ax_b.set_ylabel("Dispersion reduction (%)")
    ax_b.set_ylim(-8, 88)
    ax_b.text(xb[0] - width / 2, obs[0] + 2.0, "observed", ha="center", fontsize=7.5, color="#2f7d32")
    ax_b.text(xb[0] + width / 2, null[0] - 5.0, "matched AR(1)", ha="center", fontsize=7.5, color="#8f2450")
    panel_label(ax_b, "b", f"Matched-AR(1) null sets a higher artifact floor")

    # Panel c: gauge-level residual distribution after subtracting each gauge's AR(1) reference.
    clipped = scores.copy()
    clipped["score_percent"] = np.clip(clipped["ar1_residual_de_reduction"] * 100.0, -250, 120)
    data = [clipped[clipped["archive"].eq(a)]["score_percent"].to_numpy() for a in r32["archive"]]
    parts = ax_c.violinplot(data, positions=np.arange(len(data)), widths=0.72, showmeans=False, showmedians=False)
    for body, archive in zip(parts["bodies"], r32["archive"]):
        body.set_facecolor(colors.get(archive, "#33658a"))
        body.set_alpha(0.32)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.6)
    rng = np.random.default_rng(532)
    for idx, archive in enumerate(r32["archive"]):
        vals = clipped[clipped["archive"].eq(archive)]["score_percent"].to_numpy()
        sample = rng.choice(vals, size=min(180, len(vals)), replace=False)
        jitter = rng.normal(idx, 0.045, size=len(sample))
        ax_c.scatter(jitter, sample, s=7, alpha=0.18, color=colors.get(archive, "#33658a"), linewidth=0)
        med = float(r32.loc[r32["archive"].eq(archive), "per_gauge_ar1_residual_median_percent"].iloc[0])
        pos = float(r32.loc[r32["archive"].eq(archive), "per_gauge_positive_fraction_percent"].iloc[0])
        ax_c.plot([idx - 0.26, idx + 0.26], [med, med], color="#111111", lw=2.0)
        ax_c.text(idx, 112, f"median {med:.1f}%\npositive {pos:.1f}%", ha="center", va="top", fontsize=8)
    ax_c.axhline(0, color="#222222", lw=0.8, ls="--")
    ax_c.set_xticks(np.arange(len(data)), r32["archive"].str.replace("CAMELS-", "", regex=False))
    ax_c.set_ylabel("Per-gauge residual De benefit (%)\n(clipped for display)")
    ax_c.set_ylim(-255, 125)
    panel_label(ax_c, "c", "Gauge-level residual tightening is mostly negative")

    # Panel d: decision ledger.
    ax_d.axis("off")
    panel_label(ax_d, "d", "Interpretation ledger")
    draw_decision_box(ax_d, (0.02, 0.75), "PASS: measured De reduces dispersion under support-matched bins", colors["pass"])
    draw_decision_box(ax_d, (0.02, 0.55), "PASS: shuffled-memory and proxy coordinates do not reproduce the effect", colors["pass"])
    draw_decision_box(ax_d, (0.02, 0.35), "FAIL FOR MECHANISM: matched AR(1) null exceeds observed reduction", colors["fail"])
    draw_decision_box(ax_d, (0.02, 0.15), "USE CASE: compare model timescale structure alongside NSE/KGE", colors["use"])
    ax_d.text(
        0.02,
        0.02,
        "Conclusion: benchmark utility is retained; storage-causality and excess-over-AR(1) claims are rejected.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#202020",
        wrap=True,
    )

    fig.suptitle("Support, artifact floor and null-adjusted interpretation of the memory-normalized benchmark", fontsize=12.5, fontweight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        out = FIGURES / f"{OUT}.{ext}"
        fig.savefig(out, dpi=320 if ext == "png" else None)
        shutil.copy2(out, LATEX_FIGURES / f"{OUT}.{ext}")
    plt.close(fig)
    print(f"Wrote and copied {OUT}.pdf/.png/.svg")


if __name__ == "__main__":
    main()
