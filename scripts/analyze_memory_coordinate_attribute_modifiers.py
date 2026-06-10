"""Attribute-modification audit for memory-normalized spectral alignment.

This R15 analysis asks a reviewer-driven question: which catchments move
closer to the archive-level beta curve after frequency is mapped to
De = tau_acf * f, and do simple catchment attributes explain that change?

The gauge-level score is exploratory and does not replace the manuscript's
headline weighted beta-variance metric. It is designed to expose boundary
conditions and help decide whether a new main figure should focus on effect
modification rather than another aggregate bar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"

CURVES_CSV = TABLES / "camels_673_beta_curve_points.csv"
ATTR_CSV = TABLES / "camels_673_attributed_diagnostics.csv"
GAUGE_OUT = TABLES / "camels_673_memory_coordinate_effect_by_gauge.csv"
CORR_OUT = TABLES / "camels_673_memory_coordinate_attribute_modifiers.csv"
FIG_OUT = FIGURES / "camels_673_memory_coordinate_attribute_modifiers"
NOTE_OUT = NOTES / "camels_673_memory_coordinate_attribute_modifiers.md"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)

PALETTE = {
    "ink": "#202124",
    "muted": "#6B7280",
    "grid": "#D6D9DE",
    "blue": "#3B6EA8",
    "cyan": "#5FB4A2",
    "red": "#C75146",
    "gold": "#D49B38",
    "purple": "#7B67A8",
}


def _gauge_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(8)


def station_bin_residuals(
    curves: pd.DataFrame,
    axis_col: str,
    n_bins: int = 24,
    min_units: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = curves[["gauge_id", axis_col, "beta"]].dropna().copy()
    work = work[(work[axis_col] > 0) & np.isfinite(work[axis_col]) & np.isfinite(work["beta"])]
    log_axis = np.log10(work[axis_col])
    edges = np.linspace(log_axis.min(), log_axis.max(), n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work["bin_id"] = pd.cut(log_axis, bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)

    station_bins = (
        work.groupby(["gauge_id", "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"))
        .assign(axis=axis_col)
    )
    station_bins["axis_value"] = station_bins["bin_id"].map(
        {i: 10 ** labels[i] for i in range(len(labels))}
    )

    bin_summary = (
        station_bins.groupby("bin_id", as_index=False)
        .agg(
            n_units=("gauge_id", "nunique"),
            beta_ref=("beta_median", "median"),
            beta_var=("beta_median", "var"),
            axis_value=("axis_value", "first"),
        )
    )
    usable = bin_summary[bin_summary["n_units"] >= min_units].copy()
    merged = station_bins.merge(usable[["bin_id", "beta_ref"]], on="bin_id", how="inner")
    merged["residual_sq"] = (merged["beta_median"] - merged["beta_ref"]) ** 2
    scores = (
        merged.groupby("gauge_id", as_index=False)
        .agg(
            mean_residual_sq=("residual_sq", "mean"),
            median_abs_residual=("residual_sq", lambda x: float(np.sqrt(np.median(x)))),
            n_bins=("bin_id", "nunique"),
        )
        .rename(
            columns={
                "mean_residual_sq": f"{axis_col}_mean_residual_sq",
                "median_abs_residual": f"{axis_col}_median_abs_residual",
                "n_bins": f"{axis_col}_n_bins",
            }
        )
    )
    return scores, usable.assign(axis=axis_col)


def compute_gauge_scores(curves: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    raw_scores, _ = station_bin_residuals(curves, "frequency_cpd")
    de_scores, _ = station_bin_residuals(curves, "de")
    score = raw_scores.merge(de_scores, on="gauge_id", how="inner")
    score = score[
        (score["frequency_cpd_n_bins"] >= 8)
        & (score["de_n_bins"] >= 6)
        & (score["frequency_cpd_mean_residual_sq"] > 0)
    ].copy()
    score["de_residual_reduction"] = 1.0 - (
        score["de_mean_residual_sq"] / score["frequency_cpd_mean_residual_sq"]
    )
    score["de_abs_residual_reduction"] = 1.0 - (
        score["de_median_abs_residual"] / score["frequency_cpd_median_abs_residual"]
    )

    attrs = attrs.copy()
    attrs["gauge_id"] = _gauge_str(attrs["gauge_id"])
    attr_cols = [
        "gauge_id",
        "tau_acf_days",
        "tau_recession_days",
        "drainage_area_km2",
        "baseflow_index_x",
        "aridity",
        "frac_snow",
        "elev_mean",
        "slope_mean",
        "runoff_ratio",
        "p_seasonality",
        "phase_reject_zero",
        "phase_reject_resolution",
    ]
    return score.merge(attrs[attr_cols], on="gauge_id", how="left")


def attribute_correlations(gauge_scores: pd.DataFrame) -> pd.DataFrame:
    work = gauge_scores.copy()
    work["log10_area"] = np.log10(work["drainage_area_km2"].where(work["drainage_area_km2"] > 0))
    work["log10_tau_acf"] = np.log10(work["tau_acf_days"].where(work["tau_acf_days"] > 0))
    candidates = {
        "log10_tau_acf": "log10 tau_acf",
        "baseflow_index_x": "BFI",
        "aridity": "aridity",
        "frac_snow": "snow fraction",
        "log10_area": "log10 area",
        "elev_mean": "elevation",
        "slope_mean": "slope",
        "runoff_ratio": "runoff ratio",
        "p_seasonality": "precip seasonality",
    }
    rows: list[dict[str, object]] = []
    for col, label in candidates.items():
        sub = work[["de_residual_reduction", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 25:
            continue
        rho, p_value = spearmanr(sub[col], sub["de_residual_reduction"])
        rows.append(
            {
                "attribute": col,
                "label": label,
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "abs_rho": abs(float(rho)),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    c: str,
    xlabel: str,
    clabel: str,
    y_clip: tuple[float, float],
) -> None:
    sub = df[[x, y, c]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    sub["_plot_y"] = (100 * sub[y]).clip(*y_clip)
    sc = ax.scatter(
        sub[x],
        sub["_plot_y"],
        c=sub[c],
        s=13,
        alpha=0.78,
        linewidths=0,
        cmap="viridis",
    )
    ax.axhline(0, color=PALETTE["muted"], lw=0.7, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("gauge-level residual reduction [%]")
    ax.set_ylim(y_clip[0] - 5, y_clip[1] + 5)
    ax.grid(True, color=PALETTE["grid"], lw=0.35, alpha=0.8)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(clabel)


def plot(gauge_scores: pd.DataFrame, corr: pd.DataFrame) -> None:
    plot_df = gauge_scores.copy()
    plot_df["log10_tau_acf"] = np.log10(plot_df["tau_acf_days"].where(plot_df["tau_acf_days"] > 0))
    plot_df["log10_area"] = np.log10(
        plot_df["drainage_area_km2"].where(plot_df["drainage_area_km2"] > 0)
    )

    fig = plt.figure(figsize=(7.6, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[0.92, 1.0])
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[:, 2])

    vals = 100 * plot_df["de_residual_reduction"].dropna()
    y_clip = (-250.0, 120.0)
    hist_clip = (-250.0, 120.0)
    bins = np.linspace(hist_clip[0], hist_clip[1], 38)
    ax_a.hist(vals.clip(*hist_clip), bins=bins, color=PALETTE["blue"], alpha=0.86)
    ax_a.axvline(float(vals.median()), color=PALETTE["red"], lw=1.6, label=f"median {vals.median():.1f}%")
    ax_a.axvline(0, color=PALETTE["muted"], lw=0.8, ls="--")
    ax_a.set_xlabel("gauge-level residual reduction after De mapping [%]")
    ax_a.set_ylabel("number of gauges")
    ax_a.legend(loc="upper right")
    ax_a.grid(True, axis="y", color=PALETTE["grid"], lw=0.35, alpha=0.8)
    ax_a.set_title("a  Memory coordinate benefit is heterogeneous", loc="left", fontweight="bold")
    ax_a.text(
        0.01,
        0.94,
        "display clipped at -250% and 120%; source data retain full values",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=PALETTE["muted"],
    )

    _scatter(
        ax_b,
        plot_df,
        "baseflow_index_x",
        "de_residual_reduction",
        "log10_tau_acf",
        "BFI",
        "log10 tau_acf",
        y_clip,
    )
    ax_b.set_title("b  Storage proxy", loc="left", fontweight="bold")

    _scatter(
        ax_c,
        plot_df,
        "aridity",
        "de_residual_reduction",
        "frac_snow",
        "aridity",
        "snow fraction",
        y_clip,
    )
    ax_c.set_title("c  Hydroclimate context", loc="left", fontweight="bold")

    top = corr.sort_values("spearman_rho").copy()
    colors = [PALETTE["cyan"] if v >= 0 else PALETTE["red"] for v in top["spearman_rho"]]
    ax_d.barh(top["label"], top["spearman_rho"], color=colors, alpha=0.88)
    ax_d.axvline(0, color=PALETTE["muted"], lw=0.8)
    ax_d.set_xlabel("Spearman rho")
    ax_d.set_title("d  Attribute modifiers", loc="left", fontweight="bold")
    ax_d.grid(True, axis="x", color=PALETTE["grid"], lw=0.35, alpha=0.8)
    ax_d.set_xlim(-0.20, 0.34)
    for _, row in top.iterrows():
        star = "*" if row["p_value"] < 0.05 else ""
        ax_d.text(
            0.33,
            row["label"],
            f"n={int(row['n'])}{star}",
            va="center",
            ha="right",
            fontsize=6,
            color=PALETTE["muted"],
        )

    fig.suptitle(
        "R15 exploratory audit: where does memory-normalized alignment help?",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{FIG_OUT}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIG_OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{FIG_OUT}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_note(gauge_scores: pd.DataFrame, corr: pd.DataFrame) -> None:
    vals = gauge_scores["de_residual_reduction"].dropna()
    positive = float((vals > 0).mean())
    med = float(vals.median())
    q25, q75 = np.nanpercentile(vals, [25, 75])
    top = corr.sort_values("abs_rho", ascending=False).head(5)
    lines = [
        "# R15 Attribute-Modification Audit",
        "",
        "Date: 2026-06-03",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: the memory coordinate has an aggregate benefit, but gauge-level benefit is heterogeneous and should be tested against catchment attributes before any global framing.",
        "- Archetype: quantitative grid with one distribution panel, two attribute scatter panels and one modifier-ranking panel.",
        "- Evidence boundary: this gauge-level residual score is exploratory and does not replace the manuscript's weighted beta-variance metric.",
        "",
        "## Gauge-Level Score",
        "",
        "For each coordinate axis, station-level median beta values were binned in log-spaced coordinate bins and compared with the archive-level median beta curve. The exploratory score is `1 - mean_squared_residual_De / mean_squared_residual_raw_frequency` for gauges with sufficient raw-frequency and De bins.",
        "",
        "## Result",
        "",
        f"- Gauges scored: {len(vals)}",
        f"- Median gauge-level residual reduction: {med:.1%}",
        f"- Interquartile range: {q25:.1%} to {q75:.1%}",
        f"- Fraction of gauges with positive residual reduction: {positive:.1%}",
        "",
        "Top attribute associations with gauge-level residual reduction:",
        "",
        "| Attribute | Spearman rho | p value | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['label']} | {row['spearman_rho']:.3f} | {row['p_value']:.3g} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This result should be treated as a reviewer-facing heterogeneity audit. If used in the manuscript, it should support a boundary statement: the aggregate De gain is real under the current metric, but the paper still needs explicit analysis of where the coordinate works best or fails. It should not be used as proof of universal hydrological scaling.",
            "",
            "## Outputs",
            "",
            f"- `{GAUGE_OUT.relative_to(ROOT)}`",
            f"- `{CORR_OUT.relative_to(ROOT)}`",
            f"- `{FIG_OUT.relative_to(ROOT)}.png/svg/pdf`",
        ]
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    curves = pd.read_csv(CURVES_CSV, dtype={"gauge_id": str})
    curves["gauge_id"] = _gauge_str(curves["gauge_id"])
    attrs = pd.read_csv(ATTR_CSV)

    gauge_scores = compute_gauge_scores(curves, attrs)
    corr = attribute_correlations(gauge_scores)

    TABLES.mkdir(parents=True, exist_ok=True)
    gauge_scores.to_csv(GAUGE_OUT, index=False)
    corr.to_csv(CORR_OUT, index=False)
    plot(gauge_scores, corr)
    write_note(gauge_scores, corr)

    print(f"Wrote {GAUGE_OUT.relative_to(ROOT)}")
    print(f"Wrote {CORR_OUT.relative_to(ROOT)}")
    print(f"Wrote {FIG_OUT.relative_to(ROOT)}.png/svg/pdf")
    print(f"Wrote {NOTE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
