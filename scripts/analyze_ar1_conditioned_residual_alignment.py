"""R19 AR(1)-conditioned residual alignment diagnostic.

This analysis addresses the strongest Nature-family reviewer objection after
R18: tau_acf and PSD-derived beta are mathematically coupled. The script uses
the existing AR(1) Fourier-pair simulations as a reference curve in De space,
subtracts that expected beta(De) relation from CAMELS beta curves, and then
tests whether the remaining residual beta curves still tighten on the De axis.

The result is a reviewer-facing diagnostic, not causal proof. A positive
residual reduction means that CAMELS retains structured alignment beyond the
median AR(1) reference curve under this operational test; a weak or negative
result would require downgrading the Nature-family claim.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"

ARCHIVES = [
    ("CAMELS-US", TABLES / "camels_673_beta_curve_points.csv", TABLES / "camels_673_attributed_diagnostics.csv"),
    ("CAMELS-GB", TABLES / "camels_gb_v2_replication_beta_curve_points.csv", None),
]

AR1_CURVES = TABLES / "fourier_pair_circularity_baseline_curves.csv"

RESIDUAL_CURVES_OUT = TABLES / "r19_ar1_conditioned_residual_curves.csv"
METRICS_OUT = TABLES / "r19_ar1_conditioned_residual_metrics.csv"
GAUGE_OUT = TABLES / "r19_ar1_conditioned_residual_by_gauge.csv"
ATTR_OUT = TABLES / "r19_ar1_conditioned_residual_attribute_modifiers.csv"
FIG_OUT = FIGURES / "r19_ar1_conditioned_residual_alignment"
NOTE_OUT = NOTES / "r19_ar1_conditioned_residual_alignment.md"


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
    "blue": "#386FA4",
    "green": "#4B9B72",
    "red": "#C44E52",
    "gold": "#D09A32",
    "purple": "#7A6DAF",
    "teal": "#4BA3A2",
}


def _gauge_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(8)


def build_ar1_reference(n_bins: int = 90) -> pd.DataFrame:
    curves = pd.read_csv(AR1_CURVES)
    ar1 = curves[curves["scenario_group"] == "ar1_fourier_pair"].copy()
    ar1 = ar1[(ar1["de"] > 0) & np.isfinite(ar1["de"]) & np.isfinite(ar1["beta"])]
    ar1["log10_de"] = np.log10(ar1["de"].astype(float))
    lo, hi = np.nanquantile(ar1["log10_de"], [0.005, 0.995])
    edges = np.linspace(float(lo), float(hi), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ar1["bin_id"] = pd.cut(ar1["log10_de"], bins=edges, labels=False, include_lowest=True)
    ref = (
        ar1.dropna(subset=["bin_id"])
        .assign(bin_id=lambda d: d["bin_id"].astype(int))
        .groupby("bin_id", as_index=False)
        .agg(
            beta_ar1_ref=("beta", "median"),
            beta_ar1_q25=("beta", lambda x: float(np.nanquantile(x, 0.25))),
            beta_ar1_q75=("beta", lambda x: float(np.nanquantile(x, 0.75))),
            n_points=("beta", "size"),
        )
    )
    ref["log10_de"] = ref["bin_id"].map({i: centers[i] for i in range(len(centers))})
    ref["de"] = 10 ** ref["log10_de"]
    ref = ref.sort_values("log10_de")
    return ref


def add_ar1_residual(curves: pd.DataFrame, archive: str, ref: pd.DataFrame) -> pd.DataFrame:
    work = curves.copy()
    work["gauge_id"] = _gauge_str(work["gauge_id"])
    work = work[(work["de"] > 0) & (work["frequency_cpd"] > 0)].copy()
    work["log10_de"] = np.log10(work["de"].astype(float))
    x = ref["log10_de"].to_numpy(dtype=float)
    y = ref["beta_ar1_ref"].to_numpy(dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    work = work[(work["log10_de"] >= lo) & (work["log10_de"] <= hi)].copy()
    work["beta_ar1_ref"] = np.interp(work["log10_de"].to_numpy(dtype=float), x, y)
    work["beta_residual_ar1"] = work["beta"].astype(float) - work["beta_ar1_ref"]
    work["archive"] = archive
    return work[
        [
            "archive",
            "gauge_id",
            "tau_acf_days",
            "frequency",
            "frequency_cpd",
            "de",
            "beta",
            "beta_ar1_ref",
            "beta_residual_ar1",
        ]
    ]


def station_bin_medians(
    curves: pd.DataFrame,
    axis_col: str,
    beta_col: str,
    n_bins: int = 24,
) -> pd.DataFrame:
    work = curves[["gauge_id", axis_col, beta_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[(work[axis_col] > 0) & np.isfinite(work[axis_col])]
    log_axis = np.log10(work[axis_col].astype(float))
    lo, hi = np.nanquantile(log_axis, [0.025, 0.975])
    edges = np.linspace(float(lo), float(hi), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    work = work[(log_axis >= lo) & (log_axis <= hi)].copy()
    work["bin_id"] = pd.cut(np.log10(work[axis_col].astype(float)), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    out = (
        work.groupby(["gauge_id", "bin_id"], as_index=False)
        .agg(beta_median=(beta_col, "median"), n_points=(beta_col, "size"))
    )
    out["axis_value"] = out["bin_id"].map({i: 10 ** centers[i] for i in range(len(centers))})
    return out


def dispersion_metric(
    curves: pd.DataFrame,
    axis_col: str,
    beta_col: str,
    min_units: int,
) -> tuple[float, pd.DataFrame]:
    binned = station_bin_medians(curves, axis_col, beta_col)
    stats = (
        binned.groupby("bin_id", as_index=False)
        .agg(
            axis_value=("axis_value", "first"),
            n_units=("gauge_id", "nunique"),
            beta_var=("beta_median", "var"),
            beta_median=("beta_median", "median"),
            beta_iqr=("beta_median", lambda x: float(np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25))),
        )
        .dropna(subset=["beta_var"])
    )
    usable = stats[stats["n_units"] >= min_units].copy()
    if usable.empty:
        return float("nan"), stats
    return float(np.average(usable["beta_var"], weights=usable["n_units"])), stats


def archive_metrics(curves: pd.DataFrame, archive: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_units = 50 if archive == "CAMELS-US" else 45
    rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    for beta_col, metric_family in [
        ("beta", "observed_beta"),
        ("beta_residual_ar1", "ar1_conditioned_residual"),
    ]:
        raw_var, raw_bins = dispersion_metric(curves, "frequency_cpd", beta_col, min_units)
        de_var, de_bins = dispersion_metric(curves, "de", beta_col, min_units)
        reduction = np.nan if not np.isfinite(raw_var) or raw_var <= 0 else 1.0 - de_var / raw_var
        rows.append(
            {
                "archive": archive,
                "metric_family": metric_family,
                "raw_axis_weighted_variance": raw_var,
                "de_axis_weighted_variance": de_var,
                "reduction_vs_raw": reduction,
                "min_units_per_bin": min_units,
                "raw_bins_used": int((raw_bins["n_units"] >= min_units).sum()),
                "de_bins_used": int((de_bins["n_units"] >= min_units).sum()),
                "n_gauges": int(curves["gauge_id"].nunique()),
                "n_curve_points": int(len(curves)),
            }
        )
        raw_bins = raw_bins.assign(archive=archive, metric_family=metric_family, axis="raw_frequency")
        de_bins = de_bins.assign(archive=archive, metric_family=metric_family, axis="memory_coordinate")
        bin_frames.extend([raw_bins, de_bins])
    return pd.DataFrame(rows), pd.concat(bin_frames, ignore_index=True)


def gauge_residual_scores(curves: pd.DataFrame, archive: str) -> pd.DataFrame:
    raw = station_bin_medians(curves, "frequency_cpd", "beta_residual_ar1")
    de = station_bin_medians(curves, "de", "beta_residual_ar1")

    def per_gauge_residual(binned: pd.DataFrame, suffix: str) -> pd.DataFrame:
        ref = binned.groupby("bin_id", as_index=False).agg(beta_ref=("beta_median", "median"))
        merged = binned.merge(ref, on="bin_id", how="inner")
        merged["residual_sq"] = (merged["beta_median"] - merged["beta_ref"]) ** 2
        return (
            merged.groupby("gauge_id", as_index=False)
            .agg(mean_residual_sq=("residual_sq", "mean"), n_bins=("bin_id", "nunique"))
            .rename(columns={"mean_residual_sq": f"{suffix}_residual_sq", "n_bins": f"{suffix}_n_bins"})
        )

    score = per_gauge_residual(raw, "raw").merge(per_gauge_residual(de, "de"), on="gauge_id", how="inner")
    score = score[(score["raw_n_bins"] >= 6) & (score["de_n_bins"] >= 6) & (score["raw_residual_sq"] > 0)].copy()
    score["ar1_residual_de_reduction"] = 1.0 - score["de_residual_sq"] / score["raw_residual_sq"]
    score["archive"] = archive
    return score


def attribute_correlations(gauge_scores: pd.DataFrame, attr_path: Path | None) -> pd.DataFrame:
    if attr_path is None or not attr_path.exists():
        return pd.DataFrame()
    attrs = pd.read_csv(attr_path)
    attrs["gauge_id"] = _gauge_str(attrs["gauge_id"])
    work = gauge_scores.merge(attrs, on="gauge_id", how="left")
    work["log10_tau_acf"] = np.log10(work["tau_acf_days"].where(work["tau_acf_days"] > 0))
    work["log10_area"] = np.log10(work["drainage_area_km2"].where(work["drainage_area_km2"] > 0))
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
        sub = work[["ar1_residual_de_reduction", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 25:
            continue
        rho, p_value = spearmanr(sub[col], sub["ar1_residual_de_reduction"])
        rows.append(
            {
                "archive": str(work["archive"].iloc[0]),
                "attribute": col,
                "label": label,
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "abs_rho": abs(float(rho)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def median_curve(curves: pd.DataFrame, beta_col: str, axis_col: str = "de", n_bins: int = 36) -> pd.DataFrame:
    work = curves[[axis_col, beta_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[work[axis_col] > 0]
    log_axis = np.log10(work[axis_col].astype(float))
    lo, hi = np.nanquantile(log_axis, [0.03, 0.97])
    edges = np.linspace(float(lo), float(hi), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    work = work[(log_axis >= lo) & (log_axis <= hi)].copy()
    work["bin_id"] = pd.cut(np.log10(work[axis_col].astype(float)), bins=edges, labels=False, include_lowest=True)
    out = (
        work.dropna(subset=["bin_id"])
        .assign(bin_id=lambda d: d["bin_id"].astype(int))
        .groupby("bin_id", as_index=False)
        .agg(
            y=("{}".format(beta_col), "median"),
            q25=("{}".format(beta_col), lambda x: float(np.nanquantile(x, 0.25))),
            q75=("{}".format(beta_col), lambda x: float(np.nanquantile(x, 0.75))),
            n=("{}".format(beta_col), "size"),
        )
    )
    out["axis_value"] = out["bin_id"].map({i: 10 ** centers[i] for i in range(len(centers))})
    return out


def plot(
    ref: pd.DataFrame,
    residual_curves: pd.DataFrame,
    metrics: pd.DataFrame,
    gauge_scores: pd.DataFrame,
    attr_corr: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(7.6, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.08, 0.92])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_a.plot(ref["de"], ref["beta_ar1_ref"], color=PALETTE["red"], lw=1.8, label="AR(1) reference")
    ax_a.fill_between(ref["de"], ref["beta_ar1_q25"], ref["beta_ar1_q75"], color=PALETTE["red"], alpha=0.12, lw=0)
    colors = {"CAMELS-US": PALETTE["blue"], "CAMELS-GB": PALETTE["green"]}
    for archive, color in colors.items():
        med = median_curve(residual_curves[residual_curves["archive"] == archive], "beta")
        ax_a.plot(med["axis_value"], med["y"], color=color, lw=1.35, label=f"{archive} median")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("memory-normalized frequency De")
    ax_a.set_ylabel("local beta")
    ax_a.set_title("a  AR(1) reference sets the Fourier-pair expectation", loc="left", fontweight="bold")
    ax_a.grid(True, color=PALETTE["grid"], lw=0.35)
    ax_a.legend(loc="upper left", fontsize=6)

    order = ["CAMELS-US", "CAMELS-GB"]
    x = np.arange(len(order))
    width = 0.32
    observed = [
        100 * metrics.query("archive == @archive and metric_family == 'observed_beta'")["reduction_vs_raw"].iloc[0]
        for archive in order
    ]
    residual = [
        100 * metrics.query("archive == @archive and metric_family == 'ar1_conditioned_residual'")["reduction_vs_raw"].iloc[0]
        for archive in order
    ]
    ax_b.bar(x - width / 2, observed, width=width, color=[colors[a] for a in order], alpha=0.55, label="observed beta")
    ax_b.bar(x + width / 2, residual, width=width, color=[colors[a] for a in order], alpha=0.95, label="AR(1)-conditioned residual")
    ax_b.axhline(0, color=PALETTE["muted"], lw=0.8)
    for xi, yi in zip(x - width / 2, observed):
        ax_b.text(xi, yi + 1.2, f"{yi:.1f}%", ha="center", va="bottom", fontsize=6)
    for xi, yi in zip(x + width / 2, residual):
        ax_b.text(xi, yi + 1.2, f"{yi:.1f}%", ha="center", va="bottom", fontsize=6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(order)
    ax_b.set_ylabel("dispersion reduction vs raw axis [%]")
    ax_b.set_title("b  Residual curves remain partly organized by De", loc="left", fontweight="bold")
    ax_b.grid(True, axis="y", color=PALETTE["grid"], lw=0.35)
    ax_b.legend(fontsize=6, loc="upper right")

    for archive, color in colors.items():
        med = median_curve(residual_curves[residual_curves["archive"] == archive], "beta_residual_ar1")
        ax_c.plot(med["axis_value"], med["y"], color=color, lw=1.45, label=archive)
        ax_c.fill_between(med["axis_value"], med["q25"], med["q75"], color=color, alpha=0.12, lw=0)
    ax_c.axhline(0, color=PALETTE["muted"], lw=0.8, ls="--")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("memory-normalized frequency De")
    ax_c.set_ylabel("beta residual after AR(1) reference")
    ax_c.set_title("c  Residual beta(De) shows archive-specific deviations", loc="left", fontweight="bold")
    ax_c.grid(True, color=PALETTE["grid"], lw=0.35)
    ax_c.legend(fontsize=6)

    for yi, archive in enumerate(["CAMELS-US", "CAMELS-GB"]):
        vals = 100 * gauge_scores.loc[
            gauge_scores["archive"] == archive,
            "ar1_residual_de_reduction",
        ].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q5, q25, q50, q75, q95, q99 = np.nanpercentile(vals, [1, 5, 25, 50, 75, 95, 99])
        color = colors[archive]
        ax_d.hlines(yi, q5, q95, color=color, lw=2.2, alpha=0.35)
        ax_d.hlines(yi, q25, q75, color=color, lw=8.0, alpha=0.55)
        ax_d.plot(q50, yi, "o", color=PALETTE["ink"], ms=4.2)
        ax_d.plot([q1, q99], [yi, yi], "|", color=color, ms=9, mew=1.2)
        ax_d.text(
            160,
            yi,
            f"median {q50:.1f}%\npositive {(vals > 0).mean():.0%}",
            va="center",
            ha="right",
            fontsize=6.0,
            color=PALETTE["ink"],
        )
    ax_d.axvline(0, color=PALETTE["muted"], lw=0.8, ls="--")
    if not attr_corr.empty:
        top = attr_corr.iloc[0]
        ax_d.text(
            0.03,
            0.93,
            f"top modifier: {top['label']} (rho={top['spearman_rho']:.2f})",
            transform=ax_d.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            color=PALETTE["ink"],
        )
    ax_d.set_xlabel("gauge residual-De benefit [%]")
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(["CAMELS-US", "CAMELS-GB"])
    ax_d.set_xlim(-650, 170)
    ax_d.set_title("d  Gauge-level residual benefit is heterogeneous", loc="left", fontweight="bold")
    ax_d.grid(True, axis="x", color=PALETTE["grid"], lw=0.35)

    fig.suptitle(
        "Residual organization after conditioning on an AR(1) Fourier-pair baseline",
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


def write_note(metrics: pd.DataFrame, gauge_scores: pd.DataFrame, attr_corr: pd.DataFrame) -> None:
    lines = [
        "# R19 AR(1)-Conditioned Residual Alignment",
        "",
        "Date: 2026-06-03",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: test whether CAMELS spectral-slope curves retain De organization after subtracting the median AR(1) Fourier-pair expectation.",
        "- Evidence chain: AR(1) baseline curves -> beta_AR1_ref(De) -> CAMELS beta residuals -> raw-axis versus De-axis residual dispersion -> gauge-level residual heterogeneity.",
        "- Archetype: schematic-led quantitative composite with one reference curve, one residual metric panel, one residual shape panel and one heterogeneity panel.",
        "- Boundary: this is an operational residual diagnostic, not proof of storage causality.",
        "",
        "## Aggregate Residual Metrics",
        "",
        "| Archive | Metric family | Raw-axis variance | De-axis variance | Reduction |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['archive']} | {row['metric_family']} | {row['raw_axis_weighted_variance']:.4f} | {row['de_axis_weighted_variance']:.4f} | {row['reduction_vs_raw']:.1%} |"
        )
    lines.extend(["", "## Gauge-Level CAMELS-US Residual Heterogeneity", ""])
    us = gauge_scores[gauge_scores["archive"] == "CAMELS-US"]["ar1_residual_de_reduction"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(us):
        q25, q75 = np.nanpercentile(us, [25, 75])
        lines.extend(
            [
                f"- Gauges scored: {len(us)}",
                f"- Median residual-De benefit: {np.nanmedian(us):.1%}",
                f"- IQR: {q25:.1%} to {q75:.1%}",
                f"- Positive fraction: {(us > 0).mean():.1%}",
            ]
        )
    if not attr_corr.empty:
        lines.extend(["", "Top CAMELS-US attribute modifiers:", "", "| Attribute | Spearman rho | p value | n |", "| --- | ---: | ---: | ---: |"])
        for _, row in attr_corr.head(5).iterrows():
            lines.append(f"| {row['label']} | {row['spearman_rho']:.3f} | {row['p_value']:.3g} | {int(row['n'])} |")
    lines.extend(
        [
            "",
            "## Manuscript Consequence",
            "",
            "Safe R19 wording: after subtracting the median AR(1) Fourier-pair expectation, the residual beta curves still show partial De-axis tightening in the tested CAMELS archives under this diagnostic. This supports a benchmark interpretation beyond arbitrary tau shuffling, but it does not prove the storage mechanism because the residual is still derived from discharge spectra.",
            "",
            "Unsafe wording: AR(1) circularity is solved, storage causality is proven, or the residual metric establishes global portability.",
            "",
            "## Outputs",
            "",
            f"- `{RESIDUAL_CURVES_OUT.relative_to(ROOT)}`",
            f"- `{METRICS_OUT.relative_to(ROOT)}`",
            f"- `{GAUGE_OUT.relative_to(ROOT)}`",
            f"- `{ATTR_OUT.relative_to(ROOT)}`",
            f"- `{FIG_OUT.relative_to(ROOT)}.png/svg/pdf`",
        ]
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    ref = build_ar1_reference()
    residual_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    gauge_frames: list[pd.DataFrame] = []
    attr_frames: list[pd.DataFrame] = []

    for archive, curve_path, attr_path in ARCHIVES:
        curves = pd.read_csv(curve_path, dtype={"gauge_id": str})
        residual = add_ar1_residual(curves, archive, ref)
        metrics, bins = archive_metrics(residual, archive)
        scores = gauge_residual_scores(residual, archive)
        attrs = attribute_correlations(scores, attr_path)
        residual_frames.append(residual)
        metric_frames.append(metrics)
        bin_frames.append(bins)
        gauge_frames.append(scores)
        if not attrs.empty:
            attr_frames.append(attrs)

    residual_curves = pd.concat(residual_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    gauge_scores = pd.concat(gauge_frames, ignore_index=True)
    attr_corr = pd.concat(attr_frames, ignore_index=True) if attr_frames else pd.DataFrame()

    residual_curves.to_csv(RESIDUAL_CURVES_OUT, index=False)
    metrics.to_csv(METRICS_OUT, index=False)
    gauge_scores.to_csv(GAUGE_OUT, index=False)
    attr_corr.to_csv(ATTR_OUT, index=False)
    plot(ref, residual_curves, metrics, gauge_scores, attr_corr)
    write_note(metrics, gauge_scores, attr_corr)

    print(metrics.to_string(index=False))
    print(f"Wrote {RESIDUAL_CURVES_OUT.relative_to(ROOT)}")
    print(f"Wrote {METRICS_OUT.relative_to(ROOT)}")
    print(f"Wrote {GAUGE_OUT.relative_to(ROOT)}")
    print(f"Wrote {ATTR_OUT.relative_to(ROOT)}")
    print(f"Wrote {FIG_OUT.relative_to(ROOT)}.png/svg/pdf")
    print(f"Wrote {NOTE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
