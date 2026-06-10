"""R31 gauge-level excess-over-matched-AR(1) boundary analysis.

R20 showed that the archive-level residual after subtracting each gauge's
analytical AR(1) reference is negative on the De axis. R31 turns that global
failure into a reviewer-facing distributional and stratified boundary:
which gauges improve, which do not, and whether common catchment attributes
explain the residual.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX_FIGURES = ROOT / "figures"
LATEX_SI_FIGURES = ROOT / "figures"
LATEX_SOURCE = ROOT / "source_data"
OUT = "r31_gauge_level_ar1_boundary"


def _gauge_str(s: pd.Series, archive: str) -> pd.Series:
    if archive == "CAMELS-US":
        return s.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(8)
    return s.astype(str)


def _bh_qvalues(pvals: list[float]) -> list[float]:
    p = np.asarray([1.0 if not np.isfinite(x) else x for x in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = n - rank + 1
        val = min(prev, p[idx] * n / true_rank)
        q[idx] = val
        prev = val
    return q.tolist()


def station_bin_medians(curves: pd.DataFrame, axis_col: str, beta_col: str, n_bins: int = 24) -> pd.DataFrame:
    work = curves[["gauge_id", axis_col, beta_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[work[axis_col] > 0]
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


def per_gauge_scores(curves: pd.DataFrame, archive: str) -> pd.DataFrame:
    rows = []
    for axis_col, prefix in [("frequency_cpd", "raw"), ("de", "de")]:
        binned = station_bin_medians(curves, axis_col, "beta_residual_gauge_ar1")
        reference = binned.groupby("bin_id", as_index=False).agg(reference_beta=("beta_median", "median"))
        merged = binned.merge(reference, on="bin_id", how="inner")
        merged["residual_sq"] = (merged["beta_median"] - merged["reference_beta"]) ** 2
        score = (
            merged.groupby("gauge_id", as_index=False)
            .agg(
                mean_residual_sq=("residual_sq", "mean"),
                median_abs_residual=("residual_sq", lambda s: float(np.sqrt(np.nanmedian(s)))),
                n_bins=("bin_id", "nunique"),
            )
            .rename(
                columns={
                    "mean_residual_sq": f"{prefix}_residual_sq",
                    "median_abs_residual": f"{prefix}_median_abs_residual",
                    "n_bins": f"{prefix}_n_bins",
                }
            )
        )
        rows.append(score)
    scores = rows[0].merge(rows[1], on="gauge_id", how="inner")
    scores = scores[(scores["raw_n_bins"] >= 6) & (scores["de_n_bins"] >= 6) & (scores["raw_residual_sq"] > 0)].copy()
    scores["ar1_residual_de_reduction"] = 1.0 - scores["de_residual_sq"] / scores["raw_residual_sq"]
    tau = curves.groupby("gauge_id", as_index=False).agg(tau_acf_days=("tau_acf_days", "first"))
    scores = scores.merge(tau, on="gauge_id", how="left")
    scores["archive"] = archive
    return scores


def bootstrap_ci(values: np.ndarray, func, n_boot: int = 5000, seed: int = 531) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    observed = float(func(x))
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = func(rng.choice(x, size=len(x), replace=True))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return observed, float(lo), float(hi)


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for archive, group in scores.groupby("archive"):
        x = group["ar1_residual_de_reduction"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = int(len(x))
        n_pos = int(np.sum(x > 0))
        med, med_lo, med_hi = bootstrap_ci(x, np.nanmedian, seed=531)
        frac, frac_lo, frac_hi = bootstrap_ci(x, lambda v: np.mean(v > 0), seed=532)
        try:
            wilcox_less = float(wilcoxon(x, alternative="less").pvalue)
        except ValueError:
            wilcox_less = float("nan")
        rows.append(
            {
                "archive": archive,
                "n_gauges": n,
                "median_reduction": med,
                "median_reduction_ci_low": med_lo,
                "median_reduction_ci_high": med_hi,
                "mean_reduction": float(np.nanmean(x)),
                "positive_gauges": n_pos,
                "positive_fraction": frac,
                "positive_fraction_ci_low": frac_lo,
                "positive_fraction_ci_high": frac_hi,
                "binomial_p_positive_less_than_half": float(binomtest(n_pos, n, 0.5, alternative="less").pvalue),
                "wilcoxon_p_median_less_than_zero": wilcox_less,
                "q10": float(np.nanquantile(x, 0.10)),
                "q25": float(np.nanquantile(x, 0.25)),
                "q75": float(np.nanquantile(x, 0.75)),
                "q90": float(np.nanquantile(x, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def load_attributes(archive: str) -> pd.DataFrame:
    if archive == "CAMELS-US":
        attrs = pd.read_csv(TABLES / "camels_673_attributed_diagnostics.csv")
        attrs["gauge_id"] = _gauge_str(attrs["gauge_id"], archive)
        cols = {
            "drainage_area_km2": "area",
            "aridity": "aridity",
            "frac_snow": "snow fraction",
            "p_seasonality": "precipitation seasonality",
            "runoff_ratio": "runoff ratio",
            "baseflow_index_x": "baseflow index",
            "zero_q_freq_x": "zero-flow frequency",
            "geol_permeability_x": "geologic permeability",
            "geol_porostiy_x": "geologic porosity",
        }
    else:
        attrs = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv")
        attrs["gauge_id"] = attrs["gauge_id"].astype(str)
        cols = {
            "drainage_area_km2": "area",
            "aridity": "aridity",
            "frac_snow": "snow fraction",
            "p_seasonality": "precipitation seasonality",
            "runoff_ratio": "runoff ratio",
            "baseflow_index": "baseflow index",
            "zero_q_freq": "zero-flow frequency",
            "dpsbar": "mean slope",
            "elev_mean": "mean elevation",
        }
    keep = ["gauge_id"] + [c for c in cols if c in attrs.columns]
    out = attrs[keep].copy()
    return out.rename(columns=cols)


def attribute_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for archive, group in scores.groupby("archive"):
        attrs = load_attributes(archive)
        work = group.merge(attrs, on="gauge_id", how="left")
        work["log10 tau_acf"] = np.log10(work["tau_acf_days"].where(work["tau_acf_days"] > 0))
        candidates = [c for c in work.columns if c not in {
            "archive",
            "gauge_id",
            "raw_residual_sq",
            "raw_median_abs_residual",
            "raw_n_bins",
            "de_residual_sq",
            "de_median_abs_residual",
            "de_n_bins",
            "ar1_residual_de_reduction",
            "tau_acf_days",
        }]
        pvals = []
        tmp = []
        for col in candidates:
            pair = work[["ar1_residual_de_reduction", col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 40 or pair[col].nunique() < 5:
                continue
            rho, p = spearmanr(pair[col], pair["ar1_residual_de_reduction"])
            tmp.append(
                {
                    "archive": archive,
                    "attribute": col,
                    "n": int(len(pair)),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                }
            )
            pvals.append(float(p))
        qvals = _bh_qvalues(pvals)
        for row, q in zip(tmp, qvals):
            row["q_value"] = q
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["archive", "q_value", "attribute"])


def strata(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for archive, group in scores.groupby("archive"):
        work = group.copy()
        work["tau quartile"] = pd.qcut(work["tau_acf_days"], q=4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"], duplicates="drop")
        attrs = load_attributes(archive)
        if "baseflow index" in attrs.columns:
            work = work.merge(attrs[["gauge_id", "baseflow index"]], on="gauge_id", how="left")
            if work["baseflow index"].notna().sum() > 40:
                work["baseflow quartile"] = pd.qcut(
                    work["baseflow index"],
                    q=4,
                    labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
                    duplicates="drop",
                )
        for variable in ["tau quartile", "baseflow quartile"]:
            if variable not in work.columns:
                continue
            sub = (
                work.dropna(subset=[variable])
                .groupby(variable, observed=False)
                .agg(
                    n=("gauge_id", "size"),
                    median_reduction=("ar1_residual_de_reduction", "median"),
                    positive_fraction=("ar1_residual_de_reduction", lambda s: float(np.mean(np.asarray(s) > 0))),
                    q25=("ar1_residual_de_reduction", lambda s: float(np.nanquantile(s, 0.25))),
                    q75=("ar1_residual_de_reduction", lambda s: float(np.nanquantile(s, 0.75))),
                )
                .reset_index()
                .rename(columns={variable: "stratum"})
            )
            sub["archive"] = archive
            sub["stratifier"] = variable
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def plot(summary_df: pd.DataFrame, scores: pd.DataFrame, corr: pd.DataFrame, strata_df: pd.DataFrame) -> None:
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
    colors = {"CAMELS-US": "#386FA4", "CAMELS-GB": "#4B9B72"}
    red = "#C44E52"
    gold = "#D09A32"
    grid = "#D6D9DE"
    ink = "#202124"

    fig = plt.figure(figsize=(7.55, 5.9))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        top=0.92,
        bottom=0.09,
        hspace=0.48,
        wspace=0.62,
        width_ratios=[1.0, 1.08],
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel a: medians and positive fractions.
    archives = ["CAMELS-US", "CAMELS-GB"]
    x = np.arange(len(archives))
    med = summary_df.set_index("archive").loc[archives, "median_reduction"].to_numpy()
    err_lo = med - summary_df.set_index("archive").loc[archives, "median_reduction_ci_low"].to_numpy()
    err_hi = summary_df.set_index("archive").loc[archives, "median_reduction_ci_high"].to_numpy() - med
    ax0.bar(x, med, color=[colors[a] for a in archives], alpha=0.88)
    ax0.errorbar(x, med, yerr=[err_lo, err_hi], fmt="none", ecolor=ink, capsize=3, lw=0.9)
    ax0.axhline(0, color=ink, lw=0.8)
    ax0.set_xticks(x, archives)
    ax0.set_ylabel("Gauge-level residual reduction")
    ax0.set_title("a  Median residual test after matched AR(1)", loc="left", fontsize=9, fontweight="bold")
    ax0.grid(axis="y", color=grid, lw=0.55)
    ax0.text(
        0.02,
        0.05,
        "positive = excess tightening\nbeyond matched AR(1)",
        transform=ax0.transAxes,
        fontsize=6.2,
        color=ink,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F7F9FB", edgecolor=grid, linewidth=0.5),
    )
    for xx, yy, archive in zip(x, med, archives):
        frac = summary_df.set_index("archive").loc[archive, "positive_fraction"]
        ax0.text(xx, yy - 0.10, f"{yy:.2f}\n{frac:.0%} positive", ha="center", va="top", fontsize=6.4, color="white")

    # Panel b: distribution.
    data = [scores.loc[scores["archive"] == a, "ar1_residual_de_reduction"].to_numpy(dtype=float) for a in archives]
    parts = ax1.violinplot(data, positions=x, widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    for body, archive in zip(parts["bodies"], archives):
        body.set_facecolor(colors[archive])
        body.set_edgecolor("none")
        body.set_alpha(0.28)
    rng = np.random.default_rng(531)
    for i, (archive, vals) in enumerate(zip(archives, data)):
        vals = vals[np.isfinite(vals)]
        sample = rng.choice(vals, size=min(180, len(vals)), replace=False)
        jitter = rng.normal(0, 0.045, size=len(sample))
        ax1.scatter(np.full(len(sample), i) + jitter, np.clip(sample, -6, 2), s=5, color=colors[archive], alpha=0.45, lw=0)
        q = np.nanquantile(vals, [0.25, 0.5, 0.75])
        ax1.plot([i - 0.22, i + 0.22], [q[1], q[1]], color=ink, lw=1.3)
        ax1.plot([i, i], [q[0], q[2]], color=ink, lw=1.0)
    ax1.axhline(0, color=ink, lw=0.8)
    ax1.set_xticks(x, archives)
    ax1.set_ylim(-6, 1.5)
    ax1.set_ylabel("Per-gauge reduction (clipped)")
    ax1.set_title("b  Most gauges remain below zero", loc="left", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", color=grid, lw=0.55)
    ax1.text(
        0.98,
        0.93,
        "Matched-AR(1)\nexcess gate: failed",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="white",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=red, edgecolor="none", alpha=0.95),
    )

    # Panel c: tau quartiles.
    tau = strata_df[strata_df["stratifier"] == "tau quartile"].copy()
    labels = ["Q1 shortest", "Q2", "Q3", "Q4 longest"]
    offset = {"CAMELS-US": -0.17, "CAMELS-GB": 0.17}
    for archive in archives:
        sub = tau[tau["archive"] == archive].set_index("stratum").reindex(labels)
        xx = np.arange(len(labels)) + offset[archive]
        ax2.errorbar(
            xx,
            sub["median_reduction"],
            yerr=[sub["median_reduction"] - sub["q25"], sub["q75"] - sub["median_reduction"]],
            fmt="o",
            color=colors[archive],
            label=archive,
            capsize=2.5,
            ms=4,
        )
    ax2.axhline(0, color=ink, lw=0.8)
    ax2.set_xticks(np.arange(len(labels)), ["Q1", "Q2", "Q3", "Q4"])
    ax2.set_xlabel("tau_acf quartile")
    ax2.set_ylabel("Median residual reduction")
    ax2.set_title("c  Residual signal is heterogeneous by memory", loc="left", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", color=grid, lw=0.55)
    ax2.legend(fontsize=6.2)

    # Panel d: strongest attribute associations, if any.
    top = corr.copy()
    top = top.sort_values("q_value").groupby("archive", as_index=False).head(5)
    short_attr = {
        "precipitation seasonality": "seasonality",
        "zero-flow frequency": "zero-flow",
        "baseflow index": "BFI",
        "geologic permeability": "permeability",
        "geologic porosity": "porosity",
        "mean elevation": "elevation",
        "snow fraction": "snow",
        "runoff ratio": "runoff",
        "log10 tau_acf": "log tau",
    }
    top["attribute_short"] = top["attribute"].map(short_attr).fillna(top["attribute"])
    top["label"] = top["archive"].str.replace("CAMELS-", "") + ": " + top["attribute_short"]
    top = top.sort_values("spearman_rho")
    y = np.arange(len(top))
    bar_colors = [colors[a] if q < 0.1 else gold for a, q in zip(top["archive"], top["q_value"])]
    ax3.barh(y, top["spearman_rho"], color=bar_colors, alpha=0.86)
    ax3.axvline(0, color=ink, lw=0.8)
    ax3.set_yticks(y, top["label"], fontsize=6.0)
    ax3.set_xlabel("Spearman rho with residual reduction")
    ax3.set_title("d  Attribute-linked residual heterogeneity", loc="left", fontsize=9, fontweight="bold")
    ax3.grid(axis="x", color=grid, lw=0.55)
    ax3.legend(
        handles=[
            Patch(facecolor=colors["CAMELS-US"], label="CAMELS-US"),
            Patch(facecolor=colors["CAMELS-GB"], label="CAMELS-GB v2"),
        ],
        loc="lower right",
        fontsize=6.2,
    )

    for suffix in (".pdf", ".svg", ".png"):
        path = FIGURES / f"{OUT}{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None)
        shutil.copy2(path, LATEX_FIGURES / path.name)
        shutil.copy2(path, LATEX_SI_FIGURES / path.name)
    plt.close(fig)


def write_note(summary_df: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines = [
        "# R31 Gauge-Level Matched-AR(1) Boundary",
        "",
        "Date: 2026-06-04",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Attribute Screen",
        "",
        corr.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Manuscript Consequence",
        "",
        "The strict gauge-level residual gate is negative in both CAMELS-US and CAMELS-GB.",
        "This strengthens the claim boundary: De is defensible as a diagnostic benchmark",
        "and coordinate test, but not as an excess-over-matched-AR(1) mechanism proof.",
    ]
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in (TABLES, FIGURES, NOTES, LATEX_FIGURES, LATEX_SI_FIGURES, LATEX_SOURCE):
        path.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(TABLES / "r20_gauge_matched_ar1_residual_curves.csv")
    frames = []
    for archive, group in curves.groupby("archive"):
        group = group.copy()
        group["gauge_id"] = _gauge_str(group["gauge_id"], archive)
        frames.append(per_gauge_scores(group, archive))
    scores = pd.concat(frames, ignore_index=True)
    summary_df = summary(scores)
    corr = attribute_correlations(scores)
    strata_df = strata(scores)

    scores.to_csv(TABLES / f"{OUT}_gauge_scores.csv", index=False)
    summary_df.to_csv(TABLES / f"{OUT}_summary.csv", index=False)
    corr.to_csv(TABLES / f"{OUT}_attribute_correlations.csv", index=False)
    strata_df.to_csv(TABLES / f"{OUT}_strata.csv", index=False)
    shutil.copy2(TABLES / f"{OUT}_summary.csv", LATEX_SOURCE / f"{OUT}_summary.csv")
    shutil.copy2(TABLES / f"{OUT}_gauge_scores.csv", LATEX_SOURCE / f"{OUT}_gauge_scores.csv")
    shutil.copy2(TABLES / f"{OUT}_attribute_correlations.csv", LATEX_SOURCE / f"{OUT}_attribute_correlations.csv")
    shutil.copy2(TABLES / f"{OUT}_strata.csv", LATEX_SOURCE / f"{OUT}_strata.csv")
    plot(summary_df, scores, corr, strata_df)
    write_note(summary_df, corr)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
