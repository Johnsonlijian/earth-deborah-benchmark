"""R26 four-archive uncertainty hardening.

This script uses already-derived beta-curve source tables, not raw third-party
archives, to quantify gauge-bootstrap uncertainty in the archive-level
raw-frequency versus De beta-variance reduction. The goal is a submission-facing
uncertainty layer for the four-archive portability claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r26_four_archive_uncertainty"


@dataclass(frozen=True)
class ArchiveSpec:
    archive: str
    curve_file: str
    n_bins: int = 24
    min_bin_gauges: int = 50


ARCHIVES = [
    ArchiveSpec("CAMELS-US", "camels_673_beta_curve_points.csv", min_bin_gauges=50),
    ArchiveSpec("CAMELS-GB v2", "camels_gb_v2_replication_beta_curve_points.csv", min_bin_gauges=50),
    ArchiveSpec("CAMELS-BR v1.2", "r23_camels_br_third_archive_curves.csv", min_bin_gauges=50),
    ArchiveSpec("CAMELS-AUS v2", "r25_camels_aus_fourth_archive_curves.csv", min_bin_gauges=40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def load_curves(spec: ArchiveSpec) -> pd.DataFrame:
    path = TABLES / spec.curve_file
    curves = pd.read_csv(path)
    curves["gauge_id"] = curves["gauge_id"].astype(str)
    curves["frequency_cpd"] = pd.to_numeric(curves["frequency_cpd"], errors="coerce")
    curves["de"] = pd.to_numeric(curves["de"], errors="coerce")
    curves["beta"] = pd.to_numeric(curves["beta"], errors="coerce")
    curves = curves.replace([np.inf, -np.inf], np.nan).dropna(subset=["gauge_id", "frequency_cpd", "de", "beta"])
    curves = curves[(curves["frequency_cpd"] > 0) & (curves["de"] > 0)].copy()
    if curves.empty:
        raise RuntimeError(f"No usable curves for {spec.archive}: {path}")
    return curves[["gauge_id", "frequency_cpd", "de", "beta"]]


def fixed_bin_medians(curves: pd.DataFrame, coord: str, spec: ArchiveSpec) -> pd.DataFrame:
    work = curves[["gauge_id", coord, "beta"]].dropna().copy()
    logx = np.log10(work[coord].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(logx, [2, 98])
    edges = np.linspace(lo, hi, spec.n_bins + 1)
    work["bin"] = pd.cut(logx, bins=edges, labels=False, include_lowest=True)
    med = work.dropna(subset=["bin"]).groupby(["bin", "gauge_id"], as_index=False)["beta"].median()
    med["bin"] = med["bin"].astype(int)
    counts = med.groupby("bin")["gauge_id"].nunique()
    keep = counts[counts >= spec.min_bin_gauges].index
    med = med[med["bin"].isin(keep)].copy()
    med["axis"] = coord
    if med.empty:
        raise RuntimeError(f"No retained bins for {spec.archive} / {coord}")
    return med


def weighted_variance_for_bins(med: pd.DataFrame, counts: pd.Series | None = None) -> tuple[float, int, int]:
    rows = []
    for bin_id, group in med.groupby("bin"):
        values = group["beta"].to_numpy(dtype=float)
        if counts is None:
            weights = np.ones_like(values)
        else:
            weights = group["gauge_id"].map(counts).fillna(0).to_numpy(dtype=float)
        keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        values = values[keep]
        weights = weights[keep]
        n_eff = float(weights.sum())
        if n_eff < 2:
            continue
        mean = float(np.average(values, weights=weights))
        # Frequency-weighted sample variance; duplicate gauges in a bootstrap
        # draw are counted as repeated gauge instances.
        var = float(np.sum(weights * (values - mean) ** 2) / max(n_eff - 1.0, 1.0))
        rows.append({"bin": int(bin_id), "n": int(n_eff), "variance": var})
    if not rows:
        return np.nan, 0, 0
    stats = pd.DataFrame(rows)
    weighted = float(np.average(stats["variance"], weights=stats["n"]))
    return weighted, int(len(stats)), int(stats["n"].sum())


def reduction_from_medians(raw_med: pd.DataFrame, de_med: pd.DataFrame, counts: pd.Series | None = None) -> dict:
    raw_var, raw_bins, raw_pairs = weighted_variance_for_bins(raw_med, counts)
    de_var, de_bins, de_pairs = weighted_variance_for_bins(de_med, counts)
    reduction = 1.0 - de_var / raw_var if np.isfinite(raw_var) and raw_var > 0 else np.nan
    return {
        "raw_weighted_beta_variance": raw_var,
        "de_weighted_beta_variance": de_var,
        "variance_reduction_fraction": reduction,
        "raw_n_bins": raw_bins,
        "de_n_bins": de_bins,
        "raw_n_gauge_bin_pairs": raw_pairs,
        "de_n_gauge_bin_pairs": de_pairs,
    }


def bootstrap_archive(spec: ArchiveSpec, curves: pd.DataFrame, args: argparse.Namespace, seed_offset: int) -> tuple[pd.DataFrame, dict]:
    raw_med = fixed_bin_medians(curves, "frequency_cpd", spec)
    de_med = fixed_bin_medians(curves, "de", spec)
    observed = reduction_from_medians(raw_med, de_med)
    gids = np.array(sorted(curves["gauge_id"].unique()))
    rng = np.random.default_rng(args.seed + seed_offset)
    rows = []
    for draw in range(args.n_boot):
        sample = rng.choice(gids, size=len(gids), replace=True)
        counts = pd.Series(sample).value_counts()
        res = reduction_from_medians(raw_med, de_med, counts)
        res.update({"archive": spec.archive, "draw": draw, "n_gauges": len(gids)})
        rows.append(res)
    observed.update({"archive": spec.archive, "n_gauges": len(gids)})
    return pd.DataFrame(rows), observed


def summarize_bootstrap(boot: pd.DataFrame, observed_rows: list[dict]) -> pd.DataFrame:
    observed = pd.DataFrame(observed_rows).set_index("archive")
    rows = []
    for archive, group in boot.groupby("archive"):
        vals = pd.to_numeric(group["variance_reduction_fraction"], errors="coerce").dropna()
        rows.append(
            {
                "archive": archive,
                "n_gauges": int(observed.loc[archive, "n_gauges"]),
                "observed_reduction_fraction": float(observed.loc[archive, "variance_reduction_fraction"]),
                "bootstrap_median_reduction_fraction": float(vals.median()),
                "bootstrap_ci_low_fraction": float(vals.quantile(0.025)),
                "bootstrap_ci_high_fraction": float(vals.quantile(0.975)),
                "probability_positive": float((vals > 0).mean()),
                "bootstrap_sd_fraction": float(vals.std(ddof=1)),
                "raw_weighted_beta_variance": float(observed.loc[archive, "raw_weighted_beta_variance"]),
                "de_weighted_beta_variance": float(observed.loc[archive, "de_weighted_beta_variance"]),
            }
        )
    order = {spec.archive: i for i, spec in enumerate(ARCHIVES)}
    summary = pd.DataFrame(rows)
    summary["order"] = summary["archive"].map(order)
    return summary.sort_values("order").drop(columns=["order"])


def plot_uncertainty(summary: pd.DataFrame, boot: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    colors = {
        "CAMELS-US": "#3F7CAC",
        "CAMELS-GB v2": "#7B8CE3",
        "CAMELS-BR v1.2": "#49A88C",
        "CAMELS-AUS v2": "#C96352",
    }
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    plot = summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    vals = plot["observed_reduction_fraction"].to_numpy() * 100
    lo = plot["bootstrap_ci_low_fraction"].to_numpy() * 100
    hi = plot["bootstrap_ci_high_fraction"].to_numpy() * 100
    xerr = np.vstack([vals - lo, hi - vals])
    ax0.axvline(0, color="#6b7280", lw=0.8)
    ax0.errorbar(
        vals,
        y,
        xerr=xerr,
        fmt="o",
        ms=6.2,
        lw=1.5,
        capsize=3,
        color="#1f2937",
        ecolor="#334155",
        zorder=3,
    )
    for idx, row in plot.iterrows():
        ax0.scatter(
            row["observed_reduction_fraction"] * 100,
            idx,
            s=58,
            color=colors[row["archive"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax0.text(
            max(row["bootstrap_ci_high_fraction"] * 100 + 0.8, row["observed_reduction_fraction"] * 100 + 1.0),
            idx,
            f"Pr>0={row['probability_positive']:.2f}",
            va="center",
            fontsize=6.8,
            color="#475569",
        )
    ax0.set_yticks(y, plot["archive"])
    ax0.set_xlabel("De beta-variance reduction [%]")
    ax0.set_title("a  Archive-level effect with gauge-bootstrap uncertainty", loc="left", fontweight="bold")
    ax0.grid(axis="x", color="#dbe3ea", lw=0.8)
    ax0.set_xlim(min(-8, np.nanmin(lo) - 3), max(32, np.nanmax(hi) + 9))

    for archive in summary["archive"]:
        vals = boot.loc[boot["archive"] == archive, "variance_reduction_fraction"].to_numpy(dtype=float) * 100
        ax1.kde = None
        ax1.hist(vals, bins=32, density=True, alpha=0.32, color=colors[archive], label=archive)
    ax1.axvline(0, color="#6b7280", lw=0.8)
    ax1.set_xlabel("bootstrap reduction [%]")
    ax1.set_ylabel("density")
    ax1.set_title("b  Bootstrap distributions", loc="left", fontweight="bold")
    ax1.legend(fontsize=6.2, ncol=1, loc="upper left")

    ratio = summary.copy().reset_index(drop=True)
    ratio["variance_ratio"] = ratio["de_weighted_beta_variance"] / ratio["raw_weighted_beta_variance"]
    ax2.bar(
        np.arange(len(ratio)),
        ratio["variance_ratio"],
        color=[colors[a] for a in ratio["archive"]],
        alpha=0.86,
    )
    ax2.axhline(1.0, color="#6b7280", lw=0.8, ls="--")
    ax2.set_xticks(np.arange(len(ratio)), ratio["archive"], rotation=25, ha="right")
    ax2.set_ylabel("De variance / raw variance")
    ax2.set_title("c  Dispersion ratio retained after normalization", loc="left", fontweight="bold")
    ax2.grid(axis="y", color="#dbe3ea", lw=0.8)
    for pos, row in ratio.iterrows():
        ax2.text(pos, row["variance_ratio"] + 0.025, f"{row['variance_ratio']:.2f}", ha="center", fontsize=6.6)

    for suffix, kwargs in [
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# R26 Four-Archive Uncertainty",
        "",
        f"Gauge-bootstrap draws per archive: {args.n_boot}.",
        "",
        "| Archive | Observed reduction | Bootstrap median | 95% CI | Pr(reduction > 0) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {archive} | {obs:.1%} | {med:.1%} | {lo:.1%} to {hi:.1%} | {prob:.2f} |".format(
                archive=row["archive"],
                obs=row["observed_reduction_fraction"],
                med=row["bootstrap_median_reduction_fraction"],
                lo=row["bootstrap_ci_low_fraction"],
                hi=row["bootstrap_ci_high_fraction"],
                prob=row["probability_positive"],
            )
        )
    lines += [
        "",
        "Interpretation: the four-archive portability claim is supported as an",
        "uncertainty-aware benchmark result, but effect sizes are heterogeneous.",
        "CAMELS-AUS has the smallest aggregate effect and should remain linked to",
        "the dry/intermittent-regime boundary rather than universal-collapse",
        "language.",
        "",
        "Boundary: this is a gauge-level bootstrap over already-derived beta curves.",
        "It does not estimate direct tracer/storage-state mechanism uncertainty and",
        "does not change the matched-AR(1) artifact boundary.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    all_boot = []
    observed_rows = []
    for idx, spec in enumerate(ARCHIVES):
        curves = load_curves(spec)
        boot, observed = bootstrap_archive(spec, curves, args, seed_offset=idx * 10_000)
        all_boot.append(boot)
        observed_rows.append(observed)
    boot = pd.concat(all_boot, ignore_index=True)
    summary = summarize_bootstrap(boot, observed_rows)
    boot.to_csv(TABLES / f"{OUT}_bootstrap_draws.csv", index=False)
    summary.to_csv(TABLES / f"{OUT}_summary.csv", index=False)
    plot_uncertainty(summary, boot)
    write_note(summary, args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
