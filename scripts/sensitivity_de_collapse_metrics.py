"""Sensitivity tests for the CAMELS De-collapse dispersion metric."""

from __future__ import annotations

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
CURVES_CSV = TABLES / "camels_673_beta_curve_points.csv"
OUT_CSV = TABLES / "camels_673_collapse_sensitivity.csv"
OUT_MD = NOTES / "de_collapse_sensitivity_20260531.md"


def station_bin_medians(curves: pd.DataFrame, value_col: str, n_bins: int) -> pd.DataFrame:
    values = curves[value_col].to_numpy(dtype=float)
    keep = np.isfinite(values) & (values > 0)
    work = curves.loc[keep].copy()
    x = np.log10(work[value_col].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work["bin_id"] = pd.cut(np.log10(work[value_col]), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    grouped = (
        work.groupby(["gauge_id", "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"), n_points=("beta", "size"))
    )
    grouped["axis_value"] = grouped["bin_id"].map({i: 10 ** labels[i] for i in range(len(labels))})
    return grouped


def weighted_variance_by_bin(binned: pd.DataFrame, min_stations: int) -> tuple[float, int, int]:
    rows: list[tuple[float, int]] = []
    for _, group in binned.groupby("bin_id"):
        n_stations = int(group["gauge_id"].nunique())
        if n_stations < min_stations:
            continue
        var = float(group["beta_median"].var())
        if np.isfinite(var):
            rows.append((var, n_stations))
    if not rows:
        return np.nan, 0, 0
    variances = np.array([item[0] for item in rows], dtype=float)
    weights = np.array([item[1] for item in rows], dtype=float)
    return float(np.average(variances, weights=weights)), len(rows), int(weights.sum())


def iqr_width_by_bin(binned: pd.DataFrame, min_stations: int) -> tuple[float, int, int]:
    rows: list[tuple[float, int]] = []
    for _, group in binned.groupby("bin_id"):
        n_stations = int(group["gauge_id"].nunique())
        if n_stations < min_stations:
            continue
        vals = group["beta_median"].to_numpy(dtype=float)
        width = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
        if np.isfinite(width):
            rows.append((width, n_stations))
    if not rows:
        return np.nan, 0, 0
    widths = np.array([item[0] for item in rows], dtype=float)
    weights = np.array([item[1] for item in rows], dtype=float)
    return float(np.average(widths, weights=weights)), len(rows), int(weights.sum())


def run_sensitivity(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_bins in [12, 16, 20, 24, 28, 32, 36]:
        raw = station_bin_medians(curves, "frequency_cpd", n_bins)
        de = station_bin_medians(curves, "de", n_bins)
        for min_stations in [30, 50, 80, 100, 150]:
            raw_var, raw_bins, raw_values = weighted_variance_by_bin(raw, min_stations)
            de_var, de_bins, de_values = weighted_variance_by_bin(de, min_stations)
            raw_iqr, _, _ = iqr_width_by_bin(raw, min_stations)
            de_iqr, _, _ = iqr_width_by_bin(de, min_stations)
            rows.append(
                {
                    "n_bins": n_bins,
                    "min_stations": min_stations,
                    "raw_weighted_variance": raw_var,
                    "de_weighted_variance": de_var,
                    "variance_reduction": 1 - de_var / raw_var if np.isfinite(raw_var) and raw_var > 0 else np.nan,
                    "raw_weighted_iqr": raw_iqr,
                    "de_weighted_iqr": de_iqr,
                    "iqr_reduction": 1 - de_iqr / raw_iqr if np.isfinite(raw_iqr) and raw_iqr > 0 else np.nan,
                    "raw_bins": raw_bins,
                    "de_bins": de_bins,
                    "raw_station_bin_values": raw_values,
                    "de_station_bin_values": de_values,
                }
            )
    return pd.DataFrame(rows)


def plot_sensitivity(sens: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for min_stations, group in sens.groupby("min_stations"):
        axes[0].plot(group["n_bins"], 100 * group["variance_reduction"], marker="o", lw=1.2, label=f"n>={min_stations}")
        axes[1].plot(group["n_bins"], 100 * group["iqr_reduction"], marker="o", lw=1.2, label=f"n>={min_stations}")
    for ax, title, ylabel in [
        (axes[0], "Variance metric", "reduction [%]"),
        (axes[1], "IQR metric", "reduction [%]"),
    ]:
        ax.axhline(0, color="#7a7a7a", ls="--", lw=0.8)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel("log-spaced bins")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e5e2dc", lw=0.6)
    axes[1].legend(frameon=False, fontsize=6, ncol=1, loc="best")
    out = FIGURES / "nature_extended_collapse_sensitivity"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write_report(sens: pd.DataFrame) -> None:
    valid = sens.dropna(subset=["variance_reduction", "iqr_reduction"]).copy()
    var_min = valid["variance_reduction"].min()
    var_med = valid["variance_reduction"].median()
    var_max = valid["variance_reduction"].max()
    iqr_min = valid["iqr_reduction"].min()
    iqr_med = valid["iqr_reduction"].median()
    iqr_max = valid["iqr_reduction"].max()
    negative_var = int((valid["variance_reduction"] <= 0).sum())
    negative_iqr = int((valid["iqr_reduction"] <= 0).sum())
    verdict = "stable positive" if negative_var == 0 and negative_iqr == 0 else "mixed"
    lines = [
        "# De Collapse Sensitivity",
        "",
        "Date: 2026-05-31",
        "",
        "## Method",
        "",
        "The first-pass collapse metric was recomputed across log-bin counts (12-36) and minimum station thresholds (30-150). Two dispersion measures were tracked: weighted cross-catchment variance and weighted interquartile width of station-level median beta in each bin.",
        "",
        "## Result",
        "",
        f"- Variance-reduction range: {var_min:.1%} to {var_max:.1%}; median {var_med:.1%}.",
        f"- IQR-reduction range: {iqr_min:.1%} to {iqr_max:.1%}; median {iqr_med:.1%}.",
        f"- Non-positive variance-reduction configurations: {negative_var}/{len(valid)}.",
        f"- Non-positive IQR-reduction configurations: {negative_iqr}/{len(valid)}.",
        f"- Verdict: {verdict}.",
        "",
        "## Interpretation",
        "",
        "If all configurations remain positive, the manuscript can state that De normalization reduces cross-catchment beta dispersion across reasonable binning choices. The safer wording remains 'dispersion reduction' until alternative memory normalizations and global replication are tested.",
        "",
        "## Outputs",
        "",
        "- `reports/tables/camels_673_collapse_sensitivity.csv`",
        "- `reports/figures/nature_extended_collapse_sensitivity.png`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(CURVES_CSV)
    curves["gauge_id"] = curves["gauge_id"].astype(str).str.zfill(8)
    sens = run_sensitivity(curves)
    sens.to_csv(OUT_CSV, index=False)
    plot_sensitivity(sens)
    write_report(sens)


if __name__ == "__main__":
    main()
