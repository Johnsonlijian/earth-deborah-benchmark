"""R20 support-matched coordinate sensitivity checks.

Reviewers can ask whether De mapping reduces dispersion by changing coordinate
support or bin occupancy rather than by organizing the same local-slope points.
This script repeats the archive dispersion metric under stricter support
contracts: the same point set must be used for both axes, and equal-count bins
are used as an alternative to fixed log-spaced bins.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_de_collapse_metrics import dispersion_from_binned, station_bin_medians

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r20_support_matched_coordinate_sensitivity"


def load_archive(archive: str) -> tuple[pd.DataFrame, int]:
    if archive == "CAMELS-US":
        path = TABLES / "camels_673_beta_curve_points.csv"
        min_units = 50
    elif archive == "CAMELS-GB":
        path = TABLES / "camels_gb_v2_replication_beta_curve_points.csv"
        min_units = 45
    else:
        raise ValueError(archive)
    curves = pd.read_csv(path)
    curves["gauge_id"] = curves["gauge_id"].astype(str).str.zfill(8) if archive == "CAMELS-US" else curves["gauge_id"].astype(str)
    return curves.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency_cpd", "de", "beta"]), min_units


def metric_for(curves: pd.DataFrame, value_col: str, min_units: int, binning: str, n_bins: int = 24) -> tuple[float, int, int]:
    if binning == "log_spaced":
        binned = station_bin_medians(curves, value_col, n_bins=n_bins)
    elif binning == "equal_count":
        work = curves.copy()
        work["coord"] = np.log10(work[value_col].to_numpy(dtype=float))
        work = work.dropna(subset=["coord"])
        work["bin_id"] = pd.qcut(work["coord"], q=n_bins, labels=False, duplicates="drop")
        work = work.dropna(subset=["bin_id"])
        work["bin_id"] = work["bin_id"].astype(int)
        centers = work.groupby("bin_id")["coord"].median().to_dict()
        binned = (
            work.groupby(["gauge_id", "bin_id"], as_index=False)
            .agg(beta_median=("beta", "median"), n_points=("beta", "size"))
        )
        binned["axis_value"] = binned["bin_id"].map({int(k): 10 ** float(v) for k, v in centers.items()})
    else:
        raise ValueError(binning)
    result, _ = dispersion_from_binned(binned, min_stations=min_units)
    return result.weighted_variance, result.n_bins, result.n_station_bin_values


def central_support_filter(curves: pd.DataFrame) -> pd.DataFrame:
    work = curves.copy()
    log_f = np.log10(work["frequency_cpd"].to_numpy(dtype=float))
    log_de = np.log10(work["de"].to_numpy(dtype=float))
    f_lo, f_hi = np.nanpercentile(log_f, [2.5, 97.5])
    de_lo, de_hi = np.nanpercentile(log_de, [2.5, 97.5])
    keep = (log_f >= f_lo) & (log_f <= f_hi) & (log_de >= de_lo) & (log_de <= de_hi)
    return work.loc[keep].copy()


def summarize_archive(archive: str) -> pd.DataFrame:
    curves, min_units = load_archive(archive)
    rows = []
    for support_name, data in [
        ("standard_axis_specific_support", curves),
        ("same_point_central_support", central_support_filter(curves)),
    ]:
        for binning in ["log_spaced", "equal_count"]:
            raw_var, raw_bins, raw_units = metric_for(data, "frequency_cpd", min_units, binning)
            de_var, de_bins, de_units = metric_for(data, "de", min_units, binning)
            rows.append(
                {
                    "archive": archive,
                    "support_contract": support_name,
                    "binning": binning,
                    "n_points": int(len(data)),
                    "n_gauges": int(data["gauge_id"].nunique()),
                    "raw_axis_weighted_variance": raw_var,
                    "de_axis_weighted_variance": de_var,
                    "reduction_vs_raw": 1.0 - de_var / raw_var,
                    "raw_bins_used": raw_bins,
                    "de_bins_used": de_bins,
                    "raw_station_bin_values": raw_units,
                    "de_station_bin_values": de_units,
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
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
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    labels = []
    values = []
    colors = []
    palette = {
        "standard_axis_specific_support": "#8AAFD1",
        "same_point_central_support": "#4477AA",
    }
    for archive in ["CAMELS-US", "CAMELS-GB"]:
        data = summary[summary["archive"] == archive]
        for row in data.itertuples(index=False):
            labels.append(f"{archive}\n{row.support_contract.replace('_', ' ')}\n{row.binning.replace('_', ' ')}")
            values.append(100 * row.reduction_vs_raw)
            colors.append(palette[row.support_contract])
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.72)
    ax.axhline(0, color="#7B8794", lw=0.8)
    for xx, value in zip(x, values):
        ax.text(xx, value + 0.7, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.2)
    ax.set_ylabel("dispersion reduction vs raw axis [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=5.8)
    ax.set_title("R20 support-matched coordinate sensitivity", loc="left", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)
    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{OUT}{suffix}", **kwargs)
    plt.close(fig)


def write_note(summary: pd.DataFrame) -> None:
    lines = [
        "# R20 Support-Matched Coordinate Sensitivity",
        "",
        "This analysis tests whether the De reduction depends on changing the",
        "coordinate support or bin occupancy. The same-point central-support",
        "contract filters local-slope points once and then applies both raw and",
        "De axes to that identical point set. Equal-count bins test whether fixed",
        "log-spaced bins create the reduction.",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "Safe manuscript consequence: if reductions remain positive under the",
        "same-point and equal-count contracts, coordinate-support artifacts are",
        "less likely to explain the benchmark. This remains an observational",
        "dispersion test, not a causal storage proof.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([summarize_archive("CAMELS-US"), summarize_archive("CAMELS-GB")], ignore_index=True)
    summary.to_csv(TABLES / f"{OUT}.csv", index=False)
    plot_summary(summary)
    write_note(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
