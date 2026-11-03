"""Quantify CAMELS beta-curve dispersion before and after De normalization.

This script is a first-pass, reproducible collapse metric for the manuscript.
It recomputes local beta curves for CAMELS daily-discharge anomalies, maps them
to De using the already vetted tau_acf values, and compares cross-catchment
dispersion on raw-frequency and De-normalized axes.
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

from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope

ROOT = Path(__file__).resolve().parents[1]
FLOW_DIR = ROOT / "data" / "external" / "camels" / "usgs_streamflow"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
SUMMARY_CSV = TABLES / "camels_674_n15_summary.csv"
CURVES_CSV = TABLES / "camels_673_beta_curve_points.csv"
BIN_CSV = TABLES / "camels_673_collapse_bin_stats.csv"
METRICS_CSV = TABLES / "camels_673_collapse_metrics.csv"
REPORT_MD = NOTES / "de_collapse_metrics_20260531.md"
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class DispersionResult:
    axis: str
    weighted_variance: float
    mean_variance: float
    median_variance: float
    n_bins: int
    n_station_bin_values: int


def load_one_camels_flow(filepath: Path) -> tuple[str, pd.Series]:
    gauge_id = filepath.stem
    rows: list[tuple[pd.Timestamp, float]] = []
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                date = pd.Timestamp(year=int(parts[1]), month=int(parts[2]), day=int(parts[3]))
                q = float(parts[4])
            except ValueError:
                continue
            if q >= 0:
                rows.append((date, q))
    if not rows:
        return gauge_id, pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "q"]).sort_values("date")
    return gauge_id, df.set_index("date")["q"]


def prepare_daily_anomaly(series: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    df = series.to_frame("q").dropna()
    df["doy"] = df.index.dayofyear
    climo = df.groupby("doy")["q"].transform("mean")
    df["anomaly"] = df["q"] - climo
    df["anomaly"] = df["anomaly"].asfreq("D").interpolate(method="linear", limit=30)
    df = df.dropna(subset=["anomaly"])
    med = float(df["anomaly"].median())
    mad = float(np.median(np.abs(df["anomaly"].to_numpy() - med)))
    if mad > 0:
        df["anomaly"] = (df["anomaly"] - med) / mad
    return df.index, df["anomaly"].to_numpy(dtype=float)


def compute_curves(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    tau_by_gid = {
        str(row.gauge_id).zfill(8): float(row.tau_acf_days)
        for row in summary.itertuples(index=False)
        if np.isfinite(float(row.tau_acf_days)) and float(row.tau_acf_days) > 0
    }
    wanted = set(tau_by_gid)
    files = [fp for fp in sorted(FLOW_DIR.glob("*.txt")) if fp.stem in wanted]
    for idx, filepath in enumerate(files, start=1):
        gauge_id, series = load_one_camels_flow(filepath)
        tau_days = tau_by_gid.get(gauge_id)
        if tau_days is None or series.empty:
            continue
        try:
            dates, values = prepare_daily_anomaly(series)
            psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
            beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
            curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * SECONDS_PER_DAY)
        except Exception:
            continue
        curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
        curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
        if curve.empty:
            continue
        curve["gauge_id"] = gauge_id
        curve["tau_acf_days"] = tau_days
        curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
        rows.append(curve[["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]])
        if idx % 200 == 0:
            print(f"  beta curves computed for {idx}/{len(files)} files")
    if not rows:
        raise RuntimeError("No beta curve rows were computed.")
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=150, help="Bootstrap resamples of catchments.")
    parser.add_argument("--force-curves", action="store_true", help="Recompute beta curves even if cache exists.")
    return parser.parse_args()


def station_bin_medians(curves: pd.DataFrame, value_col: str, n_bins: int = 24) -> pd.DataFrame:
    x = np.log10(curves[value_col].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work = curves.copy()
    work["bin_id"] = pd.cut(np.log10(work[value_col]), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    grouped = (
        work.groupby(["gauge_id", "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"), n_points=("beta", "size"))
    )
    grouped["bin_center_log10"] = grouped["bin_id"].map({i: labels[i] for i in range(len(labels))})
    grouped["axis_value"] = 10 ** grouped["bin_center_log10"]
    return grouped


def dispersion_from_binned(binned: pd.DataFrame, min_stations: int = 50) -> tuple[DispersionResult, pd.DataFrame]:
    stats = (
        binned.groupby("bin_id", as_index=False)
        .agg(
            axis_value=("axis_value", "first"),
            n_stations=("gauge_id", "nunique"),
            beta_mean=("beta_median", "mean"),
            beta_median=("beta_median", "median"),
            beta_var=("beta_median", "var"),
            beta_iqr=("beta_median", lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))),
        )
        .dropna(subset=["beta_var"])
    )
    usable = stats[stats["n_stations"] >= min_stations].copy()
    if usable.empty:
        raise RuntimeError("No bins have enough stations for dispersion calculation.")
    weights = usable["n_stations"].to_numpy(dtype=float)
    variances = usable["beta_var"].to_numpy(dtype=float)
    result = DispersionResult(
        axis="",
        weighted_variance=float(np.average(variances, weights=weights)),
        mean_variance=float(np.mean(variances)),
        median_variance=float(np.median(variances)),
        n_bins=int(len(usable)),
        n_station_bin_values=int(weights.sum()),
    )
    return result, stats


def weighted_bin_dispersion(binned: pd.DataFrame, gauge_weights: dict[str, int], min_stations: int = 50) -> float:
    rows = []
    for _, group in binned.groupby("bin_id"):
        weights = group["gauge_id"].map(gauge_weights).fillna(0).to_numpy(dtype=float)
        keep = weights > 0
        if keep.sum() < min_stations:
            continue
        values = group.loc[keep, "beta_median"].to_numpy(dtype=float)
        weights = weights[keep]
        mean = np.average(values, weights=weights)
        var = np.average((values - mean) ** 2, weights=weights)
        rows.append((float(var), float(weights.sum())))
    if not rows:
        raise RuntimeError("No bootstrap bins have enough stations.")
    variances = np.array([item[0] for item in rows], dtype=float)
    weights = np.array([item[1] for item in rows], dtype=float)
    return float(np.average(variances, weights=weights))


def bootstrap_reduction(raw_binned: pd.DataFrame, de_binned: pd.DataFrame, n_boot: int = 150, seed: int = 20260531) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gauges = np.array(sorted(set(raw_binned["gauge_id"]).intersection(de_binned["gauge_id"])))
    rows = []
    for i in range(n_boot):
        sampled = rng.choice(gauges, size=len(gauges), replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        weights = {gid: int(count) for gid, count in zip(unique, counts)}
        raw_var = weighted_bin_dispersion(raw_binned, weights)
        de_var = weighted_bin_dispersion(de_binned, weights)
        rows.append(
            {
                "draw": i,
                "raw_weighted_variance": raw_var,
                "de_weighted_variance": de_var,
                "variance_reduction": 1.0 - de_var / raw_var,
            }
        )
    return pd.DataFrame(rows)


def plot_bin_stats(bin_stats: pd.DataFrame, bootstrap: pd.DataFrame, out_path: Path) -> None:
    raw = bin_stats[bin_stats["axis"] == "raw_frequency"].copy()
    de = bin_stats[bin_stats["axis"] == "de_normalized"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    for ax, data, title, xlabel in [
        (axes[0], raw, "Raw frequency", "frequency [cycles d$^{-1}$]"),
        (axes[1], de, "Deborah coordinate", "De"),
    ]:
        ax.semilogx(data["axis_value"], data["beta_median"], color="#1f6f78", lw=1.8)
        ax.fill_between(
            data["axis_value"],
            data["beta_median"] - 0.5 * data["beta_iqr"],
            data["beta_median"] + 0.5 * data["beta_iqr"],
            color="#6aa6a8",
            alpha=0.25,
            linewidth=0,
        )
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("median local beta")
        ax.grid(True, which="major", color="#e5e2dc", linewidth=0.6)
    vals = bootstrap["variance_reduction"].to_numpy(dtype=float)
    axes[2].hist(vals, bins=28, color="#c4513f", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[2].axvline(np.median(vals), color="#282828", lw=1.2)
    axes[2].axvline(0, color="#7a7a7a", lw=0.8, ls="--")
    axes[2].set_title("Bootstrap reduction", loc="left", fontsize=8)
    axes[2].set_xlabel("1 - Var_De / Var_raw")
    axes[2].set_ylabel("bootstrap draws")
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    if CURVES_CSV.exists() and not args.force_curves:
        curves = pd.read_csv(CURVES_CSV)
        curves["gauge_id"] = curves["gauge_id"].astype(str).str.zfill(8)
    else:
        curves = compute_curves(summary)
        curves.to_csv(CURVES_CSV, index=False)

    raw_binned = station_bin_medians(curves, "frequency_cpd")
    de_binned = station_bin_medians(curves, "de")
    raw_res, raw_stats = dispersion_from_binned(raw_binned)
    de_res, de_stats = dispersion_from_binned(de_binned)
    raw_res = DispersionResult(
        axis="raw_frequency",
        weighted_variance=raw_res.weighted_variance,
        mean_variance=raw_res.mean_variance,
        median_variance=raw_res.median_variance,
        n_bins=raw_res.n_bins,
        n_station_bin_values=raw_res.n_station_bin_values,
    )
    de_res = DispersionResult(
        axis="de_normalized",
        weighted_variance=de_res.weighted_variance,
        mean_variance=de_res.mean_variance,
        median_variance=de_res.median_variance,
        n_bins=de_res.n_bins,
        n_station_bin_values=de_res.n_station_bin_values,
    )

    raw_stats["axis"] = "raw_frequency"
    de_stats["axis"] = "de_normalized"
    bin_stats = pd.concat([raw_stats, de_stats], ignore_index=True)
    bin_stats.to_csv(BIN_CSV, index=False)

    bootstrap = bootstrap_reduction(raw_binned, de_binned, n_boot=args.n_boot)
    reduction = 1.0 - de_res.weighted_variance / raw_res.weighted_variance
    lo, med, hi = np.nanpercentile(bootstrap["variance_reduction"], [2.5, 50, 97.5])
    metrics = pd.DataFrame(
        [
            raw_res.__dict__,
            de_res.__dict__,
            {
                "axis": "variance_reduction",
                "weighted_variance": reduction,
                "mean_variance": med,
                "median_variance": med,
                "n_bins": np.nan,
                "n_station_bin_values": np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
            },
        ]
    )
    metrics.to_csv(METRICS_CSV, index=False)
    plot_bin_stats(bin_stats, bootstrap, FIGURES / "nature_extended_collapse_metric")

    verdict = "supports a strong dispersion-reduction claim" if lo > 0.0 else "does not yet support a strong dispersion-reduction claim"
    REPORT_MD.write_text(
        "\n".join(
            [
                "# De Collapse Dispersion Metric",
                "",
                "Date: 2026-05-31",
                "",
                "## Method",
                "",
                f"Local beta curves were recomputed for CAMELS discharge anomalies. For each catchment, beta was summarized in log-spaced bins on two axes: raw frequency (cycles per day) and Deborah-normalized frequency De = tau_acf f. The headline statistic is the weighted mean cross-catchment variance of station-level median beta values across bins with at least 50 catchments. Uncertainty comes from {args.n_boot} bootstrap resamples of catchments.",
                "",
                "## Result",
                "",
                f"- Raw-frequency weighted beta variance: {raw_res.weighted_variance:.4f}",
                f"- De-normalized weighted beta variance: {de_res.weighted_variance:.4f}",
                f"- Variance reduction: {reduction:.1%}",
                f"- Bootstrap median reduction: {med:.1%}",
                f"- Bootstrap 95% interval: {lo:.1%} to {hi:.1%}",
                f"- Verdict: {verdict}.",
                "",
                "## Interpretation",
                "",
                "This is a first-pass collapse metric, not the final submission statistic. It should be supplemented by sensitivity tests over bin counts, minimum station thresholds and alternative normalizations. If the confidence interval remains positive under these sensitivities, the manuscript can use a quantified dispersion-reduction claim. If it does not, the manuscript should downgrade from 'collapse' to 'De-organized spectral ordering'.",
                "",
                "## Outputs",
                "",
                f"- `{CURVES_CSV.relative_to(ROOT)}`",
                f"- `{BIN_CSV.relative_to(ROOT)}`",
                f"- `{METRICS_CSV.relative_to(ROOT)}`",
                "- `reports/figures/nature_extended_collapse_metric.png`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
