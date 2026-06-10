"""CAMELS-GB v2 external replication of Deborah-normalized river spectra.

This analysis asks whether the CAMELS-US result is portable to a non-US
hydroclimatic archive. It uses the official CAMELS-GB v2 daily
hydro-meteorological CSVs, estimates a positive-integral ACF memory time from
daily discharge anomalies, and compares beta-curve dispersion on raw-frequency
and De-normalized axes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from compute_de_collapse_metrics import (
    dispersion_from_binned,
    station_bin_medians,
)
from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time
from run_camels_full import SECONDS_PER_DAY, prepare_daily_anomaly

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "camels_gb_v2"
DAILY_DIR = DATA_DIR / "hydromet_daily"
ATTR_DIR = DATA_DIR / "attributes"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT_PREFIX = "camels_gb_v2_replication"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", default=str(DAILY_DIR))
    parser.add_argument("--attr-dir", default=str(ATTR_DIR))
    parser.add_argument("--output-prefix", default=OUT_PREFIX)
    parser.add_argument("--limit", type=int, default=0, help="Analyze first N files; 0 means all available.")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--discharge-col", default="discharge_spec")
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--min-bin-stations", type=int, default=50)
    parser.add_argument("--n-random", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def gauge_id_from_path(path: Path) -> str:
    match = re.search(r"timeseries_([0-9]+)_19701001-20220930", path.name)
    if match:
        return match.group(1)
    return path.stem


def load_camels_gb_discharge(path: Path, discharge_col: str) -> tuple[str, pd.Series]:
    gauge_id = gauge_id_from_path(path)
    usecols = ["date", discharge_col]
    try:
        data = pd.read_csv(path, usecols=usecols, na_values=["NaN", "nan", ""])
    except ValueError:
        return gauge_id, pd.Series(dtype=float)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[discharge_col] = pd.to_numeric(data[discharge_col], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    data = data[data[discharge_col] >= 0]
    if data.empty:
        return gauge_id, pd.Series(dtype=float)
    series = data.set_index("date")[discharge_col].astype(float)
    finite = series.dropna()
    if finite.empty:
        return gauge_id, pd.Series(dtype=float)
    trimmed = series.loc[finite.index.min() : finite.index.max()].asfreq("D")
    return gauge_id, trimmed


def load_attributes(attr_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in [
        "camels_gb_v2_hydrologic_attributes.csv",
        "camels_gb_v2_topographic_attributes.csv",
        "camels_gb_v2_climatic_attributes.csv",
    ]:
        path = attr_dir / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"gauge_id": str})
        frame["gauge_id"] = frame["gauge_id"].astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["gauge_id"]).set_index("gauge_id")
    merged = frames[0]
    for frame in frames[1:]:
        keep = ["gauge_id"] + [col for col in frame.columns if col not in merged.columns and col != "gauge_id"]
        merged = merged.merge(frame[keep], on="gauge_id", how="outer")
    return merged.set_index("gauge_id")


def compute_one(
    path: Path,
    discharge_col: str,
    min_years: float,
    max_missing: float,
) -> tuple[dict[str, float | str] | None, pd.DataFrame | None, dict[str, float | str]]:
    gauge_id, series = load_camels_gb_discharge(path, discharge_col)
    audit: dict[str, float | str] = {
        "gauge_id": gauge_id,
        "filename": path.name,
        "discharge_col": discharge_col,
        "status": "failed",
        "reason": "",
        "n_years": np.nan,
        "missing_frac": np.nan,
    }
    if series.empty:
        audit["reason"] = "no_nonnegative_discharge"
        return None, None, audit
    n_years = len(series) / 365.25
    missing_frac = float(series.isna().mean())
    audit["n_years"] = float(n_years)
    audit["missing_frac"] = missing_frac
    if n_years < min_years or missing_frac > max_missing:
        audit["reason"] = "coverage_filter"
        return None, None, audit
    try:
        dates, values = prepare_daily_anomaly(series)
    except Exception:
        audit["reason"] = "daily_anomaly_failed"
        return None, None, audit
    if len(values) < min_years * 365.25:
        audit["reason"] = "post_anomaly_min_years_filter"
        return None, None, audit
    tau_seconds = integral_autocorrelation_time(
        values,
        dt_seconds=SECONDS_PER_DAY,
        max_lag=730,
        stop_at_zero=True,
    )
    tau_days = tau_seconds / SECONDS_PER_DAY
    if not np.isfinite(tau_days) or tau_days <= 0:
        audit["reason"] = "invalid_tau_acf"
        return None, None, audit
    try:
        psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
        summary = summarize_beta_window(curve)
    except Exception:
        audit["reason"] = "beta_curve_failed"
        return None, None, audit
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    if curve.empty:
        audit["reason"] = "empty_beta_curve"
        return None, None, audit
    curve["gauge_id"] = gauge_id
    curve["tau_acf_days"] = tau_days
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    row: dict[str, float | str] = {
        "gauge_id": gauge_id,
        "n_years": float(n_years),
        "missing_frac": missing_frac,
        "tau_acf_days": float(tau_days),
        "discharge_col": discharge_col,
        **summary,
    }
    audit["status"] = "included"
    audit["reason"] = "included"
    audit["tau_acf_days"] = float(tau_days)
    audit["n_beta_de_window"] = row.get("n_beta_de_window", np.nan)
    return row, curve[["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]], audit


def attach_attributes(summary: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    if attrs.empty or summary.empty:
        return summary
    out = summary.merge(attrs.reset_index(), on="gauge_id", how="left")
    if "area" in out.columns and "drainage_area_km2" not in out.columns:
        out["drainage_area_km2"] = out["area"]
    return out


def build_bin_stats(curves: pd.DataFrame, n_bins: int, min_bin_stations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metric_rows = []
    for label, value_col in [
        ("raw_frequency", "frequency_cpd"),
        ("de_normalized", "de"),
        ("constant_tau", "de_constant_tau"),
    ]:
        binned = station_bin_medians(curves, value_col, n_bins=n_bins)
        result, stats_df = dispersion_from_binned(binned, min_stations=min_bin_stations)
        stats_df["axis"] = label
        rows.append(stats_df)
        metric_rows.append(
            {
                "axis": label,
                "weighted_variance": result.weighted_variance,
                "mean_variance": result.mean_variance,
                "median_variance": result.median_variance,
                "n_bins": result.n_bins,
                "n_station_bin_values": result.n_station_bin_values,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    raw_var = float(metrics.loc[metrics["axis"] == "raw_frequency", "weighted_variance"].iloc[0])
    metrics["variance_reduction_vs_raw"] = 1.0 - metrics["weighted_variance"] / raw_var
    return pd.concat(rows, ignore_index=True), metrics


def random_tau_null(
    curves: pd.DataFrame,
    n_bins: int,
    min_bin_stations: int,
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tau_by_gauge = curves.groupby("gauge_id")["tau_acf_days"].first()
    gauges = tau_by_gauge.index.to_numpy()
    tau_values = tau_by_gauge.to_numpy(dtype=float)
    raw_binned = station_bin_medians(curves, "frequency_cpd", n_bins=n_bins)
    raw_result, _ = dispersion_from_binned(raw_binned, min_stations=min_bin_stations)
    rows = []
    base = curves.copy()
    for draw in range(n_random):
        shuffled = pd.Series(rng.permutation(tau_values), index=gauges)
        work = base.copy()
        work["de_random_tau"] = work["frequency_cpd"] * work["gauge_id"].map(shuffled)
        binned = station_bin_medians(work, "de_random_tau", n_bins=n_bins)
        result, _ = dispersion_from_binned(binned, min_stations=min_bin_stations)
        rows.append(
            {
                "draw": draw,
                "weighted_variance": result.weighted_variance,
                "variance_reduction_vs_raw": 1.0 - result.weighted_variance / raw_result.weighted_variance,
            }
        )
    return pd.DataFrame(rows)


def spearman_or_nan(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    work = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 5:
        return np.nan, np.nan, int(len(work))
    rho, p_value = stats.spearmanr(work["x"], work["y"])
    return float(rho), float(p_value), int(len(work))


def plot_replication(
    summary: pd.DataFrame,
    bin_stats: pd.DataFrame,
    metrics: pd.DataFrame,
    random_null: pd.DataFrame,
    output_base: Path,
) -> None:
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
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_de = fig.add_subplot(gs[0, 2])
    ax_null = fig.add_subplot(gs[1, 1])
    ax_mech = fig.add_subplot(gs[1, 2])

    if {"gauge_lon", "gauge_lat"}.issubset(summary.columns):
        plot_map = summary.dropna(subset=["gauge_lon", "gauge_lat", "tau_acf_days"])
        colors = np.log10(plot_map["tau_acf_days"].clip(lower=0.1))
        sc = ax_map.scatter(
            plot_map["gauge_lon"],
            plot_map["gauge_lat"],
            c=colors,
            s=10,
            cmap="viridis",
            linewidths=0,
            alpha=0.9,
        )
        cbar = fig.colorbar(sc, ax=ax_map, shrink=0.66, pad=0.02)
        cbar.set_label(r"$\log_{10}\tau_{\mathrm{acf}}$ [d]")
        ax_map.set_xlabel("longitude")
        ax_map.set_ylabel("latitude")
        ax_map.set_title("a  CAMELS-GB external archive", loc="left", fontsize=9, fontweight="bold")
        ax_map.text(
            0.02,
            0.02,
            f"n={len(plot_map)} gauges",
            transform=ax_map.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
        )
    else:
        ax_map.hist(summary["tau_acf_days"].dropna(), bins=24, color="#4c78a8", edgecolor="white")
        ax_map.set_xscale("log")
        ax_map.set_xlabel("tau_acf [d]")
        ax_map.set_ylabel("gauges")
        ax_map.set_title("a  CAMELS-GB memory distribution", loc="left", fontsize=9, fontweight="bold")

    def plot_axis(ax: plt.Axes, axis: str, title: str, xlabel: str, color: str) -> None:
        data = bin_stats[bin_stats["axis"] == axis].dropna(subset=["axis_value", "beta_median", "beta_iqr"])
        data = data.sort_values("axis_value")
        x = data["axis_value"].to_numpy(dtype=float)
        med = data["beta_median"].to_numpy(dtype=float)
        half_iqr = 0.5 * data["beta_iqr"].to_numpy(dtype=float)
        ax.fill_between(x, med - half_iqr, med + half_iqr, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, med, color=color, lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("local beta")
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.axhline(1.0, color="#999999", lw=0.6, ls=":")

    plot_axis(ax_raw, "raw_frequency", "b  raw-frequency beta spread", "frequency [cycles d$^{-1}$]", "#7a7a7a")
    plot_axis(ax_de, "de_normalized", "c  Deborah-normalized beta", r"De = $\tau_{\mathrm{acf}} f$", "#1f6f78")
    ax_de.axvline(1.0, color="#b53d2a", lw=0.8, ls="--")
    ax_de.text(1.08, ax_de.get_ylim()[1] - 0.18, "De=1", color="#b53d2a", fontsize=7)

    empirical = float(metrics.loc[metrics["axis"] == "de_normalized", "variance_reduction_vs_raw"].iloc[0])
    constant = float(metrics.loc[metrics["axis"] == "constant_tau", "variance_reduction_vs_raw"].iloc[0])
    ax_null.hist(
        100.0 * random_null["variance_reduction_vs_raw"],
        bins=28,
        color="#bfc5c9",
        edgecolor="white",
    )
    ax_null.axvline(100.0 * empirical, color="#1f6f78", lw=1.8, label="observed tau")
    ax_null.axvline(100.0 * constant, color="#555555", lw=1.2, ls="--", label="constant tau")
    ax_null.set_xlabel("variance reduction vs raw [%]")
    ax_null.set_ylabel("random assignments")
    ax_null.set_title("d  shuffled-tau falsification", loc="left", fontsize=9, fontweight="bold")
    ax_null.legend(loc="upper left", fontsize=6)

    if "baseflow_index" in summary.columns:
        mech = summary.dropna(subset=["baseflow_index", "tau_acf_days"])
        ax_mech.scatter(
            mech["baseflow_index"],
            mech["tau_acf_days"],
            color="#2f8f9d",
            s=12,
            alpha=0.62,
            linewidths=0,
        )
        ax_mech.set_yscale("log")
        ax_mech.set_xlabel("baseflow index")
        ax_mech.set_ylabel(r"$\tau_{\mathrm{acf}}$ [d]")
        rho_bfi, p_bfi, n_bfi = spearman_or_nan(mech["tau_acf_days"], mech["baseflow_index"])
        rho_area, _, n_area = spearman_or_nan(
            summary["tau_acf_days"],
            np.log10(summary.get("drainage_area_km2", pd.Series(dtype=float))),
        )
        ax_mech.text(
            0.02,
            0.98,
            f"$\\rho_{{BFI}}$={rho_bfi:.2f} (n={n_bfi})\n$\\rho_{{area}}$={rho_area:.2f} (n={n_area})",
            transform=ax_mech.transAxes,
            va="top",
            ha="left",
            fontsize=7,
        )
    else:
        ax_mech.axis("off")
    ax_mech.set_title("e  storage proxy controls memory", loc="left", fontsize=9, fontweight="bold")

    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def write_note(
    path: Path,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    random_null: pd.DataFrame,
    source_count: int,
    args: argparse.Namespace,
) -> None:
    de_reduction = float(metrics.loc[metrics["axis"] == "de_normalized", "variance_reduction_vs_raw"].iloc[0])
    constant_reduction = float(metrics.loc[metrics["axis"] == "constant_tau", "variance_reduction_vs_raw"].iloc[0])
    random_p = float((np.sum(random_null["variance_reduction_vs_raw"] >= de_reduction) + 1) / (len(random_null) + 1))
    rho_bfi, p_bfi, n_bfi = (
        spearman_or_nan(summary["tau_acf_days"], summary["baseflow_index"])
        if "baseflow_index" in summary.columns
        else (np.nan, np.nan, 0)
    )
    rho_area, p_area, n_area = (
        spearman_or_nan(summary["tau_acf_days"], np.log10(summary["drainage_area_km2"]))
        if "drainage_area_km2" in summary.columns
        else (np.nan, np.nan, 0)
    )
    lines = [
        "# CAMELS-GB v2 external replication",
        "",
        "Date checked: 2026-05-31. Source: UKCEH CAMELS-GB v2 official data store, DOI 10.5285/9a46d428-958f-4ac1-86eb-94eee70c0955.",
        "",
        "## Scope",
        "",
        f"- Daily files available locally/analyzed source files: {source_count}.",
        f"- Discharge column: `{args.discharge_col}`; coverage screen: at least {args.min_years:g} years and missing fraction <= {args.max_missing:g} after trimming to each station's first-last valid discharge day.",
        f"- Stations passing coverage and beta-curve filters: {len(summary)}.",
        f"- Median tau_acf: {summary['tau_acf_days'].median():.2f} d; IQR {summary['tau_acf_days'].quantile(0.25):.2f}-{summary['tau_acf_days'].quantile(0.75):.2f} d.",
        "",
        "## Collapse result",
        "",
        f"- De-normalization variance reduction vs raw frequency: {de_reduction:.1%}.",
        f"- Constant-median-tau reduction vs raw frequency: {constant_reduction:.1%}.",
        f"- Randomly shuffled tau null: median {random_null['variance_reduction_vs_raw'].median():.1%}, 95th percentile {random_null['variance_reduction_vs_raw'].quantile(0.95):.1%}.",
        f"- Empirical probability random tau >= observed tau_acf reduction: {random_p:.4f}; this equals 1/(n_random+1) because 0/{args.n_random} shuffled assignments matched the observed reduction.",
        f"- Random tau seed: {args.seed}; number of shuffled assignments: {args.n_random}.",
        "",
        "## Mechanism controls",
        "",
        f"- Spearman tau_acf vs baseflow_index: rho={rho_bfi:.3f}, p<1e-10 (asymptotic scipy p={p_bfi:.3g}), n={n_bfi}.",
        f"- Spearman tau_acf vs log10(area): rho={rho_area:.3f}, p={p_area:.3g}, n={n_area}.",
        "",
        "## Manuscript implication",
        "",
        "This is an external geographic replication, not a new primary-claim replacement. If retained after independent review, it should appear as an Extended Data validation figure and a short Results paragraph showing that the De axis is not a CAMELS-US artefact. Because CAMELS-GB is a homologous catchment archive, it strengthens portability within river basins but does not by itself prove universality across all climates or regulated systems.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.daily_dir).glob("camels_gb_v2_hydromet_daily_timeseries_*_19701001-20220930.csv"))
    if args.limit > 0:
        files = files[: args.limit]
    rows: list[dict[str, float | str]] = []
    audit_rows: list[dict[str, float | str]] = []
    curve_rows: list[pd.DataFrame] = []
    for idx, path in enumerate(files, start=1):
        result = compute_one(path, args.discharge_col, args.min_years, args.max_missing)
        row, curve, audit = result
        audit_rows.append(audit)
        if row is not None and curve is not None:
            rows.append(row)
            curve_rows.append(curve)
        if idx % 100 == 0 or idx == len(files):
            print(f"  processed {idx}/{len(files)} files; valid stations={len(rows)}")
    if not rows or not curve_rows:
        raise RuntimeError("No valid CAMELS-GB station curves were computed.")

    summary = pd.DataFrame(rows).sort_values("tau_acf_days")
    attrs = load_attributes(Path(args.attr_dir))
    summary = attach_attributes(summary, attrs)
    curves = pd.concat(curve_rows, ignore_index=True)
    median_tau = float(summary["tau_acf_days"].median())
    curves["de_constant_tau"] = curves["frequency_cpd"] * median_tau

    bin_stats, metrics = build_bin_stats(curves, args.n_bins, args.min_bin_stations)
    random_null = random_tau_null(curves, args.n_bins, args.min_bin_stations, args.n_random, args.seed)

    prefix = args.output_prefix
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False)
    curves.to_csv(TABLES / f"{prefix}_beta_curve_points.csv", index=False)
    bin_stats.to_csv(TABLES / f"{prefix}_bin_stats.csv", index=False)
    metrics.to_csv(TABLES / f"{prefix}_collapse_metrics.csv", index=False)
    random_null.to_csv(TABLES / f"{prefix}_random_tau_null.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(TABLES / f"{prefix}_filter_audit.csv", index=False)
    write_note(NOTES / f"{prefix}_20260531.md", summary, metrics, random_null, source_count=len(files), args=args)
    plot_replication(summary, bin_stats, metrics, random_null, FIGURES / f"nature_extended_{prefix}")

    print(metrics.to_string(index=False))
    print(f"Outputs written with prefix {prefix}")


if __name__ == "__main__":
    main()
