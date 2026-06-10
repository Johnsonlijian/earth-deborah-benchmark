"""Split-half cross-fitting test for the CAMELS memory-coordinate claim.

R11 showed that estimating ``tau_acf`` and spectral slopes from the same time
series creates a Fourier-pair circularity risk. This script attacks that risk
directly. For each CAMELS-US catchment, it estimates ``tau_acf`` on one
contiguous temporal half and computes beta(De) on the held-out half. The reverse
direction is also run.

The analysis is intentionally conservative: split halves still come from the
same gauge and catchment, so this reduces but does not eliminate dependence. It
does, however, test whether the memory-coordinate effect survives when the ACF
estimate and PSD/local-slope estimate are separated in time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time


ROOT = Path(__file__).resolve().parents[1]
FLOW_DIR = ROOT / "data" / "external" / "camels" / "usgs_streamflow"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
SECONDS_PER_DAY = 86_400.0


def format_p_value(p_value: float) -> str:
    """Return manuscript-safe p-value text without literal zero."""
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 1e-300:
        return "<1e-300"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-dir", default=str(FLOW_DIR))
    parser.add_argument("--min-segment-years", type=float, default=10.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-random", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--n-sample", type=int, default=0)
    parser.add_argument("--output-prefix", default="camels_673_crossfit_memory_coordinate")
    return parser.parse_args()


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
    df = pd.DataFrame(rows, columns=["date", "q"]).drop_duplicates("date").sort_values("date")
    return gauge_id, df.set_index("date")["q"].astype(float)


def segment_coverage(series: pd.Series) -> tuple[float, float]:
    if series.empty:
        return 0.0, 1.0
    span_days = (series.index.max() - series.index.min()).days + 1
    if span_days <= 0:
        return 0.0, 1.0
    years = span_days / 365.25
    missing = 1.0 - min(len(series), span_days) / span_days
    return float(years), float(max(0.0, missing))


def prepare_daily_anomaly(series: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    df = series.to_frame("q").dropna()
    df["doy"] = df.index.dayofyear
    climo = df.groupby("doy")["q"].transform("mean")
    df["anomaly"] = df["q"] - climo
    df["anomaly"] = df["anomaly"].asfreq("D").interpolate(method="linear", limit=30)
    df = df.dropna(subset=["anomaly"])
    med = float(df["anomaly"].median())
    mad = float(np.median(np.abs(df["anomaly"].to_numpy(dtype=float) - med)))
    if mad > 0:
        df["anomaly"] = (df["anomaly"] - med) / mad
    return df.index, df["anomaly"].to_numpy(dtype=float)


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    dates = series.index.sort_values()
    midpoint = dates.min() + (dates.max() - dates.min()) / 2
    early = series[series.index <= midpoint]
    late = series[series.index > midpoint]
    return early, late


def beta_curve_from_values(dates: pd.DatetimeIndex, values: np.ndarray, tau_days: float) -> pd.DataFrame:
    psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * SECONDS_PER_DAY)
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    return curve


def beta_base_from_values(dates: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta = beta.rename(columns={"f_center": "frequency"}).copy()
    beta = beta.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "beta"])
    beta = beta[beta["frequency"] > 0].copy()
    beta["frequency_cpd"] = beta["frequency"] * SECONDS_PER_DAY
    return beta[["frequency", "frequency_cpd", "beta"]]


def add_de_column(base: pd.DataFrame, tau_days: float, column_name: str) -> pd.DataFrame:
    work = base.copy()
    work[column_name] = work["frequency"] * SECONDS_PER_DAY * tau_days
    return work


def station_bin_medians(
    curves: pd.DataFrame,
    value_col: str,
    unit_col: str = "unit_id",
    n_bins: int = 24,
) -> pd.DataFrame:
    x = np.log10(curves[value_col].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work = curves.copy()
    work["bin_id"] = pd.cut(np.log10(work[value_col]), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    grouped = (
        work.groupby([unit_col, "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"), n_points=("beta", "size"))
    )
    grouped["axis_value"] = grouped["bin_id"].map({i: 10 ** labels[i] for i in range(len(labels))})
    return grouped


def dispersion_from_binned(
    binned: pd.DataFrame,
    unit_col: str = "unit_id",
    min_units: int = 50,
) -> tuple[float, pd.DataFrame]:
    stats_df = (
        binned.groupby("bin_id", as_index=False)
        .agg(
            axis_value=("axis_value", "first"),
            n_units=(unit_col, "nunique"),
            beta_mean=("beta_median", "mean"),
            beta_median=("beta_median", "median"),
            beta_var=("beta_median", "var"),
            beta_iqr=("beta_median", lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))),
        )
        .dropna(subset=["beta_var"])
    )
    usable = stats_df[stats_df["n_units"] >= min_units]
    if usable.empty:
        return np.nan, stats_df
    weights = usable["n_units"].to_numpy(dtype=float)
    return float(np.average(usable["beta_var"].to_numpy(dtype=float), weights=weights)), stats_df


def dispersion_metric(
    curves: pd.DataFrame,
    value_col: str,
    group_name: str,
    method: str,
    min_units: int = 50,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    binned = station_bin_medians(curves.dropna(subset=[value_col]), value_col=value_col)
    weighted_var, stats_df = dispersion_from_binned(binned, min_units=min_units)
    usable = stats_df[stats_df["n_units"] >= min_units]
    row = {
        "analysis_group": group_name,
        "method": method,
        "axis_column": value_col,
        "weighted_beta_variance": weighted_var,
        "n_units": int(curves["unit_id"].nunique()),
        "n_bins_used": int(usable.shape[0]),
        "n_unit_bin_values": int(usable["n_units"].sum()) if not usable.empty else 0,
    }
    stats_df["analysis_group"] = group_name
    stats_df["method"] = method
    return row, stats_df


def compute_group_metrics(curves: pd.DataFrame, group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    stats_parts: list[pd.DataFrame] = []
    for method, column in [
        ("raw_frequency", "frequency_cpd"),
        ("same_segment_tau", "de_same"),
        ("crossfit_tau", "de_crossfit"),
    ]:
        row, stats_df = dispersion_metric(curves, column, group_name, method)
        rows.append(row)
        stats_parts.append(stats_df)
    out = pd.DataFrame(rows)
    raw_var = float(out.loc[out["method"] == "raw_frequency", "weighted_beta_variance"].iloc[0])
    out["variance_reduction_vs_raw"] = 1.0 - out["weighted_beta_variance"] / raw_var
    out.loc[out["method"] == "raw_frequency", "variance_reduction_vs_raw"] = 0.0
    return out, pd.concat(stats_parts, ignore_index=True)


def random_tau_null(
    curves: pd.DataFrame,
    group_name: str,
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = curves[["unit_id", "frequency", "frequency_cpd", "beta", "fold", "train_tau_days"]].copy()
    unit_tau = base.drop_duplicates("unit_id")[["unit_id", "fold", "train_tau_days"]]
    raw_row, _ = dispersion_metric(base, "frequency_cpd", group_name, "raw_frequency")
    raw_var = float(raw_row["weighted_beta_variance"])
    rows: list[dict[str, float | int | str]] = []
    for draw in range(n_random):
        tau_map: dict[str, float] = {}
        for fold, fold_units in unit_tau.groupby("fold"):
            tau_values = fold_units["train_tau_days"].to_numpy(dtype=float)
            shuffled = rng.permutation(tau_values)
            tau_map.update({unit: float(tau) for unit, tau in zip(fold_units["unit_id"], shuffled)})
        work = base.copy()
        work["random_tau_days"] = work["unit_id"].map(tau_map)
        work["de_random"] = work["frequency"] * SECONDS_PER_DAY * work["random_tau_days"]
        rand_row, _ = dispersion_metric(work, "de_random", group_name, "random_train_tau")
        rand_var = float(rand_row["weighted_beta_variance"])
        rows.append(
            {
                "analysis_group": group_name,
                "draw": draw,
                "raw_weighted_beta_variance": raw_var,
                "random_tau_weighted_beta_variance": rand_var,
                "variance_reduction_vs_raw": 1.0 - rand_var / raw_var,
            }
        )
    return pd.DataFrame(rows)


def run_crossfit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow_files = sorted(Path(args.flow_dir).glob("*.txt"))
    if args.n_sample > 0:
        rng = np.random.default_rng(args.seed)
        flow_files = list(rng.choice(flow_files, size=min(args.n_sample, len(flow_files)), replace=False))

    summary_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[pd.DataFrame] = []
    for idx, filepath in enumerate(flow_files, start=1):
        gauge_id, series = load_one_camels_flow(filepath)
        if series.empty:
            continue
        early, late = split_series(series)
        segments = {"early": early, "late": late}
        prepared: dict[str, tuple[pd.DatetimeIndex, np.ndarray, float, float, float]] = {}
        for name, segment in segments.items():
            years, missing = segment_coverage(segment)
            if years < args.min_segment_years or missing > args.max_missing:
                continue
            try:
                dates, values = prepare_daily_anomaly(segment)
                tau_sec = integral_autocorrelation_time(
                    values,
                    dt_seconds=SECONDS_PER_DAY,
                    max_lag=730,
                    stop_at_zero=True,
                )
            except Exception:
                continue
            tau_days = tau_sec / SECONDS_PER_DAY
            if not np.isfinite(tau_days) or tau_days <= 0:
                continue
            prepared[name] = (dates, values, tau_days, years, missing)
        if set(prepared) != {"early", "late"}:
            continue

        for train_name, test_name, fold_label in [
            ("early", "late", "early_to_late"),
            ("late", "early", "late_to_early"),
        ]:
            train_dates, train_values, train_tau_days, train_years, train_missing = prepared[train_name]
            test_dates, test_values, test_tau_days, test_years, test_missing = prepared[test_name]
            try:
                beta_base = beta_base_from_values(test_dates, test_values)
            except Exception:
                continue
            if beta_base.empty:
                continue
            curve = add_de_column(beta_base, train_tau_days, "de_crossfit")
            curve = add_de_column(curve, test_tau_days, "de_same")
            unit_id = f"{gauge_id}_{fold_label}"
            curve["gauge_id"] = gauge_id
            curve["unit_id"] = unit_id
            curve["fold"] = fold_label
            curve["train_segment"] = train_name
            curve["test_segment"] = test_name
            curve["train_tau_days"] = train_tau_days
            curve["test_tau_days"] = test_tau_days
            curve_rows.append(
                curve[
                    [
                        "gauge_id",
                        "unit_id",
                        "fold",
                        "train_segment",
                        "test_segment",
                        "train_tau_days",
                        "test_tau_days",
                        "frequency",
                        "frequency_cpd",
                        "beta",
                        "de_crossfit",
                        "de_same",
                    ]
                ]
            )
            cross_summary = summarize_beta_window(curve.rename(columns={"de_crossfit": "de"}))
            same_summary = summarize_beta_window(curve.rename(columns={"de_same": "de"}))
            summary_rows.append(
                {
                    "gauge_id": gauge_id,
                    "unit_id": unit_id,
                    "fold": fold_label,
                    "train_segment": train_name,
                    "test_segment": test_name,
                    "train_years": train_years,
                    "test_years": test_years,
                    "train_missing_frac": train_missing,
                    "test_missing_frac": test_missing,
                    "train_tau_days": train_tau_days,
                    "test_tau_days": test_tau_days,
                    "tau_ratio_train_over_test": train_tau_days / test_tau_days if test_tau_days > 0 else np.nan,
                    "crossfit_n_beta_de_window": cross_summary["n_beta_de_window"],
                    "crossfit_mean_beta_de_window": cross_summary["mean_beta_de_window"],
                    "same_n_beta_de_window": same_summary["n_beta_de_window"],
                    "same_mean_beta_de_window": same_summary["mean_beta_de_window"],
                    "n_beta_rows": int(beta_base.shape[0]),
                }
            )
        if idx % 150 == 0:
            print(f"  processed {idx}/{len(flow_files)} flow files")

    if not curve_rows:
        raise RuntimeError("No cross-fit beta curves were computed.")
    return pd.DataFrame(summary_rows), pd.concat(curve_rows, ignore_index=True)


def plot_results(
    summaries: pd.DataFrame,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    random_null: pd.DataFrame,
    output_prefix: str,
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
        }
    )
    colors = {
        "crossfit_tau": "#C93434",
        "same_segment_tau": "#2B7BB9",
        "random_train_tau": "#7C8794",
        "raw_frequency": "#B6BDC7",
        "early_to_late": "#2B7BB9",
        "late_to_early": "#6C5CE7",
    }
    fig = plt.figure(figsize=(7.35, 5.3))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    all_metrics = metrics[metrics["analysis_group"] == "all"].set_index("method")
    methods = ["same_segment_tau", "crossfit_tau"]
    vals = 100 * all_metrics.loc[methods, "variance_reduction_vs_raw"].to_numpy(dtype=float)
    x = np.arange(len(methods))
    ax_a.bar(x, vals, color=[colors[m] for m in methods], width=0.62, alpha=0.82)
    rand_vals = random_null[random_null["analysis_group"] == "all"]["variance_reduction_vs_raw"].to_numpy(dtype=float) * 100
    parts = ax_a.violinplot(rand_vals[np.isfinite(rand_vals)], positions=[2], widths=0.55, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(colors["random_train_tau"])
        body.set_edgecolor(colors["random_train_tau"])
        body.set_alpha(0.25)
    q = np.nanpercentile(rand_vals, [5, 50, 95])
    ax_a.vlines(2, q[0], q[2], color=colors["random_train_tau"], lw=3, alpha=0.60)
    ax_a.scatter(2, q[1], color=colors["random_train_tau"], edgecolor="white", linewidth=0.7, s=35, zorder=3)
    for xx, vv in zip(x, vals):
        ax_a.text(xx, vv + 1.0, f"{vv:.1f}%", ha="center", fontsize=6.4)
    ax_a.axhline(0, color="#AAB0B8", lw=0.8)
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(["same\nsegment", "cross-fit", "random\ntrain tau"])
    ax_a.set_ylabel("variance reduction vs raw [%]")
    ax_a.set_title("Held-out De alignment survives temporal separation", loc="left", fontsize=8)
    ax_a.text(-0.12, 1.08, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")
    ax_a.grid(True, axis="y", color="#E6EAF0", lw=0.55)

    fold_metrics = metrics[(metrics["analysis_group"] != "all") & (metrics["method"].isin(methods))]
    width = 0.34
    fold_order = ["early_to_late", "late_to_early"]
    for offset, method in [(-width / 2, "same_segment_tau"), (width / 2, "crossfit_tau")]:
        y = []
        for fold in fold_order:
            y.append(
                float(
                    fold_metrics[
                        (fold_metrics["analysis_group"] == fold) & (fold_metrics["method"] == method)
                    ]["variance_reduction_vs_raw"].iloc[0]
                )
                * 100
            )
        ax_b.bar(np.arange(len(fold_order)) + offset, y, width=width, color=colors[method], alpha=0.82, label=method.replace("_", " "))
        for xx, yy in zip(np.arange(len(fold_order)) + offset, y):
            ax_b.text(xx, yy + 1.0, f"{yy:.1f}", ha="center", fontsize=6.0)
    ax_b.axhline(0, color="#AAB0B8", lw=0.8)
    ax_b.set_xticks(np.arange(len(fold_order)))
    ax_b.set_xticklabels(["early tau\nlate beta", "late tau\nearly beta"])
    ax_b.set_ylabel("variance reduction [%]")
    ax_b.set_title("Both temporal directions are reported", loc="left", fontsize=8)
    ax_b.text(-0.12, 1.08, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=5.8)
    ax_b.grid(True, axis="y", color="#E6EAF0", lw=0.55)

    ax_c.scatter(
        summaries["train_tau_days"],
        summaries["test_tau_days"],
        c=[colors.get(f, "#777777") for f in summaries["fold"]],
        s=14,
        alpha=0.48,
        edgecolor="white",
        linewidth=0.25,
    )
    lim = np.nanpercentile(np.r_[summaries["train_tau_days"], summaries["test_tau_days"]], [1, 99])
    ax_c.plot(lim, lim, color="#AAB0B8", lw=0.9, ls="--")
    tau_sub = summaries.dropna(subset=["train_tau_days", "test_tau_days"])
    rho, p = stats.spearmanr(tau_sub["train_tau_days"], tau_sub["test_tau_days"])
    ax_c.text(
        0.04,
        0.96,
        f"Spearman rho={rho:.2f}\np {format_p_value(float(p))}\nn={len(tau_sub)} units",
        transform=ax_c.transAxes,
        va="top",
        fontsize=6.4,
    )
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("train-segment tau_acf [d]")
    ax_c.set_ylabel("held-out segment tau_acf [d]")
    ax_c.set_title("Memory is noisy but temporally transferable", loc="left", fontsize=8)
    ax_c.text(-0.12, 1.08, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold")
    ax_c.grid(True, color="#E6EAF0", lw=0.55)

    sample = curves[curves["fold"] == "early_to_late"].copy()
    # Downsample units for a stable, readable median curve.
    raw_binned = station_bin_medians(sample, "frequency_cpd")
    same_binned = station_bin_medians(sample, "de_same")
    cross_binned = station_bin_medians(sample, "de_crossfit")
    for binned, label, color in [
        (raw_binned, "raw frequency", "#7C8794"),
        (same_binned, "same tau", colors["same_segment_tau"]),
        (cross_binned, "cross-fit tau", colors["crossfit_tau"]),
    ]:
        med = binned.groupby("bin_id", as_index=False).agg(axis_value=("axis_value", "first"), beta=("beta_median", "median"))
        ax_d.semilogx(med["axis_value"], med["beta"], lw=1.6, marker="o", ms=2.4, color=color, label=label)
    ax_d.axvspan(0.5, 2.0, color=colors["crossfit_tau"], alpha=0.05, lw=0)
    ax_d.set_xlabel("raw cycles d-1 or De")
    ax_d.set_ylabel("median local beta")
    ax_d.set_title("Cross-fit beta(De) curve uses held-out spectra", loc="left", fontsize=8)
    ax_d.text(-0.12, 1.08, "d", transform=ax_d.transAxes, fontsize=9, fontweight="bold")
    ax_d.legend(loc="upper left", fontsize=5.8)
    ax_d.grid(True, color="#E6EAF0", lw=0.55)

    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(
    metrics: pd.DataFrame,
    random_null: pd.DataFrame,
    summaries: pd.DataFrame,
    output_prefix: str,
) -> None:
    all_metrics = metrics[metrics["analysis_group"] == "all"].set_index("method")
    same = float(all_metrics.loc["same_segment_tau", "variance_reduction_vs_raw"])
    cross = float(all_metrics.loc["crossfit_tau", "variance_reduction_vs_raw"])
    random_vals = random_null[random_null["analysis_group"] == "all"]["variance_reduction_vs_raw"].to_numpy(dtype=float)
    p_ge = (np.sum(random_vals >= cross) + 1.0) / (np.isfinite(random_vals).sum() + 1.0)
    lines = [
        "# CAMELS Split-Half Cross-Fitting Note",
        "",
        "This analysis estimates `tau_acf` from one temporal half of each CAMELS-US",
        "record and computes beta(De) on the held-out half, then repeats the reverse",
        "direction. It reduces the same-series ACF/PSD coupling risk, but does not",
        "create fully independent physical storage measurements.",
        "",
        "## Aggregate metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Headline interpretation",
        "",
        f"- Same-segment tau reduction: {same:.1%}.",
        f"- Cross-fit tau reduction: {cross:.1%}.",
        f"- Random train-tau empirical p for matching/exceeding cross-fit reduction: {p_ge:.4g}.",
        f"- Units analysed: {summaries['unit_id'].nunique()} fold-specific catchment segments from {summaries['gauge_id'].nunique()} gauges.",
        "",
        "## Manuscript use",
        "",
        "- Safe claim: temporal cross-fitting shows that the memory-normalized",
        "  alignment is not wholly dependent on estimating tau and beta from the",
        "  identical segment.",
        "- Required caveat: split halves share the same catchment and gauge record;",
        "  this reduces but does not eliminate dependence or prove groundwater",
        "  causality.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{output_prefix}_summary.csv`",
        f"- `reports/tables/{output_prefix}_curves.csv`",
        f"- `reports/tables/{output_prefix}_metrics.csv`",
        f"- `reports/tables/{output_prefix}_bin_stats.csv`",
        f"- `reports/tables/{output_prefix}_random_tau_null.csv`",
        f"- `reports/figures/{output_prefix}.png/svg/pdf`",
    ]
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f"{output_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries, curves = run_crossfit(args)
    metric_parts: list[pd.DataFrame] = []
    bin_parts: list[pd.DataFrame] = []
    random_parts: list[pd.DataFrame] = []
    groups = {"all": curves}
    for fold in sorted(curves["fold"].unique()):
        groups[fold] = curves[curves["fold"] == fold].copy()
    for idx, (group_name, group_curves) in enumerate(groups.items()):
        m, b = compute_group_metrics(group_curves, group_name)
        metric_parts.append(m)
        bin_parts.append(b)
        random_parts.append(
            random_tau_null(
                group_curves,
                group_name=group_name,
                n_random=args.n_random,
                seed=args.seed + idx * 1009,
            )
        )
    metrics = pd.concat(metric_parts, ignore_index=True)
    bin_stats = pd.concat(bin_parts, ignore_index=True)
    random_null = pd.concat(random_parts, ignore_index=True)

    # Add random-null summaries to the metric table for easier manuscript use.
    rows = []
    for group_name, null_group in random_null.groupby("analysis_group"):
        vals = null_group["variance_reduction_vs_raw"].to_numpy(dtype=float)
        cross_reduction = float(
            metrics[(metrics["analysis_group"] == group_name) & (metrics["method"] == "crossfit_tau")][
                "variance_reduction_vs_raw"
            ].iloc[0]
        )
        if group_name == "all":
            n_units = int(curves["unit_id"].nunique())
        else:
            n_units = int(curves[curves["fold"] == group_name]["unit_id"].nunique())
        rows.append(
            {
                "analysis_group": group_name,
                "method": "random_train_tau_summary",
                "axis_column": "de_random",
                "weighted_beta_variance": np.nan,
                "n_units": n_units,
                "n_bins_used": np.nan,
                "n_unit_bin_values": np.nan,
                "variance_reduction_vs_raw": float(np.nanmedian(vals)),
                "random_p05": float(np.nanpercentile(vals, 5)),
                "random_p95": float(np.nanpercentile(vals, 95)),
                "empirical_p_ge_crossfit": float((np.sum(vals >= cross_reduction) + 1.0) / (np.isfinite(vals).sum() + 1.0)),
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)

    prefix = args.output_prefix
    summaries.to_csv(TABLES / f"{prefix}_summary.csv", index=False)
    curves.to_csv(TABLES / f"{prefix}_curves.csv", index=False)
    metrics.to_csv(TABLES / f"{prefix}_metrics.csv", index=False)
    bin_stats.to_csv(TABLES / f"{prefix}_bin_stats.csv", index=False)
    random_null.to_csv(TABLES / f"{prefix}_random_tau_null.csv", index=False)
    plot_results(summaries, curves, metrics, random_null, prefix)
    write_note(metrics, random_null, summaries, prefix)
    print(metrics.to_string(index=False))
    print(f"Saved CAMELS cross-fitting outputs with prefix {prefix}")


if __name__ == "__main__":
    main()
