"""R25 CAMELS-AUS v2 fourth-archive memory-coordinate replication.

This script extends the streamflow spectral-memory benchmark to CAMELS-AUS v2.
It uses the public Zenodo streamflow_mmd wide table and master attributes,
without redistributing raw third-party files. The analysis mirrors the
CAMELS-BR R23 contract: estimate discharge memory, local spectral slopes,
raw-frequency versus De dispersion, shuffled-memory specificity, subsample
stability and independent storage/proxy associations.
"""

from __future__ import annotations

import argparse
import zipfile
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
from edb.signal.timescales import integral_autocorrelation_time

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "camels_aus_v2"
STREAM_ZIP = DATA_DIR / "03_streamflow.zip"
MASTER = DATA_DIR / "CAMELS_AUS_Attributes&Indices_MasterTable.csv"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r25_camels_aus_fourth_archive"
SECONDS_PER_DAY = 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Maximum stations; 0 means all.")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--min-bin-gauges", type=int, default=40)
    parser.add_argument("--n-subsample", type=int, default=300)
    parser.add_argument("--subsample-size", type=int, default=150)
    parser.add_argument("--n-random-tau", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def load_streamflow_wide() -> pd.DataFrame:
    with zipfile.ZipFile(STREAM_ZIP) as zf:
        with zf.open("streamflow_mmd.csv") as fh:
            data = pd.read_csv(fh, na_values=["-99.99", -99.99, "n/a", "NA"])
    dates = pd.to_datetime(dict(year=data["year"], month=data["month"], day=data["day"]), errors="coerce")
    data = data.drop(columns=["year", "month", "day"])
    data.index = dates
    data = data[~data.index.isna()].sort_index()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").where(lambda s: s >= 0)
    return data.asfreq("D")


def load_master() -> pd.DataFrame:
    attrs = pd.read_csv(MASTER, dtype={"station_id": str})
    return attrs.rename(columns={"station_id": "gauge_id"})


def station_span(q: pd.Series) -> pd.Series:
    valid = q.dropna()
    if valid.empty:
        return valid
    return q.loc[valid.index.min() : valid.index.max()]


def coverage(q: pd.Series) -> tuple[float, float, float]:
    if q.empty:
        return 0.0, 1.0, np.nan
    span_days = (q.index.max() - q.index.min()).days + 1
    n_years = span_days / 365.25
    missing = float(q.isna().mean())
    valid = q.dropna()
    zero_fraction = float((valid <= 0).mean()) if len(valid) else np.nan
    return n_years, missing, zero_fraction


def prepare_anomaly(q: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.DataFrame({"q": q}).replace([np.inf, -np.inf], np.nan)
    frame["q"] = frame["q"].interpolate(limit=30, limit_direction="both")
    frame = frame.dropna()
    frame["doy"] = frame.index.dayofyear
    frame["anom"] = frame["q"] - frame.groupby("doy")["q"].transform("mean")
    med = float(frame["anom"].median())
    mad = float(np.median(np.abs(frame["anom"].to_numpy(dtype=float) - med)))
    if mad > 0:
        frame["anom"] = (frame["anom"] - med) / mad
    return frame.index, frame["anom"].to_numpy(dtype=float)


def analyze_one(gauge_id: str, q_all: pd.Series, args: argparse.Namespace) -> tuple[dict | None, pd.DataFrame | None]:
    q = station_span(q_all)
    n_years, missing, zero_fraction = coverage(q)
    if n_years < args.min_years or missing > args.max_missing:
        return None, None
    try:
        dates, x = prepare_anomaly(q)
        if len(x) < int(365 * args.min_years):
            return None, None
        tau_seconds = integral_autocorrelation_time(x, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_days = tau_seconds / SECONDS_PER_DAY
        if not np.isfinite(tau_days) or tau_days <= 0:
            return None, None
        psd = welch_psd(dates, x, fs=1.0 / SECONDS_PER_DAY)
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    except Exception:
        return None, None
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    if len(curve) < 20:
        return None, None
    curve["gauge_id"] = gauge_id
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["tau_days"] = tau_days
    row = {
        "gauge_id": gauge_id,
        "start_date_used": q.index.min().date().isoformat(),
        "end_date_used": q.index.max().date().isoformat(),
        "n_years": n_years,
        "missing_fraction": missing,
        "zero_flow_fraction": zero_fraction,
        "tau_acf_days": tau_days,
        "n_beta_rows": int(len(curve)),
        "median_beta_all": float(curve["beta"].median()),
    }
    return row, curve[["gauge_id", "frequency", "frequency_cpd", "de", "beta", "tau_days"]]


def dispersion_metrics(curves: pd.DataFrame, coord: str, n_bins: int, min_bin_gauges: int) -> tuple[pd.DataFrame, dict]:
    work = curves[["gauge_id", coord, "beta"]].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[coord] > 0) & np.isfinite(work["beta"])]
    if work.empty:
        return pd.DataFrame(), {"weighted_beta_variance": np.nan, "n_bins": 0, "n_gauge_bin_pairs": 0}
    logx = np.log10(work[coord].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(logx, [2, 98])
    edges = np.linspace(lo, hi, n_bins + 1)
    work["bin"] = pd.cut(logx, bins=edges, labels=False, include_lowest=True)
    med = work.dropna(subset=["bin"]).groupby(["bin", "gauge_id"])["beta"].median().reset_index()
    rows = []
    for bin_id, group in med.groupby("bin"):
        n = group["gauge_id"].nunique()
        if n < min_bin_gauges:
            continue
        rows.append(
            {
                "axis": coord,
                "bin": int(bin_id),
                "n_gauges": int(n),
                "coord_median": float(10 ** np.nanmedian(logx[work["bin"].to_numpy() == bin_id])),
                "beta_median": float(group["beta"].median()),
                "beta_iqr": float(group["beta"].quantile(0.75) - group["beta"].quantile(0.25)),
                "beta_variance": float(group["beta"].var(ddof=1)),
            }
        )
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df, {"weighted_beta_variance": np.nan, "n_bins": 0, "n_gauge_bin_pairs": 0}
    weighted = float(np.average(stats_df["beta_variance"], weights=stats_df["n_gauges"]))
    return stats_df, {
        "weighted_beta_variance": weighted,
        "n_bins": int(len(stats_df)),
        "n_gauge_bin_pairs": int(stats_df["n_gauges"].sum()),
    }


def reduction_for_curves(curves: pd.DataFrame, args: argparse.Namespace, de_col: str = "de") -> dict:
    _raw_stats, raw = dispersion_metrics(curves, "frequency_cpd", args.n_bins, args.min_bin_gauges)
    _de_stats, de = dispersion_metrics(curves.rename(columns={de_col: "de_eval"}), "de_eval", args.n_bins, args.min_bin_gauges)
    raw_var = raw["weighted_beta_variance"]
    de_var = de["weighted_beta_variance"]
    reduction = (raw_var - de_var) / raw_var if raw_var and np.isfinite(raw_var) else np.nan
    return {
        "raw_weighted_beta_variance": raw_var,
        "de_weighted_beta_variance": de_var,
        "variance_reduction_fraction": reduction,
        "raw_n_bins": raw["n_bins"],
        "de_n_bins": de["n_bins"],
    }


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return pd.Series(q, index=pvals.index)
    order = np.argsort(p[mask])
    ranked = p[mask][order]
    m = ranked.size
    vals = ranked * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.clip(vals, 0.0, 1.0)
    q_masked = np.empty_like(ranked)
    q_masked[order] = vals
    q[mask] = q_masked
    return pd.Series(q, index=pvals.index)


def rank_residual(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    data = pd.concat([y.rename("y"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return pd.Series(dtype=float)
    yr = data["y"].rank().to_numpy(dtype=float)
    xr = data.drop(columns=["y"]).rank().to_numpy(dtype=float)
    xr = np.column_stack([np.ones(len(data)), xr])
    beta, *_ = np.linalg.lstsq(xr, yr, rcond=None)
    resid = yr - xr @ beta
    return pd.Series(resid, index=data.index)


def stability_tables(curves: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    gids = np.array(sorted(curves["gauge_id"].unique()))
    subsample_rows = []
    n_take = min(args.subsample_size, len(gids))
    for i in range(args.n_subsample):
        sample = rng.choice(gids, size=n_take, replace=False)
        result = reduction_for_curves(curves[curves["gauge_id"].isin(sample)].copy(), args)
        result["draw"] = i
        result["n_gauges"] = n_take
        subsample_rows.append(result)

    tau_by_gauge = curves.groupby("gauge_id")["tau_days"].first()
    random_rows = []
    for i in range(args.n_random_tau):
        shuffled = pd.Series(rng.permutation(tau_by_gauge.to_numpy()), index=tau_by_gauge.index)
        work = curves.copy()
        work["tau_random"] = work["gauge_id"].map(shuffled)
        work["de_random"] = work["frequency_cpd"] * work["tau_random"]
        result = reduction_for_curves(work, args, de_col="de_random")
        result["draw"] = i
        random_rows.append(result)
    return pd.DataFrame(subsample_rows), pd.DataFrame(random_rows)


def attribute_associations(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy().replace([np.inf, -np.inf], np.nan)
    work["log_area"] = np.log10(pd.to_numeric(work["catchment_area"], errors="coerce"))
    work["log_tau"] = np.log10(pd.to_numeric(work["tau_acf_days"], errors="coerce"))
    candidates = [
        ("sig_mag_BFI", "streamflow_derived"),
        ("sig_roc_AC1", "streamflow_derived"),
        ("sig_roc_BaseRecesK", "streamflow_derived"),
        ("sig_other_StorageFromBase", "streamflow_derived"),
        ("ksat", "independent_soil_proxy"),
        ("solpawhc", "independent_soil_proxy"),
        ("sanda", "independent_soil_proxy"),
        ("claya", "independent_soil_proxy"),
        ("clayb", "independent_soil_proxy"),
        ("unconsoldted", "independent_geology_proxy"),
        ("carbnatesed", "independent_geology_proxy"),
        ("oldrock", "independent_geology_proxy"),
        ("mrvbf_prop_7", "independent_topographic_valley_proxy"),
        ("mrvbf_prop_8", "independent_topographic_valley_proxy"),
        ("aridity", "hydroclimatic_covariate"),
        ("zero_flow_fraction", "streamflow_regime_boundary"),
        ("impound_fac", "anthropogenic_boundary"),
        ("flow_regime_di", "anthropogenic_boundary"),
    ]
    controls = work[["log_area", "aridity", "p_seasonality"]].apply(pd.to_numeric, errors="coerce")
    rows = []
    for attr, attr_type in candidates:
        if attr not in work.columns:
            continue
        work[attr] = pd.to_numeric(work[attr], errors="coerce")
        data = work[["log_tau", attr]].dropna()
        if len(data) < 50:
            continue
        rho, pval = stats.spearmanr(data[attr], data["log_tau"])
        control_cols = [c for c in ["log_area", "aridity", "p_seasonality"] if c != attr]
        common = pd.concat(
            [work["log_tau"].rename("log_tau"), work[attr].rename("x"), controls[control_cols]],
            axis=1,
        ).dropna()
        if len(common) >= 50:
            yres = rank_residual(common["log_tau"], common[control_cols])
            xres = rank_residual(common["x"], common[control_cols])
            aligned = pd.concat([yres.rename("y"), xres.rename("x")], axis=1).dropna()
            prho, pp = stats.spearmanr(aligned["x"], aligned["y"]) if len(aligned) >= 50 else (np.nan, np.nan)
        else:
            prho, pp = np.nan, np.nan
        rows.append(
            {
                "attribute": attr,
                "attribute_type": attr_type,
                "n": int(len(data)),
                "spearman_rho_logtau": float(rho),
                "p_value": float(pval),
                "partial_rank_rho_logtau": float(prho),
                "partial_p_value": float(pp),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh"] = bh_qvalues(out["p_value"])
        out["partial_q_value_bh"] = bh_qvalues(out["partial_p_value"])
    return out


def strata_metrics(curves: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    meta = summary[["gauge_id", "aridity", "zero_flow_fraction", "catchment_area"]].copy()
    rows = []
    for attr, labels in [
        ("aridity", ["humid_or_low_aridity", "intermediate", "dry_or_high_aridity"]),
        ("zero_flow_fraction", ["low_zero_flow", "intermediate", "high_zero_flow"]),
    ]:
        data = meta.dropna(subset=[attr]).copy()
        if len(data) < 90:
            continue
        qs = data[attr].quantile([1 / 3, 2 / 3]).to_numpy()
        data["stratum"] = pd.cut(data[attr], bins=[-np.inf, qs[0], qs[1], np.inf], labels=labels)
        for stratum, group in data.groupby("stratum", observed=True):
            subset = curves[curves["gauge_id"].isin(group["gauge_id"])].copy()
            if group["gauge_id"].nunique() < args.min_bin_gauges:
                continue
            res = reduction_for_curves(subset, args)
            res.update(
                {
                    "stratifier": attr,
                    "stratum": str(stratum),
                    "n_gauges": int(group["gauge_id"].nunique()),
                    "median_aridity": float(group["aridity"].median()),
                    "median_zero_flow_fraction": float(group["zero_flow_fraction"].median()),
                }
            )
            rows.append(res)
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, bin_stats: pd.DataFrame, subsample: pd.DataFrame, random_tau: pd.DataFrame, assoc: pd.DataFrame, strata: pd.DataFrame) -> None:
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
    fig = plt.figure(figsize=(7.55, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    sc = ax0.scatter(
        summary["long_outlet"],
        summary["lat_outlet"],
        c=np.log10(summary["tau_acf_days"].clip(lower=0.1)),
        s=11 + 28 * summary["zero_flow_fraction"].fillna(0).clip(0, 1),
        cmap="viridis",
        alpha=0.82,
        linewidths=0,
    )
    ax0.set_xlabel("longitude")
    ax0.set_ylabel("latitude")
    ax0.set_title("a  CAMELS-AUS adds a dry-continent archive", loc="left", fontsize=8.8, fontweight="bold")
    fig.colorbar(sc, ax=ax0, shrink=0.76, pad=0.02, label=r"$\log_{10}\tau_{\mathrm{acf}}$ [d]")

    for axis, color, label in [("frequency_cpd", "#7B8794", "raw frequency"), ("de_eval", "#2F6FA8", "De")]:
        data = bin_stats[bin_stats["axis"] == axis]
        if data.empty:
            continue
        ax1.plot(data["coord_median"], data["beta_variance"], color=color, lw=1.8, label=label)
        ax1.scatter(data["coord_median"], data["beta_variance"], color=color, s=15, alpha=0.82, linewidths=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("coordinate value")
    ax1.set_ylabel("cross-gauge beta variance")
    ax1.set_title("b  Fourth-archive dispersion test", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)
    ax1.legend(loc="best", fontsize=6)

    obs = float((bin_stats[bin_stats["axis"] == "frequency_cpd"]["beta_variance"].mul(bin_stats[bin_stats["axis"] == "frequency_cpd"]["n_gauges"]).sum() / bin_stats[bin_stats["axis"] == "frequency_cpd"]["n_gauges"].sum() - bin_stats[bin_stats["axis"] == "de_eval"]["beta_variance"].mul(bin_stats[bin_stats["axis"] == "de_eval"]["n_gauges"]).sum() / bin_stats[bin_stats["axis"] == "de_eval"]["n_gauges"].sum()) / (bin_stats[bin_stats["axis"] == "frequency_cpd"]["beta_variance"].mul(bin_stats[bin_stats["axis"] == "frequency_cpd"]["n_gauges"]).sum() / bin_stats[bin_stats["axis"] == "frequency_cpd"]["n_gauges"].sum()))
    ax2.hist(subsample["variance_reduction_fraction"], bins=24, color="#2F6FA8", alpha=0.65, label="150-gauge subsamples")
    ax2.hist(random_tau["variance_reduction_fraction"], bins=24, color="#C4513F", alpha=0.45, label="random tau")
    ax2.axvline(obs, color="#111827", lw=1.4, label=f"observed {obs*100:.1f}%")
    ax2.set_xlabel("variance reduction fraction")
    ax2.set_ylabel("draws")
    ax2.set_title("c  Stability and shuffled-memory specificity", loc="left", fontsize=8.8, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=6)

    top = assoc.sort_values("partial_rank_rho_logtau", key=lambda s: s.abs(), ascending=False).head(9)
    top = top.sort_values("partial_rank_rho_logtau")
    y = np.arange(len(top))
    colors = ["#C4513F" if v > 0 else "#2F6FA8" for v in top["partial_rank_rho_logtau"]]
    ax3.axvline(0, color="#667085", lw=0.9)
    ax3.barh(y, top["partial_rank_rho_logtau"], color=colors, alpha=0.86)
    ax3.set_yticks(y, [a.replace("_", " ") for a in top["attribute"]])
    for yi, row in zip(y, top.itertuples(index=False)):
        ax3.text(row.partial_rank_rho_logtau, yi, f"  q={row.partial_q_value_bh:.2g}", va="center", fontsize=6)
    ax3.set_xlabel("partial rank rho with log tau")
    ax3.set_title("d  Storage and dry-regime proxies", loc="left", fontsize=8.8, fontweight="bold")
    ax3.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    if not strata.empty:
        fig2, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
        plot = strata.copy().sort_values(["stratifier", "median_aridity", "median_zero_flow_fraction"])
        labels = [f"{r.stratifier}: {r.stratum}\n(n={r.n_gauges})" for r in plot.itertuples(index=False)]
        colors = ["#2F6FA8" if "aridity" == r.stratifier else "#C4513F" for r in plot.itertuples(index=False)]
        ax.axhline(0, color="#667085", lw=0.9)
        ax.bar(np.arange(len(plot)), plot["variance_reduction_fraction"] * 100, color=colors, alpha=0.84)
        ax.set_xticks(np.arange(len(plot)), labels, rotation=25, ha="right")
        ax.set_ylabel("De variance reduction [%]")
        ax.set_title("R25 CAMELS-AUS dry-regime boundary test", loc="left", fontsize=9.2, fontweight="bold")
        ax.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)
        for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
            fig2.savefig(FIGURES / f"{OUT}_strata{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig2)


def write_note(summary: pd.DataFrame, archive_metrics: pd.DataFrame, assoc: pd.DataFrame, strata: pd.DataFrame) -> None:
    lines = [
        "# R25 CAMELS-AUS Fourth Archive",
        "",
        "R25 adds CAMELS-AUS v2 as an Australian dry-continent archive.",
        "Raw third-party files are retained locally and should not be redistributed.",
        "",
        "## Archive metrics",
        "",
        archive_metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Storage and boundary associations",
        "",
        assoc.to_markdown(index=False, floatfmt=".4g") if not assoc.empty else "No associations available.",
        "",
        "## Dry-regime strata",
        "",
        strata.to_markdown(index=False, floatfmt=".4g") if not strata.empty else "No strata metrics available.",
        "",
        "## Safe interpretation",
        "",
        "CAMELS-AUS is a real fourth-archive portability test. Positive De",
        "dispersion reduction would strengthen archive breadth; weak or negative",
        "arid/zero-flow strata must be treated as a boundary of continuous",
        "storage-memory benchmarking rather than edited into a universal claim.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    wide = load_streamflow_wide()
    attrs = load_master()
    gauge_ids = [c for c in wide.columns if c in set(attrs["gauge_id"])]
    if args.limit:
        gauge_ids = gauge_ids[: args.limit]
    rows = []
    curve_frames = []
    for idx, gid in enumerate(gauge_ids, start=1):
        row, curve = analyze_one(gid, wide[gid], args)
        if row is not None and curve is not None:
            rows.append(row)
            curve_frames.append(curve)
        if idx % 50 == 0:
            print(f"  processed {idx}/{len(gauge_ids)} stations; usable={len(rows)}", flush=True)
    if not rows:
        raise RuntimeError("No usable CAMELS-AUS streamflow stations.")

    summary = pd.DataFrame(rows).merge(attrs, on="gauge_id", how="left")
    curves = pd.concat(curve_frames, ignore_index=True)
    summary.to_csv(TABLES / f"{OUT}_gauge_summary.csv", index=False)
    curves.to_csv(TABLES / f"{OUT}_curves.csv", index=False)

    raw_stats, raw = dispersion_metrics(curves, "frequency_cpd", args.n_bins, args.min_bin_gauges)
    de_stats, de = dispersion_metrics(curves.rename(columns={"de": "de_eval"}), "de_eval", args.n_bins, args.min_bin_gauges)
    bin_stats = pd.concat([raw_stats, de_stats], ignore_index=True)
    bin_stats.to_csv(TABLES / f"{OUT}_bin_stats.csv", index=False)
    archive_metrics = pd.DataFrame(
        [
            {
                "archive": "CAMELS-AUS v2",
                "n_gauges": int(summary["gauge_id"].nunique()),
                "median_tau_acf_days": float(summary["tau_acf_days"].median()),
                "median_aridity": float(summary["aridity"].median()),
                "median_zero_flow_fraction": float(summary["zero_flow_fraction"].median()),
                "raw_weighted_beta_variance": raw["weighted_beta_variance"],
                "de_weighted_beta_variance": de["weighted_beta_variance"],
                "variance_reduction_fraction": (raw["weighted_beta_variance"] - de["weighted_beta_variance"]) / raw["weighted_beta_variance"],
                "raw_n_bins": raw["n_bins"],
                "de_n_bins": de["n_bins"],
            }
        ]
    )
    archive_metrics.to_csv(TABLES / f"{OUT}_archive_metrics.csv", index=False)
    subsample, random_tau = stability_tables(curves, args)
    subsample.to_csv(TABLES / f"{OUT}_subsample_stability.csv", index=False)
    random_tau.to_csv(TABLES / f"{OUT}_random_tau_null.csv", index=False)
    assoc = attribute_associations(summary)
    assoc.to_csv(TABLES / f"{OUT}_storage_proxy_associations.csv", index=False)
    strata = strata_metrics(curves, summary, args)
    strata.to_csv(TABLES / f"{OUT}_strata_metrics.csv", index=False)
    plot_results(summary, bin_stats, subsample, random_tau, assoc, strata)
    write_note(summary, archive_metrics, assoc, strata)
    print(archive_metrics.to_string(index=False))
    if not assoc.empty:
        print(assoc.sort_values("partial_q_value_bh").head(12).to_string(index=False))
    if not strata.empty:
        print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
