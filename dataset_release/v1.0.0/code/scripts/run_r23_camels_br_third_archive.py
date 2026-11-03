"""R23 CAMELS-BR third-archive memory-coordinate replication.

This script adds a hydroclimatically distinct third archive to the project:
CAMELS-BR v1.2 selected Brazilian catchments. It estimates discharge memory
times, local spectral slopes and beta(De) curves from observed daily streamflow
in mm/day, then computes the same raw-frequency versus memory-coordinate
dispersion reduction used in the CAMELS-US/GB manuscript.

It also tests independent storage-proxy associations using CAMELS-BR geology
and soil attributes (geological porosity/permeability, carbonate rock fraction,
bedrock depth and water-table depth). These attributes are not estimated from
the same discharge spectrum, so they are stronger mechanism-compatibility
evidence than BFI alone, while still remaining observational.
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

from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.timescales import integral_autocorrelation_time

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "camels_br_v12"
STREAM_DIR = DATA_DIR / "streamflow_selected" / "03_CAMELS_BR_streamflow_selected_catchments"
ATTR_DIR = DATA_DIR / "attributes" / "01_CAMELS_BR_attributes"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r23_camels_br_third_archive"
SECONDS_PER_DAY = 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Maximum streamflow files; 0 means all.")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--min-bin-gauges", type=int, default=50)
    parser.add_argument("--n-subsample", type=int, default=300)
    parser.add_argument("--subsample-size", type=int, default=150)
    parser.add_argument("--n-random-tau", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def gauge_id_from_path(path: Path) -> str:
    match = re.search(r"([0-9]+)_streamflow", path.name)
    return match.group(1) if match else path.stem


def load_streamflow(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", na_values=["nan", "NaN", "-9999"])
    data["date"] = pd.to_datetime(dict(year=data["year"], month=data["month"], day=data["day"]), errors="coerce")
    data["q_mm_day"] = pd.to_numeric(data["streamflow_mm"], errors="coerce").where(lambda s: s >= 0)
    return data.dropna(subset=["date"]).sort_values("date").set_index("date")[["q_mm_day"]].asfreq("D")


def coverage(q: pd.Series) -> tuple[float, float]:
    if q.empty:
        return 0.0, 1.0
    span_days = (q.index.max() - q.index.min()).days + 1
    n_years = span_days / 365.25
    missing = float(q.isna().mean())
    return n_years, missing


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


def analyze_one(path: Path, args: argparse.Namespace) -> tuple[dict | None, pd.DataFrame | None]:
    gid = gauge_id_from_path(path)
    q = load_streamflow(path)["q_mm_day"]
    n_years, missing = coverage(q)
    if n_years < args.min_years or missing > args.max_missing:
        return None, None
    try:
        dates, x = prepare_anomaly(q)
        if len(x) < 365 * args.min_years:
            return None, None
        tau_seconds = integral_autocorrelation_time(x, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_days = tau_seconds / SECONDS_PER_DAY
        psd = welch_psd(dates, x, fs=1.0 / SECONDS_PER_DAY)
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    except Exception:
        return None, None
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    if len(curve) < 20:
        return None, None
    curve["gauge_id"] = gid
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["tau_days"] = tau_days
    row = {
        "gauge_id": gid,
        "n_years": n_years,
        "missing_fraction": missing,
        "tau_acf_days": tau_days,
        "n_beta_rows": int(len(curve)),
        "median_beta_all": float(curve["beta"].median()),
    }
    return row, curve[["gauge_id", "frequency", "frequency_cpd", "de", "beta", "tau_days"]]


def load_attributes() -> pd.DataFrame:
    files = [
        "camels_br_location.txt",
        "camels_br_topography.txt",
        "camels_br_climate.txt",
        "camels_br_hydrology.txt",
        "camels_br_geology.txt",
        "camels_br_soil.txt",
        "camels_br_human_intervention.txt",
    ]
    frames = []
    for name in files:
        path = ATTR_DIR / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep=r"\s+", dtype={"gauge_id": str})
        frames.append(frame)
    attrs = frames[0]
    for frame in frames[1:]:
        attrs = attrs.merge(frame, on="gauge_id", how="outer")
    return attrs


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


def attribute_associations(summary: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    work = summary.merge(attrs, on="gauge_id", how="left").replace([np.inf, -np.inf], np.nan)
    work["log_area"] = np.log10(pd.to_numeric(work.get("area"), errors="coerce"))
    work["log_tau"] = np.log10(pd.to_numeric(work["tau_acf_days"], errors="coerce"))
    candidates = [
        "baseflow_index",
        "geol_porosity",
        "geol_permeability",
        "carb_rocks_perc",
        "bedrock_depth",
        "water_table_depth",
        "sand_perc",
        "clay_perc",
        "aridity",
        "regulation_degree",
    ]
    controls = work[["log_area", "aridity", "p_seasonality"]].apply(pd.to_numeric, errors="coerce")
    rows = []
    for attr in candidates:
        if attr not in work.columns:
            continue
        data = work[["log_tau", attr]].copy()
        data[attr] = pd.to_numeric(data[attr], errors="coerce")
        data = data.dropna()
        if len(data) < 50:
            continue
        rho, pval = stats.spearmanr(data[attr], data["log_tau"])
        control_cols = [c for c in ["log_area", "aridity", "p_seasonality"] if c != attr]
        common = pd.concat([work["log_tau"], work[attr], controls[control_cols]], axis=1).dropna()
        if len(common) >= 50:
            yres = rank_residual(common["log_tau"], common[control_cols])
            xres = rank_residual(common[attr], common[control_cols])
            aligned = pd.concat([yres.rename("y"), xres.rename("x")], axis=1).dropna()
            prho, pp = stats.spearmanr(aligned["x"], aligned["y"]) if len(aligned) >= 50 else (np.nan, np.nan)
        else:
            prho, pp = np.nan, np.nan
        rows.append(
            {
                "attribute": attr,
                "n": int(len(data)),
                "spearman_rho_logtau": float(rho),
                "p_value": float(pval),
                "partial_rank_rho_logtau": float(prho),
                "partial_p_value": float(pp),
                "attribute_type": "streamflow-derived" if attr == "baseflow_index" else "independent_proxy_or_covariate",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh"] = bh_qvalues(out["p_value"])
        out["partial_q_value_bh"] = bh_qvalues(out["partial_p_value"])
    return out


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


def plot_results(summary: pd.DataFrame, curves: pd.DataFrame, bin_stats: pd.DataFrame, subsample: pd.DataFrame, random_tau: pd.DataFrame, assoc: pd.DataFrame, attrs: pd.DataFrame) -> None:
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
    fig = plt.figure(figsize=(7.55, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    loc = summary.copy()
    if "gauge_lon" not in loc.columns or "gauge_lat" not in loc.columns:
        loc = loc.merge(attrs[["gauge_id", "gauge_lat", "gauge_lon", "aridity", "baseflow_index"]], on="gauge_id", how="left")
    sc = ax0.scatter(loc["gauge_lon"], loc["gauge_lat"], c=np.log10(loc["tau_acf_days"].clip(lower=0.1)), s=11, cmap="viridis", alpha=0.82, linewidths=0)
    ax0.set_xlabel("longitude")
    ax0.set_ylabel("latitude")
    ax0.set_title("a  CAMELS-BR adds a tropical third archive", loc="left", fontsize=8.8, fontweight="bold")
    fig.colorbar(sc, ax=ax0, shrink=0.76, pad=0.02, label=r"$\log_{10}\tau_{\mathrm{acf}}$ [d]")

    for axis, color, label in [("frequency_cpd", "#7B8794", "raw frequency"), ("de_eval", "#2F6FA8", "De")]:
        data = bin_stats[bin_stats["axis"] == axis]
        if data.empty:
            continue
        ax1.plot(data["coord_median"], data["beta_variance"], color=color, lw=1.8, label=label)
        ax1.scatter(data["coord_median"], data["beta_variance"], color=color, s=16, alpha=0.82, linewidths=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("coordinate value")
    ax1.set_ylabel("cross-gauge beta variance")
    ax1.set_title("b  Memory coordinate reduces Brazil curve dispersion", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)
    ax1.legend(loc="best", fontsize=6)

    obs = reduction_for_curves(curves, argparse.Namespace(n_bins=24, min_bin_gauges=50))["variance_reduction_fraction"]
    ax2.hist(subsample["variance_reduction_fraction"], bins=24, color="#2F6FA8", alpha=0.65, label="150-gauge subsamples")
    ax2.hist(random_tau["variance_reduction_fraction"], bins=24, color="#C4513F", alpha=0.45, label="random tau")
    ax2.axvline(obs, color="#111827", lw=1.4, label=f"observed {obs*100:.1f}%")
    ax2.set_xlabel("variance reduction fraction")
    ax2.set_ylabel("draws")
    ax2.set_title("c  Stability and shuffled-memory specificity", loc="left", fontsize=8.8, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=6)

    awork = assoc.copy()
    awork = awork.sort_values("partial_rank_rho_logtau", key=lambda s: s.abs(), ascending=False).head(8).sort_values("partial_rank_rho_logtau")
    y = np.arange(len(awork))
    colors = ["#C4513F" if v > 0 else "#2F6FA8" for v in awork["partial_rank_rho_logtau"]]
    ax3.axvline(0, color="#667085", lw=0.9)
    ax3.barh(y, awork["partial_rank_rho_logtau"], color=colors, alpha=0.86)
    ax3.set_yticks(y, [a.replace("_", " ") for a in awork["attribute"]])
    for yi, row in zip(y, awork.itertuples(index=False)):
        ax3.text(row.partial_rank_rho_logtau, yi, f"  q={row.partial_q_value_bh:.2g}", va="center", fontsize=6)
    ax3.set_xlabel("partial rank rho with log tau")
    ax3.set_title("d  Independent storage proxies constrain interpretation", loc="left", fontsize=8.8, fontweight="bold")
    ax3.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(summary_table: pd.DataFrame, archive_metrics: pd.DataFrame, assoc: pd.DataFrame) -> None:
    lines = [
        "# R23 CAMELS-BR Third Archive",
        "",
        "R23 adds CAMELS-BR v1.2 as a hydroclimatically distinct third streamflow",
        "archive. Raw third-party data are retained locally and should not be",
        "redistributed in the public reproducibility package.",
        "",
        "## Archive metrics",
        "",
        archive_metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Storage-proxy associations",
        "",
        assoc.to_markdown(index=False, floatfmt=".4g") if not assoc.empty else "No associations available.",
        "",
        "## Safe interpretation",
        "",
        "CAMELS-BR provides an independent tropical archive and independent geology",
        "and soil proxy attributes. Positive or directional associations with these",
        "attributes support storage compatibility, but they remain observational and",
        "do not replace tracer or groundwater-level validation.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    files = sorted(STREAM_DIR.glob("*_streamflow.txt"))
    if args.limit:
        files = files[: args.limit]
    rows = []
    curve_frames = []
    for idx, path in enumerate(files, start=1):
        row, curve = analyze_one(path, args)
        if row is not None and curve is not None:
            rows.append(row)
            curve_frames.append(curve)
        if idx % 100 == 0:
            print(f"  processed {idx}/{len(files)} files; usable={len(rows)}", flush=True)
    if not rows:
        raise RuntimeError("No usable CAMELS-BR streamflow gauges.")
    summary = pd.DataFrame(rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    attrs = load_attributes()
    summary = summary.merge(attrs, on="gauge_id", how="left")
    summary.to_csv(TABLES / f"{OUT}_gauge_summary.csv", index=False)
    curves.to_csv(TABLES / f"{OUT}_curves.csv", index=False)

    raw_stats, raw = dispersion_metrics(curves, "frequency_cpd", args.n_bins, args.min_bin_gauges)
    de_stats, de = dispersion_metrics(curves.rename(columns={"de": "de_eval"}), "de_eval", args.n_bins, args.min_bin_gauges)
    bin_stats = pd.concat([raw_stats, de_stats], ignore_index=True)
    bin_stats.to_csv(TABLES / f"{OUT}_bin_stats.csv", index=False)
    archive_metrics = pd.DataFrame(
        [
            {
                "archive": "CAMELS-BR v1.2 selected catchments",
                "n_gauges": int(summary["gauge_id"].nunique()),
                "median_tau_acf_days": float(summary["tau_acf_days"].median()),
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
    assoc = attribute_associations(summary[["gauge_id", "tau_acf_days"]], attrs)
    assoc.to_csv(TABLES / f"{OUT}_storage_proxy_associations.csv", index=False)
    plot_results(summary, curves, bin_stats, subsample, random_tau, assoc, attrs)
    write_note(summary, archive_metrics, assoc)
    print(archive_metrics.to_string(index=False))
    print(assoc.sort_values("partial_q_value_bh").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
