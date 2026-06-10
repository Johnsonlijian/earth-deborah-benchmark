"""R27 CAMELS-DK lowland groundwater/storage validation.

This script adds a groundwater-dominated, lowland archive to the existing
streamflow spectral-memory benchmark. CAMELS-DK includes observed discharge for
304 gauged catchments plus daily DK-model storage-state variables. The analysis
uses the gauged package only, keeps raw data out of the submission package, and
separates storage-compatible evidence from direct causal proof.
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
from edb.signal.timescales import autocorrelation, integral_autocorrelation_time

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "camels_dk"
GAUGED_ZIP = DATA_DIR / "Gauged_catchments.zip"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r27_camels_dk_groundwater_storage_validation"
SECONDS_PER_DAY = 86_400.0

STORAGE_SERIES = {
    "DKM_dtp": "phreatic-depth state",
    "DKM_wcr": "soil-water state",
    "DKM_gwh": "deep-groundwater-head state",
    "Qdkm": "DK-model runoff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Maximum catchments; 0 means all.")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--min-bin-gauges", type=int, default=30)
    parser.add_argument("--n-subsample", type=int, default=300)
    parser.add_argument("--subsample-size", type=int, default=150)
    parser.add_argument("--n-random-tau", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def coverage(q: pd.Series) -> tuple[float, float]:
    q = q.dropna()
    if q.empty:
        return 0.0, 1.0
    span_days = (q.index.max() - q.index.min()).days + 1
    n_years = span_days / 365.25
    missing = float(q.asfreq("D").isna().mean())
    return n_years, missing


def trim_to_valid(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return valid
    return series.loc[valid.index.min() : valid.index.max()].asfreq("D")


def prepare_anomaly(series: pd.Series, interpolate_limit: int = 30) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.DataFrame({"x": series}).replace([np.inf, -np.inf], np.nan)
    frame["x"] = frame["x"].interpolate(limit=interpolate_limit, limit_direction="both")
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("empty series after interpolation")
    frame["doy"] = frame.index.dayofyear
    frame["anom"] = frame["x"] - frame.groupby("doy")["x"].transform("mean")
    med = float(frame["anom"].median())
    mad = float(np.median(np.abs(frame["anom"].to_numpy(dtype=float) - med)))
    if mad > 0:
        frame["anom"] = (frame["anom"] - med) / mad
    return frame.index, frame["anom"].to_numpy(dtype=float)


def tau_days_from_series(series: pd.Series, max_lag: int) -> tuple[float, bool, int]:
    dates, x = prepare_anomaly(series)
    if len(x) < 365 * 5:
        return np.nan, False, len(x)
    tau_seconds = integral_autocorrelation_time(x, dt_seconds=SECONDS_PER_DAY, max_lag=max_lag, stop_at_zero=True)
    acf = autocorrelation(x, max_lag=max_lag)
    crossed = bool(np.any(acf[1:] <= 0))
    return tau_seconds / SECONDS_PER_DAY, not crossed, len(x)


def beta_curve_for_qobs(dates: pd.DatetimeIndex, x: np.ndarray, tau_days: float) -> pd.DataFrame:
    psd = welch_psd(dates, x, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * SECONDS_PER_DAY)
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["tau_days"] = tau_days
    return curve


def load_one_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as fh:
        data = pd.read_csv(fh, na_values=["", "NaN", "nan", -9999, "-9999"])
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time")
    data["catch_id"] = data["catch_id"].astype(str)
    for col in [*STORAGE_SERIES.keys(), "Qobs"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.set_index("time").asfreq("D")


def analyze_archive(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    curves: list[pd.DataFrame] = []
    with zipfile.ZipFile(GAUGED_ZIP) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        if args.limit:
            names = names[: args.limit]
        for name in names:
            data = load_one_from_zip(zf, name)
            catch_id = str(data["catch_id"].dropna().iloc[0])
            qobs = trim_to_valid(data["Qobs"].where(data["Qobs"] >= 0))
            n_years, missing = coverage(qobs)
            row = {
                "gauge_id": catch_id,
                "filename": Path(name).name,
                "n_years": n_years,
                "missing_fraction": missing,
                "status": "failed",
                "reason": "",
            }
            if n_years < args.min_years or missing > args.max_missing:
                row["reason"] = "coverage_filter"
                summaries.append(row)
                continue
            try:
                dates, x = prepare_anomaly(qobs)
                tau_q_days = integral_autocorrelation_time(
                    x,
                    dt_seconds=SECONDS_PER_DAY,
                    max_lag=730,
                    stop_at_zero=True,
                ) / SECONDS_PER_DAY
                if not np.isfinite(tau_q_days) or tau_q_days <= 0:
                    raise ValueError("invalid Qobs tau")
                curve = beta_curve_for_qobs(dates, x, tau_q_days)
                if len(curve) < 20:
                    raise ValueError("insufficient beta curve")
            except Exception as exc:
                row["reason"] = f"spectral_pipeline_failed:{exc}"
                summaries.append(row)
                continue

            row.update(
                {
                    "status": "accepted",
                    "reason": "",
                    "start_date_used": qobs.index.min().date().isoformat(),
                    "end_date_used": qobs.index.max().date().isoformat(),
                    "tau_acf_days": tau_q_days,
                    "n_beta_rows": int(len(curve)),
                    "median_beta_all": float(curve["beta"].median()),
                }
            )

            for col, label in STORAGE_SERIES.items():
                if col not in data.columns:
                    row[f"{col}_tau_days"] = np.nan
                    row[f"{col}_tau_censored_at_max_lag"] = np.nan
                    row[f"{col}_n_days"] = 0
                    continue
                series = trim_to_valid(data[col])
                try:
                    tau_state, censored, n_days = tau_days_from_series(series, max_lag=1825)
                except Exception:
                    tau_state, censored, n_days = np.nan, np.nan, 0
                row[f"{col}_label"] = label
                row[f"{col}_tau_days"] = tau_state
                row[f"{col}_tau_censored_at_max_lag"] = censored
                row[f"{col}_n_days"] = n_days

            curve["gauge_id"] = catch_id
            curve["tau_acf_days"] = tau_q_days
            curves.append(curve[["gauge_id", "frequency", "frequency_cpd", "de", "beta", "tau_days", "tau_acf_days"]])
            summaries.append(row)

    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True)


def dispersion_metrics(curves: pd.DataFrame, coord: str, n_bins: int, min_bin_gauges: int) -> tuple[pd.DataFrame, dict]:
    work = curves[["gauge_id", coord, "beta"]].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[coord] > 0) & np.isfinite(work["beta"])]
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
    return stats_df, {
        "weighted_beta_variance": float(np.average(stats_df["beta_variance"], weights=stats_df["n_gauges"])),
        "n_bins": int(len(stats_df)),
        "n_gauge_bin_pairs": int(stats_df["n_gauges"].sum()),
    }


def reduction_for_curves(curves: pd.DataFrame, args: argparse.Namespace, de_col: str = "de") -> dict:
    _, raw = dispersion_metrics(curves, "frequency_cpd", args.n_bins, args.min_bin_gauges)
    _, de = dispersion_metrics(curves.rename(columns={de_col: "de_eval"}), "de_eval", args.n_bins, args.min_bin_gauges)
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
    n_take = min(args.subsample_size, len(gids))
    subsample_rows = []
    for draw in range(args.n_subsample):
        sample = rng.choice(gids, size=n_take, replace=False)
        result = reduction_for_curves(curves[curves["gauge_id"].isin(sample)].copy(), args)
        result["draw"] = draw
        result["n_gauges"] = n_take
        subsample_rows.append(result)

    tau_by_gauge = curves.groupby("gauge_id")["tau_acf_days"].first()
    random_rows = []
    for draw in range(args.n_random_tau):
        shuffled = pd.Series(rng.permutation(tau_by_gauge.to_numpy()), index=tau_by_gauge.index)
        perm = curves.copy()
        perm["tau_random_days"] = perm["gauge_id"].map(shuffled)
        perm["de_random"] = perm["frequency"] * perm["tau_random_days"] * SECONDS_PER_DAY
        result = reduction_for_curves(perm, args, de_col="de_random")
        result["draw"] = draw
        result["n_gauges"] = int(len(gids))
        random_rows.append(result)
    return pd.DataFrame(subsample_rows), pd.DataFrame(random_rows)


def coordinate_sensitivity(summary: pd.DataFrame, curves: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Test whether DK-model storage-state memory is a better coordinate."""

    accepted = summary.query("status == 'accepted'").copy()
    accepted["gauge_id"] = accepted["gauge_id"].astype(str)
    curve_base = curves.copy()
    curve_base["gauge_id"] = curve_base["gauge_id"].astype(str)
    coordinate_cols = [
        ("tau_acf_days", "observed discharge tau"),
        ("DKM_dtp_tau_days", "DK-model phreatic-depth tau"),
        ("DKM_wcr_tau_days", "DK-model soil-water tau"),
        ("DKM_gwh_tau_days", "DK-model deep-groundwater-head tau"),
        ("Qdkm_tau_days", "DK-model runoff tau"),
    ]
    rows = []
    for col, label in coordinate_cols:
        tau = accepted.set_index("gauge_id")[col]
        work = curve_base.copy()
        work["tau_alt_days"] = work["gauge_id"].map(tau)
        work = work[(work["tau_alt_days"] > 0) & np.isfinite(work["tau_alt_days"])].copy()
        work["de_alt"] = work["frequency"] * work["tau_alt_days"] * SECONDS_PER_DAY
        result = reduction_for_curves(work, args, de_col="de_alt")
        rows.append(
            {
                "coordinate": col,
                "label": label,
                "n_gauges": int(work["gauge_id"].nunique()),
                **result,
            }
        )
    return pd.DataFrame(rows)


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
    if len(data) < controls.shape[1] + 5:
        return pd.Series(dtype=float)
    yr = data["y"].rank().to_numpy(dtype=float)
    xr = data.drop(columns=["y"]).rank().to_numpy(dtype=float)
    xr = np.column_stack([np.ones(len(data)), xr])
    beta, *_ = np.linalg.lstsq(xr, yr, rcond=None)
    resid = yr - xr @ beta
    return pd.Series(resid, index=data.index)


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict:
    cols = [x, y, *controls]
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        return {"n": int(len(data)), "partial_rho": np.nan, "partial_p_value": np.nan}
    xr = rank_residual(data[x], data[controls])
    yr = rank_residual(data[y], data[controls])
    both = pd.concat([xr.rename("x"), yr.rename("y")], axis=1).dropna()
    if len(both) < 20:
        return {"n": int(len(both)), "partial_rho": np.nan, "partial_p_value": np.nan}
    rho, p = stats.spearmanr(both["x"], both["y"])
    return {"n": int(len(both)), "partial_rho": float(rho), "partial_p_value": float(p)}


def load_attributes() -> pd.DataFrame:
    frames = []
    for filename in [
        "CAMELS_DK_signature_obs_based.csv",
        "CAMELS_DK_signature_sim_based.csv",
        "CAMELS_DK_geology.csv",
        "CAMELS_DK_soil.csv",
        "CAMELS_DK_climate.csv",
        "CAMELS_DK_topography.csv",
    ]:
        path = DATA_DIR / filename
        frame = pd.read_csv(path)
        frame["gauge_id"] = frame["catch_id"].astype(str)
        suffix = ""
        if "signature_obs" in filename:
            suffix = "_obs_sig"
        elif "signature_sim" in filename:
            suffix = "_sim_sig"
        keep_cols = ["gauge_id"] + [c for c in frame.columns if c not in {"catch_id", "gauge_id"}]
        frame = frame[keep_cols]
        if suffix:
            frame = frame.rename(columns={c: f"{c}{suffix}" for c in frame.columns if c != "gauge_id"})
        frames.append(frame)
    attrs = frames[0]
    for frame in frames[1:]:
        attrs = attrs.merge(frame, on="gauge_id", how="outer")
    return attrs


def association_tables(summary: pd.DataFrame, attrs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = summary.query("status == 'accepted'").copy()
    accepted["gauge_id"] = accepted["gauge_id"].astype(str)
    data = accepted.merge(attrs, on="gauge_id", how="left")
    data["log_tau_acf_days"] = np.log10(data["tau_acf_days"].where(data["tau_acf_days"] > 0))
    controls = ["catch_area", "aridity", "p_seasonality"]
    for col in controls:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    dynamic_rows = []
    for col, label in STORAGE_SERIES.items():
        tau_col = f"{col}_tau_days"
        data[f"log_{tau_col}"] = np.log10(data[tau_col].where(data[tau_col] > 0))
        pair = data[["log_tau_acf_days", f"log_{tau_col}", *controls]].dropna()
        rho, p = (np.nan, np.nan)
        if len(pair) >= 20:
            rho, p = stats.spearmanr(pair["log_tau_acf_days"], pair[f"log_{tau_col}"])
        pr = partial_spearman(data, "log_tau_acf_days", f"log_{tau_col}", controls)
        dynamic_rows.append(
            {
                "evidence_class": "daily_DK_model_storage_state",
                "variable": col,
                "label": label,
                "n": int(len(pair)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                **pr,
            }
        )
    dynamic = pd.DataFrame(dynamic_rows)
    dynamic["q_value"] = bh_qvalues(dynamic["p_value"])
    dynamic["partial_q_value"] = bh_qvalues(dynamic["partial_p_value"])

    candidates = {
        "BFI_obs_sig": ("streamflow_signature", "observed BFI"),
        "BFI_sim_sig": ("model_signature", "simulated BFI"),
        "StorageFraction_obs_sig": ("streamflow_signature", "observed storage fraction"),
        "StorageFraction_sim_sig": ("model_signature", "simulated storage fraction"),
        "AverageStorage_obs_sig": ("streamflow_signature", "observed average storage"),
        "AverageStorage_sim_sig": ("model_signature", "simulated average storage"),
        "ResponseTime_obs_sig": ("streamflow_signature", "observed response time"),
        "ResponseTime_sim_sig": ("model_signature", "simulated response time"),
        "AC1_obs_sig": ("streamflow_signature", "observed lag-1 autocorrelation"),
        "AC1_sim_sig": ("model_signature", "simulated lag-1 autocorrelation"),
        "BaseflowRecessionK_obs_sig": ("streamflow_signature", "observed recession K"),
        "BaseflowRecessionK_sim_sig": ("model_signature", "simulated recession K"),
        "uaquifer_t": ("static_hydrogeology", "upper aquifer thickness"),
        "uaquifer_d": ("static_hydrogeology", "upper aquifer depth"),
        "uclay_t": ("static_hydrogeology", "upper clay thickness"),
        "usand_t": ("static_hydrogeology", "upper sand thickness"),
        "chalk_d": ("static_hydrogeology", "chalk depth"),
        "pct_sand": ("static_soil", "soil sand fraction"),
        "pct_clay": ("static_soil", "soil clay fraction"),
        "tawc": ("static_soil", "total available water capacity"),
        "KS": ("static_soil", "saturated hydraulic conductivity"),
        "FC": ("static_soil", "field capacity"),
        "HCC": ("static_soil", "hydraulic conductivity class"),
        "pct_flat_area": ("static_topography", "flat-area fraction"),
    }
    proxy_rows = []
    for col, (eclass, label) in candidates.items():
        if col not in data.columns:
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")
        pair = data[["log_tau_acf_days", col, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p = (np.nan, np.nan)
        if len(pair) >= 20:
            rho, p = stats.spearmanr(pair["log_tau_acf_days"], pair[col])
        pr = partial_spearman(data, "log_tau_acf_days", col, controls)
        proxy_rows.append(
            {
                "evidence_class": eclass,
                "variable": col,
                "label": label,
                "n": int(len(pair)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                **pr,
            }
        )
    proxy = pd.DataFrame(proxy_rows)
    proxy["q_value"] = bh_qvalues(proxy["p_value"])
    proxy["partial_q_value"] = bh_qvalues(proxy["partial_p_value"])
    return dynamic, proxy


def load_four_archive_uncertainty() -> pd.DataFrame:
    path = TABLES / "r26_four_archive_uncertainty_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def median_curve(curves: pd.DataFrame, axis_col: str, n_bins: int = 36) -> pd.DataFrame:
    work = curves[[axis_col, "beta"]].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[axis_col] > 0) & np.isfinite(work["beta"])]
    logx = np.log10(work[axis_col])
    edges = np.linspace(np.nanpercentile(logx, 2), np.nanpercentile(logx, 98), n_bins + 1)
    work["bin"] = pd.cut(logx, edges, labels=False, include_lowest=True)
    med = (
        work.dropna(subset=["bin"])
        .groupby("bin")
        .agg(axis_value=(axis_col, "median"), beta=("beta", "median"), q25=("beta", lambda x: np.nanquantile(x, 0.25)), q75=("beta", lambda x: np.nanquantile(x, 0.75)), n=("beta", "size"))
        .reset_index(drop=True)
    )
    return med


def make_figure(
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    subsample: pd.DataFrame,
    random_tau: pd.DataFrame,
    coord_sens: pd.DataFrame,
    dynamic: pd.DataFrame,
    proxy: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.07, top=0.96, wspace=0.45, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    four = load_four_archive_uncertainty()
    archive_rows = []
    if not four.empty:
        for _, row in four.iterrows():
            archive_rows.append(
                {
                    "archive": row["archive"].replace("CAMELS-", ""),
                    "median": 100 * row["bootstrap_median_reduction_fraction"],
                    "lo": 100 * row["bootstrap_ci_low_fraction"],
                    "hi": 100 * row["bootstrap_ci_high_fraction"],
                }
            )
    dk_row = metrics.iloc[0]
    dk_dist = 100 * subsample["variance_reduction_fraction"].dropna()
    archive_rows.append(
        {
            "archive": "DK lowland",
            "median": 100 * dk_row["variance_reduction_fraction"],
            "lo": float(np.nanquantile(dk_dist, 0.025)) if len(dk_dist) else np.nan,
            "hi": float(np.nanquantile(dk_dist, 0.975)) if len(dk_dist) else np.nan,
        }
    )
    arch = pd.DataFrame(archive_rows)
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626"][: len(arch)]
    x = np.arange(len(arch))
    ax_a.bar(x, arch["median"], color=colors, alpha=0.82, width=0.68)
    err_lo = arch["median"] - arch["lo"]
    err_hi = arch["hi"] - arch["median"]
    ax_a.errorbar(x, arch["median"], yerr=[err_lo, err_hi], fmt="none", ecolor="#111827", lw=0.9, capsize=2.5)
    ax_a.axhline(0, color="#111827", lw=0.8)
    ax_a.set_xticks(x, arch["archive"], rotation=35, ha="right")
    ax_a.set_ylabel("beta-variance reduction (%)")
    ax_a.set_title("a  Fifth archive stress-tests the coordinate", loc="left", fontweight="bold")

    raw = median_curve(curves, "frequency_cpd")
    de = median_curve(curves, "de")
    ax_b.semilogx(raw["axis_value"], raw["beta"], color="#6b7280", lw=1.5, label="raw frequency")
    ax_b.fill_between(raw["axis_value"], raw["q25"], raw["q75"], color="#6b7280", alpha=0.16, lw=0)
    ax_b.semilogx(de["axis_value"], de["beta"], color="#dc2626", lw=1.6, label="De")
    ax_b.fill_between(de["axis_value"], de["q25"], de["q75"], color="#dc2626", alpha=0.14, lw=0)
    ax_b.set_xlabel("coordinate")
    ax_b.set_ylabel("median local beta")
    ax_b.legend(frameon=False)
    ax_b.set_title("b  DK observed-discharge beta curves", loc="left", fontweight="bold")
    soil = coord_sens.query("coordinate == 'DKM_wcr_tau_days'")
    if not soil.empty:
        ax_b.text(
            0.03,
            0.05,
            f"soil-water tau coordinate: {100*soil['variance_reduction_fraction'].iloc[0]:.1f}% reduction",
            transform=ax_b.transAxes,
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#e5e7eb", "alpha": 0.9},
        )

    plot = summary.query("status == 'accepted'").copy()
    plot["log_q_tau"] = np.log10(plot["tau_acf_days"].where(plot["tau_acf_days"] > 0))
    marker_map = {"DKM_dtp": "o", "DKM_wcr": "s", "DKM_gwh": "^"}
    color_map = {"DKM_dtp": "#0891b2", "DKM_wcr": "#65a30d", "DKM_gwh": "#9333ea"}
    for variable in ["DKM_dtp", "DKM_wcr", "DKM_gwh"]:
        y = np.log10(plot[f"{variable}_tau_days"].where(plot[f"{variable}_tau_days"] > 0))
        ax_c.scatter(plot["log_q_tau"], y, s=18, alpha=0.62, marker=marker_map[variable], color=color_map[variable], edgecolor="none", label=STORAGE_SERIES[variable])
    ax_c.set_xlabel("log10 discharge tau [d]")
    ax_c.set_ylabel("log10 storage-state tau [d]")
    ax_c.legend(frameon=False, loc="upper left")
    ax_c.set_title("c  Daily storage-state memory check", loc="left", fontweight="bold")

    dyn_plot = dynamic[dynamic["variable"].isin(["DKM_dtp", "DKM_wcr", "DKM_gwh"])].copy()
    dyn_plot["shown"] = dyn_plot["partial_rho"]
    ax_d.axvline(0, color="#111827", lw=0.8)
    y = np.arange(len(dyn_plot))
    ax_d.scatter(dyn_plot["shown"], y, s=70, color=[color_map[v] for v in dyn_plot["variable"]], zorder=2)
    for yi, (_, row) in zip(y, dyn_plot.iterrows()):
        x_text = row["shown"] + 0.035 * np.sign(row["shown"] if row["shown"] else 1)
        ax_d.text(x_text, yi, f"q={row['partial_q_value']:.2g}", va="center", fontsize=7, clip_on=True)
    ax_d.set_yticks(y, dyn_plot["label"])
    ax_d.set_xlabel("partial Spearman rho")
    ax_d.set_xlim(-0.42, 0.78)
    ax_d.set_ylim(-0.5, len(dyn_plot) - 0.5)
    ax_d.set_title("d  Area-climate partial associations", loc="left", fontweight="bold")

    proxy_plot = proxy.sort_values("partial_rho", key=lambda s: s.abs(), ascending=False).head(10).iloc[::-1]
    class_colors = {
        "streamflow_signature": "#64748b",
        "model_signature": "#0f766e",
        "static_hydrogeology": "#7c3aed",
        "static_soil": "#ca8a04",
        "static_topography": "#475569",
    }
    y = np.arange(len(proxy_plot))
    ax_e.axvline(0, color="#111827", lw=0.8)
    ax_e.scatter(proxy_plot["partial_rho"], y, s=55, color=[class_colors.get(v, "#334155") for v in proxy_plot["evidence_class"]])
    ax_e.set_yticks(y, proxy_plot["label"])
    ax_e.set_xlabel("partial Spearman rho")
    ax_e.set_xlim(-0.85, 0.85)
    ax_e.set_title("e  DK signatures and static proxies", loc="left", fontweight="bold")

    levels = evidence.copy()
    short_labels = {
        "DK daily storage states": "DK daily states",
        "DK storage-state coordinates": "DK state coords",
        "cross-archive static proxies": "static proxies",
        "streamflow-derived BFI/recession": "BFI/recession",
        "GB matched groundwater wells": "GB wells",
        "direct tracer transit time": "tracer boundary",
    }
    short_boundary_by_level = {
        "DK daily storage states": "compatible",
        "DK storage-state coordinates": "soil-water only",
        "cross-archive static proxies": "attributes",
        "streamflow-derived BFI/recession": "streamflow-derived",
        "GB matched groundwater wells": "negative boundary",
        "direct tracer transit time": "screen mixed; causality unclosed",
    }
    levels["short_level"] = levels["evidence_level"].map(short_labels).fillna(levels["evidence_level"])
    levels["short_boundary"] = levels["evidence_level"].map(short_boundary_by_level).fillna(levels["claim_boundary"])
    y = np.arange(len(levels))[::-1]
    status_colors = {"strong": "#16a34a", "bounded": "#d97706", "negative": "#dc2626", "missing": "#64748b"}
    ax_f.barh(y, levels["score"], color=[status_colors[s] for s in levels["status"]], alpha=0.82)
    ax_f.set_yticks(y, levels["short_level"])
    ax_f.set_xlim(0, 4.35)
    ax_f.set_xlabel("evidence strength tier")
    for yi, (_, row) in zip(y, levels.iterrows()):
        ax_f.text(min(row["score"] + 0.06, 3.35), yi, row["short_boundary"], va="center", ha="left", fontsize=7, color="#111827", clip_on=True)
    ax_f.set_title("f  Evidence ladder and claim boundary", loc="left", fontweight="bold")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.5)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"{OUT}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_note(metrics: pd.DataFrame, coord_sens: pd.DataFrame, dynamic: pd.DataFrame, proxy: pd.DataFrame, evidence: pd.DataFrame) -> None:
    m = metrics.iloc[0]
    dyn = dynamic.set_index("variable")
    proxy_top = proxy.sort_values("partial_rho", key=lambda s: s.abs(), ascending=False).head(8)
    lines = [
        "# R27 CAMELS-DK groundwater/storage validation",
        "",
        "## What was added",
        "",
        "- Downloaded the official CAMELS-DK gauged-catchment package from GEUS Dataverse (DOI 10.22008/FK2/AZXSYP), using only the 304 observed-runoff catchments.",
        "- Recomputed observed-discharge tau_acf, local beta curves, raw-frequency dispersion and De-axis dispersion.",
        "- Computed daily storage-state memory from DK-model phreatic depth, root-zone/soil-water state and deep groundwater head.",
        "- Tested storage-state memory and static soil/hydrogeology signatures against observed-discharge memory with Spearman and area-climate partial-rank controls.",
        "",
        "## Headline results",
        "",
        f"- Accepted DK catchments: {int(m['n_gauges'])}.",
        f"- DK De reduction: {100*m['variance_reduction_fraction']:.1f}% (raw variance {m['raw_weighted_beta_variance']:.3f}; De variance {m['de_weighted_beta_variance']:.3f}).",
        f"- Random-tau median reduction: {100*m['random_tau_median_reduction_fraction']:.1f}%; observed percentile in random null: {m['observed_reduction_percentile_vs_random_tau']:.3f}.",
    ]
    for _, row in coord_sens.iterrows():
        lines.append(
            f"- Coordinate sensitivity: {row['label']} gives {100*row['variance_reduction_fraction']:.1f}% reduction."
        )
    for variable in ["DKM_dtp", "DKM_wcr", "DKM_gwh"]:
        row = dyn.loc[variable]
        lines.append(
            f"- {row['label']}: Spearman rho={row['spearman_rho']:.3f}, p={row['p_value']:.2g}; partial rho={row['partial_rho']:.3f}, partial q={row['partial_q_value']:.2g}."
        )
    lines.extend(["", "## Strongest proxy/signature rows", ""])
    for _, row in proxy_top.iterrows():
        lines.append(
            f"- {row['label']} ({row['evidence_class']}): partial rho={row['partial_rho']:.3f}, partial q={row['partial_q_value']:.2g}, n={int(row['n'])}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- This is stronger than BFI-only storage compatibility because it uses a lowland groundwater-dominated archive and daily process-model storage states calibrated in a national hydrogeological model context.",
            "- It is still not tracer causality and not direct causal groundwater proof. The R24 CAMELS-GB well-to-catchment test remains a negative/non-significant boundary and should stay visible.",
            "",
            "## Evidence ladder",
            "",
            evidence.to_markdown(index=False),
        ]
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    summary, curves = analyze_archive(args)
    accepted = summary.query("status == 'accepted'").copy()
    curves.to_csv(TABLES / f"{OUT}_curves.csv", index=False)
    summary.to_csv(TABLES / f"{OUT}_gauge_summary.csv", index=False)

    raw_bins, raw_metric = dispersion_metrics(curves, "frequency_cpd", args.n_bins, args.min_bin_gauges)
    de_bins, de_metric = dispersion_metrics(curves, "de", args.n_bins, args.min_bin_gauges)
    bins = pd.concat([raw_bins, de_bins], ignore_index=True)
    bins.to_csv(TABLES / f"{OUT}_bin_stats.csv", index=False)

    subsample, random_tau = stability_tables(curves, args)
    subsample.to_csv(TABLES / f"{OUT}_subsample_stability.csv", index=False)
    random_tau.to_csv(TABLES / f"{OUT}_random_tau_null.csv", index=False)

    attrs = load_attributes()
    dynamic, proxy = association_tables(summary, attrs)
    dynamic.to_csv(TABLES / f"{OUT}_storage_state_memory_associations.csv", index=False)
    proxy.to_csv(TABLES / f"{OUT}_storage_proxy_associations.csv", index=False)
    coord_sens = coordinate_sensitivity(summary, curves, args)
    coord_sens.to_csv(TABLES / f"{OUT}_coordinate_sensitivity.csv", index=False)

    reduction = (raw_metric["weighted_beta_variance"] - de_metric["weighted_beta_variance"]) / raw_metric["weighted_beta_variance"]
    rand = random_tau["variance_reduction_fraction"].dropna()
    sub = subsample["variance_reduction_fraction"].dropna()
    archive_metrics = pd.DataFrame(
        [
            {
                "archive": "CAMELS-DK gauged lowland",
                "n_gauges": int(accepted["gauge_id"].nunique()),
                "median_tau_acf_days": float(accepted["tau_acf_days"].median()),
                "raw_weighted_beta_variance": raw_metric["weighted_beta_variance"],
                "de_weighted_beta_variance": de_metric["weighted_beta_variance"],
                "variance_reduction_fraction": float(reduction),
                "subsample_median_reduction_fraction": float(sub.median()) if len(sub) else np.nan,
                "subsample_ci95_low_reduction_fraction": float(np.nanquantile(sub, 0.025)) if len(sub) else np.nan,
                "subsample_ci95_high_reduction_fraction": float(np.nanquantile(sub, 0.975)) if len(sub) else np.nan,
                "random_tau_median_reduction_fraction": float(rand.median()) if len(rand) else np.nan,
                "random_tau_ci95_low_reduction_fraction": float(np.nanquantile(rand, 0.025)) if len(rand) else np.nan,
                "random_tau_ci95_high_reduction_fraction": float(np.nanquantile(rand, 0.975)) if len(rand) else np.nan,
                "observed_reduction_percentile_vs_random_tau": float((rand <= reduction).mean()) if len(rand) else np.nan,
                "raw_n_bins": raw_metric["n_bins"],
                "de_n_bins": de_metric["n_bins"],
            }
        ]
    )
    archive_metrics.to_csv(TABLES / f"{OUT}_archive_metrics.csv", index=False)

    r24 = pd.read_csv(TABLES / "r24_groundwater_boundary_validation_association_summary.csv")
    gb_well_rho = float(r24.query("contract == 'well_level_smallest_containing_catchment'")["spearman_rho_log_gw_tau_vs_log_discharge_tau"].iloc[0])
    gb_well_p = float(r24.query("contract == 'well_level_smallest_containing_catchment'")["p_value"].iloc[0])
    evidence = pd.DataFrame(
        [
            {
                "evidence_level": "DK daily storage states",
                "status": "strong",
                "score": 3.5,
                "claim_boundary": "process-model storage-state compatibility",
                "support": "CAMELS-DK gauged catchments; daily phreatic/soil-water/deep-groundwater memory",
            },
            {
                "evidence_level": "DK storage-state coordinates",
                "status": "bounded",
                "score": 2.2,
                "claim_boundary": "soil-water tau helps; deeper groundwater tau does not",
                "support": "alternative De coordinates from DK-model phreatic, soil-water and groundwater-head states",
            },
            {
                "evidence_level": "cross-archive static proxies",
                "status": "strong",
                "score": 3.0,
                "claim_boundary": "independent catchment-property support",
                "support": "US/GB/BR/AUS/DK soil, geology and hydrogeology associations",
            },
            {
                "evidence_level": "streamflow-derived BFI/recession",
                "status": "bounded",
                "score": 2.0,
                "claim_boundary": "useful but not independent",
                "support": "large-sample BFI/recession association with tau_acf",
            },
            {
                "evidence_level": "GB matched groundwater wells",
                "status": "negative",
                "score": 1.0,
                "claim_boundary": f"explicit non-significant boundary rho={gb_well_rho:.3f}, p={gb_well_p:.3f}",
                "support": "official CAMELS-GB v2 boundaries and 53 monthly wells",
            },
            {
                "evidence_level": "direct tracer transit time",
                "status": "bounded",
                "score": 1.4,
                "claim_boundary": "CAMELS-Chem screen mixed; direct De-tracer causality unclosed",
                "support": "CAMELS-Chem passive-solute screen and Plynlimon tracer anchor are not large-sample De-tracer causality",
            },
        ]
    )
    evidence.to_csv(TABLES / f"{OUT}_evidence_ladder.csv", index=False)

    make_figure(summary, curves, archive_metrics, subsample, random_tau, coord_sens, dynamic, proxy, evidence)
    write_note(archive_metrics, coord_sens, dynamic, proxy, evidence)

    print(archive_metrics.to_string(index=False))
    print(coord_sens.to_string(index=False))
    print(dynamic.to_string(index=False))
    print(proxy.sort_values("partial_rho", key=lambda s: s.abs(), ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
