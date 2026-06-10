"""CAMELS 671-station storage-memory collapse analysis.

Extracts daily streamflow from CAMELS basin_timeseries dataset, computes
tau_acf, tau_recession, and beta(De) for all 671 CONUS catchments.

Usage:
  python scripts/run_camels_analysis.py
  python scripts/run_camels_analysis.py --n-sample 100  # quick test
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from edb.signal.de import beta_vs_de
from edb.signal.nulls import (
    ar1_surrogate,
    phase_randomized_surrogate,
    seasonal_noise_surrogate,
)
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time

SECONDS_PER_DAY = 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", default="data/external/camels/camels_timeseries.zip")
    parser.add_argument("--attrs-csv", default="data/external/camels/camels_attributes_v2p0.csv")
    parser.add_argument("--output-prefix", default="camels_671_memory_collapse")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-sample", type=int, default=0)
    parser.add_argument("--n-null", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def load_camels_streamflow(zip_path: Path, n_sample: int = 0) -> pd.DataFrame:
    """Extract all usgs_streamflow files from CAMELS zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        flow_files = sorted([n for n in zf.namelist()
                            if "usgs_streamflow" in n and n.endswith(".txt")])
        if n_sample > 0:
            rng = np.random.default_rng(42)
            flow_files = list(rng.choice(flow_files, min(n_sample, len(flow_files)), replace=False))

        print(f"Extracting {len(flow_files)} streamflow files...")
        all_data = []
        for fname in flow_files:
            gauge_id = Path(fname).stem.split("_")[0]
            with zf.open(fname) as f:
                content = io.TextIOWrapper(f, encoding="utf-8").read()
            lines = content.strip().split("\n")
            if len(lines) < 4:
                continue
            # CAMELS format: 4 header lines, then "Year  Mnth  Day  Hr  QObs(mm/d)"
            data_lines = [l for l in lines[4:] if l.strip() and not l.startswith("#")]
            if not data_lines:
                continue
            rows = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                        q = float(parts[3])
                        if q >= 0:  # valid discharge
                            rows.append({
                                "gauge_id": gauge_id,
                                "date": pd.Timestamp(year=year, month=month, day=day),
                                "discharge_mm_d": q,
                            })
                    except (ValueError, IndexError):
                        continue
            if rows:
                all_data.extend(rows)
    return pd.DataFrame(all_data)


def prepare_daily_series(df: pd.DataFrame, gauge_id: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Extract daily discharge for one gauge. Returns dates and values."""
    gdf = df[df["gauge_id"] == gauge_id].copy()
    if gdf.empty:
        return pd.DatetimeIndex([]), np.array([])
    gdf = gdf.sort_values("date")
    gdf = gdf.set_index("date")
    # Remove day-of-year climatology
    gdf["doy"] = gdf.index.dayofyear
    climatology = gdf.groupby("doy")["discharge_mm_d"].transform("mean")
    gdf["anomaly"] = gdf["discharge_mm_d"] - climatology
    # Interpolate short gaps
    gdf["anomaly"] = gdf["anomaly"].interpolate(method="linear", limit=30)
    gdf = gdf.dropna(subset=["anomaly"])
    # MAD scale
    med = gdf["anomaly"].median()
    mad = np.median(np.abs(gdf["anomaly"] - med))
    if mad > 0:
        gdf["anomaly"] = (gdf["anomaly"] - med) / mad

    return gdf.index, gdf["anomaly"].values


def estimate_recession_tau(discharge: np.ndarray, qtile: float = 0.3) -> float:
    """Estimate recession constant from log-discharge decay."""
    q = np.asarray(discharge, dtype=float)
    valid = np.isfinite(q) & (q > 0)
    q_valid = q[valid]
    if len(q_valid) < 365:
        return np.nan
    threshold = np.percentile(q_valid, qtile * 100)
    log_q = np.log(q_valid)
    dlogq = np.diff(log_q, prepend=log_q[0])
    recession = (q_valid < threshold) & (dlogq < 0)
    if recession.sum() < 10:
        return np.nan
    k = -np.mean(dlogq[recession])
    if k <= 0 or not np.isfinite(k):
        return np.nan
    return 1.0 / k


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise SystemExit(f"CAMELS zip not found: {zip_path}. Download first.")

    df = load_camels_streamflow(zip_path, n_sample=args.n_sample)
    gauges = sorted(df["gauge_id"].unique())
    print(f"Loaded {len(df):,} records from {len(gauges)} CAMELS gauges")

    rng = np.random.default_rng(args.seed)
    rows = []
    for i, gauge in enumerate(gauges):
        dates, values = prepare_daily_series(df, gauge)
        if len(values) < 365 * args.min_years:
            continue
        missing = np.isnan(values).mean()
        if missing > args.max_missing:
            continue

        tau_sec = integral_autocorrelation_time(values, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_acf = tau_sec / SECONDS_PER_DAY
        tau_rec = estimate_recession_tau(values)

        # PSD + beta(De)
        psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_sec)
        obs_summary = summarize_beta_window(beta_de)

        # Null models
        null_results = {}
        for nm, null_gen in [
            ("seasonal_noise", lambda s: seasonal_noise_surrogate(pd.Series(dates), values, seed=s, seasonal_key="dayofyear")),
            ("ar1", lambda s: ar1_surrogate(values, seed=s)),
            ("phase_randomized", lambda s: phase_randomized_surrogate(values, seed=s)),
        ]:
            null_betas = []
            for _ in range(args.n_null):
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                null_vals = null_gen(seed)
                ns = summarize_beta_window(beta_vs_de(beta["f_center"], beta["beta"], tau=tau_sec))
                null_betas.append(ns["mean_beta_de_window"])
            arr = np.asarray(null_betas, dtype=float)
            valid_arr = arr[np.isfinite(arr)]
            null_results[nm] = {
                "n": int(valid_arr.size),
                "median": float(np.median(valid_arr)) if valid_arr.size else np.nan,
                "p95": float(np.percentile(valid_arr, 95)) if valid_arr.size else np.nan,
                "p_ge_obs": float(np.mean(valid_arr >= obs_summary["mean_beta_de_window"]))
                if valid_arr.size and np.isfinite(obs_summary["mean_beta_de_window"]) else np.nan,
            }

        rec_years = len(values) / 365.25
        rows.append({
            "gauge_id": gauge,
            "n_years": rec_years,
            "missing_frac": missing,
            "tau_acf_days": tau_acf,
            "tau_recession_days": tau_rec,
            **obs_summary,
            **{f"{k}_{kk}": vv for k, v in null_results.items() for kk, vv in v.items()},
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(gauges)} processed...")

    summary = pd.DataFrame(rows).sort_values("tau_acf_days")
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}_summary.csv"
    summary.to_csv(out_path, index=False)

    n_valid = len(summary)
    n_de = (summary["n_beta_de_window"].fillna(0) > 0).sum()
    n_phase = (summary["phase_randomized_p_ge_obs"].fillna(1) == 0).sum() if "phase_randomized_p_ge_obs" in summary.columns else 0
    print(f"\nCAMELS stations passed filters: {n_valid}")
    print(f"Populated De windows: {n_de}")
    print(f"Phase-randomized rejected: {n_phase}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
