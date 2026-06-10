"""CAMELS 674-station storage-memory collapse analysis.

Reads extracted CAMELS usgs_streamflow files + attributes v2.0, computes
tau_acf, beta(De), tau_recession (on raw discharge), and null-model comparisons.

Usage:
  python scripts/run_camels_full.py
  python scripts/run_camels_full.py --n-sample 50 --n-null 10  # quick test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    parser.add_argument("--flow-dir", default="data/external/camels/usgs_streamflow")
    parser.add_argument("--attr-dir", default="data/external/camels/camels_attributes_v2.0")
    parser.add_argument("--output-prefix", default="camels_674_memory_collapse")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-sample", type=int, default=0)
    parser.add_argument("--n-null", type=int, default=30)
    parser.add_argument(
        "--max-null-large-sample",
        type=int,
        default=0,
        help=(
            "Optional computational cap for full CAMELS runs. "
            "Default 0 means use --n-null exactly."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def load_one_camels_flow(filepath: Path):
    """Parse a CAMELS usgs_streamflow txt file. Format: gauge_id year month day discharge flag (no header)."""
    gauge_id = filepath.stem
    with open(filepath, "r") as f:
        lines = f.readlines()
    # No header lines in QC files; data starts directly
    data_lines = [l.strip() for l in lines if l.strip()]
    if not data_lines:
        return gauge_id, pd.DataFrame()
    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 5:
            try:
                yr, mo, dy = int(parts[1]), int(parts[2]), int(parts[3])
                q = float(parts[4])
                if q >= 0:
                    rows.append({"date": pd.Timestamp(year=yr, month=mo, day=dy), "q": q})
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return gauge_id, df
    return gauge_id, df.sort_values("date").set_index("date")


def load_all_flows(flow_dir, n_sample=0):
    files = sorted(Path(flow_dir).glob("*.txt"))
    if n_sample > 0:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, min(n_sample, len(files)), replace=False))
    print(f"Loading {len(files)} CAMELS flow files...")
    flows = {}
    for i, fp in enumerate(files):
        gid, df = load_one_camels_flow(fp)
        if not df.empty and len(df) > 365:
            flows[gid] = df["q"]
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} loaded ({len(flows)} valid)")
    print(f"Loaded {len(flows)} valid gauge series")
    return flows


def load_camels_attributes(attr_dir):
    dfs = {}
    attr_dir = Path(attr_dir)
    for fname in ["camels_name", "camels_clim", "camels_hydro", "camels_topo", "camels_geol"]:
        fp = attr_dir / f"{fname}.txt"
        if fp.exists():
            df = pd.read_csv(fp, sep=";")
            id_col = "gauge_id" if "gauge_id" in df.columns else df.columns[0]
            df["gauge_id_str"] = df[id_col].astype(str).str.strip().str.zfill(8)
            dfs[fname] = df
            print(f"  Loaded {fname}: {len(df)} rows")
    if not dfs:
        return pd.DataFrame()
    merged = dfs.pop(list(dfs.keys())[0])
    for key, df in dfs.items():
        new_cols = ["gauge_id_str"] + [c for c in df.columns
                    if c not in merged.columns and c != "gauge_id"]
        if new_cols:
            merged = merged.merge(df[new_cols], on="gauge_id_str", how="outer")
    # Rename key columns
    col_map = {
        "gauge_name": "gauge_name", "huc_02": "huc_02",
        "p_mean": "p_mean", "pet_mean": "pet_mean", "aridity": "aridity",
        "frac_snow": "frac_snow", "p_seasonality": "p_seasonality",
        "q_mean": "q_mean", "runoff_ratio": "runoff_ratio",
        "elev_mean": "elev_mean", "slope_mean": "slope_mean",
        "area_gages2": "drainage_area_km2",
        "baseflow_index": "baseflow_index",
        "slope_fdc": "slope_fdc",
        "stream_elas": "stream_elas",
        "hfd_mean": "hfd_mean",
        "high_q_freq": "high_q_freq",
        "low_q_freq": "low_q_freq",
        "zero_q_freq": "zero_q_freq",
        "carbonate_rocks_frac": "carbonate_rocks_frac",
        "geol_porostiy": "geol_porostiy",
        "geol_permeability": "geol_permeability",
    }
    for src, dst in col_map.items():
        if src in merged.columns and dst not in merged.columns:
            merged[dst] = merged[src]
    return merged.set_index("gauge_id_str")


def prepare_daily_anomaly(series):
    df = series.to_frame("q").dropna()
    df["doy"] = df.index.dayofyear
    climo = df.groupby("doy")["q"].transform("mean")
    df["anomaly"] = df["q"] - climo
    df["anomaly"] = df["anomaly"].asfreq("D").interpolate(method="linear", limit=30)
    df = df.dropna(subset=["anomaly"])
    med = df["anomaly"].median()
    mad_val = np.median(np.abs(df["anomaly"] - med))
    if mad_val > 0:
        df["anomaly"] = (df["anomaly"] - med) / mad_val
    return df.index, df["anomaly"].values


def estimate_recession_tau_raw(series):
    q = series.dropna().values
    if len(q) < 365:
        return np.nan
    threshold = np.percentile(q, 30)
    log_q = np.log(np.maximum(q, 1e-6))
    dlogq = np.diff(log_q, prepend=log_q[0])
    is_rec = (dlogq < 0) & (q < threshold)
    k_vals = []
    in_seg, start = False, 0
    for i in range(len(is_rec)):
        if is_rec[i] and not in_seg:
            start, in_seg = i, True
        elif (not is_rec[i] or i == len(is_rec) - 1) and in_seg:
            end = i
            if end - start >= 5:
                t = np.arange(end - start)
                seg = log_q[start:end]
                coeffs = np.polyfit(t, seg, 1)
                k = -coeffs[0]
                if 0 < k < 1.0:
                    k_vals.append(k)
            in_seg = False
    if len(k_vals) < 3:
        return np.nan
    return round(1.0 / np.mean(k_vals), 1)


def wy_shuffle(values, dates, seed):
    rng = np.random.default_rng(seed)
    wy = dates.year.where(dates.month >= 10, dates.year - 1).values
    wy_series = pd.Series(wy, index=range(len(values)))
    grouped = [(w, values[wy_series == w]) for w in sorted(set(wy)) if (wy_series == w).sum() > 100]
    if len(grouped) < 2:
        return values.copy()
    val_list = [g[1] for g in grouped]
    perm = rng.permutation(len(val_list))
    result = np.concatenate([val_list[p] for p in perm])
    if len(result) < len(values):
        result = np.concatenate([result, values[len(result):]])
    return result[:len(values)]


def beta_quick(dates, values, tau_sec):
    psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    bd = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_sec)
    return bd, summarize_beta_window(bd)


def main():
    args = parse_args()
    flows = load_all_flows(args.flow_dir, n_sample=args.n_sample)
    attrs = load_camels_attributes(args.attr_dir)
    print(f"Attributes loaded: {len(attrs)} gauges")

    rng = np.random.default_rng(args.seed)
    rows = []
    if args.max_null_large_sample > 0 and len(flows) > 200:
        null_n = min(args.n_null, args.max_null_large_sample)
    else:
        null_n = args.n_null
    print(f"Null-model realizations per station: {null_n}")

    for i, (gid, series) in enumerate(flows.items()):
        dates, values = prepare_daily_anomaly(series)
        n_years = len(values) / 365.25
        missing = np.isnan(values).mean()
        if n_years < args.min_years or missing > args.max_missing:
            continue
        tau_sec = integral_autocorrelation_time(values, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_acf = tau_sec / SECONDS_PER_DAY
        tau_rec = estimate_recession_tau_raw(series)
        try:
            bd, obs = beta_quick(dates, values, tau_sec)
        except Exception:
            continue
        nulls = {}
        for nm, gen in [
            ("seasonal", lambda s: seasonal_noise_surrogate(pd.Series(dates), values, seed=s, seasonal_key="dayofyear")),
            ("ar1", lambda s: ar1_surrogate(values, seed=s)),
            ("wy_shuffle", lambda s: wy_shuffle(values, dates, seed=s)),
            ("phase_rand", lambda s: phase_randomized_surrogate(values, seed=s)),
        ]:
            nbs = []
            for _ in range(null_n):
                sd = int(rng.integers(0, np.iinfo(np.int32).max))
                try:
                    _, ns = beta_quick(dates, gen(sd), tau_sec)
                    nbs.append(ns["mean_beta_de_window"])
                except Exception:
                    continue
            arr = np.asarray(nbs, dtype=float)
            va = arr[np.isfinite(arr)]
            om = obs.get("mean_beta_de_window", np.nan)
            nulls[f"{nm}_n"] = int(va.size)
            nulls[f"{nm}_median"] = float(np.median(va)) if va.size else np.nan
            nulls[f"{nm}_p_ge"] = float(np.mean(va >= om)) if va.size and np.isfinite(om) else np.nan

        row = {"gauge_id": gid, "n_years": n_years, "missing_frac": missing,
               "tau_acf_days": tau_acf, "tau_recession_days": tau_rec, **obs, **nulls}
        if gid in attrs.index:
            for col in [
                "gauge_name", "drainage_area_km2", "p_mean", "pet_mean", "aridity",
                "frac_snow", "elev_mean", "slope_mean", "q_mean", "runoff_ratio", "huc_02",
                "baseflow_index", "slope_fdc", "stream_elas", "hfd_mean", "high_q_freq",
                "low_q_freq", "zero_q_freq", "carbonate_rocks_frac", "geol_porostiy",
                "geol_permeability",
            ]:
                if col in attrs.columns:
                    row[col] = attrs.loc[gid, col]
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(flows)}: {len(rows)} passed filters")

    summary = pd.DataFrame(rows).sort_values("tau_acf_days")
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}_summary.csv"
    summary.to_csv(out_path, index=False)

    nv = len(summary)
    nd = (summary["n_beta_de_window"].fillna(0) > 0).sum()
    nph = 0
    for c in summary.columns:
        if "phase" in c and "p_ge" in c:
            nph = (summary[c].fillna(1) == 0).sum()
            break
    print(f"\n{nv} stations passed filters ({nd} with De window)")
    print(f"tau_acf: {summary['tau_acf_days'].min():.1f} - {summary['tau_acf_days'].max():.1f} d (median {summary['tau_acf_days'].median():.1f})")
    if "tau_recession_days" in summary.columns:
        tr = summary["tau_recession_days"].dropna()
        print(f"tau_recession: {tr.median():.1f} d median (n={len(tr)})")
    print(f"Phase-randomized rejected: {nph}")
    if "drainage_area_km2" in summary.columns:
        vd = summary.dropna(subset=["tau_acf_days", "drainage_area_km2"])
        if len(vd) >= 10:
            rho, p = stats.spearmanr(vd["tau_acf_days"], np.log10(vd["drainage_area_km2"]))
            print(f"tau_acf vs area: rho={rho:.3f} p={p:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
