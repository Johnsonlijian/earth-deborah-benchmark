"""Enhanced analysis: water-year shuffle, recession tau, beta(De) curve fitting.

Adds to the NWIS expanded analysis:
1. Water-year shuffle surrogate (preserves intra-annual, destroys inter-annual)
2. Recession-based memory time (tau_recession)
3. Functional form fit to beta(De) curves (transition characterization)

Usage:
  python scripts/run_enhanced_nulls.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.data.grdc import build_discharge_series, remove_daily_climatology, robust_mad_scale, summarize_station_coverage
from edb.data.nwis import parse_nwis_daily_values_json
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.nulls import ar1_surrogate, phase_randomized_surrogate, seasonal_noise_surrogate
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time

SECONDS_PER_DAY = 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/usgs_nwis_expanded")
    parser.add_argument("--pattern", default="nwis_dv_*_1975-01-01_2026-05-28.json")
    parser.add_argument("--output-prefix", default="nwis_enhanced_1975_20260528")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument("--n-null", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def daily_sample_rate_hz() -> float:
    return 1.0 / SECONDS_PER_DAY


# ── Water-year shuffle ────────────────────────────────────────────

def water_year_shuffle(values: np.ndarray, dates: pd.Series, seed: int) -> np.ndarray:
    """Shuffle water years (Oct-Sep). Preserves intra-annual, destroys inter-annual memory."""
    rng = np.random.default_rng(seed)
    wy = dates.dt.year.where(dates.dt.month >= 10, dates.dt.year - 1)
    # Build list of (water_year, values_array) sorted by WY
    wy_series = pd.Series(wy.values, index=dates.index, name="wy")
    grouped = [(w, values[wy_series == w]) for w in sorted(wy_series.unique()) if (wy_series == w).sum() > 100]
    if len(grouped) < 2:
        return values.copy()
    wy_list = [g[0] for g in grouped]
    val_list = [g[1] for g in grouped]
    # Build shuffled array with same total length
    perm = rng.permutation(len(wy_list))
    shuffled_parts = [val_list[p] for p in perm]
    result = np.concatenate(shuffled_parts)
    # Pad or truncate to match original length
    if len(result) < len(values):
        result = np.concatenate([result, values[len(result):]])
    return result[:len(values)]


# ── Recession-based memory time ───────────────────────────────────

def estimate_recession_tau(discharge: np.ndarray, dates: pd.Series, qtile: float = 0.3) -> float:
    """Estimate recession constant from segments of >=5 consecutive declining days.

    Fits log(Q) = const - k*t on each segment, pools k values, returns tau = 1/k_mean.
    """
    q = np.asarray(discharge, dtype=float)
    valid = np.isfinite(q) & (q > 0)
    if valid.sum() < 365:
        return np.nan
    qv = q[valid]
    threshold = np.percentile(qv, qtile * 100)
    log_q = np.log(qv)
    dlogq = np.diff(log_q, prepend=log_q[0])
    is_rec = (dlogq < 0) & (qv < threshold)

    k_values = []
    in_seg = False
    start = 0
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
                    k_values.append(k)
            in_seg = False

    if len(k_values) < 3:
        return np.nan
    k_mean = np.mean(k_values)
    return round(1.0 / k_mean, 1) if k_mean > 0 else np.nan


# ── Beta(De) curve shape characterization ─────────────────────────

def characterize_beta_de_curve(beta_de: pd.DataFrame) -> dict[str, float]:
    """Extract curve shape features from beta(De)."""
    if beta_de.empty:
        return {}
    de = beta_de["de"].values
    b = beta_de["beta"].values
    valid = np.isfinite(de) & np.isfinite(b) & (de > 0)
    de, b = de[valid], b[valid]
    if len(de) < 5:
        return {}

    # Slope in three regimes
    high_de = de > 2.0
    mid_de = (de >= 0.5) & (de <= 2.0)
    low_de = de < 0.5

    features = {}
    if high_de.sum() >= 3:
        coeffs = np.polyfit(np.log10(de[high_de]), b[high_de], 1)
        features["high_de_slope"] = float(coeffs[0])
    if mid_de.sum() >= 3:
        features["mean_beta_mid"] = float(np.mean(b[mid_de]))
        features["beta_range_mid"] = float(np.ptp(b[mid_de]))
    if low_de.sum() >= 3:
        coeffs = np.polyfit(np.log10(de[low_de]), b[low_de], 1)
        features["low_de_slope"] = float(coeffs[0])

    # De at minimum beta (spectral transition)
    if len(b) >= 10:
        try:
            smooth_b = np.convolve(b, np.ones(3)/3, mode='same')
            min_idx = np.argmin(smooth_b)
            features["de_at_min_beta"] = float(de[min_idx])
            features["min_beta"] = float(b[min_idx])
        except Exception:
            pass

    return features


# ── Main ──────────────────────────────────────────────────────────

def prepare_anomaly(discharge: pd.DataFrame) -> pd.DataFrame:
    anomaly = remove_daily_climatology(discharge)
    anomaly["value"] = pd.to_numeric(anomaly["value"], errors="coerce")
    anomaly["value"] = anomaly["value"].interpolate(method="linear", limit=30, limit_direction="both")
    anomaly = robust_mad_scale(anomaly)
    return anomaly.dropna(subset=["time", "value"]).reset_index(drop=True)


def beta_summary_for_values(dates: pd.Series, values: np.ndarray, tau_seconds: float) -> dict:
    psd = welch_psd(dates, values, fs=daily_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    return summarize_beta_window(beta_de)


def main():
    args = parse_args()
    raw_paths = sorted(Path(args.raw_dir).glob(args.pattern))
    if not raw_paths:
        raise SystemExit(f"No files matched")

    frames = []
    for path in raw_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = parse_nwis_daily_values_json(payload)
            if not parsed.empty:
                frames.append(parsed)
        except Exception:
            pass
    if not frames:
        raise SystemExit("No records found")
    records = pd.concat(frames, ignore_index=True)

    coverage = summarize_station_coverage(records)
    eligible = coverage[
        (coverage["n_years"] >= args.min_years) & (coverage["missing_daily_fraction"] <= args.max_missing_fraction)
    ].copy()
    if eligible.empty:
        raise SystemExit("No stations passed filters")

    rng = np.random.default_rng(args.seed)
    all_rows = []

    for _, station in eligible.iterrows():
        station_id = str(station["station_id"])
        try:
            station_records = records[records["station_id"].astype(str) == station_id]
            discharge = build_discharge_series(station_records, station_id=station_id, rule="D")
            anomaly_df = prepare_anomaly(discharge)
            values = anomaly_df["value"].to_numpy(float)
            dates = anomaly_df["time"]
        except Exception:
            continue

        tau_acf = integral_autocorrelation_time(values, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
        tau_recession = estimate_recession_tau(values, dates)

        # Beta(De) with tau_acf
        psd = welch_psd(dates, values, fs=daily_sample_rate_hz())
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_acf)
        obs_summary = summarize_beta_window(beta_de)
        curve_features = characterize_beta_de_curve(beta_de)

        # Null models
        null_results = {}
        for null_model, null_gen in [
            ("seasonal_noise", lambda s: seasonal_noise_surrogate(dates, values, seed=s, seasonal_key="dayofyear")),
            ("ar1", lambda s: ar1_surrogate(values, seed=s)),
            ("phase_randomized", lambda s: phase_randomized_surrogate(values, seed=s)),
            ("water_year_shuffle", lambda s: water_year_shuffle(values, dates, seed=s)),
        ]:
            null_betas = []
            for ni in range(args.n_null):
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                null_vals = null_gen(seed)
                null_s = beta_summary_for_values(dates, null_vals, tau_acf)
                null_betas.append(null_s["mean_beta_de_window"])
            arr = np.asarray(null_betas, dtype=float)
            valid_arr = arr[np.isfinite(arr)]
            null_results[null_model] = {
                "n": int(valid_arr.size),
                "median": float(np.nanmedian(valid_arr)) if valid_arr.size else np.nan,
                "p05": float(np.nanpercentile(valid_arr, 5)) if valid_arr.size else np.nan,
                "p95": float(np.nanpercentile(valid_arr, 95)) if valid_arr.size else np.nan,
                "p_ge_observed": float(np.mean(valid_arr >= obs_summary["mean_beta_de_window"]))
                if valid_arr.size and np.isfinite(obs_summary["mean_beta_de_window"]) else np.nan,
            }

        row = {
            "station_id": station_id,
            "station_name": station.get("station_name"),
            "tau_acf_days": tau_acf / SECONDS_PER_DAY,
            "tau_recession_days": tau_recession,
            **obs_summary,
            **curve_features,
        }
        for nm, nr in null_results.items():
            for k, v in nr.items():
                row[f"{nm}_{k}"] = v

        all_rows.append(row)
        print(f"  {station_id}: tau_acf={tau_acf/SECONDS_PER_DAY:.1f}d, tau_rec={tau_recession:.1f}d, "
              f"wy_shuffle_p={null_results.get('water_year_shuffle', {}).get('p_ge_observed', np.nan):.3f}")

    summary = pd.DataFrame(all_rows).sort_values("tau_acf_days")
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / f"{args.output_prefix}_summary.csv"
    summary.to_csv(out_path, index=False)

    n_phase_rej = (summary["phase_randomized_p_ge_observed"].fillna(1) == 0).sum()
    n_wy_rej = (summary["water_year_shuffle_p_ge_observed"].fillna(1) == 0).sum()
    print(f"\nStations: {len(summary)}")
    print(f"Phase-randomized rejected: {n_phase_rej}")
    print(f"Water-year-shuffle rejected: {n_wy_rej}")
    if summary["tau_recession_days"].notna().sum() > 5:
        rho = summary[["tau_acf_days", "tau_recession_days"]].corr(method="spearman").iloc[0, 1]
        print(f"tau_acf vs tau_recession Spearman rho = {rho:.3f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
