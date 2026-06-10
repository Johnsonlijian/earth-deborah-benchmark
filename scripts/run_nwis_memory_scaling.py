"""Analyze NWIS river spectra with data-driven autocorrelation memory times."""

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
    parser.add_argument("--raw-dir", default="data/raw/usgs_nwis")
    parser.add_argument("--pattern", default="nwis_dv_*_1980-01-01_2026-05-28.json")
    parser.add_argument("--output-prefix", default="nwis_memory_scaling_1980_20260528")
    parser.add_argument("--max-acf-lag-days", type=int, default=730)
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.20)
    parser.add_argument("--n-null", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def daily_sample_rate_hz() -> float:
    """Return daily sample rate in Hz."""

    return 1.0 / SECONDS_PER_DAY


def prepare_anomaly(discharge: pd.DataFrame) -> pd.DataFrame:
    """Create finite daily discharge anomaly values."""

    anomaly = remove_daily_climatology(discharge)
    anomaly["value"] = pd.to_numeric(anomaly["value"], errors="coerce")
    anomaly["value"] = anomaly["value"].interpolate(method="linear", limit=30, limit_direction="both")
    anomaly = robust_mad_scale(anomaly)
    return anomaly.dropna(subset=["time", "value"]).reset_index(drop=True)


def beta_summary_for_values(dates: pd.Series, values: np.ndarray, tau_seconds: float) -> dict[str, float | int]:
    """Compute beta-vs-De summary for one daily sequence and memory time."""

    psd = welch_psd(dates, values, fs=daily_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    return summarize_beta_window(beta_de)


def main() -> None:
    args = parse_args()
    raw_paths = sorted(Path(args.raw_dir).glob(args.pattern))
    if not raw_paths:
        raise SystemExit(f"No NWIS raw JSON files matched {Path(args.raw_dir) / args.pattern}")

    frames: list[pd.DataFrame] = []
    for path in raw_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parsed = parse_nwis_daily_values_json(payload)
        if not parsed.empty:
            frames.append(parsed)
    if not frames:
        raise SystemExit("No parseable NWIS records found")

    records = pd.concat(frames, ignore_index=True)
    coverage = summarize_station_coverage(records)
    eligible = coverage[
        (coverage["n_years"] >= args.min_years) & (coverage["missing_daily_fraction"] <= args.max_missing_fraction)
    ].copy()
    if eligible.empty:
        raise SystemExit("No station passed coverage filters")

    rng = np.random.default_rng(args.seed)
    station_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    beta_de_frames: list[pd.DataFrame] = []

    for _, station in eligible.iterrows():
        station_id = str(station["station_id"])
        station_records = records[records["station_id"].astype(str) == station_id]
        discharge = build_discharge_series(station_records, station_id=station_id, rule="D")
        anomaly = prepare_anomaly(discharge)
        values = anomaly["value"].to_numpy(float)
        tau_seconds = integral_autocorrelation_time(
            values,
            dt_seconds=SECONDS_PER_DAY,
            max_lag=args.max_acf_lag_days,
            stop_at_zero=True,
        )
        tau_days = tau_seconds / SECONDS_PER_DAY

        psd = welch_psd(anomaly["time"], values, fs=daily_sample_rate_hz())
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
        beta_de["process"] = "river"
        beta_de["region"] = station_id
        beta_de["variable"] = "nwis_discharge_anomaly"
        beta_de["tau_days"] = tau_days
        beta_de["tau_source"] = "integral_acf_positive"
        beta_de["station_name"] = station.get("station_name")
        beta_de_frames.append(beta_de)
        observed_summary = summarize_beta_window(beta_de)

        seasonal_null: list[float] = []
        phase_null: list[float] = []
        for null_idx in range(args.n_null):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            null_specs = [
                ("seasonal_noise", seasonal_noise_surrogate(anomaly["time"], values, seed=seed, seasonal_key="dayofyear")),
                ("ar1", ar1_surrogate(values, seed=seed)),
                ("phase_randomized", phase_randomized_surrogate(values, seed=seed)),
            ]
            for null_model, null_values in null_specs:
                null_summary = beta_summary_for_values(anomaly["time"], null_values, tau_seconds)
                null_mean = null_summary["mean_beta_de_window"]
                null_rows.append(
                    {
                        "station_id": station_id,
                        "null_model": null_model,
                        "null_idx": null_idx,
                        "tau_days_observed": tau_days,
                        **null_summary,
                    }
                )
                if np.isfinite(null_mean):
                    if null_model == "seasonal_noise":
                        seasonal_null.append(float(null_mean))
                    elif null_model == "phase_randomized":
                        phase_null.append(float(null_mean))

        row: dict[str, object] = {
            "station_id": station_id,
            "station_name": station.get("station_name"),
            "n_records": int(station["n_records"]),
            "n_years": float(station["n_years"]),
            "missing_daily_fraction": float(station["missing_daily_fraction"]),
            "tau_acf_days": tau_days,
            **observed_summary,
        }
        ar1_samples = [
            float(item["mean_beta_de_window"])
            for item in null_rows
            if item["station_id"] == station_id
            and item["null_model"] == "ar1"
            and np.isfinite(float(item["mean_beta_de_window"]))
        ]
        for null_model, samples in [
            ("seasonal_noise", seasonal_null),
            ("ar1", ar1_samples),
            ("phase_randomized", phase_null),
        ]:
            arr = np.asarray(samples, dtype=float)
            row[f"{null_model}_n"] = int(arr.size)
            row[f"{null_model}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
            row[f"{null_model}_p05"] = float(np.nanpercentile(arr, 5)) if arr.size else np.nan
            row[f"{null_model}_p95"] = float(np.nanpercentile(arr, 95)) if arr.size else np.nan
            observed_mean = observed_summary["mean_beta_de_window"]
            row[f"{null_model}_p_ge_observed"] = (
                float(np.mean(arr >= float(observed_mean))) if arr.size and np.isfinite(observed_mean) else np.nan
            )
        station_rows.append(row)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    station_summary = pd.DataFrame(station_rows).sort_values("tau_acf_days")
    null_summary = pd.DataFrame(null_rows)
    beta_de_all = pd.concat(beta_de_frames, ignore_index=True)
    station_summary.to_csv(tables_dir / f"{prefix}_summary.csv", index=False)
    null_summary.to_csv(tables_dir / f"{prefix}_null_samples.csv", index=False)
    beta_de_all.to_csv(tables_dir / f"{prefix}_beta_vs_de.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].scatter(station_summary["tau_acf_days"], station_summary["mean_beta_de_window"], s=45, color="#1f77b4")
    for _, row in station_summary.iterrows():
        axes[0].annotate(str(row["station_id"]), (row["tau_acf_days"], row["mean_beta_de_window"]), fontsize=7)
    axes[0].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("tau_acf [days]")
    axes[0].set_ylabel("mean beta in 0.5<=De<=2")
    axes[0].set_title("Data-driven memory normalization")

    axes[1].hist(station_summary["mean_beta_de_window"].dropna(), bins=8, color="#2ca02c", alpha=0.8)
    axes[1].axvline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("mean beta in 0.5<=De<=2")
    axes[1].set_ylabel("stations")
    axes[1].set_title("NWIS station distribution")
    fig.tight_layout()
    figure_path = save_figure(fig, f"{prefix}_diagnostic.png")
    plt.close(fig)

    print(f"Eligible stations: {eligible.shape[0]}")
    print(f"Saved {tables_dir / f'{prefix}_summary.csv'}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
