"""Run a GRDC-compatible river discharge MVP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.data.grdc import (
    build_discharge_series,
    filter_river_records,
    parse_river_dataframe,
    read_river_csv,
    remove_daily_climatology,
    robust_mad_scale,
    summarize_station_coverage,
)
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated list of floats."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="GRDC-compatible CSV. If omitted, a synthetic demo is generated.")
    parser.add_argument("--station-id", default=None, help="Station id to analyze. Defaults to first station.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tau-days", default="7,30,90,180")
    parser.add_argument("--output-prefix", default="river_demo_mvp")
    parser.add_argument("--demo-years", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260526)
    return parser.parse_args()


def generate_demo_river(years: int = 20, seed: int = 20260526) -> pd.DataFrame:
    """Generate a synthetic daily river-discharge station for pipeline testing."""

    rng = np.random.default_rng(seed)
    n_days = int(years * 365.25)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    day = dates.dayofyear.to_numpy()
    seasonal = 80.0 + 35.0 * np.sin(2.0 * np.pi * (day - 80) / 365.25)
    ar = np.zeros(n_days)
    for idx in range(1, n_days):
        ar[idx] = 0.92 * ar[idx - 1] + rng.normal(0.0, 4.0)
    pulses = np.zeros(n_days)
    pulse_days = rng.choice(np.arange(20, n_days - 20), size=max(10, years * 3), replace=False)
    for start in pulse_days:
        length = int(rng.integers(3, 18))
        amp = float(rng.lognormal(mean=2.5, sigma=0.45))
        decay = amp * np.exp(-np.arange(length) / rng.uniform(2.0, 7.0))
        pulses[start : start + length] += decay
    discharge = np.maximum(seasonal + ar + pulses, 0.1)
    return pd.DataFrame(
        {
            "station_id": "DEMO_RIVER_001",
            "station_name": "Synthetic Demo River",
            "river_name": "Demo River",
            "country_code": "XX",
            "time": dates,
            "discharge_m3s": discharge,
            "quality_flag": "demo",
        }
    )


def daily_sample_rate_hz() -> float:
    """Return daily sample rate in Hz."""

    return 1.0 / 86_400.0


def main() -> None:
    args = parse_args()
    tau_days_values = parse_float_list(args.tau_days)

    if args.input is None:
        parsed = parse_river_dataframe(generate_demo_river(years=args.demo_years, seed=args.seed))
        input_kind = "synthetic_demo"
    else:
        parsed = parse_river_dataframe(read_river_csv(args.input))
        input_kind = "local_csv"

    coverage = summarize_station_coverage(parsed)
    station_id = args.station_id or str(coverage.loc[0, "station_id"])
    filtered = filter_river_records(parsed, station_id=station_id, start_date=args.start_date, end_date=args.end_date)
    if filtered.empty:
        raise SystemExit(f"No river records found for station_id={station_id}")

    discharge = build_discharge_series(filtered, station_id=station_id, rule="D")
    anomaly = remove_daily_climatology(discharge)
    anomaly = robust_mad_scale(anomaly)

    psd = welch_psd(anomaly["time"], anomaly["value"], fs=daily_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    coverage.to_csv(tables_dir / f"{prefix}_coverage.csv", index=False)
    discharge.to_csv(tables_dir / f"{prefix}_discharge_series.csv", index=False)
    anomaly.to_csv(tables_dir / f"{prefix}_anomaly_series.csv", index=False)
    psd.to_csv(tables_dir / f"{prefix}_psd.csv", index=False)
    beta.to_csv(tables_dir / f"{prefix}_local_beta.csv", index=False)

    beta_de_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for tau_days in tau_days_values:
        beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)
        beta_de["process"] = "river"
        beta_de["region"] = station_id
        beta_de["variable"] = "discharge_anomaly"
        beta_de["tau_days"] = tau_days
        beta_de["input_kind"] = input_kind
        beta_de_frames.append(beta_de)
        summary_rows.append(
            {
                "station_id": station_id,
                "input_kind": input_kind,
                "tau_days": tau_days,
                "n_records": int(filtered.shape[0]),
                **summarize_beta_window(beta_de),
            }
        )

    beta_de_all = pd.concat(beta_de_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    beta_de_all.to_csv(tables_dir / f"{prefix}_beta_vs_de.csv", index=False)
    summary.to_csv(tables_dir / f"{prefix}_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(discharge["time"], discharge["discharge_m3s"], color="#1f77b4", linewidth=0.7)
    axes[0].set_title("Daily discharge")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("m3/s")

    axes[1].plot(anomaly["time"], anomaly["value"], color="#2ca02c", linewidth=0.7)
    axes[1].set_title("Seasonal anomaly")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("MAD z")

    for tau_days, group in beta_de_all.groupby("tau_days"):
        axes[2].semilogx(group["de"], group["beta"], linewidth=1.0, label=f"{tau_days:g} d")
    axes[2].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[2].axvline(1.0, color="#666666", linestyle=":", linewidth=1.0)
    axes[2].set_title("Beta vs De")
    axes[2].set_xlabel("De")
    axes[2].set_ylabel("local beta")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    figure_path = save_figure(fig, f"{prefix}_diagnostic.png")
    plt.close(fig)

    print(f"Input kind: {input_kind}")
    print(f"Station: {station_id}")
    print(f"Records: {len(filtered)}")
    print(f"Saved {tables_dir / f'{prefix}_summary.csv'}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
