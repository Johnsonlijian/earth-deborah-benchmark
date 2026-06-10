"""Run an ITS_LIVE-compatible glacier irregular-series MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.data.glacier import (
    build_velocity_anomaly,
    filter_glacier_velocity,
    parse_glacier_velocity_dataframe,
    read_glacier_velocity_csv,
    summarize_glacier_points,
)
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.psd import lomb_scargle_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated list of floats."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def generate_demo_glacier(years: int = 12, seed: int = 20260527) -> pd.DataFrame:
    """Generate an irregular synthetic glacier velocity time series."""

    rng = np.random.default_rng(seed)
    dates = []
    current = pd.Timestamp("2008-01-01")
    end = pd.Timestamp("2008-01-01") + pd.Timedelta(days=int(years * 365.25))
    while current < end:
        current = current + pd.Timedelta(days=int(rng.integers(18, 95)))
        if current < end:
            dates.append(current)
    dates = pd.to_datetime(dates)
    t_years = (dates - dates.min()).days.to_numpy() / 365.25
    seasonal = 25.0 * np.sin(2.0 * np.pi * t_years)
    low_freq = 18.0 * np.sin(2.0 * np.pi * t_years / 4.0)
    noise = rng.normal(0.0, 10.0, size=len(dates))
    velocity = 180.0 + seasonal + low_freq + noise
    error = rng.uniform(5.0, 15.0, size=len(dates))
    return pd.DataFrame(
        {
            "glacier_id": "DEMO_GLACIER_001",
            "point_id": "POINT_001",
            "time": dates,
            "velocity_myr": velocity,
            "velocity_error_myr": error,
            "source": "synthetic_demo",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="ITS_LIVE-compatible point velocity CSV")
    parser.add_argument("--glacier-id", default=None)
    parser.add_argument("--point-id", default=None)
    parser.add_argument("--tau-days", default="90,180,365,730")
    parser.add_argument("--output-prefix", default="glacier_demo_mvp")
    parser.add_argument("--demo-years", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau_days_values = parse_float_list(args.tau_days)

    if args.input is None:
        parsed = parse_glacier_velocity_dataframe(generate_demo_glacier(years=args.demo_years, seed=args.seed))
        input_kind = "synthetic_demo"
    else:
        parsed = parse_glacier_velocity_dataframe(read_glacier_velocity_csv(args.input))
        input_kind = "local_csv"

    coverage = summarize_glacier_points(parsed).reset_index(drop=True)
    glacier_id = args.glacier_id or str(coverage.loc[0, "glacier_id"])
    point_id = args.point_id or str(coverage.loc[0, "point_id"])
    filtered = filter_glacier_velocity(parsed, glacier_id=glacier_id, point_id=point_id)
    anomaly = build_velocity_anomaly(filtered)
    if anomaly.shape[0] < 8:
        raise SystemExit("Need at least 8 velocity observations for MVP PSD")

    time_seconds = (anomaly["time"] - anomaly["time"].min()).dt.total_seconds()
    span_seconds = float(time_seconds.max() - time_seconds.min())
    median_dt = float(time_seconds.diff().median())
    min_freq = 1.0 / span_seconds
    max_freq = 0.5 / median_dt
    psd = lomb_scargle_psd(anomaly["time"], anomaly["value"], min_freq=min_freq, max_freq=max_freq, n_freqs=256)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    coverage.to_csv(tables_dir / f"{prefix}_coverage.csv", index=False)
    anomaly.to_csv(tables_dir / f"{prefix}_velocity_anomaly.csv", index=False)
    psd.to_csv(tables_dir / f"{prefix}_psd.csv", index=False)
    beta.to_csv(tables_dir / f"{prefix}_local_beta.csv", index=False)

    beta_de_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for tau_days in tau_days_values:
        beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)
        beta_de["process"] = "glacier"
        beta_de["region"] = glacier_id
        beta_de["variable"] = "velocity_anomaly"
        beta_de["tau_days"] = tau_days
        beta_de["input_kind"] = input_kind
        beta_de_frames.append(beta_de)
        summary_rows.append(
            {
                "glacier_id": glacier_id,
                "point_id": point_id,
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
    axes[0].errorbar(
        filtered["time"],
        filtered["velocity_myr"],
        yerr=filtered["velocity_error_myr"],
        fmt="o-",
        markersize=3,
        linewidth=0.8,
        color="#1f77b4",
    )
    axes[0].set_title("Irregular velocity")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("m/yr")

    axes[1].loglog(psd["frequency"], psd["psd"], color="#2ca02c", linewidth=1.0)
    axes[1].set_title("Lomb-Scargle PSD")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("PSD")

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
    print(f"Glacier point: {glacier_id}/{point_id}")
    print(f"Records: {len(filtered)}")
    print(f"Saved {tables_dir / f'{prefix}_summary.csv'}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
