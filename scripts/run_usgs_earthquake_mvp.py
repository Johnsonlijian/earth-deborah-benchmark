"""Fetch a USGS ComCat earthquake MVP and compute beta-vs-De diagnostics."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from edb.data.usgs import (
    build_energy_rate_series,
    build_event_rate_series,
    fetch_comcat_events_paginated,
    parse_comcat_geojson,
)
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.preprocess import robust_zscore
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starttime", default="2000-01-01", help="USGS start time, e.g. 2000-01-01")
    parser.add_argument("--endtime", default=datetime.now(UTC).date().isoformat(), help="USGS end time")
    parser.add_argument("--minmagnitude", type=float, default=5.5, help="Minimum magnitude")
    parser.add_argument("--rule", default="D", choices=["D", "MS"], help="Resampling rule: D or MS")
    parser.add_argument("--tau-days", type=float, default=30.0, help="Working tau for De mapping")
    parser.add_argument("--page-limit", type=int, default=20000, help="USGS page limit")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum USGS pages")
    parser.add_argument("--output-prefix", default="earthquake_global_m55", help="Output file prefix")
    return parser.parse_args()


def compute_beta_de(series, variable: str, fs: float, tau_seconds: float, args: argparse.Namespace):
    psd = welch_psd(series["time"], series["value"], fs=fs)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    beta_de["process"] = "earthquake"
    beta_de["region"] = "global"
    beta_de["variable"] = variable
    beta_de["minmagnitude"] = args.minmagnitude
    beta_de["resample_rule"] = args.rule
    beta_de["tau_source"] = f"working sensitivity value: {args.tau_days} days"
    return psd, beta, beta_de


def main() -> None:
    args = parse_args()
    payload = fetch_comcat_events_paginated(
        starttime=args.starttime,
        endtime=args.endtime,
        minmagnitude=args.minmagnitude,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    )
    events = parse_comcat_geojson(payload)
    if events.empty:
        raise SystemExit("USGS query returned no events")

    event_rate = build_event_rate_series(events, rule=args.rule)
    energy_rate = build_energy_rate_series(events, rule=args.rule)
    event_rate["value"] = robust_zscore(event_rate["value"])
    energy_rate["value"] = robust_zscore(energy_rate["value"])

    if args.rule == "D":
        fs = 1.0 / 86_400.0
        tau_seconds = args.tau_days * 86_400.0
    else:
        fs = 1.0 / (30.4375 * 86_400.0)
        tau_seconds = args.tau_days * 86_400.0

    event_psd, event_beta, event_beta_de = compute_beta_de(event_rate, "event_count", fs, tau_seconds, args)
    energy_psd, energy_beta, energy_beta_de = compute_beta_de(energy_rate, "energy_rate", fs, tau_seconds, args)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    event_rate.to_csv(tables_dir / f"{prefix}_event_rate.csv", index=False)
    energy_rate.to_csv(tables_dir / f"{prefix}_energy_rate.csv", index=False)
    event_psd.to_csv(tables_dir / f"{prefix}_event_psd.csv", index=False)
    event_beta.to_csv(tables_dir / f"{prefix}_event_local_beta.csv", index=False)
    event_beta_de.to_csv(tables_dir / f"{prefix}_event_beta_vs_de.csv", index=False)
    energy_psd.to_csv(tables_dir / f"{prefix}_energy_psd.csv", index=False)
    energy_beta.to_csv(tables_dir / f"{prefix}_energy_local_beta.csv", index=False)
    energy_beta_de.to_csv(tables_dir / f"{prefix}_energy_beta_vs_de.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    rows = [
        ("Event count", event_rate, event_psd, event_beta_de),
        ("Energy rate", energy_rate, energy_psd, energy_beta_de),
    ]
    for row_idx, (label, series, psd, beta_de) in enumerate(rows):
        axes[row_idx, 0].plot(series["time"], series["value"], color="#1f77b4", linewidth=0.8)
        axes[row_idx, 0].set_title(f"{label} z-score")
        axes[row_idx, 0].set_xlabel("time")
        axes[row_idx, 0].set_ylabel("robust z")

        axes[row_idx, 1].loglog(psd["frequency"], psd["psd"], color="#2ca02c", linewidth=1.1)
        axes[row_idx, 1].set_title("Welch PSD")
        axes[row_idx, 1].set_xlabel("frequency [Hz]")
        axes[row_idx, 1].set_ylabel("PSD")

        axes[row_idx, 2].semilogx(beta_de["de"], beta_de["beta"], color="#d62728", linewidth=1.1)
        axes[row_idx, 2].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
        axes[row_idx, 2].axvline(1.0, color="#666666", linestyle=":", linewidth=1.0)
        axes[row_idx, 2].set_title("Beta vs De")
        axes[row_idx, 2].set_xlabel("De")
        axes[row_idx, 2].set_ylabel("local beta")

    fig.tight_layout()
    output_path = save_figure(fig, f"{prefix}_diagnostic.png")
    plt.close(fig)

    print(f"Fetched {len(events)} events")
    print(f"Saved {tables_dir / f'{prefix}_event_beta_vs_de.csv'}")
    print(f"Saved {tables_dir / f'{prefix}_energy_beta_vs_de.csv'}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
