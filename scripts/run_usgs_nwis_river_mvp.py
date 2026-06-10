"""Download and analyze public USGS NWIS daily river discharge stations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.data.grdc import (
    build_discharge_series,
    remove_daily_climatology,
    robust_mad_scale,
    summarize_station_coverage,
)
from edb.data.nwis import NWIS_DAILY_VALUES_URL, fetch_nwis_daily_values, parse_nwis_daily_values_json
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window


DEFAULT_SITES = [
    "01646500",  # Potomac River near Washington, DC
    "09380000",  # Colorado River at Lees Ferry, AZ
    "14105700",  # Columbia River at The Dalles, OR
    "02037500",  # James River near Richmond, VA
    "03339000",  # Vermilion River near Danville, IL
    "06192500",  # Yellowstone River near Livingston, MT
    "12301933",  # Kootenai River below Libby Dam, MT
    "11447650",  # Sacramento River at Freeport, CA
    "09522000",  # Colorado River above Morelos Dam, AZ
]


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated string list."""

    return [item.strip() for item in raw.split(",") if item.strip()]


def file_sha256(path: Path) -> str:
    """Return SHA256 for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON cache file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def daily_sample_rate_hz() -> float:
    """Return daily sample rate in Hz."""

    return 1.0 / 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-ids", default=",".join(DEFAULT_SITES))
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--tau-days", default="7,30,90,180,365")
    parser.add_argument("--raw-dir", default="data/raw/usgs_nwis")
    parser.add_argument("--output-prefix", default="usgs_nwis_river_mvp")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.20)
    parser.add_argument("--refresh", action="store_true", help="Re-download even if a matching raw cache exists.")
    return parser.parse_args()


def load_or_fetch_site(site_id: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load cached NWIS JSON or fetch one station."""

    raw_dir = Path(args.raw_dir)
    raw_path = raw_dir / f"nwis_dv_{site_id}_{args.start_date}_{args.end_date}.json"
    status = "cached_raw"
    if args.refresh or not raw_path.exists():
        payload = fetch_nwis_daily_values(site_id, start_date=args.start_date, end_date=args.end_date)
        write_json(raw_path, payload)
        status = "downloaded_raw_cache"
    else:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))

    parsed = parse_nwis_daily_values_json(payload)
    row = {
        "dataset": "usgs_nwis_daily_values",
        "station_id": site_id,
        "path": str(raw_path),
        "source_url": NWIS_DAILY_VALUES_URL,
        "status": status,
        "size_bytes": raw_path.stat().st_size,
        "sha256": file_sha256(raw_path),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "notes": f"parameterCd=00060 discharge; statCd=00003 daily mean; rows={len(parsed)}",
    }
    return parsed, row


def failed_manifest_row(site_id: str, args: argparse.Namespace, error: Exception) -> dict[str, object]:
    """Build a manifest row for a station download failure."""

    raw_path = Path(args.raw_dir) / f"nwis_dv_{site_id}_{args.start_date}_{args.end_date}.json"
    return {
        "dataset": "usgs_nwis_daily_values",
        "station_id": site_id,
        "path": str(raw_path),
        "source_url": NWIS_DAILY_VALUES_URL,
        "status": "failed",
        "size_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
        "sha256": file_sha256(raw_path) if raw_path.exists() else "",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "notes": f"{type(error).__name__}: {error}",
    }


def prepare_anomaly(discharge: pd.DataFrame) -> pd.DataFrame:
    """Create a finite daily anomaly series suitable for Welch PSD."""

    anomaly = remove_daily_climatology(discharge)
    anomaly["value"] = pd.to_numeric(anomaly["value"], errors="coerce")
    anomaly["value"] = anomaly["value"].interpolate(method="linear", limit=30, limit_direction="both")
    anomaly = robust_mad_scale(anomaly)
    return anomaly.dropna(subset=["time", "value"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    site_ids = parse_str_list(args.site_ids)
    tau_days_values = parse_float_list(args.tau_days)
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    parsed_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for site_id in site_ids:
        try:
            parsed, row = load_or_fetch_site(site_id, args)
            manifest_rows.append(row)
            if not parsed.empty:
                parsed_frames.append(parsed)
        except Exception as exc:
            manifest_rows.append(failed_manifest_row(site_id, args, exc))

    if not parsed_frames:
        raise SystemExit("No NWIS station data were downloaded or parsed")

    all_records = pd.concat(parsed_frames, ignore_index=True)
    coverage = summarize_station_coverage(all_records)
    eligible = coverage[
        (coverage["n_years"] >= args.min_years) & (coverage["missing_daily_fraction"] <= args.max_missing_fraction)
    ].copy()
    if eligible.empty:
        raise SystemExit("No NWIS stations passed coverage filters")

    summary_rows: list[dict[str, object]] = []
    beta_de_frames: list[pd.DataFrame] = []
    local_beta_frames: list[pd.DataFrame] = []

    for _, station in eligible.iterrows():
        station_id = str(station["station_id"])
        station_records = all_records[all_records["station_id"].astype(str) == station_id]
        discharge = build_discharge_series(station_records, station_id=station_id, rule="D")
        anomaly = prepare_anomaly(discharge)
        if anomaly.shape[0] < 365:
            continue
        psd = welch_psd(anomaly["time"], anomaly["value"], fs=daily_sample_rate_hz())
        beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
        beta["station_id"] = station_id
        local_beta_frames.append(beta)
        for tau_days in tau_days_values:
            beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)
            beta_de["process"] = "river"
            beta_de["region"] = station_id
            beta_de["variable"] = "nwis_discharge_anomaly"
            beta_de["tau_days"] = tau_days
            beta_de["station_name"] = station.get("station_name")
            beta_de["input_kind"] = "usgs_nwis_daily_values"
            beta_de_frames.append(beta_de)
            summary_rows.append(
                {
                    "station_id": station_id,
                    "station_name": station.get("station_name"),
                    "tau_days": tau_days,
                    "n_records": int(station["n_records"]),
                    "n_years": float(station["n_years"]),
                    "missing_daily_fraction": float(station["missing_daily_fraction"]),
                    **summarize_beta_window(beta_de),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    beta_de_all = pd.concat(beta_de_frames, ignore_index=True)
    local_beta_all = pd.concat(local_beta_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    prefix = args.output_prefix
    manifest.to_csv(tables_dir / f"{prefix}_raw_manifest.csv", index=False)
    coverage.to_csv(tables_dir / f"{prefix}_coverage.csv", index=False)
    eligible.to_csv(tables_dir / f"{prefix}_eligible_stations.csv", index=False)
    all_records.to_csv(tables_dir / f"{prefix}_daily_values_derived.csv", index=False)
    local_beta_all.to_csv(tables_dir / f"{prefix}_local_beta.csv", index=False)
    beta_de_all.to_csv(tables_dir / f"{prefix}_beta_vs_de.csv", index=False)
    summary.to_csv(tables_dir / f"{prefix}_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_summary = summary[summary["tau_days"] == 90].sort_values("mean_beta_de_window")
    axes[0].barh(plot_summary["station_id"], plot_summary["mean_beta_de_window"], color="#1f77b4")
    axes[0].axvline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[0].set_title("NWIS river beta, tau=90 d")
    axes[0].set_xlabel("mean beta in 0.5<=De<=2")

    axes[1].scatter(summary["tau_days"], summary["mean_beta_de_window"], s=18, alpha=0.75)
    axes[1].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_title("All eligible station/tau windows")
    axes[1].set_xlabel("tau [days]")
    axes[1].set_ylabel("mean beta in 0.5<=De<=2")
    fig.tight_layout()
    figure_path = save_figure(fig, f"{prefix}_diagnostic.png")
    plt.close(fig)

    print(f"Downloaded/loaded stations: {len(site_ids)}")
    print(f"Eligible stations: {eligible.shape[0]}")
    print(f"Saved {tables_dir / f'{prefix}_summary.csv'}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
