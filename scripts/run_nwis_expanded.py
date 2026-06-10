"""Expanded USGS NWIS memory-scaling analysis with 50+ stations.

Downloads daily discharge for a curated set of USGS reference and long-record
stations, estimates tau_acf and beta(De), runs null-model comparisons, and
produces summary tables and figures.

Usage:
  python scripts/run_nwis_expanded.py
  python scripts/run_nwis_expanded.py --skip-download  # use cached raw files only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from edb.data.grdc import build_discharge_series, remove_daily_climatology, robust_mad_scale, summarize_station_coverage
from edb.data.nwis import build_nwis_daily_values_url, parse_nwis_daily_values_json
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.nulls import ar1_surrogate, phase_randomized_surrogate, seasonal_noise_surrogate
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time

# ── curated station list ──────────────────────────────────────────────
# Selected for: long record (40+ years), diverse drainage areas,
# mix of reference and regulated, continental US coverage.
# Each entry: (station_id, station_name, notes)
CURATED_STATIONS: list[tuple[str, str, str]] = [
    # --- Northeast / Mid-Atlantic ---
    ("01013500", "Fish River near Fort Kent, ME", "small, cold, unregulated"),
    ("01022500", "Narraguagus River at Cherryfield, ME", "coastal, unregulated"),
    ("01073000", "Oyster River near Durham, NH", "small coastal"),
    ("01134500", "Moose River at Victory, VT", "northern forest"),
    ("01333000", "Green River at Williamstown, MA", "small mountain"),
    ("01413500", "East Branch Delaware River at Margaretville, NY", "reservoir-influenced"),
    ("01439500", "Bush Kill at Shoemakers, PA", "small pocono"),
    ("01503000", "Susquehanna River at Conklin, NY", "large basin, mixed"),
    ("01548005", "Spring Creek at Milesburg, PA", "limestone spring-fed"),
    ("01646500", "Potomac River near Wash, DC", "large, mixed regulation"),

    # --- Southeast ---
    ("02037500", "James River near Richmond, VA", "large, piedmont"),
    ("02055000", "Roanoke River at Roanoke, VA", "mid-size, regulated"),
    ("02126000", "Rocky River near Norwood, NC", "piedmont"),
    ("02169000", "Saluda River near Greenville, SC", "piedmont, reservoir"),
    ("02202500", "Ogeechee River near Eden, GA", "coastal plain, unregulated"),
    ("02231000", "St. Marys River near Macclenny, FL", "coastal plain, swampy"),
    ("02358000", "Chattahoochee River at Columbus, GA", "large, regulated"),
    ("02472000", "Leaf River near McLain, MS", "gulf coastal plain"),
    ("03339000", "Vermilion River near Danville, IL", "agricultural, unregulated"),

    # --- Great Lakes / Midwest ---
    ("04027000", "Bad River near Odanah, WI", "northern forest"),
    ("04119000", "Grand River at Grand Rapids, MI", "large, mixed"),
    ("04193500", "Maumee River at Waterville, OH", "agricultural, large"),
    ("04213500", "Cattaraugus Creek at Gowanda, NY", "Lake Erie tributary"),
    ("05082500", "Red River of the North at Grand Forks, ND", "northern plains"),
    ("05330000", "Minnesota River near Jordan, MN", "large, agricultural"),
    ("05420500", "Mississippi River at Clinton, IA", "very large, regulated"),
    ("05464500", "Cedar River at Cedar Rapids, IA", "midwest agricultural"),
    ("05586100", "Illinois River at Valley City, IL", "large, regulated, navigation"),

    # --- Missouri Basin / Plains ---
    ("06054500", "Missouri River at Toston, MT", "headwaters, regulated"),
    ("06192500", "Yellowstone River near Livingston, MT", "mountain, snowmelt"),
    ("06334500", "Little Missouri River near Watford City, ND", "badlands, flashy"),
    ("06478500", "Big Sioux River near Brookings, SD", "northern plains"),
    ("06800500", "Platte River at Louisville, NE", "large, heavily regulated"),
    ("06846500", "Solomon River at Niles, KS", "plains, agricultural"),

    # --- Southern Plains / Texas ---
    ("07022000", "Mississippi River at Thebes, IL", "very large, regulated"),
    ("07152500", "Cimarron River near Waynoka, OK", "southern plains, flashy"),
    ("07227500", "Canadian River near Amarillo, TX", "semi-arid plains"),
    ("07374000", "Mississippi River at Baton Rouge, LA", "largest, regulated"),
    ("08055500", "Trinity River at Dallas, TX", "urban, regulated"),

    # --- Rocky Mountains ---
    ("09014050", "Colorado River at Granby, CO", "headwaters, reservoir"),
    ("09180500", "Colorado River near Cisco, UT", "upper basin"),
    ("09380000", "Colorado River at Lees Ferry, AZ", "large, heavily regulated"),
    ("09402000", "Little Colorado River near Cameron, AZ", "semi-arid"),
    ("09471500", "San Pedro River at Charleston, AZ", "semi-arid, groundwater-fed"),

    # --- Northwest ---
    ("11447650", "Sacramento River at Freeport, CA", "large, regulated"),
    ("11482500", "Klamath River near Klamath, CA", "coastal, regulated"),
    ("11532500", "Smith River near Crescent City, CA", "coastal, unregulated"),
    ("12010000", "Naselle River near Naselle, WA", "coastal rainforest"),
    ("12113000", "Green River near Auburn, WA", "regulated, snowmelt"),
    ("12301933", "Kootenai River bl Libby Dam nr Libby MT", "regulated, large reservoir"),
    ("12400500", "Pend Oreille River at Newport, WA", "regulated lake outlet"),
    ("12452500", "Chelan River at Chelan, WA", "regulated hydropower"),
    ("12472900", "Columbia River at Vernita Bridge nr Priest Rapids, WA", "large, regulated"),
    ("13011000", "Snake River near Moran, WY", "headwaters, snowmelt"),
    ("13185000", "Boise River near Twin Springs, ID", "regulated, irrigation"),
    ("13213000", "Donner und Blitzen River near Frenchglen, OR", "high desert, snowmelt"),
    ("13317000", "Salmon River at White Bird, ID", "wild and scenic, unregulated"),
    ("14105700", "Columbia River at The Dalles, OR", "very large, heavily regulated"),
    ("14211010", "Clackamas River near Estacada, OR", "regulated, snowmelt"),
    ("14246900", "Columbia River at Beaver Army Terminal nr Quincy, OR", "tidal influence"),

    # --- Alaska ---
    ("15356000", "Yukon River at Eagle, AK", "large, subarctic, unregulated"),
    ("15453500", "Tanana River at Nenana, AK", "subarctic, glacier-fed"),

    # --- Southwest ---
    ("09471000", "San Pedro River at Charleston, AZ", "semi-arid, intermittent"),
    ("08313000", "Rio Grande at Otowi Bridge, NM", "large, regulated, semi-arid"),
]

SECONDS_PER_DAY = 86_400.0
START_DATE = "1975-01-01"
END_DATE = "2026-05-28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/usgs_nwis_expanded")
    parser.add_argument("--output-prefix", default="nwis_expanded_1975_20260528")
    parser.add_argument("--max-acf-lag-days", type=int, default=730)
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.30)
    parser.add_argument("--n-null", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    return parser.parse_args()


def daily_sample_rate_hz() -> float:
    return 1.0 / SECONDS_PER_DAY


def download_one_station(session: requests.Session, station_id: str, raw_dir: Path) -> Path:
    """Download one station's daily values and cache as JSON."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"nwis_dv_{station_id}_{START_DATE}_{END_DATE}.json"
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"  [{station_id}] cached ({out_path.stat().st_size} bytes)")
        return out_path

    url = build_nwis_daily_values_url(sites=station_id, start_date=START_DATE, end_date=END_DATE)
    try:
        resp = session.get(url, timeout=180.0, headers={"User-Agent": "Mozilla/5.0 earth-deborah-benchmark/0.2"})
        resp.raise_for_status()
        payload = resp.json()
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        n_ts = len(payload.get("value", {}).get("timeSeries", []))
        print(f"  [{station_id}] downloaded ({n_ts} time series, {out_path.stat().st_size} bytes)")
        return out_path
    except Exception as exc:
        print(f"  [{station_id}] FAILED: {exc}", file=sys.stderr)
        if out_path.exists():
            out_path.unlink()
        return out_path  # return path even on failure so manifest can record it


def download_all(args: argparse.Namespace) -> list[Path]:
    """Download all curated stations, respecting batch size."""
    import os

    # Bypass desktop-level global VPN proxy — force direct connection
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(var, None)

    raw_dir = Path(args.raw_dir)
    session = requests.Session()
    session.trust_env = False           # ignore system proxy settings
    session.proxies = {}                # force direct connection
    paths: list[Path] = []
    for i, (station_id, _name, _notes) in enumerate(CURATED_STATIONS):
        if i > 0 and i % args.batch_size == 0:
            print(f"  [batch pause {args.batch_size}s]")
            sleep(args.batch_size)
        path = download_one_station(session, station_id, raw_dir)
        paths.append(path)
        sleep(0.5)
    return paths


def prepare_anomaly(discharge: pd.DataFrame) -> pd.DataFrame:
    anomaly = remove_daily_climatology(discharge)
    anomaly["value"] = pd.to_numeric(anomaly["value"], errors="coerce")
    anomaly["value"] = anomaly["value"].interpolate(method="linear", limit=30, limit_direction="both")
    anomaly = robust_mad_scale(anomaly)
    return anomaly.dropna(subset=["time", "value"]).reset_index(drop=True)


def beta_summary_for_values(dates: pd.Series, values: np.ndarray, tau_seconds: float) -> dict[str, float | int]:
    psd = welch_psd(dates, values, fs=daily_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    return summarize_beta_window(beta_de)


def process_station(records: pd.DataFrame, station_id: str, station_name: str, args: argparse.Namespace, rng: np.random.Generator):
    """Process one station: coverage, tau_acf, beta(De), nulls."""
    coverage = summarize_station_coverage(records[records["station_id"].astype(str) == station_id])
    if coverage.empty:
        return None, None, None, None

    station = coverage.iloc[0]
    n_years = float(station["n_years"])
    missing = float(station["missing_daily_fraction"])
    if n_years < args.min_years or missing > args.max_missing_fraction:
        return None, None, None, None

    station_records = records[records["station_id"].astype(str) == station_id]
    try:
        discharge = build_discharge_series(station_records, station_id=station_id, rule="D")
        anomaly = prepare_anomaly(discharge)
    except Exception as exc:
        print(f"  [{station_id}] prep failed: {exc}", file=sys.stderr)
        return None, None, None, None

    if len(anomaly) < 365:
        return None, None, None, None

    values = anomaly["value"].to_numpy(float)
    tau_seconds = integral_autocorrelation_time(values, dt_seconds=SECONDS_PER_DAY, max_lag=args.max_acf_lag_days, stop_at_zero=True)
    tau_days = tau_seconds / SECONDS_PER_DAY

    psd = welch_psd(anomaly["time"], values, fs=daily_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    beta_de["process"] = "river"
    beta_de["region"] = station_id
    beta_de["variable"] = "nwis_discharge_anomaly"
    beta_de["tau_days"] = tau_days
    beta_de["tau_source"] = "integral_acf_positive"
    beta_de["station_name"] = station_name
    observed_summary = summarize_beta_window(beta_de)

    # Null models
    seasonal_null: list[float] = []
    phase_null: list[float] = []
    ar1_samples_list: list[float] = []
    null_rows: list[dict[str, object]] = []
    for null_idx in range(args.n_null):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        null_specs = [
            ("seasonal_noise", seasonal_noise_surrogate(anomaly["time"], values, seed=seed, seasonal_key="dayofyear")),
            ("ar1", ar1_surrogate(values, seed=seed)),
            ("phase_randomized", phase_randomized_surrogate(values, seed=seed)),
        ]
        for null_model, null_values in null_specs:
            null_s = beta_summary_for_values(anomaly["time"], null_values, tau_seconds)
            null_mean = null_s["mean_beta_de_window"]
            null_rows.append({"station_id": station_id, "null_model": null_model, "null_idx": null_idx, "tau_days_observed": tau_days, **null_s})
            if np.isfinite(null_mean):
                if null_model == "seasonal_noise":
                    seasonal_null.append(float(null_mean))
                elif null_model == "phase_randomized":
                    phase_null.append(float(null_mean))
                elif null_model == "ar1":
                    ar1_samples_list.append(float(null_mean))

    row: dict[str, object] = {
        "station_id": station_id,
        "station_name": station_name,
        "n_records": int(station["n_records"]),
        "n_years": n_years,
        "missing_daily_fraction": missing,
        "tau_acf_days": tau_days,
        **observed_summary,
    }
    for null_model, samples in [("seasonal_noise", seasonal_null), ("ar1", ar1_samples_list), ("phase_randomized", phase_null)]:
        arr = np.asarray(samples, dtype=float)
        row[f"{null_model}_n"] = int(arr.size)
        row[f"{null_model}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
        row[f"{null_model}_p05"] = float(np.nanpercentile(arr, 5)) if arr.size else np.nan
        row[f"{null_model}_p95"] = float(np.nanpercentile(arr, 95)) if arr.size else np.nan
        observed_mean = observed_summary["mean_beta_de_window"]
        row[f"{null_model}_p_ge_observed"] = float(np.mean(arr >= float(observed_mean))) if arr.size and np.isfinite(observed_mean) else np.nan

    return row, null_rows, beta_de, tau_days


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)

    # Download
    if not args.skip_download:
        print(f"Downloading {len(CURATED_STATIONS)} stations...")
        download_all(args)

    # Parse cached raw files
    raw_paths = sorted(raw_dir.glob("nwis_dv_*.json"))
    if not raw_paths:
        raise SystemExit(f"No raw JSON files found in {raw_dir}")

    frames: list[pd.DataFrame] = []
    station_lookup: dict[str, str] = {}
    for sid, name, _notes in CURATED_STATIONS:
        station_lookup[sid] = name

    for path in raw_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = parse_nwis_daily_values_json(payload)
            if not parsed.empty:
                frames.append(parsed)
        except Exception as exc:
            print(f"Parse failed for {path}: {exc}", file=sys.stderr)

    if not frames:
        raise SystemExit("No parseable records found")
    records = pd.concat(frames, ignore_index=True)
    print(f"Parsed {len(records):,} daily discharge records from {records['station_id'].nunique()} stations")

    # Process
    rng = np.random.default_rng(args.seed)
    station_rows: list[dict[str, object]] = []
    all_null_rows: list[dict[str, object]] = []
    all_beta_de: list[pd.DataFrame] = []

    for station_id in sorted(records["station_id"].astype(str).unique()):
        name = station_lookup.get(station_id, station_id)
        print(f"Processing {station_id} ({name})...")
        try:
            row, null_rows, beta_de, tau_days = process_station(records, station_id, name, args, rng)
            if row is not None:
                station_rows.append(row)
                if null_rows:
                    all_null_rows.extend(null_rows)
                if beta_de is not None:
                    all_beta_de.append(beta_de)
                print(f"  tau_acf={tau_days:.1f}d, n_beta_de={row.get('n_beta_de_window', 0)}, mean_beta={row.get('mean_beta_de_window', np.nan):.3f}")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)

    # Save outputs
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    station_summary = pd.DataFrame(station_rows).sort_values("tau_acf_days")
    null_summary_df = pd.DataFrame(all_null_rows)
    beta_de_all = pd.concat(all_beta_de, ignore_index=True) if all_beta_de else pd.DataFrame()

    station_summary.to_csv(tables_dir / f"{prefix}_summary.csv", index=False)
    if not null_summary_df.empty:
        null_summary_df.to_csv(tables_dir / f"{prefix}_null_samples.csv", index=False)
    if not beta_de_all.empty:
        beta_de_all.to_csv(tables_dir / f"{prefix}_beta_vs_de.csv", index=False)

    n_eligible = len(station_summary)
    n_de_window = station_summary["n_beta_de_window"].gt(0).sum()
    print(f"\nEligible stations: {n_eligible}")
    print(f"Stations with populated De window: {n_de_window}")

    # Diagnostic plots
    valid = station_summary.dropna(subset=["mean_beta_de_window"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel 1: tau_acf distribution
    axes[0].hist(station_summary["tau_acf_days"], bins=20, color="#1f77b4", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("tau_acf [days]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Memory time distribution (n={n_eligible})")

    # Panel 2: beta vs tau_acf
    axes[1].scatter(valid["tau_acf_days"], valid["mean_beta_de_window"], s=30, color="#d62728", alpha=0.7)
    axes[1].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0, label="beta = 5/3")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("tau_acf [days]")
    axes[1].set_ylabel("mean beta in 0.5 <= De <= 2")
    axes[1].set_title("Storage-memory vs spectral slope")
    axes[1].legend(fontsize=8)

    # Panel 3: beta distribution
    axes[2].hist(valid["mean_beta_de_window"].dropna(), bins=15, color="#2ca02c", alpha=0.8, edgecolor="white")
    axes[2].axvline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("mean beta in 0.5 <= De <= 2")
    axes[2].set_ylabel("stations")
    axes[2].set_title(f"Deborah-normalized beta distribution (n={len(valid)})")

    fig.tight_layout()
    figure_path = save_figure(fig, f"{prefix}_diagnostic.png")
    plt.close(fig)
    print(f"Saved {figure_path}")
    print(f"Saved {tables_dir / f'{prefix}_summary.csv'}")

    # Print top stations
    print("\n=== Stations with greatest distance from null models ===")
    for col in ["seasonal_noise_p_ge_observed", "ar1_p_ge_observed", "phase_randomized_p_ge_observed"]:
        if col in valid.columns:
            best = valid.nsmallest(5, col)[["station_id", "station_name", "tau_acf_days", "mean_beta_de_window", col]]
            print(f"\n  Lowest {col}:")
            print(best.to_string())


if __name__ == "__main__":
    main()
