"""Run region/depth earthquake beta-vs-De diagnostics from cached ComCat events."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.data.usgs import build_event_rate_series, fetch_comcat_events_paginated, parse_comcat_geojson
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.preprocess import robust_zscore
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window


REGION_BOXES = {
    "global": (-90.0, 90.0, -180.0, 180.0),
    "japan_kuril": (25.0, 50.0, 125.0, 155.0),
    "chile_andes": (-56.0, -15.0, -82.0, -65.0),
    "indonesia_sunda": (-12.0, 8.0, 92.0, 132.0),
    "california_western_us": (30.0, 45.0, -128.0, -112.0),
    "himalaya_tibet": (25.0, 40.0, 70.0, 102.0),
}


DEPTH_CLASSES = {
    "all_depths": (-np.inf, np.inf),
    "shallow_0_70km": (0.0, 70.0),
    "intermediate_70_300km": (70.0, 300.0),
    "deep_gt_300km": (300.0, np.inf),
}


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated string list."""

    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Cached ComCat GeoJSON. If omitted, fetches from USGS.")
    parser.add_argument("--starttime", default="2000-01-01")
    parser.add_argument("--endtime", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--minmagnitudes", default="5.0,5.5")
    parser.add_argument("--tau-days", default="7,30,90,365")
    parser.add_argument("--rule", default="D")
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--output-prefix", default="earthquake_regions")
    return parser.parse_args()


def sample_rate_for_rule(rule: str) -> float:
    """Return approximate sample rate in Hz for a pandas resampling rule."""

    if rule == "D":
        return 1.0 / 86_400.0
    if rule == "MS":
        return 1.0 / (30.4375 * 86_400.0)
    raise ValueError(f"unsupported rule: {rule}")


def load_events(args: argparse.Namespace) -> pd.DataFrame:
    """Load cached or live ComCat events."""

    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        payload = fetch_comcat_events_paginated(starttime=args.starttime, endtime=args.endtime, minmagnitude=5.0)
    return parse_comcat_geojson(payload)


def filter_region(events: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """Filter events by a named bounding box."""

    minlat, maxlat, minlon, maxlon = REGION_BOXES[region_name]
    return events[
        (events["latitude"] >= minlat)
        & (events["latitude"] <= maxlat)
        & (events["longitude"] >= minlon)
        & (events["longitude"] <= maxlon)
    ].copy()


def filter_depth(events: pd.DataFrame, depth_class: str) -> pd.DataFrame:
    """Filter events by a depth class."""

    min_depth, max_depth = DEPTH_CLASSES[depth_class]
    return events[(events["depth_km"] >= min_depth) & (events["depth_km"] < max_depth)].copy()


def main() -> None:
    args = parse_args()
    minmagnitudes = parse_float_list(args.minmagnitudes)
    tau_days_values = parse_float_list(args.tau_days)
    fs = sample_rate_for_rule(args.rule)
    events = load_events(args)
    if events.empty:
        raise SystemExit("No ComCat events available")

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    subset_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    beta_de_frames: list[pd.DataFrame] = []

    for minmag in minmagnitudes:
        mag_events = events[events["magnitude"] >= minmag].copy()
        for region_name in REGION_BOXES:
            region_events = filter_region(mag_events, region_name)
            for depth_class in DEPTH_CLASSES:
                subset = filter_depth(region_events, depth_class)
                subset_rows.append(
                    {
                        "minmagnitude": minmag,
                        "region": region_name,
                        "depth_class": depth_class,
                        "n_events": int(subset.shape[0]),
                    }
                )
                if subset.shape[0] < args.min_events:
                    continue
                series = build_event_rate_series(subset, rule=args.rule)
                series["value"] = robust_zscore(series["value"])
                psd = welch_psd(series["time"], series["value"], fs=fs)
                beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
                for tau_days in tau_days_values:
                    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)
                    beta_de["process"] = "earthquake"
                    beta_de["region"] = region_name
                    beta_de["variable"] = "event_count"
                    beta_de["depth_class"] = depth_class
                    beta_de["minmagnitude"] = minmag
                    beta_de["tau_days"] = tau_days
                    beta_de["resample_rule"] = args.rule
                    beta_de_frames.append(beta_de)
                    summary_rows.append(
                        {
                            "minmagnitude": minmag,
                            "region": region_name,
                            "depth_class": depth_class,
                            "tau_days": tau_days,
                            "resample_rule": args.rule,
                            "n_events": int(subset.shape[0]),
                            **summarize_beta_window(beta_de),
                        }
                    )

    prefix = args.output_prefix
    subset_counts = pd.DataFrame(subset_rows)
    summary = pd.DataFrame(summary_rows)
    subset_counts.to_csv(tables_dir / f"{prefix}_subset_counts.csv", index=False)
    if summary.empty:
        raise SystemExit("No region/depth subset passed the minimum-event threshold")
    beta_de_all = pd.concat(beta_de_frames, ignore_index=True)
    summary.to_csv(tables_dir / f"{prefix}_summary.csv", index=False)
    beta_de_all.to_csv(tables_dir / f"{prefix}_beta_vs_de.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    plot_df = summary[(summary["tau_days"] == 30) & (summary["depth_class"] == "all_depths")].copy()
    plot_df = plot_df.sort_values("mean_beta_de_window")
    labels = plot_df["region"] + " M>=" + plot_df["minmagnitude"].astype(str)
    ax.barh(labels, plot_df["mean_beta_de_window"], color="#9467bd")
    ax.axvline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("mean beta in 0.5<=De<=2")
    ax.set_title("Earthquake regional event-count beta, tau=30 d")
    fig.tight_layout()
    figure_path = save_figure(fig, f"{prefix}_summary.png")
    plt.close(fig)

    print(f"Subsets evaluated: {subset_counts.shape[0]}")
    print(f"Subsets analyzed: {summary[['region', 'depth_class', 'minmagnitude']].drop_duplicates().shape[0]}")
    print(f"Saved {tables_dir / f'{prefix}_summary.csv'}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()

