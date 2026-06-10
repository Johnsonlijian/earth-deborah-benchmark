"""Run earthquake magnitude/tau sensitivity diagnostics from USGS ComCat."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
from edb.signal.summary import summarize_beta_window


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    """Parse a comma-separated string list."""

    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starttime", default="2000-01-01")
    parser.add_argument("--endtime", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--minmagnitudes", default="5.0,5.5,6.0")
    parser.add_argument("--tau-days", default="7,30,365")
    parser.add_argument("--rules", default="D,MS")
    parser.add_argument("--page-limit", type=int, default=20000)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--output-prefix", default="earthquake_sensitivity")
    return parser.parse_args()


def sample_rate_for_rule(rule: str) -> float:
    """Return approximate sample rate in Hz for a pandas resampling rule."""

    if rule == "D":
        return 1.0 / 86_400.0
    if rule == "MS":
        return 1.0 / (30.4375 * 86_400.0)
    raise ValueError(f"unsupported rule: {rule}")


def compute_variable_outputs(series: pd.DataFrame, fs: float, tau_days: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute local beta and beta-vs-De outputs for one series/tau."""

    psd = welch_psd(series["time"], series["value"], fs=fs)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)
    return beta, beta_de


def main() -> None:
    args = parse_args()
    minmagnitudes = parse_float_list(args.minmagnitudes)
    tau_days_values = parse_float_list(args.tau_days)
    rules = parse_str_list(args.rules)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []

    for minmag in minmagnitudes:
        payload = fetch_comcat_events_paginated(
            starttime=args.starttime,
            endtime=args.endtime,
            minmagnitude=minmag,
            page_limit=args.page_limit,
            max_pages=args.max_pages,
        )
        events = parse_comcat_geojson(payload)
        if events.empty:
            continue

        for rule in rules:
            fs = sample_rate_for_rule(rule)
            event_rate = build_event_rate_series(events, rule=rule)
            energy_rate = build_energy_rate_series(events, rule=rule)
            event_rate["value"] = robust_zscore(event_rate["value"])
            energy_rate["value"] = robust_zscore(energy_rate["value"])

            variables = {
                "event_count": event_rate,
                "energy_rate": energy_rate,
            }
            for variable, series in variables.items():
                for tau_days in tau_days_values:
                    beta, beta_de = compute_variable_outputs(series, fs, tau_days)
                    beta_de["process"] = "earthquake"
                    beta_de["region"] = "global"
                    beta_de["variable"] = variable
                    beta_de["minmagnitude"] = minmag
                    beta_de["resample_rule"] = rule
                    beta_de["tau_days"] = tau_days
                    beta_de["starttime"] = args.starttime
                    beta_de["endtime"] = args.endtime
                    beta_de["n_events"] = len(events)
                    curve_rows.append(beta_de)

                    summary = summarize_beta_window(beta_de)
                    summary_rows.append(
                        {
                            "starttime": args.starttime,
                            "endtime": args.endtime,
                            "minmagnitude": minmag,
                            "resample_rule": rule,
                            "variable": variable,
                            "tau_days": tau_days,
                            "n_events": len(events),
                            **summary,
                        }
                    )

    if not summary_rows:
        raise SystemExit("No sensitivity outputs were generated")

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.concat(curve_rows, ignore_index=True)
    summary_path = tables_dir / f"{args.output_prefix}_summary.csv"
    curves_path = tables_dir / f"{args.output_prefix}_beta_vs_de.csv"
    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=True)
    axes_flat = axes.ravel()
    panels = [
        ("D", "event_count"),
        ("D", "energy_rate"),
        ("MS", "event_count"),
        ("MS", "energy_rate"),
    ]
    for ax, (rule, variable) in zip(axes_flat, panels, strict=True):
        subset = summary_df[(summary_df["resample_rule"] == rule) & (summary_df["variable"] == variable)]
        if subset.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        for minmag, group in subset.groupby("minmagnitude"):
            group = group.sort_values("tau_days")
            ax.plot(group["tau_days"], group["mean_beta_de_window"], marker="o", label=f"M>={minmag:g}")
        ax.axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_title(f"{rule} {variable}")
        ax.set_xlabel("tau [days]")
        ax.set_ylabel("mean beta in 0.5<=De<=2")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.tight_layout()
    figure_path = save_figure(fig, f"{args.output_prefix}_summary.png")
    plt.close(fig)

    print(f"Saved {summary_path}")
    print(f"Saved {curves_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
