"""Run the NASA GLC landslide monthly-count MVP diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from edb.data.landslide_glc import (
    GLC_LEGACY_CSV_URL,
    build_landslide_count_series,
    filter_glc_records,
    parse_glc_dataframe,
    read_glc_csv,
    summarize_glc_reporting,
)
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.preprocess import robust_zscore
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated list of floats."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=GLC_LEGACY_CSV_URL, help="NASA GLC CSV URL or local path")
    parser.add_argument("--start-date", default="2007-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--rule", default="MS", choices=["MS"])
    parser.add_argument("--tau-days", default="30,90,180,365")
    parser.add_argument("--top-countries", type=int, default=5)
    parser.add_argument("--min-country-events", type=int, default=100)
    parser.add_argument("--output-prefix", default="landslide_glc_mvp")
    return parser.parse_args()


def monthly_sample_rate_hz() -> float:
    """Return approximate monthly sample rate in Hz."""

    return 1.0 / (30.4375 * 86_400.0)


def compute_beta_de(monthly: pd.DataFrame, tau_days: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute PSD, local beta, and beta-vs-De for one monthly series."""

    work = monthly.copy()
    work["value"] = robust_zscore(work["value"])
    psd = welch_psd(work["time"], work["value"], fs=monthly_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    beta_de = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)
    return psd, beta, beta_de


def main() -> None:
    args = parse_args()
    tau_days_values = parse_float_list(args.tau_days)

    raw = read_glc_csv(args.source)
    parsed = parse_glc_dataframe(raw)
    filtered = filter_glc_records(parsed, start_date=args.start_date, end_date=args.end_date)
    if filtered.empty:
        raise SystemExit("No GLC records remain after filtering")

    country_counts = filtered["country_code"].value_counts(dropna=True)
    top_country_codes = [
        code for code, count in country_counts.head(args.top_countries).items() if count >= args.min_country_events
    ]
    groups: list[tuple[str, pd.DataFrame]] = [("global", filtered)]
    groups.extend((str(code), filtered[filtered["country_code"] == code]) for code in top_country_codes)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    reporting = summarize_glc_reporting(filtered)
    reporting_path = tables_dir / f"{args.output_prefix}_reporting_summary.csv"
    reporting.to_csv(reporting_path, index=False)

    country_summary = (
        filtered.groupby(["country_code", "country_name"], dropna=False)
        .agg(n_events=("event_id", "count"))
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    country_summary.to_csv(tables_dir / f"{args.output_prefix}_country_counts.csv", index=False)

    monthly_frames: list[pd.DataFrame] = []
    beta_curve_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for region, group in groups:
        monthly = build_landslide_count_series(group, rule=args.rule)
        monthly["region"] = region
        monthly["variable"] = "landslide_count"
        monthly_frames.append(monthly)

        for tau_days in tau_days_values:
            psd, beta, beta_de = compute_beta_de(monthly, tau_days)
            beta_de["process"] = "landslide"
            beta_de["region"] = region
            beta_de["variable"] = "landslide_count"
            beta_de["tau_days"] = tau_days
            beta_de["resample_rule"] = args.rule
            beta_de["n_events"] = int(group.shape[0])
            beta_curve_frames.append(beta_de)

            summary = summarize_beta_window(beta_de)
            summary_rows.append(
                {
                    "region": region,
                    "variable": "landslide_count",
                    "start_date": args.start_date,
                    "end_date": args.end_date or filtered["event_date"].max().date().isoformat(),
                    "resample_rule": args.rule,
                    "tau_days": tau_days,
                    "n_events": int(group.shape[0]),
                    **summary,
                }
            )

    monthly_counts = pd.concat(monthly_frames, ignore_index=True)
    beta_curves = pd.concat(beta_curve_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    monthly_counts.to_csv(tables_dir / f"{args.output_prefix}_monthly_counts.csv", index=False)
    beta_curves.to_csv(tables_dir / f"{args.output_prefix}_beta_vs_de.csv", index=False)
    summary_df.to_csv(tables_dir / f"{args.output_prefix}_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(reporting["event_year"].astype(str), reporting["n_events"], color="#1f77b4")
    axes[0].set_title("Annual GLC records")
    axes[0].set_xlabel("event year")
    axes[0].set_ylabel("records")
    axes[0].tick_params(axis="x", rotation=45)

    for region, group in summary_df.groupby("region"):
        group = group.sort_values("tau_days")
        axes[1].plot(group["tau_days"], group["mean_beta_de_window"], marker="o", linewidth=1.0, label=region)
    axes[1].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_title("Mean beta in 0.5<=De<=2")
    axes[1].set_xlabel("tau [days]")
    axes[1].set_ylabel("mean local beta")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    figure_path = save_figure(fig, f"{args.output_prefix}_diagnostic.png")
    plt.close(fig)

    print(f"Parsed {len(parsed)} GLC records")
    print(f"Filtered {len(filtered)} records")
    print(f"Top country groups: {', '.join(top_country_codes) if top_country_codes else 'none'}")
    print(f"Saved {tables_dir / f'{args.output_prefix}_summary.csv'}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
