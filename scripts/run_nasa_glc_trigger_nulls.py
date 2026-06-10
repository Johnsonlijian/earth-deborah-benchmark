"""Run trigger-specific NASA GLC landslide null-model diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.data.landslide_glc import (
    GLC_LEGACY_CSV_URL,
    build_landslide_count_series,
    filter_glc_records,
    parse_glc_dataframe,
    read_glc_csv,
)
from edb.plots.figures import save_figure
from edb.signal.de import beta_vs_de
from edb.signal.nulls import phase_randomized_surrogate, seasonal_noise_surrogate
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
    parser.add_argument("--tau-days", default="90,180,365")
    parser.add_argument("--top-triggers", type=int, default=6)
    parser.add_argument("--min-trigger-events", type=int, default=100)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--output-prefix", default="landslide_glc_trigger_nulls")
    return parser.parse_args()


def monthly_sample_rate_hz() -> float:
    """Return approximate monthly sample rate in Hz."""

    return 1.0 / (30.4375 * 86_400.0)


def beta_de_for_values(dates: pd.Series, values: np.ndarray, tau_days: float) -> pd.DataFrame:
    """Compute beta-vs-De for one monthly count-like sequence."""

    z_values = robust_zscore(values)
    psd = welch_psd(dates, z_values, fs=monthly_sample_rate_hz())
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    return beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * 86_400.0)


def main() -> None:
    args = parse_args()
    tau_days_values = parse_float_list(args.tau_days)
    if args.n_null <= 0:
        raise ValueError("n-null must be positive")

    raw = read_glc_csv(args.source)
    parsed = parse_glc_dataframe(raw)
    filtered = filter_glc_records(parsed, start_date=args.start_date, end_date=args.end_date)
    if filtered.empty:
        raise SystemExit("No GLC records remain after filtering")

    trigger_counts = filtered["landslide_trigger"].value_counts(dropna=True)
    triggers = [
        trigger
        for trigger, count in trigger_counts.head(args.top_triggers).items()
        if trigger != "unknown" and count >= args.min_trigger_events
    ]
    groups: list[tuple[str, pd.DataFrame]] = [("global", filtered)]
    groups.extend((f"trigger:{trigger}", filtered[filtered["landslide_trigger"] == trigger]) for trigger in triggers)

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for group_name, group_df in groups:
        monthly = build_landslide_count_series(group_df, rule="MS")
        dates = monthly["time"]
        values = monthly["value"].to_numpy(float)

        for tau_days in tau_days_values:
            observed_beta_de = beta_de_for_values(dates, values, tau_days)
            observed_summary = summarize_beta_window(observed_beta_de)
            observed_mean = observed_summary["mean_beta_de_window"]
            observed_median = observed_summary["median_beta_de_window"]

            null_values: dict[str, list[float]] = {"seasonal_noise": [], "phase_randomized": []}
            for null_idx in range(args.n_null):
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                seasonal = seasonal_noise_surrogate(dates, values, seed=seed, seasonal_key="month")
                phase = phase_randomized_surrogate(values, seed=seed)
                for null_name, null_series in [("seasonal_noise", seasonal), ("phase_randomized", phase)]:
                    null_beta_de = beta_de_for_values(dates, null_series, tau_days)
                    null_summary = summarize_beta_window(null_beta_de)
                    null_mean = null_summary["mean_beta_de_window"]
                    sample_rows.append(
                        {
                            "group": group_name,
                            "tau_days": tau_days,
                            "null_model": null_name,
                            "null_idx": null_idx,
                            "mean_beta_de_window": null_mean,
                            "median_beta_de_window": null_summary["median_beta_de_window"],
                            "n_beta_de_window": null_summary["n_beta_de_window"],
                        }
                    )
                    if np.isfinite(null_mean):
                        null_values[null_name].append(float(null_mean))

            row: dict[str, object] = {
                "group": group_name,
                "tau_days": tau_days,
                "n_events": int(group_df.shape[0]),
                "n_months": int(monthly.shape[0]),
                "observed_mean_beta_de_window": observed_mean,
                "observed_median_beta_de_window": observed_median,
                "observed_n_beta_de_window": observed_summary["n_beta_de_window"],
            }
            for null_name, values_list in null_values.items():
                arr = np.asarray(values_list, dtype=float)
                row[f"{null_name}_n"] = int(arr.size)
                row[f"{null_name}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
                row[f"{null_name}_p05"] = float(np.nanpercentile(arr, 5)) if arr.size else np.nan
                row[f"{null_name}_p95"] = float(np.nanpercentile(arr, 95)) if arr.size else np.nan
                row[f"{null_name}_p_ge_observed"] = (
                    float(np.mean(arr >= float(observed_mean))) if arr.size and np.isfinite(observed_mean) else np.nan
                )
            summary_rows.append(row)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    samples_df = pd.DataFrame(sample_rows)
    summary_path = tables_dir / f"{args.output_prefix}_summary.csv"
    samples_path = tables_dir / f"{args.output_prefix}_null_samples.csv"
    trigger_counts_path = tables_dir / f"{args.output_prefix}_trigger_counts.csv"
    summary_df.to_csv(summary_path, index=False)
    samples_df.to_csv(samples_path, index=False)
    trigger_counts.rename_axis("landslide_trigger").reset_index(name="n_events").to_csv(trigger_counts_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    top_counts = trigger_counts.head(10).sort_values()
    axes[0].barh(top_counts.index.astype(str), top_counts.values, color="#1f77b4")
    axes[0].set_title("Top GLC triggers")
    axes[0].set_xlabel("records")

    plot_df = summary_df.copy()
    for group_name, group in plot_df.groupby("group"):
        group = group.sort_values("tau_days")
        axes[1].plot(group["tau_days"], group["observed_mean_beta_de_window"], marker="o", label=group_name)
    axes[1].axhline(5.0 / 3.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_title("Observed trigger beta")
    axes[1].set_xlabel("tau [days]")
    axes[1].set_ylabel("mean beta in 0.5<=De<=2")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figure_path = save_figure(fig, f"{args.output_prefix}_diagnostic.png")
    plt.close(fig)

    print(f"Groups: {', '.join(name for name, _ in groups)}")
    print(f"Saved {summary_path}")
    print(f"Saved {samples_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
