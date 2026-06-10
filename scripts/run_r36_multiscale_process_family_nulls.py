"""R36 lightweight multiscale process-family null controls.

This script complements the heavier gauge-matched multiscale surrogate script
with a fast, reproducible process-family experiment. It uses the same
tau_acf--PSD--local-beta--De dispersion metric as the manuscript, but simulates
families whose interpretation is explicit:

* single-timescale AR(1): the Fourier-pair ceiling for one relaxation time;
* two-timescale AR mixture: fast and slow linear components with the same
  nominal target memory;
* ARFIMA-like long memory: fractional integration without a unique relaxation
  time;
* seasonal two-timescale control: a deterministic annual component added before
  the day-of-year anomaly step.

The analysis is a mechanism-boundary control, not a fitted catchment model.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX_SOURCE = ROOT / "source_data"
SECONDS_PER_DAY = 86_400.0
DEFAULT_PREFIX = "r36_multiscale_process_family_nulls"


@dataclass(frozen=True)
class Scenario:
    family: str
    target_tau_days: float
    replicate: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=14)
    parser.add_argument("--series-length", type=int, default=9000)
    parser.add_argument("--burn-in", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--arfima-d", type=float, default=0.28)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def make_scenarios(n_replicates: int) -> list[Scenario]:
    tau_grid = [2, 4, 8, 16, 32, 64, 128]
    families = [
        "single_timescale_ar1",
        "two_timescale_ar_mix",
        "arfima_fractional",
        "seasonal_two_timescale_control",
    ]
    return [
        Scenario(family=family, target_tau_days=float(tau), replicate=rep)
        for family in families
        for tau in tau_grid
        for rep in range(n_replicates)
    ]


def simulate_unit_ar1(n: int, tau_days: float, rng: np.random.Generator) -> np.ndarray:
    phi = float(np.exp(-1.0 / max(tau_days, 0.2)))
    phi = float(np.clip(phi, -0.995, 0.995))
    sigma = float(np.sqrt(max(1.0 - phi * phi, 1e-8)))
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(0.0, 1.0)
    for idx in range(1, n):
        out[idx] = phi * out[idx - 1] + rng.normal(0.0, sigma)
    out = out - np.mean(out)
    std = float(np.std(out))
    return out / std if std > 0 else out


def two_timescale_mix(n: int, target_tau_days: float, rng: np.random.Generator) -> np.ndarray:
    fast_tau = max(1.0, target_tau_days / 6.0)
    slow_tau = min(730.0, max(target_tau_days * 6.0, fast_tau + 1.0))
    fast_weight = (target_tau_days - slow_tau) / (fast_tau - slow_tau)
    fast_weight = float(np.clip(fast_weight, 0.25, 0.90))
    fast = simulate_unit_ar1(n, fast_tau, rng)
    slow = simulate_unit_ar1(n, slow_tau, rng)
    out = np.sqrt(fast_weight) * fast + np.sqrt(1.0 - fast_weight) * slow
    out = out - np.mean(out)
    std = float(np.std(out))
    return out / std if std > 0 else out


def fractional_integration_weights(d: float, n_weights: int) -> np.ndarray:
    d = float(np.clip(d, 0.02, 0.45))
    weights = np.empty(n_weights, dtype=float)
    weights[0] = 1.0
    for k in range(1, n_weights):
        weights[k] = weights[k - 1] * ((k - 1 + d) / k)
        if k > 128 and abs(weights[k]) < 1e-5:
            return weights[: k + 1]
    return weights


def arfima_fractional(n: int, d: float, rng: np.random.Generator) -> np.ndarray:
    n_weights = int(min(max(512, n // 3), 4096))
    weights = fractional_integration_weights(d, n_weights)
    eps = rng.normal(0.0, 1.0, n + len(weights))
    out = np.convolve(eps, weights, mode="valid")[:n]
    out = out - np.mean(out)
    std = float(np.std(out))
    return out / std if std > 0 else out


def seasonal_anomaly(values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    dates = pd.date_range("1980-01-01", periods=len(values), freq="D")
    frame = pd.DataFrame({"value": values}, index=dates)
    clim = frame.groupby(frame.index.dayofyear)["value"].transform("mean")
    anomaly = frame["value"] - clim
    anomaly = anomaly.to_numpy(dtype=float)
    anomaly = anomaly - np.nanmedian(anomaly)
    mad = float(np.nanmedian(np.abs(anomaly - np.nanmedian(anomaly))))
    if mad > 0:
        anomaly = anomaly / mad
    return dates, anomaly


def simulate_scenario(scenario: Scenario, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    n = int(args.series_length + args.burn_in)
    if scenario.family == "single_timescale_ar1":
        values = simulate_unit_ar1(n, scenario.target_tau_days, rng)
    elif scenario.family == "two_timescale_ar_mix":
        values = two_timescale_mix(n, scenario.target_tau_days, rng)
    elif scenario.family == "arfima_fractional":
        values = arfima_fractional(n, args.arfima_d, rng)
    elif scenario.family == "seasonal_two_timescale_control":
        day = np.arange(n)
        values = two_timescale_mix(n, scenario.target_tau_days, rng)
        values = values + 0.85 * np.sin(2.0 * np.pi * day / 365.25)
    else:
        raise ValueError(f"unknown family: {scenario.family}")
    return values[int(args.burn_in) :]


def beta_curve_for_series(dates: pd.DatetimeIndex, values: np.ndarray, tau_seconds: float) -> pd.DataFrame:
    psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    return curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])


def station_bin_medians(curves: pd.DataFrame, value_col: str, n_bins: int = 24) -> pd.DataFrame:
    x = np.log10(curves[value_col].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work = curves.copy()
    work["bin_id"] = pd.cut(np.log10(work[value_col]), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    grouped = (
        work.groupby(["series_id", "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"), n_points=("beta", "size"))
    )
    grouped["axis_value"] = grouped["bin_id"].map({idx: 10 ** labels[idx] for idx in range(len(labels))})
    return grouped


def dispersion_metric(curves: pd.DataFrame, value_col: str, min_series: int = 12) -> tuple[float, int]:
    binned = station_bin_medians(curves, value_col=value_col)
    stats = (
        binned.groupby("bin_id", as_index=False)
        .agg(n_series=("series_id", "nunique"), beta_var=("beta_median", "var"))
        .dropna(subset=["beta_var"])
    )
    usable = stats[stats["n_series"] >= min_series]
    if usable.empty:
        return np.nan, 0
    weights = usable["n_series"].to_numpy(dtype=float)
    return float(np.average(usable["beta_var"].to_numpy(dtype=float), weights=weights)), int(len(usable))


def summarize_group(curves: pd.DataFrame, group_name: str) -> dict[str, float | int | str]:
    raw_var, raw_bins = dispersion_metric(curves, "frequency_cpd")
    de_var, de_bins = dispersion_metric(curves, "de")
    reduction = np.nan if not np.isfinite(raw_var) or raw_var <= 0 else 1.0 - de_var / raw_var
    return {
        "scenario_group": group_name,
        "n_series": int(curves["series_id"].nunique()),
        "raw_weighted_beta_variance": raw_var,
        "de_weighted_beta_variance": de_var,
        "variance_reduction_vs_raw": reduction,
        "variance_ratio_de_over_raw": np.nan if raw_var <= 0 else de_var / raw_var,
        "raw_bins_used": raw_bins,
        "de_bins_used": de_bins,
    }


def plot_results(metrics: pd.DataFrame, series_summary: pd.DataFrame, curves: pd.DataFrame, output_prefix: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    colors = {
        "single_timescale_ar1": "#C4513F",
        "two_timescale_ar_mix": "#2F855A",
        "arfima_fractional": "#6C5CE7",
        "seasonal_two_timescale_control": "#D99A24",
    }
    labels = {
        "single_timescale_ar1": "single AR(1)",
        "two_timescale_ar_mix": "two-timescale AR",
        "arfima_fractional": "ARFIMA-like",
        "seasonal_two_timescale_control": "seasonal two-scale",
    }
    order = list(colors)
    plot_m = metrics.set_index("scenario_group").loc[order].reset_index()

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    x = np.arange(len(plot_m))
    vals = 100.0 * plot_m["variance_reduction_vs_raw"].to_numpy(dtype=float)
    ax0.bar(x, vals, color=[colors[g] for g in plot_m["scenario_group"]], alpha=0.80, width=0.62)
    ax0.axhline(23.7, color="#2E6FAD", lw=1.0, ls="--")
    ax0.axhline(14.7, color="#2E6FAD", lw=1.0, ls=":")
    ax0.axhline(46.6, color="#8A1F17", lw=0.9, ls="-.")
    ax0.text(3.58, 23.7, "CAMELS-US", va="center", ha="left", fontsize=6, color="#2E6FAD")
    ax0.text(3.58, 14.7, "CAMELS-GB", va="center", ha="left", fontsize=6, color="#2E6FAD")
    ax0.text(3.58, 46.6, "AR(1) R18", va="center", ha="left", fontsize=6, color="#8A1F17")
    ax0.set_xticks(x)
    ax0.set_xticklabels([labels[g] for g in plot_m["scenario_group"]], rotation=18, ha="right")
    ax0.set_ylabel("De-axis dispersion reduction [%]")
    ax0.set_title("Multiscale process nulls reduce the single-timescale ceiling", loc="left")
    ax0.set_xlim(-0.6, 4.35)
    ax0.grid(True, axis="y", color="#E4E9EF", lw=0.55)
    for xx, yy in zip(x, vals):
        ax0.text(xx, yy + (1.2 if yy >= 0 else -2.2), f"{yy:.1f}%", ha="center", va="center", fontsize=6)

    for family, data in series_summary.groupby("scenario_group"):
        ax1.scatter(
            data["tau_acf_days"],
            data["mean_beta_de_window"],
            s=12,
            alpha=0.65,
            color=colors.get(family, "#777777"),
            label=labels.get(family, family),
        )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"estimated output $\tau_{acf}$ [d]")
    ax1.set_ylabel(r"mean beta in $0.5 \leq De \leq 2$")
    ax1.set_title("All families use estimated output memory", loc="left")
    ax1.grid(True, color="#E4E9EF", lw=0.55)

    sample_parts = []
    for family, group in curves.groupby("scenario_group", sort=False):
        keep = sorted(group["series_id"].unique())[: min(14, group["series_id"].nunique())]
        sample_parts.append(group[group["series_id"].isin(keep)].copy())
    sample = pd.concat(sample_parts, ignore_index=True)
    for family, group in sample.groupby("scenario_group"):
        binned = station_bin_medians(group, "de", n_bins=28)
        med = binned.groupby("bin_id", as_index=False).agg(axis_value=("axis_value", "first"), beta=("beta_median", "median"))
        ax2.semilogx(med["axis_value"], med["beta"], lw=1.4, color=colors.get(family, "#777777"), label=labels.get(family, family))
    ax2.axvspan(0.5, 2.0, color="#C4513F", alpha=0.07, lw=0)
    ax2.set_xlabel(r"$De=\tau_{acf} f$")
    ax2.set_ylabel("median local beta")
    ax2.set_title("De curves show why observed-minus-AR(1) deficit is a multiscale boundary, not direct causality", loc="left")
    ax2.grid(True, color="#E4E9EF", lw=0.55)
    ax2.legend(ncol=4, loc="upper left", fontsize=6)

    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(metrics: pd.DataFrame, output_prefix: str, args: argparse.Namespace) -> None:
    lines = [
        "# R36 Multiscale Process-Family Nulls",
        "",
        "Date: 2026-06-05",
        "",
        "## Method",
        "",
        "Synthetic families were processed by the same tau_acf, Welch PSD,",
        "local-slope and De-axis dispersion code used in the manuscript. The goal",
        "is to test whether the observed deficit below matched AR(1) can be read",
        "as a multiscale boundary rather than a positive excess-over-null claim.",
        "",
        f"- Replicates per family/tau: {args.n_replicates}",
        f"- Series length after burn-in: {args.series_length} days",
        f"- ARFIMA-like d: {args.arfima_d}",
        f"- Seed: {args.seed}",
        "",
        "## Aggregate metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Safe manuscript conclusion",
        "",
        "Process-family controls show whether single-timescale AR(1) should be",
        "treated as a ceiling and whether multiscale or long-memory structure can",
        "lower De-axis reductions toward the observed archives. They do not prove",
        "direct storage or tracer causality.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{output_prefix}_metrics.csv`",
        f"- `reports/tables/{output_prefix}_series.csv`",
        f"- `reports/tables/{output_prefix}_curves.csv`",
        f"- `reports/figures/{output_prefix}.png/svg/pdf`",
    ]
    (NOTES / f"{output_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_source_data(prefix: str) -> None:
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)
    for suffix in ["metrics.csv", "series.csv", "curves.csv"]:
        src = TABLES / f"{prefix}_{suffix}"
        if src.exists():
            shutil.copy2(src, LATEX_SOURCE / src.name)


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    curve_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for idx, scenario in enumerate(make_scenarios(args.n_replicates), start=1):
        local_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max)))
        values = simulate_scenario(scenario, args, local_rng)
        dates, anomaly = seasonal_anomaly(values)
        tau_seconds = integral_autocorrelation_time(
            anomaly,
            dt_seconds=SECONDS_PER_DAY,
            max_lag=730,
            stop_at_zero=True,
        )
        tau_days = tau_seconds / SECONDS_PER_DAY
        if not np.isfinite(tau_days) or tau_days <= 0:
            continue
        curve = beta_curve_for_series(dates, anomaly, tau_seconds)
        if curve.empty:
            continue
        series_id = f"{scenario.family}_{scenario.target_tau_days:g}_{scenario.replicate:02d}"
        curve["series_id"] = series_id
        curve["scenario_group"] = scenario.family
        curve["target_tau_days"] = scenario.target_tau_days
        curve["replicate"] = scenario.replicate
        curve["tau_acf_days"] = tau_days
        curve_rows.append(curve[["series_id", "scenario_group", "target_tau_days", "replicate", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]])
        beta_summary = summarize_beta_window(curve)
        summary_rows.append(
            {
                "series_id": series_id,
                "scenario_group": scenario.family,
                "target_tau_days": scenario.target_tau_days,
                "replicate": scenario.replicate,
                "tau_acf_days": tau_days,
                **beta_summary,
            }
        )
        if idx % 80 == 0:
            print(f"processed {idx} scenarios")

    if not curve_rows:
        raise RuntimeError("No process-family curves were computed.")
    curves = pd.concat(curve_rows, ignore_index=True)
    series_summary = pd.DataFrame(summary_rows)
    metrics = pd.DataFrame(
        [summarize_group(group, family) for family, group in curves.groupby("scenario_group", sort=False)]
    )

    curves.to_csv(TABLES / f"{args.output_prefix}_curves.csv", index=False)
    series_summary.to_csv(TABLES / f"{args.output_prefix}_series.csv", index=False)
    metrics.to_csv(TABLES / f"{args.output_prefix}_metrics.csv", index=False)
    plot_results(metrics, series_summary, curves, args.output_prefix)
    write_note(metrics, args.output_prefix, args)
    copy_to_source_data(args.output_prefix)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
