"""Synthetic reservoir process-control test for the memory-normalized spectrum.

The goal is reviewer-facing: test whether ``De = tau_acf * f`` behaves like a
mechanistically plausible storage-memory coordinate in controlled systems, while
keeping the claim bounded. These simulations do not prove real catchment
physics; they test whether the observed coordinate is more than a purely
decorative rescaling.
"""

from __future__ import annotations

import argparse
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
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class Scenario:
    name: str
    alpha: float
    k: float
    rain_p: float
    rain_shape: float
    rain_scale: float
    seasonal_amp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=12)
    parser.add_argument("--series-length", type=int, default=12000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--output-prefix", default="synthetic_reservoir_process_control")
    return parser.parse_args()


def make_scenarios() -> list[Scenario]:
    rows: list[Scenario] = []
    for k in [0.035, 0.055, 0.085, 0.13, 0.20, 0.30]:
        rows.append(Scenario("linear_reservoir", 1.0, k, 0.35, 1.3, 2.2, 0.30))
    for k in [0.018, 0.026, 0.038, 0.055, 0.080, 0.115]:
        rows.append(Scenario("nonlinear_concave_storage", 0.65, k, 0.35, 1.3, 2.2, 0.30))
    for k in [0.010, 0.015, 0.022, 0.032, 0.047, 0.070]:
        rows.append(Scenario("nonlinear_convex_storage", 1.45, k, 0.35, 1.3, 2.2, 0.30))
    for k in [0.035, 0.070, 0.14]:
        rows.append(Scenario("seasonal_forcing_control", 1.0, k, 0.35, 1.3, 2.2, 0.75))
    return rows


def rainfall_series(n: int, scenario: Scenario, rng: np.random.Generator) -> np.ndarray:
    day = np.arange(n)
    seasonal = 1.0 + scenario.seasonal_amp * np.sin(2.0 * np.pi * day / 365.25)
    p = np.clip(scenario.rain_p * seasonal, 0.02, 0.92)
    wet = rng.random(n) < p
    depth = rng.gamma(shape=scenario.rain_shape, scale=scenario.rain_scale, size=n)
    return wet * depth


def simulate_reservoir(
    scenario: Scenario,
    rng: np.random.Generator,
    series_length: int,
    burn_in: int,
) -> np.ndarray:
    n_total = series_length + burn_in
    rain = rainfall_series(n_total, scenario, rng)
    storage = max(1.0, float(np.mean(rain) / max(scenario.k, 1e-6)))
    q = np.zeros(n_total, dtype=float)
    for i, p in enumerate(rain):
        raw = scenario.k * (storage ** scenario.alpha)
        outflow = min(max(raw, 0.0), 0.95 * storage)
        storage = max(storage + p - outflow, 1e-9)
        q[i] = outflow
    return q[burn_in:]


def seasonal_anomaly(values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    base = pd.Timestamp("1980-01-01")
    dates = pd.date_range(base, periods=len(values), freq="D")
    df = pd.DataFrame({"q": values}, index=dates)
    clim = df.groupby(df.index.dayofyear)["q"].transform("mean")
    anomaly = df["q"] - clim
    med = float(np.median(anomaly))
    mad = float(np.median(np.abs(anomaly - med)))
    if mad > 0:
        anomaly = (anomaly - med) / mad
    return dates, anomaly.to_numpy(dtype=float)


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
    grouped["axis_value"] = grouped["bin_id"].map({i: 10 ** labels[i] for i in range(len(labels))})
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
        "raw_bins_used": raw_bins,
        "de_bins_used": de_bins,
    }


def plot_results(series_summary: pd.DataFrame, metrics: pd.DataFrame, curves: pd.DataFrame, output_prefix: str) -> None:
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
        }
    )
    colors = {
        "linear_reservoir": "#2B7BB9",
        "nonlinear_concave_storage": "#2F855A",
        "nonlinear_convex_storage": "#C93434",
        "seasonal_forcing_control": "#D79A20",
    }
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.1, 1.1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    ax0.set_axis_off()
    ax0.text(0.00, 0.90, "Synthetic storage-reservoir process control", fontsize=11, fontweight="bold")
    ax0.text(
        0.00,
        0.58,
        r"Input rainfall $P(t)$ -> storage $S$ -> discharge $Q=kS^\alpha$; "
        r"$\tau_{acf}$ is estimated from simulated $Q(t)$.",
        fontsize=8,
    )
    ax0.text(
        0.00,
        0.30,
        "Purpose: mechanistic plausibility and artifact control, not proof that real catchments obey this model.",
        fontsize=7,
        color="#667085",
    )

    for group, data in series_summary.groupby("scenario_group"):
        ax1.scatter(
            data["tau_acf_days"],
            data["mean_beta_de_window"],
            s=16,
            alpha=0.70,
            color=colors.get(group, "#777777"),
            label=group.replace("_", " "),
        )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"simulated output memory $\tau_{acf}$ [d]")
    ax1.set_ylabel(r"mean beta in $0.5 \leq De \leq 2$")
    ax1.set_title("Storage controls span memory and spectral shape", loc="left")
    ax1.grid(True, color="#E7EAF0", lw=0.6)
    ax1.legend(fontsize=5.8, ncol=1)

    plot_m = metrics.copy()
    plot_m["variance_ratio"] = plot_m["de_weighted_beta_variance"] / plot_m["raw_weighted_beta_variance"]
    plot_m["log10_variance_ratio"] = np.log10(plot_m["variance_ratio"])
    order = [
        "linear_reservoir",
        "nonlinear_concave_storage",
        "nonlinear_convex_storage",
        "seasonal_forcing_control",
    ]
    plot_m = plot_m.set_index("scenario_group").loc[order].reset_index()
    x = np.arange(len(plot_m))
    bars = ax2.bar(
        x,
        plot_m["log10_variance_ratio"],
        color=[colors.get(g, "#777777") for g in plot_m["scenario_group"]],
        alpha=0.82,
    )
    ax2.axhline(0, color="#9AA0A6", lw=0.8)
    for bar, ratio in zip(bars, plot_m["variance_ratio"]):
        yy = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yy + (0.05 if yy >= 0 else -0.08),
            f"{ratio:.2g}x",
            ha="center",
            va="center",
            fontsize=6,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["linear", "concave\nnonlinear", "convex\nnonlinear", "seasonal\nforcing"])
    ax2.set_ylabel(r"$\log_{10}(\mathrm{Var}_{De}/\mathrm{Var}_{raw})$")
    ax2.set_title("De is process-specific, not automatic", loc="left")
    ax2.grid(True, axis="y", color="#E7EAF0", lw=0.6)

    sample_parts = []
    for _, group in curves.groupby("scenario_group", sort=False):
        keep_ids = sorted(group["series_id"].unique())[: min(10, group["series_id"].nunique())]
        sample_parts.append(group[group["series_id"].isin(keep_ids)].copy())
    sample = pd.concat(sample_parts, ignore_index=True)
    for group, data in sample.groupby("scenario_group"):
        binned = station_bin_medians(data, "de", n_bins=26)
        med = binned.groupby("bin_id", as_index=False).agg(axis_value=("axis_value", "first"), beta=("beta_median", "median"))
        ax3.semilogx(
            med["axis_value"],
            med["beta"],
            lw=1.4,
            color=colors.get(group, "#777777"),
            label=group.replace("_", " "),
        )
    ax3.set_xlabel("De")
    ax3.set_ylabel("median local beta")
    ax3.set_title("Synthetic master curves remain model-family dependent", loc="left")
    ax3.grid(True, color="#E7EAF0", lw=0.6)
    ax3.legend(fontsize=6, ncol=2)

    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", bbox_inches="tight", dpi=600 if suffix == ".png" else None)
    plt.close(fig)


def write_note(metrics: pd.DataFrame, output_prefix: str) -> None:
    lines = [
        "# Synthetic Reservoir Process-Control Note",
        "",
        "This analysis simulates storage reservoirs with known process structure",
        "and runs the same tau/beta/De logic used for the manuscript controls.",
        "It is a mechanistic plausibility and artifact-control test, not a proof",
        "that real catchments obey the simulated equations.",
        "",
        "## Aggregate results",
        "",
        metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Manuscript use",
        "",
        "- Safe claim: storage-process simulations show that the memory-normalized",
        "  coordinate behaves coherently when discharge is generated by controlled",
        "  linear and nonlinear storage filters.",
        "- Unsafe claim: the simulations prove that CAMELS catchments follow a",
        "  specific reservoir equation or storage-discharge exponent.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{output_prefix}_series.csv`",
        f"- `reports/tables/{output_prefix}_metrics.csv`",
        f"- `reports/tables/{output_prefix}_curves.csv`",
        f"- `reports/figures/{output_prefix}.png/svg/pdf`",
    ]
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f"{output_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    curves: list[pd.DataFrame] = []

    for scenario in make_scenarios():
        for rep in range(args.n_replicates):
            local_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int32).max))
            q = simulate_reservoir(scenario, local_rng, args.series_length, args.burn_in)
            dates, anomaly = seasonal_anomaly(q)
            tau_seconds = integral_autocorrelation_time(
                anomaly,
                dt_seconds=SECONDS_PER_DAY,
                max_lag=730,
                stop_at_zero=True,
            )
            curve = beta_curve_for_series(dates, anomaly, tau_seconds)
            summary = summarize_beta_window(curve)
            series_id = f"{scenario.name}_a{scenario.alpha:g}_k{scenario.k:g}_r{rep:02d}"
            curve["series_id"] = series_id
            curve["scenario_group"] = scenario.name
            curve["alpha"] = scenario.alpha
            curve["k"] = scenario.k
            curves.append(curve)
            rows.append(
                {
                    "series_id": series_id,
                    "scenario_group": scenario.name,
                    "alpha": scenario.alpha,
                    "k": scenario.k,
                    "rain_p": scenario.rain_p,
                    "seasonal_amp": scenario.seasonal_amp,
                    "tau_acf_days": tau_seconds / SECONDS_PER_DAY,
                    **summary,
                }
            )

    series_summary = pd.DataFrame(rows)
    curve_df = pd.concat(curves, ignore_index=True)
    metrics = pd.DataFrame(
        [
            summarize_group(group, name)
            for name, group in curve_df.groupby("scenario_group", sort=False)
        ]
    )
    series_summary.to_csv(TABLES / f"{args.output_prefix}_series.csv", index=False)
    curve_df.to_csv(TABLES / f"{args.output_prefix}_curves.csv", index=False)
    metrics.to_csv(TABLES / f"{args.output_prefix}_metrics.csv", index=False)
    plot_results(series_summary, metrics, curve_df, args.output_prefix)
    write_note(metrics, args.output_prefix)
    print(metrics.to_string(index=False))
    print(f"Saved synthetic reservoir outputs with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()
