"""Fourier-pair circularity baseline for the De spectral-coordinate claim.

The strongest reviewer objection is that ``tau_acf`` and the power spectrum are
computed from the same time series and are related by the Wiener-Khinchin
theorem. This script estimates how much dispersion reduction can be produced by
that mathematical coupling alone in controlled AR(1)/ARMA processes.

The output is a cautionary baseline: if simple stochastic processes show large
De reductions, the manuscript must describe the CAMELS result as partial
memory-normalized alignment consistent with storage filtering, not as an
independent proof of hydrological mechanism.
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
    tau_days: float
    theta: float = 0.0
    seasonal_amp: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=18)
    parser.add_argument("--series-length", type=int, default=12000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--output-prefix", default="fourier_pair_circularity_baseline")
    return parser.parse_args()


def make_scenarios() -> list[Scenario]:
    tau_grid = [2, 4, 8, 16, 32, 64, 128]
    scenarios: list[Scenario] = []
    for tau in tau_grid:
        scenarios.append(Scenario("ar1_fourier_pair", tau_days=float(tau)))
        scenarios.append(Scenario("arma_positive_ma", tau_days=float(tau), theta=0.55))
        scenarios.append(Scenario("arma_negative_ma", tau_days=float(tau), theta=-0.45))
    for tau in [8, 16, 32, 64]:
        scenarios.append(Scenario("seasonal_ar1_control", tau_days=float(tau), seasonal_amp=0.75))
    return scenarios


def simulate_process(
    scenario: Scenario,
    rng: np.random.Generator,
    series_length: int,
    burn_in: int,
) -> np.ndarray:
    n = series_length + burn_in
    phi = float(np.exp(-1.0 / scenario.tau_days))
    eps = rng.normal(0.0, 1.0, n + 1)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        innovation = eps[i] + scenario.theta * eps[i - 1]
        x[i] = phi * x[i - 1] + innovation
    if scenario.seasonal_amp:
        day = np.arange(n)
        seasonal = scenario.seasonal_amp * np.sin(2.0 * np.pi * day / 365.25)
        x = x + seasonal
    x = x[burn_in:]
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    return x


def beta_curve_for_series(values: np.ndarray, tau_seconds: float) -> pd.DataFrame:
    dates = pd.date_range("1980-01-01", periods=len(values), freq="D")
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


def dispersion_metric(curves: pd.DataFrame, value_col: str, min_series: int = 10) -> tuple[float, int]:
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


def plot_results(metrics: pd.DataFrame, curves: pd.DataFrame, output_prefix: str) -> None:
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
        "ar1_fourier_pair": "#2B7BB9",
        "arma_positive_ma": "#6C5CE7",
        "arma_negative_ma": "#9B4762",
        "seasonal_ar1_control": "#D79A20",
    }
    labels = {
        "ar1_fourier_pair": "AR(1)",
        "arma_positive_ma": "ARMA +MA",
        "arma_negative_ma": "ARMA -MA",
        "seasonal_ar1_control": "seasonal AR(1)",
    }

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.1], height_ratios=[0.8, 1.0], wspace=0.32, hspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    order = ["ar1_fourier_pair", "arma_positive_ma", "arma_negative_ma", "seasonal_ar1_control"]
    work = metrics.set_index("scenario_group").loc[order].reset_index()
    x = np.arange(len(work))
    vals = 100 * work["variance_reduction_vs_raw"].to_numpy(dtype=float)
    bar_colors = [colors[g] for g in work["scenario_group"]]
    ax0.bar(x, vals, color=bar_colors, alpha=0.78, width=0.64)
    ax0.axhline(0, color="#AAB0B8", lw=0.8)
    ax0.axhline(23.7, color="#C93434", ls="--", lw=1.0, label="CAMELS-US")
    ax0.axhline(14.7, color="#C93434", ls=":", lw=1.0, label="CAMELS-GB")
    for xx, vv in zip(x, vals):
        ax0.text(xx, vv + (1.2 if vv >= 0 else -2.0), f"{vv:.1f}%", ha="center", va="center", fontsize=6.2)
    ax0.set_xticks(x)
    ax0.set_xticklabels([labels[g] for g in work["scenario_group"]], rotation=20, ha="right")
    ax0.set_ylabel("variance reduction [%]")
    ax0.set_title("Fourier-pair baselines can also align spectra", loc="left", fontsize=8)
    ax0.legend(loc="upper right", fontsize=5.8)
    ax0.grid(True, axis="y", color="#E6EAF0", lw=0.55)

    ax1.set_axis_off()
    ax1.text(0.00, 0.92, "Reviewer-facing interpretation", fontsize=9, fontweight="bold")
    bullets = [
        "ACF-derived tau and PSD are mathematically coupled.",
        "Large AR/ARMA reductions would not falsify CAMELS; they set a baseline.",
        "The manuscript must claim partial alignment beyond arbitrary tau shuffling,",
        "not independent proof of storage mechanism from tau_acf alone.",
    ]
    for i, bullet in enumerate(bullets):
        ax1.text(0.02, 0.75 - i * 0.16, f"- {bullet}", fontsize=7, color="#1F2328")

    sample_parts = []
    for name, group in curves.groupby("scenario_group", sort=False):
        keep = sorted(group["series_id"].unique())[: min(12, group["series_id"].nunique())]
        sample_parts.append(group[group["series_id"].isin(keep)].copy())
    sample = pd.concat(sample_parts, ignore_index=True)
    for name, group in sample.groupby("scenario_group"):
        binned = station_bin_medians(group, "de", n_bins=26)
        med = binned.groupby("bin_id", as_index=False).agg(axis_value=("axis_value", "first"), beta=("beta_median", "median"))
        ax2.semilogx(med["axis_value"], med["beta"], lw=1.4, color=colors.get(name, "#777777"), label=labels.get(name, name))
    ax2.axvspan(0.5, 2.0, color="#C93434", alpha=0.06, lw=0)
    ax2.set_xlabel("De from each process' own tau_acf")
    ax2.set_ylabel("median local beta")
    ax2.set_title("Baseline master curves show mathematical and process-family components", loc="left", fontsize=8)
    ax2.grid(True, color="#E6EAF0", lw=0.55)
    ax2.legend(ncol=4, loc="upper left", fontsize=5.8)

    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(metrics: pd.DataFrame, output_prefix: str) -> None:
    lines = [
        "# Fourier-Pair Circularity Baseline",
        "",
        "This analysis directly addresses the reviewer objection that tau_acf and",
        "the power spectrum are computed from the same time series and therefore",
        "are linked through the Wiener-Khinchin theorem.",
        "",
        "## Aggregate results",
        "",
        metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Manuscript consequence",
        "",
        "- Safe claim: De normalization reduces dispersion in CAMELS beyond random",
        "  tau shuffling and is compatible with storage-filtered dynamics.",
        "- Required caveat: AR/ARMA baselines quantify the mathematical alignment",
        "  that can arise from scaling a spectrum by an autocorrelation timescale.",
        "- Unsafe claim: the CAMELS reduction alone proves a physical storage",
        "  mechanism independently of signal-processing coupling.",
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
            values = simulate_process(scenario, local_rng, args.series_length, args.burn_in)
            tau_seconds = integral_autocorrelation_time(
                values,
                dt_seconds=SECONDS_PER_DAY,
                max_lag=730,
                stop_at_zero=True,
            )
            curve = beta_curve_for_series(values, tau_seconds)
            summary = summarize_beta_window(curve)
            series_id = f"{scenario.name}_tau{scenario.tau_days:g}_rep{rep:02d}"
            curve["series_id"] = series_id
            curve["scenario_group"] = scenario.name
            curve["true_tau_days"] = scenario.tau_days
            curve["theta"] = scenario.theta
            curve["seasonal_amp"] = scenario.seasonal_amp
            curves.append(curve)
            rows.append(
                {
                    "series_id": series_id,
                    "scenario_group": scenario.name,
                    "true_tau_days": scenario.tau_days,
                    "theta": scenario.theta,
                    "seasonal_amp": scenario.seasonal_amp,
                    "est_tau_days": tau_seconds / SECONDS_PER_DAY,
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
    plot_results(metrics, curve_df, args.output_prefix)
    write_note(metrics, args.output_prefix)
    print(metrics.to_string(index=False))
    print(f"Saved Fourier-pair baseline outputs with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()
