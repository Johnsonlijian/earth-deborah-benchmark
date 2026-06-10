"""R20 gauge-matched stochastic surrogate ladder for the memory-coordinate claim.

This script upgrades the global median AR(1) residual check into a stricter
archive-level null ladder. For each included CAMELS gauge it generates
gauge-matched surrogate discharge-anomaly series, recomputes tau_acf, PSD,
local beta and De, and compares archive-level raw-frequency versus De
dispersion against the observed archive.

The output is an artifact-bound robustness diagnostic. It should not be
interpreted as independent storage-mechanism evidence.
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

from compute_de_collapse_metrics import (
    dispersion_from_binned,
    load_one_camels_flow,
    prepare_daily_anomaly as prepare_us_anomaly,
    station_bin_medians,
)
from edb.signal.de import beta_vs_de
from edb.signal.nulls import ar1_surrogate, phase_randomized_surrogate
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.timescales import integral_autocorrelation_time
from run_camels_gb_replication import (
    load_camels_gb_discharge,
    prepare_daily_anomaly as prepare_gb_anomaly,
)

ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_DAY = 86_400.0
US_FLOW_DIR = ROOT / "data" / "external" / "camels" / "usgs_streamflow"
GB_DAILY_DIR = ROOT / "data" / "external" / "camels_gb_v2" / "hydromet_daily"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"


@dataclass(frozen=True)
class GaugeSeries:
    archive: str
    gauge_id: str
    dates: pd.DatetimeIndex
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", choices=["both", "us", "gb"], default="both")
    parser.add_argument("--n-surrogates", type=int, default=3)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["matched_ar1", "block_bootstrap", "phase_randomized"],
        choices=["matched_ar1", "block_bootstrap", "phase_randomized", "iaaft"],
    )
    parser.add_argument("--block-length", type=int, default=90)
    parser.add_argument("--iaaft-iterations", type=int, default=20)
    parser.add_argument("--limit-per-archive", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--output-prefix", default="r20_gauge_matched_surrogate_ladder")
    return parser.parse_args()


def beta_curve_for_values(dates: pd.DatetimeIndex, values: np.ndarray) -> tuple[float, pd.DataFrame]:
    tau_seconds = integral_autocorrelation_time(
        values,
        dt_seconds=SECONDS_PER_DAY,
        max_lag=730,
        stop_at_zero=True,
    )
    tau_days = tau_seconds / SECONDS_PER_DAY
    if not np.isfinite(tau_days) or tau_days <= 0:
        raise ValueError("invalid tau_acf")
    psd = welch_psd(dates, values, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    if curve.empty:
        raise ValueError("empty beta curve")
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["tau_acf_days"] = tau_days
    return tau_days, curve[["frequency", "frequency_cpd", "de", "beta", "tau_acf_days"]]


def archive_dispersion(curves: pd.DataFrame, min_units: int, n_bins: int = 24) -> dict[str, float | int]:
    raw = station_bin_medians(curves, "frequency_cpd", n_bins=n_bins)
    de = station_bin_medians(curves, "de", n_bins=n_bins)
    raw_res, _ = dispersion_from_binned(raw, min_stations=min_units)
    de_res, _ = dispersion_from_binned(de, min_stations=min_units)
    return {
        "raw_axis_weighted_variance": raw_res.weighted_variance,
        "de_axis_weighted_variance": de_res.weighted_variance,
        "reduction_vs_raw": 1.0 - de_res.weighted_variance / raw_res.weighted_variance,
        "raw_bins_used": raw_res.n_bins,
        "de_bins_used": de_res.n_bins,
        "n_units": int(curves["gauge_id"].nunique()),
        "n_curve_points": int(len(curves)),
    }


def load_us_series(limit: int = 0) -> list[GaugeSeries]:
    wanted = set(pd.read_csv(TABLES / "camels_673_beta_curve_points.csv", usecols=["gauge_id"])["gauge_id"].astype(str).str.zfill(8))
    files = [fp for fp in sorted(US_FLOW_DIR.glob("*.txt")) if fp.stem in wanted]
    if limit:
        files = files[:limit]
    out: list[GaugeSeries] = []
    for path in files:
        gid, series = load_one_camels_flow(path)
        if series.empty:
            continue
        try:
            dates, values = prepare_us_anomaly(series)
        except Exception:
            continue
        if len(values) >= 365 * 20:
            out.append(GaugeSeries("CAMELS-US", gid, dates, values))
    return out


def load_gb_series(limit: int = 0) -> list[GaugeSeries]:
    wanted = set(pd.read_csv(TABLES / "camels_gb_v2_replication_beta_curve_points.csv", usecols=["gauge_id"])["gauge_id"].astype(str))
    files = sorted(GB_DAILY_DIR.glob("*.csv"))
    out: list[GaugeSeries] = []
    for path in files:
        gid, series = load_camels_gb_discharge(path, "discharge_spec")
        if gid not in wanted or series.empty:
            continue
        try:
            dates, values = prepare_gb_anomaly(series)
        except Exception:
            continue
        if len(values) >= 365 * 20:
            out.append(GaugeSeries("CAMELS-GB", gid, dates, values))
        if limit and len(out) >= limit:
            break
    return out


def match_mean_std(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    v_std = float(np.nanstd(values))
    t_std = float(np.nanstd(target))
    if not np.isfinite(v_std) or v_std <= 0 or not np.isfinite(t_std) or t_std <= 0:
        return np.zeros_like(target)
    return (values - np.nanmean(values)) / v_std * t_std + np.nanmean(target)


def block_bootstrap_surrogate(values: np.ndarray, seed: int, block_length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < block_length * 2:
        return rng.choice(values, size=n, replace=True)
    starts = rng.integers(0, max(1, n - block_length + 1), size=int(np.ceil(n / block_length)))
    sampled = np.concatenate([values[start : start + block_length] for start in starts])[:n]
    return match_mean_std(sampled, values)


def iaaft_surrogate(values: np.ndarray, seed: int, iterations: int) -> np.ndarray:
    """Simple IAAFT surrogate preserving the rank distribution and spectrum approximately."""
    rng = np.random.default_rng(seed)
    target = np.asarray(values, dtype=float)
    sorted_target = np.sort(target)
    target_amp = np.abs(np.fft.rfft(target - np.mean(target)))
    current = rng.permutation(target)
    for _ in range(iterations):
        spectrum = np.fft.rfft(current - np.mean(current))
        phases = np.exp(1j * np.angle(spectrum))
        current = np.fft.irfft(target_amp * phases, n=target.size)
        ranks = np.argsort(np.argsort(current))
        current = sorted_target[ranks]
    return match_mean_std(current, target)


def make_surrogate(family: str, item: GaugeSeries, seed: int, block_length: int, iaaft_iterations: int) -> np.ndarray:
    if family == "matched_ar1":
        return ar1_surrogate(item.values, seed=seed)
    if family == "block_bootstrap":
        return block_bootstrap_surrogate(item.values, seed=seed, block_length=block_length)
    if family == "phase_randomized":
        return phase_randomized_surrogate(item.values, seed=seed)
    if family == "iaaft":
        return iaaft_surrogate(item.values, seed=seed, iterations=iaaft_iterations)
    raise ValueError(f"unknown surrogate family: {family}")


def observed_curves_for_archive(archive: str) -> pd.DataFrame:
    if archive == "CAMELS-US":
        path = TABLES / "camels_673_beta_curve_points.csv"
    elif archive == "CAMELS-GB":
        path = TABLES / "camels_gb_v2_replication_beta_curve_points.csv"
    else:
        raise ValueError(archive)
    curves = pd.read_csv(path)
    curves["gauge_id"] = curves["gauge_id"].astype(str).str.zfill(8) if archive == "CAMELS-US" else curves["gauge_id"].astype(str)
    return curves[["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]].copy()


def run_family_draw(
    series: list[GaugeSeries],
    family: str,
    draw: int,
    seed: int,
    block_length: int,
    iaaft_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    diag_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    for idx, item in enumerate(series, start=1):
        local_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        try:
            surrogate = make_surrogate(family, item, local_seed, block_length, iaaft_iterations)
            tau_days, curve = beta_curve_for_values(item.dates, surrogate)
        except Exception as exc:
            diag_rows.append(
                {
                    "archive": item.archive,
                    "gauge_id": item.gauge_id,
                    "family": family,
                    "draw": draw,
                    "status": "failed",
                    "reason": type(exc).__name__,
                    "tau_acf_days": np.nan,
                    "n_curve_points": 0,
                }
            )
            continue
        curve["archive"] = item.archive
        curve["gauge_id"] = item.gauge_id
        curve["family"] = family
        curve["draw"] = draw
        rows.append(curve)
        diag_rows.append(
            {
                "archive": item.archive,
                "gauge_id": item.gauge_id,
                "family": family,
                "draw": draw,
                "status": "included",
                "reason": "included",
                "tau_acf_days": tau_days,
                "n_curve_points": len(curve),
            }
        )
        if idx % 200 == 0:
            print(f"  {item.archive} {family} draw {draw}: {idx}/{len(series)} gauges")
    if not rows:
        raise RuntimeError(f"no curves for {series[0].archive} {family} draw {draw}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    observed = metrics[metrics["family"] == "observed"].set_index("archive")["reduction_vs_raw"].to_dict()
    rows = []
    for (archive, family), group in metrics[metrics["family"] != "observed"].groupby(["archive", "family"]):
        values = group["reduction_vs_raw"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        obs = float(observed.get(archive, np.nan))
        if not values.size or not np.isfinite(obs):
            continue
        exceed = int(np.sum(values >= obs))
        rows.append(
            {
                "archive": archive,
                "family": family,
                "observed_reduction": obs,
                "null_median_reduction": float(np.median(values)),
                "null_p05_reduction": float(np.percentile(values, 5)),
                "null_p95_reduction": float(np.percentile(values, 95)),
                "observed_minus_null_median": obs - float(np.median(values)),
                "n_null_draws": int(values.size),
                "null_draws_ge_observed": exceed,
                "empirical_p_ge_observed": (exceed + 1.0) / (values.size + 1.0),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_prefix: str) -> None:
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
    order = ["matched_ar1", "block_bootstrap", "phase_randomized", "iaaft"]
    colors = {
        "observed": "#2E6FAD",
        "matched_ar1": "#C4513F",
        "block_bootstrap": "#D99A24",
        "phase_randomized": "#7E5AA8",
        "iaaft": "#4E9F74",
    }
    archives = [a for a in ["CAMELS-US", "CAMELS-GB"] if a in set(summary["archive"])]
    fig, axes = plt.subplots(1, len(archives), figsize=(3.7 * len(archives), 3.0), sharey=True, constrained_layout=True)
    if len(archives) == 1:
        axes = [axes]
    for ax, archive in zip(axes, archives):
        data = summary[summary["archive"] == archive].copy()
        data["family"] = pd.Categorical(data["family"], categories=order, ordered=True)
        data = data.sort_values("family")
        obs = float(data["observed_reduction"].iloc[0])
        ax.axhline(100 * obs, color=colors["observed"], lw=1.2, ls="--", label="observed")
        x = np.arange(len(data))
        med = 100 * data["null_median_reduction"].to_numpy(dtype=float)
        lo = 100 * data["null_p05_reduction"].to_numpy(dtype=float)
        hi = 100 * data["null_p95_reduction"].to_numpy(dtype=float)
        bar_colors = [colors.get(f, "#888888") for f in data["family"].astype(str)]
        ax.bar(x, med, color=bar_colors, alpha=0.76, width=0.62)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor="#28313B", elinewidth=0.8, capsize=2)
        for xx, row in zip(x, data.itertuples(index=False)):
            ax.text(
                xx,
                med[xx] + 1.2,
                f"excess {100 * row.observed_minus_null_median:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=5.8,
                rotation=90,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(f).replace("_", "\n") for f in data["family"]], fontsize=6)
        ax.set_title(archive, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("dispersion reduction vs raw axis [%]")
        ax.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)
        ax.legend(loc="upper right", fontsize=6)
    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(metrics: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# R20 Gauge-Matched Surrogate Ladder",
        "",
        f"Date: 2026-06-03",
        "",
        "## Method",
        "",
        "Each gauge is replaced by stochastic surrogates generated from its own",
        "daily discharge-anomaly series. For each surrogate archive draw, tau_acf,",
        "PSD, local beta and De are recomputed, then the same raw-frequency versus",
        "De dispersion metric is applied.",
        "",
        f"- Surrogate draws per family: {args.n_surrogates}",
        f"- Families: {', '.join(args.families)}",
        f"- Block-bootstrap length: {args.block_length} days",
        "",
        "## Aggregate Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g") if not summary.empty else "No summary rows.",
        "",
        "## Interpretation Boundary",
        "",
        "This is a stochastic artifact ladder, not independent storage evidence.",
        "Phase-randomized and IAAFT surrogates preserve the amplitude spectrum by",
        "construction, so a small observed-minus-null excess there should be read",
        "as confirmation that the current benchmark is spectral-memory evidence",
        "rather than phase/nonlinear temporal-order evidence.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{args.output_prefix}_draw_metrics.csv`",
        f"- `reports/tables/{args.output_prefix}_summary.csv`",
        f"- `reports/tables/{args.output_prefix}_gauge_diagnostics.csv`",
        f"- `reports/figures/{args.output_prefix}.png/svg/pdf`",
    ]
    (NOTES / f"{args.output_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    archives: dict[str, list[GaugeSeries]] = {}
    if args.archives in {"both", "us"}:
        archives["CAMELS-US"] = load_us_series(args.limit_per_archive)
    if args.archives in {"both", "gb"}:
        archives["CAMELS-GB"] = load_gb_series(args.limit_per_archive)
    print({archive: len(series) for archive, series in archives.items()})

    metric_rows: list[dict[str, float | int | str]] = []
    diag_frames: list[pd.DataFrame] = []
    min_units = {"CAMELS-US": 50, "CAMELS-GB": 45}

    for archive, series in archives.items():
        observed = observed_curves_for_archive(archive)
        observed = observed[observed["gauge_id"].isin({item.gauge_id for item in series})].copy()
        obs_metrics = archive_dispersion(observed, min_units=min_units[archive])
        metric_rows.append({"archive": archive, "family": "observed", "draw": -1, **obs_metrics})

        for family in args.families:
            for draw in range(args.n_surrogates):
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                print(f"Running {archive} {family} draw {draw + 1}/{args.n_surrogates}")
                curves, diagnostics = run_family_draw(
                    series,
                    family,
                    draw,
                    seed,
                    args.block_length,
                    args.iaaft_iterations,
                )
                diag_frames.append(diagnostics)
                metrics = archive_dispersion(curves, min_units=min_units[archive])
                metric_rows.append({"archive": archive, "family": family, "draw": draw, **metrics})

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    summary = summarize_metrics(metrics)
    metrics.to_csv(TABLES / f"{args.output_prefix}_draw_metrics.csv", index=False)
    diagnostics.to_csv(TABLES / f"{args.output_prefix}_gauge_diagnostics.csv", index=False)
    summary.to_csv(TABLES / f"{args.output_prefix}_summary.csv", index=False)
    if not summary.empty:
        plot_summary(summary, args.output_prefix)
    write_note(metrics, summary, args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
