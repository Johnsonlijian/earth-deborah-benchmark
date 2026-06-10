"""R36 gauge-matched multiscale memory-null controls.

This round addresses the hardest post-R35 objection: a same-series AR(1)
surrogate is only a single-timescale null, whereas streamflow can contain fast
and slow response components or long-memory persistence. The script therefore
adds two stronger stochastic families to the existing tau_acf--PSD--beta--De
pipeline:

* a two-timescale AR mixture whose approximate integrated autocorrelation time
  is matched to each gauge's observed anomaly series;
* an ARFIMA(0,d,0)-like fractional-integration surrogate that tests whether a
  long-memory spectrum can explain the observed De reduction without invoking
  storage mechanism claims.

The output is a null-calibration control, not a fitted hydrological model.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.signal.nulls import ar1_surrogate
from edb.signal.timescales import integral_autocorrelation_time
from run_r20_gauge_matched_surrogate_ladder import (
    GaugeSeries,
    archive_dispersion,
    beta_curve_for_values,
    load_gb_series,
    load_us_series,
    match_mean_std,
    observed_curves_for_archive,
    summarize_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX_SOURCE = ROOT / "source_data"
SECONDS_PER_DAY = 86_400.0
DEFAULT_PREFIX = "r36_multiscale_memory_nulls"

FAMILIES = ["matched_ar1", "two_timescale_ar_mix", "arfima_fractional"]
WORK_ARCHIVES: dict[str, list[GaugeSeries]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", choices=["both", "us", "gb"], default="both")
    parser.add_argument("--n-draws", type=int, default=24)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--limit-per-archive", type=int, default=0)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--arfima-d", type=float, default=0.28)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def _init_worker(archives: dict[str, list[GaugeSeries]]) -> None:
    global WORK_ARCHIVES
    WORK_ARCHIVES = archives


def estimate_tau_days(values: np.ndarray) -> float:
    tau_seconds = integral_autocorrelation_time(
        values,
        dt_seconds=SECONDS_PER_DAY,
        max_lag=730,
        stop_at_zero=True,
    )
    tau_days = tau_seconds / SECONDS_PER_DAY
    if not np.isfinite(tau_days) or tau_days <= 0:
        return 1.0
    return float(tau_days)


def simulate_unit_ar1(n: int, tau_days: float, rng: np.random.Generator) -> np.ndarray:
    tau = max(float(tau_days), 0.2)
    phi = float(np.exp(-1.0 / tau))
    phi = float(np.clip(phi, -0.995, 0.995))
    sigma = float(np.sqrt(max(1.0 - phi * phi, 1e-8)))
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(0.0, 1.0)
    for idx in range(1, n):
        out[idx] = phi * out[idx - 1] + rng.normal(0.0, sigma)
    out = out - np.mean(out)
    std = float(np.std(out))
    return out / std if std > 0 else out


def two_timescale_ar_mix(values: np.ndarray, seed: int) -> np.ndarray:
    """Generate a fast/slow AR mixture matched to the target's approximate tau."""

    target = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    tau = estimate_tau_days(target)
    fast_tau = max(1.0, tau / 6.0)
    slow_tau = min(730.0, max(tau * 6.0, fast_tau + 1.0))
    if slow_tau <= fast_tau:
        slow_tau = fast_tau + 1.0
    fast_weight = (tau - slow_tau) / (fast_tau - slow_tau)
    fast_weight = float(np.clip(fast_weight, 0.25, 0.90))
    fast = simulate_unit_ar1(target.size, fast_tau, rng)
    slow = simulate_unit_ar1(target.size, slow_tau, rng)
    out = np.sqrt(fast_weight) * fast + np.sqrt(1.0 - fast_weight) * slow
    return match_mean_std(out, target)


def fractional_integration_weights(d: float, n_weights: int) -> np.ndarray:
    """Weights for (1-L)^(-d), truncated for finite daily records."""

    d = float(np.clip(d, 0.02, 0.45))
    weights = np.empty(n_weights, dtype=float)
    weights[0] = 1.0
    for k in range(1, n_weights):
        weights[k] = weights[k - 1] * ((k - 1 + d) / k)
        if k > 128 and abs(weights[k]) < 1e-5:
            return weights[: k + 1]
    return weights


def arfima_fractional(values: np.ndarray, seed: int, d: float) -> np.ndarray:
    """ARFIMA(0,d,0)-like fractional surrogate with finite-record truncation."""

    target = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = target.size
    n_weights = int(min(max(256, n // 3), 4096))
    weights = fractional_integration_weights(d, n_weights)
    eps = rng.normal(0.0, 1.0, n + len(weights))
    out = np.convolve(eps, weights, mode="valid")[:n]
    out = out - np.mean(out)
    std = float(np.std(out))
    if std > 0:
        out = out / std
    return match_mean_std(out, target)


def make_surrogate(family: str, item: GaugeSeries, seed: int, arfima_d: float) -> np.ndarray:
    if family == "matched_ar1":
        return ar1_surrogate(item.values, seed=seed)
    if family == "two_timescale_ar_mix":
        return two_timescale_ar_mix(item.values, seed=seed)
    if family == "arfima_fractional":
        return arfima_fractional(item.values, seed=seed, d=arfima_d)
    raise ValueError(f"unknown family: {family}")


def run_family_draw(task: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    archive = str(task["archive"])
    family = str(task["family"])
    draw = int(task["draw"])
    min_units = int(task["min_units"])
    rng = np.random.default_rng(int(task["seed"]))
    rows: list[pd.DataFrame] = []
    diag_rows: list[dict[str, float | int | str]] = []
    for item in WORK_ARCHIVES[archive]:
        local_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        target_tau = estimate_tau_days(item.values)
        try:
            surrogate = make_surrogate(family, item, local_seed, float(task["arfima_d"]))
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
                    "target_tau_acf_days": target_tau,
                    "surrogate_tau_acf_days": np.nan,
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
                "target_tau_acf_days": target_tau,
                "surrogate_tau_acf_days": tau_days,
                "n_curve_points": len(curve),
            }
        )
    if not rows:
        raise RuntimeError(f"no curves for {archive} {family} draw {draw}")
    metrics = archive_dispersion(pd.concat(rows, ignore_index=True), min_units=min_units)
    return {"archive": archive, "family": family, "draw": draw, **metrics}, pd.DataFrame(diag_rows)


def load_archives(args: argparse.Namespace) -> dict[str, list[GaugeSeries]]:
    archives: dict[str, list[GaugeSeries]] = {}
    if args.archives in {"both", "us"}:
        archives["CAMELS-US"] = load_us_series(args.limit_per_archive)
    if args.archives in {"both", "gb"}:
        archives["CAMELS-GB"] = load_gb_series(args.limit_per_archive)
    for archive, series in archives.items():
        if not series:
            raise RuntimeError(f"No series loaded for {archive}")
    return archives


def summarize_tau_match(diagnostics: pd.DataFrame) -> pd.DataFrame:
    included = diagnostics[diagnostics["status"] == "included"].copy()
    if included.empty:
        return pd.DataFrame()
    included["log_tau_error"] = np.log10(included["surrogate_tau_acf_days"]) - np.log10(included["target_tau_acf_days"])
    rows = []
    for (archive, family), group in included.groupby(["archive", "family"]):
        rows.append(
            {
                "archive": archive,
                "family": family,
                "n_included_series_draws": int(len(group)),
                "median_target_tau_days": float(np.nanmedian(group["target_tau_acf_days"])),
                "median_surrogate_tau_days": float(np.nanmedian(group["surrogate_tau_acf_days"])),
                "median_log10_tau_error": float(np.nanmedian(group["log_tau_error"])),
                "iqr_log10_tau_error": float(
                    np.nanpercentile(group["log_tau_error"], 75) - np.nanpercentile(group["log_tau_error"], 25)
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, tau_summary: pd.DataFrame, output_prefix: str) -> None:
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
    family_order = ["matched_ar1", "two_timescale_ar_mix", "arfima_fractional"]
    labels = {
        "matched_ar1": "single-timescale\nAR(1)",
        "two_timescale_ar_mix": "two-timescale\nAR mixture",
        "arfima_fractional": "ARFIMA-like\nlong memory",
    }
    colors = {
        "matched_ar1": "#C4513F",
        "two_timescale_ar_mix": "#2F855A",
        "arfima_fractional": "#6C5CE7",
    }
    archives = [a for a in ["CAMELS-US", "CAMELS-GB"] if a in set(summary["archive"])]
    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(2, len(archives), height_ratios=[1.1, 0.9])
    if len(archives) == 1:
        axes_top = [fig.add_subplot(gs[0, 0])]
        axes_bottom = [fig.add_subplot(gs[1, 0])]
    else:
        axes_top = [fig.add_subplot(gs[0, i]) for i in range(len(archives))]
        axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(len(archives))]

    for ax, archive in zip(axes_top, archives):
        data = summary[summary["archive"] == archive].copy()
        data["family"] = pd.Categorical(data["family"], categories=family_order, ordered=True)
        data = data.sort_values("family")
        obs = float(data["observed_reduction"].iloc[0])
        x = np.arange(len(data))
        med = 100.0 * data["null_median_reduction"].to_numpy(dtype=float)
        lo = 100.0 * data["null_p05_reduction"].to_numpy(dtype=float)
        hi = 100.0 * data["null_p95_reduction"].to_numpy(dtype=float)
        ax.axhline(100.0 * obs, color="#2E6FAD", lw=1.15, ls="--", label="observed")
        ax.bar(x, med, color=[colors[str(f)] for f in data["family"]], width=0.62, alpha=0.78)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor="#1F2328", lw=0.8, capsize=2)
        for xx, row in zip(x, data.itertuples(index=False)):
            ax.text(
                xx,
                med[xx] + 1.2,
                f"{100.0 * row.observed_minus_null_median:+.1f} pp",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=5.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[str(f)] for f in data["family"]], fontsize=6)
        ax.set_title(archive, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("De-axis dispersion reduction [%]")
        ax.grid(True, axis="y", color="#E4E9EF", lw=0.55)
        ax.legend(loc="upper right", fontsize=6)

    for ax, archive in zip(axes_bottom, archives):
        data = tau_summary[tau_summary["archive"] == archive].copy()
        data["family"] = pd.Categorical(data["family"], categories=family_order, ordered=True)
        data = data.sort_values("family")
        x = np.arange(len(data))
        ax.bar(
            x,
            data["median_log10_tau_error"].to_numpy(dtype=float),
            color=[colors[str(f)] for f in data["family"]],
            width=0.62,
            alpha=0.70,
        )
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[str(f)] for f in data["family"]], fontsize=6)
        ax.set_ylabel("median log10 tau error")
        ax.set_title("Tau matching diagnostic", loc="left", fontsize=8)
        ax.grid(True, axis="y", color="#E4E9EF", lw=0.55)

    fig.suptitle("Multiscale stochastic nulls test whether the AR(1) floor is a single-timescale ceiling", fontsize=10)
    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(
    prefix: str,
    summary: pd.DataFrame,
    tau_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# R36 Multiscale Memory Nulls",
        "",
        "Date: 2026-06-05",
        "",
        "## Purpose",
        "",
        "This analysis tests the external-reviewer concern that the R35 matched",
        "AR(1) artifact floor is a single-timescale ceiling rather than a complete",
        "streamflow null. It adds a two-timescale AR mixture and an ARFIMA-like",
        "fractional long-memory surrogate, with tau_acf, PSD, beta and De all",
        "recomputed from each surrogate archive.",
        "",
        "## Parameters",
        "",
        f"- Archives: {args.archives}",
        f"- Draws per family/archive: {args.n_draws}",
        f"- Families: {', '.join(args.families)}",
        f"- ARFIMA-like fractional differencing parameter d: {args.arfima_d}",
        f"- Limit per archive: {args.limit_per_archive or 'none'}",
        f"- Seed: {args.seed}",
        "",
        "## Null-calibration summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g") if not summary.empty else "No summary rows.",
        "",
        "## Tau-matching summary",
        "",
        tau_summary.to_markdown(index=False, floatfmt=".4g") if not tau_summary.empty else "No tau summary rows.",
        "",
        "## Interpretation boundary",
        "",
        "- If a multiscale or long-memory null approaches the observed archive",
        "  reduction, the safe interpretation is that the observed deficit below",
        "  matched AR(1) is compatible with multiscale stochastic structure.",
        "- If it still exceeds the observed reduction, the result strengthens the",
        "  null-calibration/cautionary framing and weakens any mechanism claim.",
        "- This analysis does not prove direct storage or tracer causality.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{prefix}_draw_metrics.csv`",
        f"- `reports/tables/{prefix}_summary.csv`",
        f"- `reports/tables/{prefix}_gauge_diagnostics.csv`",
        f"- `reports/tables/{prefix}_tau_match_summary.csv`",
        f"- `reports/figures/{prefix}.png/svg/pdf`",
    ]
    (NOTES / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_source_data(prefix: str) -> None:
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)
    for suffix in ["draw_metrics.csv", "summary.csv", "gauge_diagnostics.csv", "tau_match_summary.csv"]:
        src = TABLES / f"{prefix}_{suffix}"
        if src.exists():
            shutil.copy2(src, LATEX_SOURCE / src.name)


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)

    archives = load_archives(args)
    print({archive: len(series) for archive, series in archives.items()})
    min_units = {"CAMELS-US": 50, "CAMELS-GB": 45}

    metric_rows: list[dict[str, Any]] = []
    for archive, series in archives.items():
        observed = observed_curves_for_archive(archive)
        observed = observed[observed["gauge_id"].isin({item.gauge_id for item in series})].copy()
        metrics = archive_dispersion(observed, min_units=min_units[archive])
        metric_rows.append({"archive": archive, "family": "observed", "draw": -1, **metrics})

    rng = np.random.default_rng(args.seed)
    tasks: list[dict[str, Any]] = []
    for archive in archives:
        for family in args.families:
            for draw in range(args.n_draws):
                tasks.append(
                    {
                        "archive": archive,
                        "family": family,
                        "draw": draw,
                        "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
                        "min_units": min_units[archive],
                        "arfima_d": args.arfima_d,
                    }
                )

    started = time.time()
    diag_frames: list[pd.DataFrame] = []
    completed = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(archives,)) as executor:
        futures = [executor.submit(run_family_draw, task) for task in tasks]
        for fut in cf.as_completed(futures):
            metrics, diagnostics = fut.result()
            metric_rows.append(metrics)
            diag_frames.append(diagnostics)
            completed += 1
            if completed % max(1, min(8, len(tasks))) == 0 or completed == len(tasks):
                elapsed = time.time() - started
                print(f"Completed {completed}/{len(tasks)} archive-family draws in {elapsed:.1f}s")

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    summary = summarize_metrics(metrics)
    tau_summary = summarize_tau_match(diagnostics)

    metrics.to_csv(TABLES / f"{args.output_prefix}_draw_metrics.csv", index=False)
    diagnostics.to_csv(TABLES / f"{args.output_prefix}_gauge_diagnostics.csv", index=False)
    summary.to_csv(TABLES / f"{args.output_prefix}_summary.csv", index=False)
    tau_summary.to_csv(TABLES / f"{args.output_prefix}_tau_match_summary.csv", index=False)
    plot_summary(summary, tau_summary, args.output_prefix)
    write_note(args.output_prefix, summary, tau_summary, args)
    copy_to_source_data(args.output_prefix)
    print(summary.to_string(index=False))
    print(tau_summary.to_string(index=False))


if __name__ == "__main__":
    main()
