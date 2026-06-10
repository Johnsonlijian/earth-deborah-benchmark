"""R33 parallel gauge-matched AR(1) ensemble.

R20 used a single stochastic full-archive draw as a hard artifact-floor
diagnostic. This script upgrades that diagnostic to a multi-draw ensemble for
CAMELS-US and CAMELS-GB v2 while keeping the exact same tau_acf--PSD--beta--De
pipeline used by R20.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_r20_gauge_matched_surrogate_ladder import (
    GaugeSeries,
    archive_dispersion,
    load_gb_series,
    load_us_series,
    observed_curves_for_archive,
    plot_summary,
    run_family_draw,
    summarize_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX_SOURCE = ROOT / "source_data"
FAMILY = "matched_ar1"
DEFAULT_PREFIX = "r33_matched_ar1_ensemble"

WORK_ARCHIVES: dict[str, list[GaugeSeries]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", choices=["both", "us", "gb"], default="both")
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--block-length", type=int, default=90)
    parser.add_argument("--iaaft-iterations", type=int, default=20)
    parser.add_argument("--limit-per-archive", type=int, default=0)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def _init_worker(archives: dict[str, list[GaugeSeries]]) -> None:
    global WORK_ARCHIVES
    WORK_ARCHIVES = archives


def _run_task(task: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    archive = task["archive"]
    draw = int(task["draw"])
    seed = int(task["seed"])
    min_units = int(task["min_units"])
    curves, diagnostics = run_family_draw(
        WORK_ARCHIVES[archive],
        FAMILY,
        draw,
        seed,
        int(task["block_length"]),
        int(task["iaaft_iterations"]),
    )
    metrics = archive_dispersion(curves, min_units=min_units)
    return {"archive": archive, "family": FAMILY, "draw": draw, **metrics}, diagnostics


def load_archives(args: argparse.Namespace) -> dict[str, list[GaugeSeries]]:
    archives: dict[str, list[GaugeSeries]] = {}
    if args.archives in {"both", "us"}:
        archives["CAMELS-US"] = load_us_series(args.limit_per_archive)
    if args.archives in {"both", "gb"}:
        archives["CAMELS-GB"] = load_gb_series(args.limit_per_archive)
    if not archives:
        raise RuntimeError("No archives selected")
    for archive, series in archives.items():
        if not series:
            raise RuntimeError(f"No series loaded for {archive}")
    return archives


def write_note(prefix: str, metrics: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# R33 Matched-AR(1) Ensemble",
        "",
        "Date: 2026-06-04",
        "",
        "## Method",
        "",
        "Each CAMELS-US and CAMELS-GB v2 gauge is replaced by stochastic AR(1)",
        "surrogates generated from that gauge's own discharge-anomaly series.",
        "For every archive draw, tau_acf, PSD, local beta and De are recomputed",
        "before applying the same raw-frequency versus De dispersion metric.",
        "",
        f"- Surrogate family: {FAMILY}",
        f"- Draws per archive: {args.n_surrogates}",
        f"- Workers: {args.workers}",
        f"- Random seed: {args.seed}",
        "",
        "## Aggregate Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g") if not summary.empty else "No summary rows.",
        "",
        "## Interpretation Boundary",
        "",
        "This ensemble calibrates the stochastic-memory artifact floor. A negative",
        "observed-minus-null score means the current evidence supports benchmark",
        "diagnosis but not empirical excess over a gauge-matched AR(1) mechanism",
        "gate.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{prefix}_draw_metrics.csv`",
        f"- `reports/tables/{prefix}_summary.csv`",
        f"- `reports/tables/{prefix}_gauge_diagnostics.csv`",
        f"- `reports/figures/{prefix}.png/svg/pdf`",
    ]
    (NOTES / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_source_data(prefix: str) -> None:
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)
    for suffix in ["draw_metrics.csv", "summary.csv", "gauge_diagnostics.csv"]:
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
        obs_metrics = archive_dispersion(observed, min_units=min_units[archive])
        metric_rows.append({"archive": archive, "family": "observed", "draw": -1, **obs_metrics})

    rng = np.random.default_rng(args.seed)
    tasks: list[dict[str, Any]] = []
    for archive in archives:
        for draw in range(args.n_surrogates):
            tasks.append(
                {
                    "archive": archive,
                    "draw": draw,
                    "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
                    "min_units": min_units[archive],
                    "block_length": args.block_length,
                    "iaaft_iterations": args.iaaft_iterations,
                }
            )

    started = time.time()
    diag_frames: list[pd.DataFrame] = []
    completed = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(archives,)) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for fut in cf.as_completed(futures):
            metrics, diagnostics = fut.result()
            metric_rows.append(metrics)
            diag_frames.append(diagnostics)
            completed += 1
            if completed % max(1, min(10, len(tasks))) == 0 or completed == len(tasks):
                elapsed = time.time() - started
                print(f"Completed {completed}/{len(tasks)} archive-draws in {elapsed:.1f}s")

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    summary = summarize_metrics(metrics)

    metrics.to_csv(TABLES / f"{args.output_prefix}_draw_metrics.csv", index=False)
    diagnostics.to_csv(TABLES / f"{args.output_prefix}_gauge_diagnostics.csv", index=False)
    summary.to_csv(TABLES / f"{args.output_prefix}_summary.csv", index=False)
    plot_summary(summary, args.output_prefix)
    write_note(args.output_prefix, metrics, summary, args)
    copy_to_source_data(args.output_prefix)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
