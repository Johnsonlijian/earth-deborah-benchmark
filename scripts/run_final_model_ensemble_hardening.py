"""Final model-diagnostic hardening for the HESS package.

Adds reviewer-facing uncertainty around the model-use case without converting
the manuscript into an operational model ranking:

* 100-draw seasonal AR(1) memory-null ensemble on CAMELS-GB same split;
* RRMPG GR4J/HBV-Edu calibration-budget sensitivity (128 vs 512 LHS samples);
* compact model-sensitivity source tables and supplementary figure.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_r21_hydrological_model_benchmark as r21
import run_r22_model_null_context as r22
import run_r23_multimodel_hydrology_benchmark as r23
import run_r39_open_model_intercomparison as r39
from run_camels_gb_replication import DAILY_DIR, gauge_id_from_path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
NW_SOURCE = ROOT / "source_data"
NW_SUPP_FIGURES = ROOT / "figures"
OUT = "final_model_ensemble_hardening"

_AR1_GAUGES: list[r39.GaugeData] = []
_AR1_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_AR1_SEED: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar1-draws", type=int, default=100)
    parser.add_argument("--rrmpg-samples", type=int, default=512)
    parser.add_argument("--min-eval-years", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--skip-rrmpg-budget", action="store_true")
    parser.add_argument("--lstm-seeds", type=int, default=0, help="Number of global-LSTM seeds to train for seed-sensitivity hardening.")
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--lstm-batches", type=int, default=260)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument("--lstm-seq-len", type=int, default=180)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_gauges(args: argparse.Namespace) -> tuple[list[Path], list[r39.GaugeData]]:
    files = sorted(Path(DAILY_DIR).glob("*.csv"))
    if args.limit:
        files = files[: args.limit]
    gauges = [g for path in files if (g := r39.load_gauge(path, args.min_eval_years)) is not None]
    return files, gauges


def observed_cache(gauges: list[r39.GaugeData]) -> dict[str, tuple[float, pd.DataFrame]]:
    cache: dict[str, tuple[float, pd.DataFrame]] = {}
    for gauge in gauges:
        try:
            cache[gauge.gauge_id] = r21.beta_curve(gauge.dates[gauge.eval_idx], gauge.q[gauge.eval_idx], "observed")
        except Exception:
            continue
    return cache


def draw_seed(seed: int, draw: int) -> int:
    return int(np.random.SeedSequence([seed, 1009, draw]).generate_state(1, dtype=np.uint32)[0])


def seasonal_ar1_draw(
    draw: int,
    gauges: list[r39.GaugeData],
    cache: dict[str, tuple[float, pd.DataFrame]],
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(draw_seed(seed, draw))
    for gauge in gauges:
        if gauge.gauge_id not in cache:
            continue
        tau_obs, obs_curve = cache[gauge.gauge_id]
        try:
            ar1_pred = r22.seasonal_ar1_null(pd.Series(gauge.q, index=gauge.dates), gauge.train_idx, gauge.eval_idx, rng)
            pred = np.full_like(gauge.q, np.nan, dtype=float)
            pred[gauge.eval_idx] = ar1_pred
            row, _curve = r39.metric_row(
                gauge,
                "seasonal_ar1_null",
                pred,
                np.nan,
                {"calibration": f"seasonal_ar1_ensemble_draw={draw}"},
                obs_curve,
                tau_obs,
            )
            if row is not None:
                row["draw"] = draw
                rows.append(row)
        except Exception:
            continue
    return rows


def init_ar1_worker(gauges: list[r39.GaugeData], cache: dict[str, tuple[float, pd.DataFrame]], seed: int) -> None:
    global _AR1_GAUGES, _AR1_CACHE, _AR1_SEED
    _AR1_GAUGES = gauges
    _AR1_CACHE = cache
    _AR1_SEED = seed


def seasonal_ar1_draw_worker(draw: int) -> list[dict]:
    return seasonal_ar1_draw(draw, _AR1_GAUGES, _AR1_CACHE, _AR1_SEED)


def summarize_ar1_draws(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    draw_summary = (
        metrics.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["eval_nse", "eval_kge", "beta_de_median_abs_distance"])
        .groupby("draw", as_index=False)
        .agg(
            n_gauges=("gauge_id", "nunique"),
            median_eval_nse=("eval_nse", "median"),
            median_eval_kge=("eval_kge", "median"),
            median_beta_de_distance=("beta_de_median_abs_distance", "median"),
            median_abs_log10_tau_error=("abs_log10_tau_error", "median"),
        )
    )
    return draw_summary


def summarize_seed_metrics(metrics: pd.DataFrame, seed_col: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["eval_nse", "eval_kge", "beta_de_median_abs_distance"])
        .groupby(seed_col, as_index=False)
        .agg(
            n_gauges=("gauge_id", "nunique"),
            median_eval_nse=("eval_nse", "median"),
            median_eval_kge=("eval_kge", "median"),
            median_beta_de_distance=("beta_de_median_abs_distance", "median"),
            median_abs_log10_tau_error=("abs_log10_tau_error", "median"),
        )
    )


def seasonal_ar1_ensemble(
    gauges: list[r39.GaugeData],
    cache: dict[str, tuple[float, pd.DataFrame]],
    draws: int,
    seed: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = TABLES / f"{OUT}_seasonal_ar1_metrics.csv"
    summary_path = TABLES / f"{OUT}_seasonal_ar1_draw_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "draw" in metrics.columns:
            metrics = metrics[metrics["draw"].astype(int) < draws].copy()
    else:
        metrics = pd.DataFrame()

    completed = set(metrics["draw"].dropna().astype(int).unique()) if "draw" in metrics.columns else set()
    missing = [draw for draw in range(draws) if draw not in completed]
    print(f"Final seasonal AR(1) ensemble has {len(completed)}/{draws} completed draws; missing={len(missing)}", flush=True)

    def record(draw: int, rows: list[dict]) -> None:
        nonlocal metrics
        new = pd.DataFrame(rows)
        if not new.empty:
            metrics = pd.concat([metrics, new], ignore_index=True, sort=False)
        metrics.to_csv(metrics_path, index=False)
        summarize_ar1_draws(metrics).to_csv(summary_path, index=False)
        done = len(set(metrics["draw"].dropna().astype(int).unique())) if "draw" in metrics.columns else 0
        print(f"Final seasonal AR(1) ensemble draw {draw + 1}/{draws}; rows={len(rows)}; completed={done}", flush=True)

    if missing and workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers, initializer=init_ar1_worker, initargs=(gauges, cache, seed)) as executor:
            future_map = {executor.submit(seasonal_ar1_draw_worker, draw): draw for draw in missing}
            for future in cf.as_completed(future_map):
                draw = future_map[future]
                try:
                    record(draw, future.result())
                except Exception as exc:
                    print(f"Final seasonal AR(1) ensemble draw {draw} failed: {exc}", flush=True)
    else:
        for draw in missing:
            record(draw, seasonal_ar1_draw(draw, gauges, cache, seed))

    draw_summary = summarize_ar1_draws(metrics)
    return metrics, draw_summary


def rrmpg_summary_and_paired(metrics: pd.DataFrame, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = r39.summarize(metrics)
    base = pd.read_csv(TABLES / "r39_open_model_intercomparison_metrics.csv")
    base = base[base["model_type"].isin(["rrmpg_gr4j", "rrmpg_hbvedu"])].copy()
    base["budget_samples"] = 128
    work = metrics.copy()
    if not work.empty:
        work["budget_samples"] = samples
    combined = pd.concat([base, work], ignore_index=True, sort=False)
    paired_rows = []
    pivot = combined.pivot_table(
        index=["gauge_id", "model_type"],
        columns="budget_samples",
        values=["eval_nse", "eval_kge", "beta_de_median_abs_distance"],
        aggfunc="first",
    )
    required = [
        ("eval_nse", samples),
        ("eval_nse", 128),
        ("eval_kge", samples),
        ("eval_kge", 128),
        ("beta_de_median_abs_distance", samples),
        ("beta_de_median_abs_distance", 128),
    ]
    for col in required:
        if col not in pivot.columns:
            return summary, pd.DataFrame()
    for (gauge_id, model_type), row in pivot.dropna(subset=required).iterrows():
        paired_rows.append(
            {
                "gauge_id": gauge_id,
                "model_type": model_type,
                f"delta_{samples}_minus_128_eval_nse": float(row[("eval_nse", samples)] - row[("eval_nse", 128)]),
                f"delta_{samples}_minus_128_eval_kge": float(row[("eval_kge", samples)] - row[("eval_kge", 128)]),
                f"delta_{samples}_minus_128_beta_de_distance": float(
                    row[("beta_de_median_abs_distance", samples)] - row[("beta_de_median_abs_distance", 128)]
                ),
            }
        )
    return summary, pd.DataFrame(paired_rows)


def rrmpg_budget_sensitivity(files: list[Path], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + 2027)
    metrics_path = TABLES / f"{OUT}_rrmpg_{args.rrmpg_samples}_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.DataFrame()
    completed_gauges = set(metrics["gauge_id"].astype(str).unique()) if "gauge_id" in metrics.columns else set()
    jobs = [
        (str(path), int(rng.integers(0, 2**31 - 1)), args.rrmpg_samples, args.min_eval_years)
        for path in files
        if gauge_id_from_path(path) not in completed_gauges
    ]
    completed = 0
    print(
        f"Final RRMPG budget has {len(completed_gauges)}/{len(files)} completed gauges; missing={len(jobs)}",
        flush=True,
    )

    def record(gauge_id: str, model_rows: list[dict], error: str | None) -> None:
        nonlocal completed, metrics
        if error:
            print(f"  [{gauge_id}] RRMPG-{args.rrmpg_samples} skipped: {error}", flush=True)
        if model_rows:
            new = pd.DataFrame(model_rows)
            new["budget_samples"] = args.rrmpg_samples
            metrics = pd.concat([metrics, new], ignore_index=True, sort=False)
            metrics.to_csv(metrics_path, index=False)
        completed += 1
        if completed % 25 == 0 or completed == len(jobs):
            print(f"Final RRMPG budget {completed}/{len(jobs)} missing files; rows={len(metrics)}", flush=True)

    if args.workers <= 1:
        iterator = map(r39.evaluate_rrmpg_job, jobs)
        for gauge_id, model_rows, _curves, error in iterator:
            record(gauge_id, model_rows, error)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(r39.evaluate_rrmpg_job, job): job[0] for job in jobs}
            for future in cf.as_completed(future_map):
                gauge_id, model_rows, _curves, error = future.result()
                record(gauge_id, model_rows, error)
    summary, paired = rrmpg_summary_and_paired(metrics, args.rrmpg_samples)
    return metrics, summary, paired


def lstm_seed_ensemble(
    gauges: list[r39.GaugeData],
    cache: dict[str, tuple[float, pd.DataFrame]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = TABLES / f"{OUT}_lstm_seed_metrics.csv"
    log_path = TABLES / f"{OUT}_lstm_training_log.csv"
    if args.lstm_seeds <= 0 or args.skip_lstm:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not r39.HAS_TORCH:
        raise RuntimeError("PyTorch is required for the Final LSTM seed ensemble.")

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    logs = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    completed = set(metrics["seed_index"].dropna().astype(int).unique()) if "seed_index" in metrics.columns else set()
    print(f"Final LSTM seed ensemble has {len(completed)}/{args.lstm_seeds} completed seeds", flush=True)

    for seed_index in range(args.lstm_seeds):
        if seed_index in completed:
            continue
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = args.seed + 5000 + seed_index
        seed_args.skip_lstm = False
        print(f"Final LSTM seed {seed_index + 1}/{args.lstm_seeds}: training seed={seed_args.seed}", flush=True)
        predictions, train_log = r39.train_global_lstm(gauges, seed_args)
        rows: list[dict] = []
        for idx, gauge in enumerate(gauges, start=1):
            if gauge.gauge_id not in predictions or gauge.gauge_id not in cache:
                continue
            tau_obs, obs_curve = cache[gauge.gauge_id]
            try:
                row, _curve = r39.metric_row(
                    gauge,
                    "global_lstm",
                    predictions[gauge.gauge_id],
                    r21.nse(predictions[gauge.gauge_id][gauge.train_idx], gauge.q[gauge.train_idx]),
                    {"calibration": f"global_lstm_seed_ensemble_seed={seed_index}", "seed": seed_args.seed},
                    obs_curve,
                    tau_obs,
                )
                if row is not None:
                    row["seed_index"] = seed_index
                    row["seed"] = seed_args.seed
                    rows.append(row)
            except Exception:
                continue
            if idx % 100 == 0:
                print(f"  LSTM seed {seed_index + 1}: evaluated {idx}/{len(gauges)} gauges", flush=True)
        new_metrics = pd.DataFrame(rows)
        if not new_metrics.empty:
            metrics = pd.concat([metrics, new_metrics], ignore_index=True, sort=False)
            metrics.to_csv(metrics_path, index=False)
        if not train_log.empty:
            train_log = train_log.copy()
            train_log["seed_index"] = seed_index
            train_log["seed"] = seed_args.seed
            logs = pd.concat([logs, train_log], ignore_index=True, sort=False)
            logs.to_csv(log_path, index=False)
        summarize_seed_metrics(metrics, "seed_index").to_csv(TABLES / f"{OUT}_lstm_seed_summary.csv", index=False)
        print(f"Final LSTM seed {seed_index + 1}/{args.lstm_seeds}; rows={len(rows)}", flush=True)

    seed_summary = summarize_seed_metrics(metrics, "seed_index")
    return metrics, seed_summary, logs


def summarize_all(
    ar1_draw_summary: pd.DataFrame,
    rrmpg_summary: pd.DataFrame,
    rrmpg_paired: pd.DataFrame,
    lstm_seed_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    if not ar1_draw_summary.empty:
        for metric in ["median_eval_nse", "median_eval_kge", "median_beta_de_distance", "median_abs_log10_tau_error"]:
            vals = ar1_draw_summary[metric].dropna()
            rows.append(
                {
                    "analysis": "seasonal_ar1_ensemble",
                    "metric": metric,
                    "n_draws": int(len(vals)),
                    "median": float(vals.median()),
                    "q05": float(vals.quantile(0.05)),
                    "q95": float(vals.quantile(0.95)),
                    "interpretation": "null-generation uncertainty for the seasonal AR(1) baseline",
                }
            )
    if not lstm_seed_summary.empty:
        for metric in ["median_eval_nse", "median_eval_kge", "median_beta_de_distance", "median_abs_log10_tau_error"]:
            vals = lstm_seed_summary[metric].dropna()
            rows.append(
                {
                    "analysis": "global_lstm_seed_ensemble",
                    "metric": metric,
                    "n_draws": int(len(vals)),
                    "median": float(vals.median()),
                    "q05": float(vals.quantile(0.05)),
                    "q95": float(vals.quantile(0.95)),
                    "interpretation": "seed-sensitivity check for the lightweight global LSTM baseline",
                }
            )
    for _, row in rrmpg_summary.iterrows():
        if int(row["n_gauges"]) <= 0 or not np.isfinite(float(row["median_beta_de_distance"])):
            continue
        rows.append(
            {
                "analysis": f"rrmpg_budget_{args.rrmpg_samples}",
                "metric": f"{row['model_type']}_median_beta_de_distance",
                "n_draws": int(row["n_gauges"]),
                "median": float(row["median_beta_de_distance"]),
                "q05": np.nan,
                "q95": np.nan,
                "interpretation": "512-sample calibration-budget sensitivity; still not an operational model ranking",
            }
        )
    if not rrmpg_paired.empty:
        for model_type, group in rrmpg_paired.groupby("model_type"):
            rows.append(
                {
                    "analysis": "rrmpg_512_minus_128_paired_delta",
                    "metric": f"{model_type}_beta_de_distance_delta",
                    "n_draws": int(len(group)),
                    "median": float(group["delta_512_minus_128_beta_de_distance"].median()),
                    "q05": float(group["delta_512_minus_128_beta_de_distance"].quantile(0.05)),
                    "q95": float(group["delta_512_minus_128_beta_de_distance"].quantile(0.95)),
                    "interpretation": "negative values mean the larger budget reduces beta(De) distance",
                }
            )
    return pd.DataFrame(rows)


def draw_figure(ar1_draw_summary: pd.DataFrame, rrmpg_paired: pd.DataFrame, all_summary: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    blue = "#2166AC"
    red = "#B2182B"
    green = "#1B7837"
    orange = "#D95F02"
    gray = "#6B7280"
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.65), constrained_layout=True)
    ax0, ax1, ax2 = axes
    if not ar1_draw_summary.empty:
        vals = ar1_draw_summary["median_beta_de_distance"].dropna()
        ax0.hist(vals, bins=20, color=red, alpha=0.75, edgecolor="white")
        ax0.axvline(vals.median(), color="#2B2B2B", lw=1)
        lstm_row = all_summary[(all_summary["analysis"] == "global_lstm_seed_ensemble") & (all_summary["metric"] == "median_beta_de_distance")]
        if not lstm_row.empty:
            ax0.axvline(float(lstm_row.iloc[0]["median"]), color=green, lw=1.2, label="LSTM seed median")
            ax0.legend(fontsize=6.4, loc="upper right")
        ax0.set_xlabel("median beta(De) distance")
        ax0.set_ylabel("AR(1) draws")
        ax0.set_title("a  seasonal AR(1) ensemble", loc="left", fontweight="bold")
    if not rrmpg_paired.empty:
        labels = []
        data = []
        colors = []
        for model_type, label, color in [("rrmpg_gr4j", "GR4J", blue), ("rrmpg_hbvedu", "HBV-Edu", green)]:
            group = rrmpg_paired[rrmpg_paired["model_type"] == model_type]
            if not group.empty:
                labels.append(label)
                data.append(group["delta_512_minus_128_beta_de_distance"].dropna().to_numpy())
                colors.append(color)
        parts = ax1.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
        for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
            parts[key].set_color("#2B2B2B")
            parts[key].set_linewidth(0.8)
        ax1.axhline(0, color="#2B2B2B", lw=0.8, ls="--")
        ax1.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax1.set_ylabel("512 - 128 beta(De) distance")
        ax1.set_title("b  RRMPG budget sensitivity", loc="left", fontweight="bold")
    ax2.axis("off")
    ax2.set_title("c  model-claim boundary", loc="left", fontweight="bold")
    bullets = [
        ("AR(1) null", "reported as ensemble interval"),
        ("RRMPG budget", "512-vs-128 sensitivity quantified"),
        ("LSTM seeds", "reported as seed-sensitivity interval"),
        ("Interpretation", "diagnostic stress test only"),
    ]
    for i, (head, body) in enumerate(bullets):
        yy = 0.82 - i * 0.2
        ax2.add_patch(plt.Rectangle((0.03, yy - 0.03), 0.05, 0.06, transform=ax2.transAxes, color=[red, orange, gray, blue][i]))
        ax2.text(0.11, yy + 0.018, head, transform=ax2.transAxes, fontsize=7.3, fontweight="bold")
        ax2.text(0.11, yy - 0.032, body, transform=ax2.transAxes, fontsize=6.6, color=gray)
    for ext in [".pdf", ".svg", ".png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{OUT}{ext}", **kwargs)
        NW_SUPP_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(NW_SUPP_FIGURES / f"supp_fig10_model_ensemble_sensitivity{ext}", **kwargs)
    plt.close(fig)


def copy_source_outputs() -> None:
    NW_SOURCE.mkdir(parents=True, exist_ok=True)
    for suffix in [
        "seasonal_ar1_metrics.csv",
        "seasonal_ar1_draw_summary.csv",
        "lstm_seed_metrics.csv",
        "lstm_seed_summary.csv",
        "lstm_training_log.csv",
        "rrmpg_512_metrics.csv",
        "rrmpg_512_summary.csv",
        "rrmpg_512_vs_128_paired_delta.csv",
        "summary.csv",
    ]:
        src = TABLES / f"{OUT}_{suffix}"
        if src.exists():
            shutil.copy2(src, NW_SOURCE / src.name)


def write_note(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Final Model Ensemble Hardening",
        "",
        "Date: 2026-06-08",
        "",
        "## Purpose",
        "",
        "Adds ensemble uncertainty around model diagnostics while preserving the implementation-level, non-ranking claim boundary.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Boundary",
        "",
        "The model section remains a diagnostic stress test: beta(De) distance and NSE/KGE are separable axes. The results do not establish operational superiority of GR4J, HBV-Edu, LSTM or seasonal AR(1).",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{OUT}_seasonal_ar1_draw_summary.csv`",
        f"- `reports/tables/{OUT}_lstm_seed_summary.csv`",
        f"- `reports/tables/{OUT}_rrmpg_512_vs_128_paired_delta.csv`",
        f"- `reports/tables/{OUT}_summary.csv`",
        f"- `reports/figures/{OUT}.pdf/svg/png`",
        "",
        f"AR(1) draws: {args.ar1_draws}; LSTM seeds: {args.lstm_seeds}; RRMPG budget: {args.rrmpg_samples}; seed: {args.seed}.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    files, gauges = load_gauges(args)
    print(f"Final load: {len(gauges)} CAMELS-GB gauges from {len(files)} files", flush=True)
    cache = observed_cache(gauges)
    print(f"Final observed curve cache: {len(cache)} gauges", flush=True)

    ar1_metrics, ar1_draw_summary = seasonal_ar1_ensemble(gauges, cache, args.ar1_draws, args.seed, args.workers)
    ar1_metrics.to_csv(TABLES / f"{OUT}_seasonal_ar1_metrics.csv", index=False)
    ar1_draw_summary.to_csv(TABLES / f"{OUT}_seasonal_ar1_draw_summary.csv", index=False)

    lstm_metrics, lstm_seed_summary, lstm_logs = lstm_seed_ensemble(gauges, cache, args)

    if args.skip_rrmpg_budget:
        rrmpg_metrics = pd.DataFrame()
        rrmpg_summary = pd.DataFrame()
        rrmpg_paired = pd.DataFrame()
    else:
        rrmpg_metrics, rrmpg_summary, rrmpg_paired = rrmpg_budget_sensitivity(files, args)
        rrmpg_metrics.to_csv(TABLES / f"{OUT}_rrmpg_512_metrics.csv", index=False)
        rrmpg_summary.to_csv(TABLES / f"{OUT}_rrmpg_512_summary.csv", index=False)
        rrmpg_paired.to_csv(TABLES / f"{OUT}_rrmpg_512_vs_128_paired_delta.csv", index=False)

    summary = summarize_all(ar1_draw_summary, rrmpg_summary, rrmpg_paired, lstm_seed_summary, args)
    summary.to_csv(TABLES / f"{OUT}_summary.csv", index=False)
    draw_figure(ar1_draw_summary, rrmpg_paired, summary)
    copy_source_outputs()
    write_note(summary, args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
