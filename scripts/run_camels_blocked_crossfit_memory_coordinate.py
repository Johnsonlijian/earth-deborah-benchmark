"""Blocked temporal cross-fitting robustness for the CAMELS memory coordinate.

R12 tested one early/late split. This script asks whether the result survives
four contiguous held-out temporal blocks. For each gauge, one block is held out
for beta(De); the remaining three blocks estimate train tau_acf through the
median of block-specific tau values. This avoids estimating train memory from a
single artificial sequence with long cross-block gaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time
from run_camels_crossfit_memory_coordinate import (
    FIGURES,
    FLOW_DIR,
    NOTES,
    SECONDS_PER_DAY,
    TABLES,
    add_de_column,
    beta_base_from_values,
    compute_group_metrics,
    format_p_value,
    load_one_camels_flow,
    prepare_daily_anomaly,
    random_tau_null,
    segment_coverage,
    station_bin_medians,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-dir", default=str(FLOW_DIR))
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--min-record-years", type=float, default=20.0)
    parser.add_argument("--min-test-years", type=float, default=4.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-random", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--n-sample", type=int, default=0)
    parser.add_argument("--output-prefix", default="camels_673_blocked_crossfit_memory_coordinate")
    return parser.parse_args()


def estimate_tau_days(series: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
    dates, values = prepare_daily_anomaly(series)
    tau_sec = integral_autocorrelation_time(
        values,
        dt_seconds=SECONDS_PER_DAY,
        max_lag=min(730, max(1, len(values) - 1)),
        stop_at_zero=True,
    )
    tau_days = tau_sec / SECONDS_PER_DAY
    if not np.isfinite(tau_days) or tau_days <= 0:
        raise ValueError("non-positive tau")
    return dates, values, float(tau_days)


def split_into_blocks(series: pd.Series, n_blocks: int) -> list[tuple[str, pd.Series]]:
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    dates = series.index.sort_values()
    start = dates.min()
    end = dates.max()
    span = end - start
    blocks: list[tuple[str, pd.Series]] = []
    for block in range(n_blocks):
        lo = start + span * block / n_blocks
        hi = start + span * (block + 1) / n_blocks
        if block == n_blocks - 1:
            mask = (series.index >= lo) & (series.index <= hi)
        else:
            mask = (series.index >= lo) & (series.index < hi)
        blocks.append((f"block_{block + 1}", series[mask]))
    return blocks


def block_metadata(blocks: list[tuple[str, pd.Series]], max_missing: float, min_test_years: float) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    for label, block in blocks:
        years, missing = segment_coverage(block)
        metadata[label] = {"years": years, "missing_frac": missing}
        if years < min_test_years or missing > max_missing:
            metadata[label]["valid"] = 0.0
        else:
            metadata[label]["valid"] = 1.0
    return metadata


def run_blocked_crossfit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow_files = sorted(Path(args.flow_dir).glob("*.txt"))
    if args.n_sample > 0:
        rng = np.random.default_rng(args.seed)
        flow_files = list(rng.choice(flow_files, size=min(args.n_sample, len(flow_files)), replace=False))

    summary_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[pd.DataFrame] = []
    for idx, filepath in enumerate(flow_files, start=1):
        gauge_id, series = load_one_camels_flow(filepath)
        if series.empty:
            continue
        record_years, record_missing = segment_coverage(series)
        if record_years < args.min_record_years or record_missing > args.max_missing:
            continue

        blocks = split_into_blocks(series, args.n_blocks)
        metadata = block_metadata(blocks, args.max_missing, args.min_test_years)
        prepared: dict[str, tuple[pd.DatetimeIndex, np.ndarray, float]] = {}
        for label, block in blocks:
            if metadata[label]["valid"] < 1:
                continue
            try:
                prepared[label] = estimate_tau_days(block)
            except Exception:
                continue
        if len(prepared) != args.n_blocks:
            continue

        for heldout_label, _heldout_block in blocks:
            if heldout_label not in prepared:
                continue
            train_labels = [label for label, _ in blocks if label != heldout_label and label in prepared]
            if len(train_labels) != args.n_blocks - 1:
                continue
            test_dates, test_values, test_tau_days = prepared[heldout_label]
            train_tau_values = np.array([prepared[label][2] for label in train_labels], dtype=float)
            train_tau_days = float(np.nanmedian(train_tau_values))
            if not np.isfinite(train_tau_days) or train_tau_days <= 0:
                continue
            try:
                beta_base = beta_base_from_values(test_dates, test_values)
            except Exception:
                continue
            if beta_base.empty:
                continue

            curve = add_de_column(beta_base, train_tau_days, "de_crossfit")
            curve = add_de_column(curve, test_tau_days, "de_same")
            fold_label = heldout_label
            unit_id = f"{gauge_id}_{fold_label}"
            curve["gauge_id"] = gauge_id
            curve["unit_id"] = unit_id
            curve["fold"] = fold_label
            curve["train_segment"] = "+".join(train_labels)
            curve["test_segment"] = heldout_label
            curve["train_tau_days"] = train_tau_days
            curve["test_tau_days"] = test_tau_days
            curve_rows.append(
                curve[
                    [
                        "gauge_id",
                        "unit_id",
                        "fold",
                        "train_segment",
                        "test_segment",
                        "train_tau_days",
                        "test_tau_days",
                        "frequency",
                        "frequency_cpd",
                        "beta",
                        "de_crossfit",
                        "de_same",
                    ]
                ]
            )

            cross_summary = summarize_beta_window(curve.rename(columns={"de_crossfit": "de"}))
            same_summary = summarize_beta_window(curve.rename(columns={"de_same": "de"}))
            train_years = float(sum(metadata[label]["years"] for label in train_labels))
            train_missing = float(np.average([metadata[label]["missing_frac"] for label in train_labels]))
            summary_rows.append(
                {
                    "gauge_id": gauge_id,
                    "unit_id": unit_id,
                    "fold": fold_label,
                    "train_segment": "+".join(train_labels),
                    "test_segment": heldout_label,
                    "record_years": record_years,
                    "record_missing_frac": record_missing,
                    "train_years_sum": train_years,
                    "test_years": metadata[heldout_label]["years"],
                    "train_missing_frac_mean": train_missing,
                    "test_missing_frac": metadata[heldout_label]["missing_frac"],
                    "train_tau_days": train_tau_days,
                    "test_tau_days": test_tau_days,
                    "train_tau_block_min_days": float(np.nanmin(train_tau_values)),
                    "train_tau_block_max_days": float(np.nanmax(train_tau_values)),
                    "tau_ratio_train_over_test": train_tau_days / test_tau_days if test_tau_days > 0 else np.nan,
                    "crossfit_n_beta_de_window": cross_summary["n_beta_de_window"],
                    "crossfit_mean_beta_de_window": cross_summary["mean_beta_de_window"],
                    "same_n_beta_de_window": same_summary["n_beta_de_window"],
                    "same_mean_beta_de_window": same_summary["mean_beta_de_window"],
                    "n_beta_rows": int(beta_base.shape[0]),
                }
            )
        if idx % 150 == 0:
            print(f"  processed {idx}/{len(flow_files)} flow files")

    if not curve_rows:
        raise RuntimeError("No blocked cross-fit beta curves were computed.")
    return pd.DataFrame(summary_rows), pd.concat(curve_rows, ignore_index=True)


def summarize_random_rows(metrics: pd.DataFrame, curves: pd.DataFrame, random_null: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group_name, null_group in random_null.groupby("analysis_group"):
        vals = null_group["variance_reduction_vs_raw"].to_numpy(dtype=float)
        cross = float(
            metrics[(metrics["analysis_group"] == group_name) & (metrics["method"] == "crossfit_tau")][
                "variance_reduction_vs_raw"
            ].iloc[0]
        )
        n_units = int(curves["unit_id"].nunique()) if group_name == "all" else int(curves[curves["fold"] == group_name]["unit_id"].nunique())
        rows.append(
            {
                "analysis_group": group_name,
                "method": "random_train_tau_summary",
                "axis_column": "de_random",
                "weighted_beta_variance": np.nan,
                "n_units": n_units,
                "n_bins_used": np.nan,
                "n_unit_bin_values": np.nan,
                "variance_reduction_vs_raw": float(np.nanmedian(vals)),
                "random_p05": float(np.nanpercentile(vals, 5)),
                "random_p95": float(np.nanpercentile(vals, 95)),
                "empirical_p_ge_crossfit": float((np.sum(vals >= cross) + 1.0) / (np.isfinite(vals).sum() + 1.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_results(
    summaries: pd.DataFrame,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    random_null: pd.DataFrame,
    output_prefix: str,
) -> None:
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
        "same_segment_tau": "#2B7BB9",
        "crossfit_tau": "#C93434",
        "random_train_tau": "#7C8794",
        "raw_frequency": "#7C8794",
    }
    fold_colors = {
        "block_1": "#276FBF",
        "block_2": "#2A9D8F",
        "block_3": "#E9A129",
        "block_4": "#8A5CF6",
    }

    fig = plt.figure(figsize=(7.35, 5.45))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    all_metrics = metrics[metrics["analysis_group"] == "all"].set_index("method")
    methods = ["same_segment_tau", "crossfit_tau"]
    vals = 100 * all_metrics.loc[methods, "variance_reduction_vs_raw"].to_numpy(dtype=float)
    ax_a.bar([0, 1], vals, color=[colors[m] for m in methods], width=0.62, alpha=0.86)
    rand_vals = 100 * random_null[random_null["analysis_group"] == "all"]["variance_reduction_vs_raw"].to_numpy(dtype=float)
    parts = ax_a.violinplot(rand_vals[np.isfinite(rand_vals)], positions=[2], widths=0.55, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(colors["random_train_tau"])
        body.set_edgecolor(colors["random_train_tau"])
        body.set_alpha(0.25)
    q = np.nanpercentile(rand_vals, [5, 50, 95])
    ax_a.vlines(2, q[0], q[2], color=colors["random_train_tau"], lw=3, alpha=0.60)
    ax_a.scatter(2, q[1], color=colors["random_train_tau"], edgecolor="white", linewidth=0.7, s=35, zorder=3)
    for xx, yy in zip([0, 1], vals):
        ax_a.text(xx, yy + 1.0, f"{yy:.1f}%", ha="center", fontsize=6.5)
    ax_a.axhline(0, color="#AAB0B8", lw=0.8)
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(["same\nblock", "blocked\ncross-fit", "random\ntrain tau"])
    ax_a.set_ylabel("variance reduction vs raw [%]")
    ax_a.set_title("Blocked cross-fit tests split robustness", loc="left", fontsize=8)
    ax_a.text(-0.12, 1.08, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")
    ax_a.grid(True, axis="y", color="#E6EAF0", lw=0.55)

    fold_metrics = metrics[(metrics["analysis_group"] != "all") & (metrics["method"] == "crossfit_tau")].copy()
    fold_order = [f"block_{i}" for i in range(1, 5) if f"block_{i}" in set(fold_metrics["analysis_group"])]
    fold_vals = [
        100
        * float(
            fold_metrics[fold_metrics["analysis_group"] == fold]["variance_reduction_vs_raw"].iloc[0]
        )
        for fold in fold_order
    ]
    ax_b.bar(np.arange(len(fold_order)), fold_vals, color=[fold_colors[f] for f in fold_order], width=0.65, alpha=0.86)
    for xx, yy in zip(np.arange(len(fold_order)), fold_vals):
        ax_b.text(xx, yy + (1.0 if yy >= 0 else -2.2), f"{yy:.1f}", ha="center", va="bottom" if yy >= 0 else "top", fontsize=6.3)
    ax_b.axhline(0, color="#AAB0B8", lw=0.8)
    ax_b.set_xticks(np.arange(len(fold_order)))
    ax_b.set_xticklabels([f.replace("_", " ") for f in fold_order])
    ax_b.set_ylabel("cross-fit reduction [%]")
    ax_b.set_title("All held-out temporal blocks are visible", loc="left", fontsize=8)
    ax_b.text(-0.12, 1.08, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")
    ax_b.grid(True, axis="y", color="#E6EAF0", lw=0.55)

    ax_c.scatter(
        summaries["train_tau_days"],
        summaries["test_tau_days"],
        c=[fold_colors.get(f, "#777777") for f in summaries["fold"]],
        s=11,
        alpha=0.42,
        edgecolor="white",
        linewidth=0.22,
    )
    lim = np.nanpercentile(np.r_[summaries["train_tau_days"], summaries["test_tau_days"]], [1, 99])
    ax_c.plot(lim, lim, color="#AAB0B8", lw=0.9, ls="--")
    tau_sub = summaries.dropna(subset=["train_tau_days", "test_tau_days"])
    rho, p = stats.spearmanr(tau_sub["train_tau_days"], tau_sub["test_tau_days"])
    ax_c.text(
        0.04,
        0.96,
        f"Spearman rho={rho:.2f}\np {format_p_value(float(p))}\nn={len(tau_sub)} folds",
        transform=ax_c.transAxes,
        va="top",
        fontsize=6.4,
    )
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("train-block median tau_acf [d]")
    ax_c.set_ylabel("held-out block tau_acf [d]")
    ax_c.set_title("Block memory transfer is noisy but ordered", loc="left", fontsize=8)
    ax_c.text(-0.12, 1.08, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold")
    ax_c.grid(True, color="#E6EAF0", lw=0.55)

    raw_binned = station_bin_medians(curves, "frequency_cpd")
    same_binned = station_bin_medians(curves, "de_same")
    cross_binned = station_bin_medians(curves, "de_crossfit")
    for binned, label, color in [
        (raw_binned, "raw frequency", colors["raw_frequency"]),
        (same_binned, "same block tau", colors["same_segment_tau"]),
        (cross_binned, "blocked cross-fit tau", colors["crossfit_tau"]),
    ]:
        med = binned.groupby("bin_id", as_index=False).agg(axis_value=("axis_value", "first"), beta=("beta_median", "median"))
        ax_d.semilogx(med["axis_value"], med["beta"], lw=1.6, marker="o", ms=2.4, color=color, label=label)
    ax_d.axvspan(0.5, 2.0, color=colors["crossfit_tau"], alpha=0.05, lw=0)
    ax_d.set_xlabel("raw cycles d-1 or De")
    ax_d.set_ylabel("median local beta")
    ax_d.set_title("Held-out blocks keep the beta(De) ordering", loc="left", fontsize=8)
    ax_d.text(-0.12, 1.08, "d", transform=ax_d.transAxes, fontsize=9, fontweight="bold")
    ax_d.legend(loc="upper left", fontsize=5.6)
    ax_d.grid(True, color="#E6EAF0", lw=0.55)

    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{output_prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(metrics: pd.DataFrame, random_null: pd.DataFrame, summaries: pd.DataFrame, output_prefix: str) -> None:
    all_metrics = metrics[metrics["analysis_group"] == "all"].set_index("method")
    same = float(all_metrics.loc["same_segment_tau", "variance_reduction_vs_raw"])
    cross = float(all_metrics.loc["crossfit_tau", "variance_reduction_vs_raw"])
    random_vals = random_null[random_null["analysis_group"] == "all"]["variance_reduction_vs_raw"].to_numpy(dtype=float)
    p_ge = (np.sum(random_vals >= cross) + 1.0) / (np.isfinite(random_vals).sum() + 1.0)
    fold_lines = []
    for fold in sorted(summaries["fold"].unique()):
        value = float(
            metrics[(metrics["analysis_group"] == fold) & (metrics["method"] == "crossfit_tau")][
                "variance_reduction_vs_raw"
            ].iloc[0]
        )
        fold_lines.append(f"- {fold}: cross-fit reduction {value:.1%}.")
    lines = [
        "# CAMELS Blocked Cross-Fitting Robustness Note",
        "",
        "This analysis extends R12 from one early/late split to four contiguous",
        "held-out temporal blocks. Train tau_acf is the median of tau_acf values",
        "estimated from the three non-held-out blocks.",
        "",
        "## Aggregate metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Headline interpretation",
        "",
        f"- Same-block tau reduction: {same:.1%}.",
        f"- Blocked cross-fit tau reduction: {cross:.1%}.",
        f"- Random train-tau empirical p for matching/exceeding blocked cross-fit reduction: {p_ge:.4g}.",
        f"- Units analysed: {summaries['unit_id'].nunique()} held-out block folds from {summaries['gauge_id'].nunique()} gauges.",
        "",
        "## Fold-specific cross-fit reductions",
        "",
        *fold_lines,
        "",
        "## Manuscript use",
        "",
        "- Safe claim: blocked temporal cross-fitting supports robustness beyond",
        "  the single early/late split used in R12.",
        "- Required caveat: each fold remains same-gauge, discharge-derived",
        "  validation and does not prove independent physical storage causality.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{output_prefix}_summary.csv`",
        f"- `reports/tables/{output_prefix}_curves.csv`",
        f"- `reports/tables/{output_prefix}_metrics.csv`",
        f"- `reports/tables/{output_prefix}_bin_stats.csv`",
        f"- `reports/tables/{output_prefix}_random_tau_null.csv`",
        f"- `reports/figures/{output_prefix}.png/svg/pdf`",
    ]
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f"{output_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries, curves = run_blocked_crossfit(args)

    groups: dict[str, pd.DataFrame] = {"all": curves}
    for fold in sorted(curves["fold"].unique()):
        groups[fold] = curves[curves["fold"] == fold].copy()

    metric_parts: list[pd.DataFrame] = []
    bin_parts: list[pd.DataFrame] = []
    random_parts: list[pd.DataFrame] = []
    for idx, (group_name, group_curves) in enumerate(groups.items()):
        metrics, bins = compute_group_metrics(group_curves, group_name)
        metric_parts.append(metrics)
        bin_parts.append(bins)
        random_parts.append(
            random_tau_null(
                group_curves,
                group_name=group_name,
                n_random=args.n_random,
                seed=args.seed + idx * 1009,
            )
        )
    metrics = pd.concat(metric_parts, ignore_index=True)
    bin_stats = pd.concat(bin_parts, ignore_index=True)
    random_null = pd.concat(random_parts, ignore_index=True)
    metrics = pd.concat([metrics, summarize_random_rows(metrics, curves, random_null)], ignore_index=True)

    prefix = args.output_prefix
    summaries.to_csv(TABLES / f"{prefix}_summary.csv", index=False)
    curves.to_csv(TABLES / f"{prefix}_curves.csv", index=False)
    metrics.to_csv(TABLES / f"{prefix}_metrics.csv", index=False)
    bin_stats.to_csv(TABLES / f"{prefix}_bin_stats.csv", index=False)
    random_null.to_csv(TABLES / f"{prefix}_random_tau_null.csv", index=False)
    plot_results(summaries, curves, metrics, random_null, prefix)
    write_note(metrics, random_null, summaries, prefix)
    print(metrics.to_string(index=False))
    print(f"Saved CAMELS blocked cross-fitting outputs with prefix {prefix}")


if __name__ == "__main__":
    main()
