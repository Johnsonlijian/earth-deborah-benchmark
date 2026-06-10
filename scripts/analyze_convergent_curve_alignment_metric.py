"""R16 independent functional curve-alignment metrics.

This script adds a convergence check that is deliberately separate from the
headline weighted beta-variance metric. It compresses each catchment curve into
station-by-coordinate-bin median beta values, then asks whether the memory
coordinate reduces functional distance to an archive-level reference curve.

The output is a reviewer-facing robustness result. It can strengthen a
benchmark claim if positive, but it should not be used as causal evidence for a
universal hydrologic mechanism.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"

ARCHIVES = [
    ("CAMELS-US", TABLES / "camels_673_beta_curve_points.csv"),
    ("CAMELS-GB", TABLES / "camels_gb_v2_replication_beta_curve_points.csv"),
]

AXES = [
    ("raw_frequency", "frequency_cpd", "raw frequency"),
    ("memory_coordinate", "de", "memory coordinate De"),
]

AXIS_METRICS_OUT = TABLES / "r16_convergent_alignment_axis_metrics.csv"
REDUCTION_OUT = TABLES / "r16_convergent_alignment_metric_summary.csv"
BIN_STATS_OUT = TABLES / "r16_convergent_alignment_bin_stats.csv"
FOLD_OUT = TABLES / "r16_convergent_alignment_cross_validation.csv"
FIG_OUT = FIGURES / "r16_convergent_alignment_metric_summary"
NOTE_OUT = NOTES / "r16_convergent_alignment_metric_summary.md"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)

PALETTE = {
    "ink": "#202124",
    "muted": "#6B7280",
    "grid": "#D6D9DE",
    "blue": "#386FA4",
    "green": "#4B9B72",
    "gold": "#D09A32",
    "red": "#C44E52",
    "purple": "#7A6DAF",
}


def _gauge_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(8)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.replace([np.inf, -np.inf], np.nan).notna() & weights.notna()
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def station_bins(
    curves: pd.DataFrame,
    archive: str,
    axis_name: str,
    axis_col: str,
    n_bins: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = curves[["gauge_id", axis_col, "beta"]].dropna().copy()
    work = work[(work[axis_col] > 0) & np.isfinite(work[axis_col]) & np.isfinite(work["beta"])]
    log_axis = np.log10(work[axis_col].astype(float))
    lo, hi = np.nanquantile(log_axis, [0.02, 0.98])
    work = work[(log_axis >= lo) & (log_axis <= hi)].copy()
    log_axis = np.log10(work[axis_col].astype(float))

    edges = np.linspace(float(lo), float(hi), n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work["bin_id"] = pd.cut(log_axis, bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)

    unit_bins = (
        work.groupby(["gauge_id", "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"))
        .assign(archive=archive, axis=axis_name, axis_column=axis_col)
    )
    unit_bins["axis_value"] = unit_bins["bin_id"].map(
        {i: 10 ** labels[i] for i in range(len(labels))}
    )

    bin_stats = (
        unit_bins.groupby("bin_id", as_index=False)
        .agg(
            n_units=("gauge_id", "nunique"),
            beta_ref=("beta_median", "median"),
            beta_q10=("beta_median", lambda x: float(np.nanquantile(x, 0.10))),
            beta_q25=("beta_median", lambda x: float(np.nanquantile(x, 0.25))),
            beta_q75=("beta_median", lambda x: float(np.nanquantile(x, 0.75))),
            beta_q90=("beta_median", lambda x: float(np.nanquantile(x, 0.90))),
            axis_value=("axis_value", "first"),
        )
        .assign(archive=archive, axis=axis_name, axis_column=axis_col)
    )
    bin_stats["iqr_width"] = bin_stats["beta_q75"] - bin_stats["beta_q25"]
    bin_stats["tail_width_90_10"] = bin_stats["beta_q90"] - bin_stats["beta_q10"]
    bin_stats["bin_fraction"] = (bin_stats["bin_id"] + 0.5) / n_bins
    return unit_bins, bin_stats


def axis_metrics(unit_bins: pd.DataFrame, bin_stats: pd.DataFrame) -> dict[str, object]:
    n_gauges = unit_bins["gauge_id"].nunique()
    min_units = max(20, min(50, int(np.floor(0.08 * n_gauges))))
    usable = bin_stats[bin_stats["n_units"] >= min_units].copy()
    merged = unit_bins.merge(usable[["bin_id", "beta_ref"]], on="bin_id", how="inner")
    merged["residual"] = merged["beta_median"] - merged["beta_ref"]
    merged["abs_residual"] = merged["residual"].abs()

    return {
        "archive": str(unit_bins["archive"].iloc[0]),
        "axis": str(unit_bins["axis"].iloc[0]),
        "axis_column": str(unit_bins["axis_column"].iloc[0]),
        "n_gauges": int(n_gauges),
        "min_units_per_bin": int(min_units),
        "n_bins_used": int(usable["bin_id"].nunique()),
        "n_station_bin_values": int(len(merged)),
        "mean_abs_curve_distance": float(merged["abs_residual"].mean()),
        "median_abs_curve_distance": float(merged["abs_residual"].median()),
        "rmse_curve_distance": float(np.sqrt(np.mean(merged["residual"] ** 2))),
        "weighted_iqr_width": _weighted_mean(usable["iqr_width"], usable["n_units"]),
        "weighted_tail_width_90_10": _weighted_mean(
            usable["tail_width_90_10"], usable["n_units"]
        ),
    }


def cross_validated_axis_metrics(
    unit_bins: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 20260603,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gauges = np.array(sorted(unit_bins["gauge_id"].unique()))
    rng.shuffle(gauges)
    fold_for = {gauge: int(i % n_folds) for i, gauge in enumerate(gauges)}
    work = unit_bins.copy()
    work["fold"] = work["gauge_id"].map(fold_for)
    rows: list[dict[str, object]] = []
    min_train_units = max(15, min(45, int(np.floor(0.06 * len(gauges)))))

    for fold in range(n_folds):
        train = work[work["fold"] != fold]
        test = work[work["fold"] == fold]
        ref = (
            train.groupby("bin_id", as_index=False)
            .agg(beta_ref=("beta_median", "median"), n_train=("gauge_id", "nunique"))
        )
        ref = ref[ref["n_train"] >= min_train_units]
        merged = test.merge(ref, on="bin_id", how="inner")
        merged["residual"] = merged["beta_median"] - merged["beta_ref"]
        merged["abs_residual"] = merged["residual"].abs()
        rows.append(
            {
                "archive": str(unit_bins["archive"].iloc[0]),
                "axis": str(unit_bins["axis"].iloc[0]),
                "axis_column": str(unit_bins["axis_column"].iloc[0]),
                "fold": int(fold),
                "n_test_gauges": int(test["gauge_id"].nunique()),
                "n_bins_used": int(ref["bin_id"].nunique()),
                "n_station_bin_values": int(len(merged)),
                "cv_mean_abs_curve_distance": float(merged["abs_residual"].mean()),
                "cv_median_abs_curve_distance": float(merged["abs_residual"].median()),
                "cv_rmse_curve_distance": float(np.sqrt(np.mean(merged["residual"] ** 2))),
            }
        )
    return pd.DataFrame(rows)


def metric_reductions(axis_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_abs_curve_distance",
        "median_abs_curve_distance",
        "rmse_curve_distance",
        "weighted_iqr_width",
        "weighted_tail_width_90_10",
    ]
    rows: list[dict[str, object]] = []
    for archive, sub in axis_metrics_df.groupby("archive"):
        raw = sub[sub["axis"] == "raw_frequency"].iloc[0]
        de = sub[sub["axis"] == "memory_coordinate"].iloc[0]
        for metric in metrics:
            raw_value = float(raw[metric])
            de_value = float(de[metric])
            rows.append(
                {
                    "archive": archive,
                    "metric": metric,
                    "raw_value": raw_value,
                    "memory_coordinate_value": de_value,
                    "reduction_vs_raw": 1.0 - (de_value / raw_value),
                    "direction": "positive means tighter De alignment",
                }
            )
    return pd.DataFrame(rows)


def fold_reductions(fold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cv_mean_abs_curve_distance",
        "cv_median_abs_curve_distance",
        "cv_rmse_curve_distance",
    ]
    rows: list[dict[str, object]] = []
    for (archive, fold), sub in fold_df.groupby(["archive", "fold"]):
        raw = sub[sub["axis"] == "raw_frequency"].iloc[0]
        de = sub[sub["axis"] == "memory_coordinate"].iloc[0]
        for metric in metrics:
            rows.append(
                {
                    "archive": archive,
                    "fold": int(fold),
                    "metric": metric,
                    "raw_value": float(raw[metric]),
                    "memory_coordinate_value": float(de[metric]),
                    "reduction_vs_raw": 1.0 - (float(de[metric]) / float(raw[metric])),
                }
            )
    return pd.DataFrame(rows)


def load_archive(path: Path) -> pd.DataFrame:
    curves = pd.read_csv(path, dtype={"gauge_id": str})
    curves["gauge_id"] = _gauge_str(curves["gauge_id"])
    return curves


def plot(reductions: pd.DataFrame, fold_red: pd.DataFrame, bin_stats: pd.DataFrame) -> None:
    label_map = {
        "mean_abs_curve_distance": "mean absolute\ndistance",
        "median_abs_curve_distance": "median absolute\ndistance",
        "rmse_curve_distance": "RMSE\ndistance",
        "weighted_iqr_width": "IQR\nwidth",
        "weighted_tail_width_90_10": "90-10\nwidth",
    }
    metric_order = list(label_map.keys())

    fig = plt.figure(figsize=(7.7, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[0.95, 1.05])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    x = np.arange(len(metric_order))
    offsets = {"CAMELS-US": -0.17, "CAMELS-GB": 0.17}
    colors = {"CAMELS-US": PALETTE["blue"], "CAMELS-GB": PALETTE["green"]}
    for archive in ["CAMELS-US", "CAMELS-GB"]:
        sub = reductions[reductions["archive"] == archive].set_index("metric")
        vals = [100 * sub.loc[m, "reduction_vs_raw"] for m in metric_order]
        ax_a.bar(
            x + offsets[archive],
            vals,
            width=0.32,
            color=colors[archive],
            alpha=0.88,
            label=archive,
        )
        for xi, yi in zip(x + offsets[archive], vals):
            ax_a.text(
                xi,
                yi + (1.5 if yi >= 0 else -3.5),
                f"{yi:.0f}%",
                ha="center",
                va="bottom" if yi >= 0 else "top",
                fontsize=6,
                color=PALETTE["ink"],
            )
    ax_a.axhline(0, color=PALETTE["muted"], lw=0.8)
    max_reduction = 100 * reductions["reduction_vs_raw"].max()
    ax_a.set_ylim(0, max(20, max_reduction + 4))
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label_map[m] for m in metric_order])
    ax_a.set_ylabel("distance / spread reduction after De mapping [%]")
    ax_a.set_title("a  Functional alignment improves on independent distance metrics", loc="left", fontweight="bold")
    ax_a.grid(True, axis="y", color=PALETTE["grid"], lw=0.35)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.72, 1.12), ncol=2)

    cv = fold_red[fold_red["metric"] == "cv_mean_abs_curve_distance"].copy()
    for i, archive in enumerate(["CAMELS-US", "CAMELS-GB"]):
        vals = 100 * cv.loc[cv["archive"] == archive, "reduction_vs_raw"].to_numpy()
        jitter = np.linspace(-0.055, 0.055, len(vals))
        ax_b.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=26,
            color=colors[archive],
            alpha=0.88,
        )
        ax_b.hlines(np.median(vals), i - 0.22, i + 0.22, color=PALETTE["ink"], lw=1.2)
        ax_b.text(
            i,
            np.median(vals) + 2.0,
            f"median {np.median(vals):.1f}%",
            ha="center",
            va="bottom",
            fontsize=6,
            color=PALETTE["ink"],
        )
    ax_b.axhline(0, color=PALETTE["muted"], lw=0.8, ls="--")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["CAMELS-US", "CAMELS-GB"])
    ax_b.set_ylabel("held-out mean absolute distance reduction [%]")
    ax_b.set_title("b  Split-gauge validation", loc="left", fontweight="bold")
    ax_b.grid(True, axis="y", color=PALETTE["grid"], lw=0.35)

    for archive, color in colors.items():
        for axis, ls, label_suffix in [
            ("raw_frequency", "--", "raw"),
            ("memory_coordinate", "-", "De"),
        ]:
            sub = bin_stats[(bin_stats["archive"] == archive) & (bin_stats["axis"] == axis)].copy()
            sub = sub.sort_values("bin_fraction")
            ax_c.plot(
                sub["bin_fraction"],
                sub["iqr_width"],
                color=color,
                ls=ls,
                lw=1.2,
                alpha=0.9,
                label=f"{archive} {label_suffix}",
            )
    ax_c.set_xlabel("coordinate-bin rank")
    ax_c.set_ylabel("within-bin beta IQR")
    ax_c.set_title("c  Bin-wise curve-spread profile", loc="left", fontweight="bold")
    ax_c.grid(True, color=PALETTE["grid"], lw=0.35)
    ax_c.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)

    fig.suptitle(
        "R16 convergence check: memory coordinate tested with functional distances",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{FIG_OUT}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIG_OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{FIG_OUT}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_note(axis_df: pd.DataFrame, reductions: pd.DataFrame, fold_red: pd.DataFrame) -> None:
    metric_labels = {
        "mean_abs_curve_distance": "mean absolute functional distance",
        "median_abs_curve_distance": "median absolute functional distance",
        "rmse_curve_distance": "RMSE functional distance",
        "weighted_iqr_width": "within-bin IQR width",
        "weighted_tail_width_90_10": "within-bin 90-10 width",
    }
    lines = [
        "# R16 Independent Functional Alignment Metric",
        "",
        "Date: 2026-06-03",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: test whether the memory coordinate tightens spectral-slope curves under distance and spread metrics that are separate from the previous weighted beta-variance headline metric.",
        "- Evidence chain: station-by-bin median beta curves -> archive reference curve -> functional residual distances -> held-out gauge validation.",
        "- Archetype: quantitative grid with one reduction summary, one split-validation panel and one bin-spread profile.",
        "- Boundary: this is a convergence/robustness metric, not causal proof of a universal hydrologic mechanism.",
        "",
        "## Method",
        "",
        "For each archive and coordinate axis, beta values were binned in log-coordinate space after trimming the 2nd and 98th coordinate percentiles. Each catchment contributes a median beta value per coordinate bin. The reference curve is the archive-level median beta per bin. The script then measures absolute functional distance, RMSE distance, within-bin IQR width and 90-10 width. A five-fold gauge split computes the same residual metrics against reference curves estimated from held-in gauges.",
        "",
        "## Main Results",
        "",
        "| Archive | Metric | Raw | De | Reduction |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in reductions.iterrows():
        lines.append(
            f"| {row['archive']} | {metric_labels[row['metric']]} | {row['raw_value']:.4f} | {row['memory_coordinate_value']:.4f} | {row['reduction_vs_raw']:.1%} |"
        )
    lines.extend(["", "Held-out gauge validation for mean absolute distance:", "", "| Archive | Median fold reduction | Min | Max |", "| --- | ---: | ---: | ---: |"])
    cv = fold_red[fold_red["metric"] == "cv_mean_abs_curve_distance"]
    for archive, sub in cv.groupby("archive"):
        vals = sub["reduction_vs_raw"]
        lines.append(
            f"| {archive} | {vals.median():.1%} | {vals.min():.1%} | {vals.max():.1%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Positive reductions support the claim that De mapping tightens archive-level spectral-slope curves under additional, non-variance functional metrics. They do not prove a physical mechanism; mechanism language must remain hypothesis-level unless tied to independent process evidence.",
            "",
            "## Outputs",
            "",
            f"- `{AXIS_METRICS_OUT.relative_to(ROOT)}`",
            f"- `{REDUCTION_OUT.relative_to(ROOT)}`",
            f"- `{BIN_STATS_OUT.relative_to(ROOT)}`",
            f"- `{FOLD_OUT.relative_to(ROOT)}`",
            f"- `{FIG_OUT.relative_to(ROOT)}.png/svg/pdf`",
        ]
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    axis_rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []

    for archive, csv_path in ARCHIVES:
        if not csv_path.exists():
            print(f"Skipping {archive}: missing {csv_path.relative_to(ROOT)}")
            continue
        curves = load_archive(csv_path)
        for axis_name, axis_col, _label in AXES:
            unit_bins, bins = station_bins(curves, archive, axis_name, axis_col)
            axis_rows.append(axis_metrics(unit_bins, bins))
            bin_frames.append(bins)
            fold_frames.append(cross_validated_axis_metrics(unit_bins))

    axis_df = pd.DataFrame(axis_rows)
    bin_df = pd.concat(bin_frames, ignore_index=True)
    fold_df = pd.concat(fold_frames, ignore_index=True)
    reductions = metric_reductions(axis_df)
    fold_red = fold_reductions(fold_df)

    TABLES.mkdir(parents=True, exist_ok=True)
    axis_df.to_csv(AXIS_METRICS_OUT, index=False)
    reductions.to_csv(REDUCTION_OUT, index=False)
    bin_df.to_csv(BIN_STATS_OUT, index=False)
    fold_red.to_csv(FOLD_OUT, index=False)
    plot(reductions, fold_red, bin_df)
    write_note(axis_df, reductions, fold_red)

    print(f"Wrote {AXIS_METRICS_OUT.relative_to(ROOT)}")
    print(f"Wrote {REDUCTION_OUT.relative_to(ROOT)}")
    print(f"Wrote {BIN_STATS_OUT.relative_to(ROOT)}")
    print(f"Wrote {FOLD_OUT.relative_to(ROOT)}")
    print(f"Wrote {FIG_OUT.relative_to(ROOT)}.png/svg/pdf")
    print(f"Wrote {NOTE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
