"""Build the R11 Nature-style main figure suite.

The four figures are visual arguments for the current bounded manuscript:

1. Mechanism and coordinate: storage-memory filtering makes beta(De), not a
   scalar beta, the comparable object.
2. Spectral reduction and uncertainty: CAMELS-US and CAMELS-GB show directional
   De dispersion reduction, while shuffled memory assignments do not.
3. Specificity and adversarial controls: null models, sensitivity checks and
   phase-predictor screens rule out simpler explanations without naming one
   unique nonlinear mechanism.
4. Portability and boundaries: river archives, event catalogues, landslide
   controls and synthetic reservoirs define where the coordinate works, where
   it fails, and how far the paper may safely claim.

All schematic elements are drawn with matplotlib primitives. No generative AI
images or non-reproducible visual assets are used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
FIGURES.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    }
)


COL = {
    "ink": "#1F2328",
    "muted": "#667085",
    "grid": "#E6EAF0",
    "water": "#2B7BB9",
    "water_light": "#D9ECF7",
    "gb": "#6C5CE7",
    "storage": "#2F855A",
    "storage_light": "#D8EEE0",
    "signal": "#C93434",
    "signal_light": "#F7D6D6",
    "event": "#9B4762",
    "landslide": "#D79A20",
    "noise": "#7C8794",
    "dark": "#2B303B",
    "pale": "#F3F5F7",
}


def save_pub(fig: plt.Figure, stem: str) -> None:
    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{stem}{suffix}", **kwargs)
    plt.close(fig)
    print(f"Saved reports/figures/{stem}.png/svg/pdf")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.085,
        1.105,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COL["ink"],
    )


def add_panel_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0,
        1.105,
        title,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        color=COL["ink"],
    )
    if subtitle:
        ax.text(
            0,
            1.045,
            subtitle,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=6.2,
            color=COL["muted"],
        )


def safe_log10(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[arr <= 0] = np.nan
    return np.log10(arr)


def load_bin_stats() -> pd.DataFrame:
    return pd.read_csv(TABLES / "camels_673_collapse_bin_stats.csv")


def load_cross_archive() -> pd.DataFrame:
    cross = pd.read_csv(TABLES / "camels_us_gb_replication_comparison.csv")
    us_metrics = pd.read_csv(TABLES / "camels_673_collapse_metrics.csv")
    gb_ci = pd.read_csv(TABLES / "camels_gb_v2_collapse_bootstrap_summary.csv").iloc[0]
    us_ci = us_metrics[us_metrics["axis"] == "variance_reduction"].iloc[0]
    rows: list[dict[str, object]] = []
    for row in cross.to_dict("records"):
        if row["archive"] == "CAMELS-US":
            row["ci95_low"] = float(us_ci["ci95_low"])
            row["ci95_high"] = float(us_ci["ci95_high"])
        else:
            row["ci95_low"] = float(gb_ci["ci95_low"])
            row["ci95_high"] = float(gb_ci["ci95_high"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_synthetic_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(TABLES / "synthetic_reservoir_process_control_metrics.csv")
    required = {
        "scenario_group",
        "raw_weighted_beta_variance",
        "de_weighted_beta_variance",
        "variance_reduction_vs_raw",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "Synthetic process-control metrics table has the wrong schema. "
            f"Missing columns: {sorted(missing)}. Rerun "
            "`python scripts/run_synthetic_reservoir_process_control.py`."
        )
    return metrics


def draw_storage_schematic(ax: plt.Axes) -> None:
    panel_label(ax, "a")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "Storage-memory coordinate", fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        0.86,
        "A catchment is treated as an input-output filter with a measured memory time.",
        fontsize=6.5,
        color=COL["muted"],
    )

    t = np.linspace(0, 1, 240)
    p = 0.60 + 0.055 * np.sin(42 * t) + 0.028 * np.sin(133 * t)
    q = 0.60 + 0.035 * np.sin(12 * t + 0.9) + 0.012 * np.sin(35 * t)
    ax.plot(0.05 + 0.22 * t, p, lw=1.6, color=COL["water"])
    ax.text(0.14, 0.72, "forcing\nP(t)", ha="center", color=COL["water"], fontsize=6.6)

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.38, 0.49),
            0.24,
            0.23,
            boxstyle="round,pad=0.03,rounding_size=0.035",
            fc=COL["storage_light"],
            ec=COL["storage"],
            lw=1.35,
        )
    )
    ax.text(0.50, 0.64, "storage\nkernel", ha="center", va="center", color=COL["storage"], fontsize=8.4, fontweight="bold")
    ax.text(0.50, 0.535, r"$K_\tau(s)$,  $\tau_{acf}$", ha="center", va="center", color=COL["storage"], fontsize=6.4)
    ax.annotate("", xy=(0.38, 0.60), xytext=(0.28, 0.60), arrowprops=dict(arrowstyle="->", lw=1.3, color=COL["water"]))
    ax.annotate("", xy=(0.70, 0.60), xytext=(0.62, 0.60), arrowprops=dict(arrowstyle="->", lw=1.3, color=COL["water"]))
    ax.plot(0.70 + 0.24 * t, q, lw=1.7, color=COL["signal"])
    ax.text(0.82, 0.72, "output\nQ(t)", ha="center", color=COL["signal"], fontsize=6.6)

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.35, 0.19),
            0.30,
            0.13,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            fc=COL["signal_light"],
            ec=COL["signal"],
            lw=1.1,
        )
    )
    ax.text(0.50, 0.255, r"$De=\tau_{acf}f$", ha="center", va="center", color=COL["signal"], fontsize=10, fontweight="bold")
    ax.annotate("", xy=(0.50, 0.49), xytext=(0.50, 0.32), arrowprops=dict(arrowstyle="->", lw=1.2, color=COL["signal"]))
    ax.text(0.02, 0.075, "Supported claim:", fontsize=6.4, fontweight="bold")
    ax.text(
        0.02,
        0.025,
        r"compare $\beta(De)$ for continuous storage-filtered fluxes",
        fontsize=6.4,
        color=COL["muted"],
    )


def draw_transfer_function(ax: plt.Axes) -> None:
    panel_label(ax, "b")
    de = np.logspace(-2, 2, 300)
    amp = 1 / np.sqrt(1 + (2 * np.pi * de) ** 2)
    slope_proxy = 2 * ((2 * np.pi * de) ** 2 / (1 + (2 * np.pi * de) ** 2))
    ax2 = ax.twinx()
    ax.plot(de, amp, color=COL["storage"], lw=2.1, label="filter gain")
    ax2.plot(de, slope_proxy, color=COL["signal"], lw=1.8, label="slope transition")
    ax.axvspan(0.5, 2.0, color=COL["signal"], alpha=0.08, lw=0)
    ax.axvline(1.0, color="#9AA0A6", ls="--", lw=0.8)
    ax.text(1.08, 0.76, "De=1", fontsize=6.4, color=COL["muted"])
    ax.set_xscale("log")
    ax.set_xlabel("memory-normalized frequency De")
    ax.set_ylabel("linear-reservoir gain", color=COL["storage"])
    ax2.set_ylabel("expected spectral transition", color=COL["signal"])
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 2.1)
    add_panel_title(ax, "Mechanistic prediction", "spectral change should align near f ~ 1/tau")
    ax.grid(True, which="major", color=COL["grid"], lw=0.55)


def draw_raw_to_de_lens(ax: plt.Axes, bins: pd.DataFrame) -> None:
    panel_label(ax, "c")
    raw = bins[bins["axis"] == "raw_frequency"].copy()
    de = bins[bins["axis"] == "de_normalized"].copy()
    raw_x = raw["axis_value"].to_numpy(dtype=float)
    de_x = de["axis_value"].to_numpy(dtype=float)
    raw_y = raw["beta_median"].to_numpy(dtype=float)
    de_y = de["beta_median"].to_numpy(dtype=float)
    ax.plot(raw_x, raw_y, color=COL["noise"], lw=1.6, marker="o", ms=2.7, label="raw f")
    ax.plot(de_x, de_y, color=COL["signal"], lw=1.9, marker="o", ms=2.7, label="De")
    if "beta_iqr" in raw.columns:
        ax.fill_between(raw_x, raw_y - raw["beta_iqr"] / 2, raw_y + raw["beta_iqr"] / 2, color=COL["noise"], alpha=0.10, lw=0)
        ax.fill_between(de_x, de_y - de["beta_iqr"] / 2, de_y + de["beta_iqr"] / 2, color=COL["signal"], alpha=0.10, lw=0)
    ax.axvspan(0.5, 2.0, color=COL["signal"], alpha=0.06, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("axis value: raw cycles d-1 or De")
    ax.set_ylabel("median local beta")
    add_panel_title(ax, "Empirical coordinate shift", "CAMELS-US bin medians")
    ax.legend(loc="upper left")
    ax.grid(True, which="major", color=COL["grid"], lw=0.55)


def draw_theory_outcome_matrix(ax: plt.Axes) -> None:
    panel_label(ax, "d")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = [
        ("River discharge", "flux", "measured", "collapse"),
        ("Synthetic reservoirs", "storage", "estimated", "conditional"),
        ("Landslide catalogue", "events", "assigned", "exploratory"),
        ("Earthquakes", "events", "assigned", "fail"),
    ]
    col_x = [0.03, 0.40, 0.64, 0.84]
    headers = ["object", "process", "memory", "prediction"]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.90, h, fontsize=6.3, color=COL["muted"], fontweight="bold")
    for i, row in enumerate(rows):
        y = 0.75 - i * 0.18
        fc = "#FFFFFF" if i % 2 == 0 else COL["pale"]
        ax.add_patch(patches.Rectangle((0.02, y - 0.065), 0.94, 0.12, fc=fc, ec="#E2E6EA", lw=0.5))
        for x, text in zip(col_x, row):
            color = COL["signal"] if text == "collapse" else COL["event"] if text == "fail" else COL["ink"]
            weight = "bold" if text in {"collapse", "fail", "conditional"} else "normal"
            ax.text(x, y, text, fontsize=6.4, va="center", color=color, fontweight=weight)
    ax.text(
        0.03,
        0.04,
        "Boundary is part of the theory: De should not rescue systems without continuous storage-filtered output.",
        fontsize=6.5,
        color=COL["muted"],
    )


def fig1_mechanism(bins: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 0.92], height_ratios=[1.05, 0.95], wspace=0.30, hspace=0.36)
    draw_storage_schematic(fig.add_subplot(gs[0, 0]))
    draw_transfer_function(fig.add_subplot(gs[0, 1]))
    draw_raw_to_de_lens(fig.add_subplot(gs[1, 0]), bins)
    draw_theory_outcome_matrix(fig.add_subplot(gs[1, 1]))
    save_pub(fig, "nature_r11_fig1_mechanism_spectral_coordinate")


def fig2_reduction_uncertainty(bins: pd.DataFrame, cross: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 5.25))
    gs = fig.add_gridspec(2, 2, wspace=0.33, hspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    for axis, color, label in [
        ("raw_frequency", COL["noise"], "raw frequency"),
        ("de_normalized", COL["signal"], "De normalized"),
    ]:
        sub = bins[bins["axis"] == axis].copy()
        x = sub["axis_value"].to_numpy(dtype=float)
        y = sub["beta_median"].to_numpy(dtype=float)
        ax_a.plot(x, y, color=color, lw=1.9, marker="o", ms=2.5, label=label)
        if "beta_iqr" in sub.columns:
            ax_a.fill_between(x, y - sub["beta_iqr"] / 2, y + sub["beta_iqr"] / 2, color=color, alpha=0.09, lw=0)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("axis value")
    ax_a.set_ylabel("median local beta")
    add_panel_title(ax_a, "Raw-to-De collapse view", "same local-slope metric, different physical coordinate")
    ax_a.legend(loc="upper left")
    ax_a.grid(True, color=COL["grid"], lw=0.55)

    panel_label(ax_b, "b")
    work = cross.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(work))
    vals = 100 * work["de_variance_reduction_vs_raw"].to_numpy(dtype=float)
    lows = 100 * work["ci95_low"].to_numpy(dtype=float)
    highs = 100 * work["ci95_high"].to_numpy(dtype=float)
    colors = [COL["gb"] if "GB" in a else COL["water"] for a in work["archive"]]
    ax_b.hlines(y, lows, highs, color=colors, lw=5, alpha=0.22)
    ax_b.scatter(vals, y, s=74, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for xx, yy in zip(vals, y):
        ax_b.text(xx + 1.2, yy, f"{xx:.1f}%", va="center", fontsize=7)
    ax_b.axvline(0, color="#AAB0B8", lw=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(work["archive"])
    ax_b.set_xlabel("weighted beta-variance reduction")
    add_panel_title(ax_b, "Cross-archive portability", "directional replication with archive-dependent magnitude")
    ax_b.set_xlim(-3, max(highs) + 8)
    ax_b.grid(True, axis="x", color=COL["grid"], lw=0.55)

    panel_label(ax_c, "c")
    us_null = pd.read_csv(TABLES / "camels_673_random_tau_normalization_null.csv")
    gb_null = pd.read_csv(TABLES / "camels_gb_v2_replication_random_tau_null.csv")
    observed = {
        "CAMELS-US": float(cross.loc[cross["archive"] == "CAMELS-US", "de_variance_reduction_vs_raw"].iloc[0]) * 100,
        "CAMELS-GB v2": float(cross.loc[cross["archive"] == "CAMELS-GB v2", "de_variance_reduction_vs_raw"].iloc[0]) * 100,
    }
    datasets = [
        ("US shuffled tau", us_null["variance_reduction_vs_matched_raw"].to_numpy(dtype=float) * 100, COL["water"]),
        ("GB shuffled tau", gb_null["variance_reduction_vs_raw"].to_numpy(dtype=float) * 100, COL["gb"]),
    ]
    for i, (label, vals_i, color) in enumerate(datasets):
        parts = ax_c.violinplot(vals_i[np.isfinite(vals_i)], positions=[i], widths=0.62, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)
        q = np.nanpercentile(vals_i, [5, 50, 95])
        ax_c.vlines(i, q[0], q[2], color=color, lw=3, alpha=0.55)
        ax_c.scatter(i, q[1], s=24, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax_c.scatter([0, 1], [observed["CAMELS-US"], observed["CAMELS-GB v2"]], marker="*", s=110, color=COL["signal"], edgecolor="white", linewidth=0.7, zorder=4, label="observed")
    ax_c.axhline(0, color="#AAB0B8", lw=0.8)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["US\nshuffle", "GB\nshuffle"])
    ax_c.set_ylabel("variance reduction [%]")
    add_panel_title(ax_c, "Random memory assignments fail", "empirical p = 0.0033 in both archives")
    ax_c.legend(loc="upper left")
    ax_c.grid(True, axis="y", color=COL["grid"], lw=0.55)

    panel_label(ax_d, "d")
    alt = pd.read_csv(TABLES / "camels_673_alternative_normalization_metrics.csv")
    order = [
        "tau_acf",
        "tau_acf_recession_subset",
        "tau_recession_matched",
        "area_rank_tau",
        "constant_tau",
        "bfi_rank_tau",
    ]
    labels = {
        "tau_acf": "tau_acf",
        "tau_acf_recession_subset": "tau_acf\nsubset",
        "tau_recession_matched": "recession",
        "area_rank_tau": "area\nrank",
        "constant_tau": "constant",
        "bfi_rank_tau": "BFI\nrank",
    }
    work = alt.set_index("method").loc[order].reset_index()
    vals = work["variance_reduction_vs_matched_raw"].to_numpy(dtype=float) * 100
    x = np.arange(len(work))
    for xx, vv, method in zip(x, vals, work["method"]):
        color = COL["signal"] if method.startswith("tau_acf") else COL["noise"]
        ax_d.vlines(xx, 0, vv, color=color, lw=2.2, alpha=0.70)
        ax_d.scatter(xx, vv, s=45, color=color, edgecolor="white", linewidth=0.7, zorder=3)
    ax_d.axhline(0, color="#AAB0B8", lw=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([labels[m] for m in work["method"]])
    ax_d.set_ylabel("variance reduction [%]")
    add_panel_title(ax_d, "Alternative coordinates underperform", "specificity against common scalings and proxies")
    ax_d.set_ylim(-7, 28)
    ax_d.grid(True, axis="y", color=COL["grid"], lw=0.55)

    save_pub(fig, "nature_r11_fig2_reduction_uncertainty")


def fig3_specificity_nulls() -> None:
    fig = plt.figure(figsize=(7.35, 5.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.96, 1.04], wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    nulls = pd.read_csv(TABLES / "camels_673_null_rejection_counts.csv")
    panel_label(ax_a, "a")
    labels = ["Seasonal", "AR(1)", "Water-year", "Phase"]
    y = np.arange(len(labels))
    counts = nulls["p_le_1_over_n_count"].to_numpy(dtype=float)
    total = 673
    ax_a.barh(y, total, color=COL["pale"], edgecolor="#D7DCE2", height=0.56)
    ax_a.barh(y, counts, color=[COL["signal"], COL["landslide"], COL["water"], COL["event"]], alpha=0.80, height=0.56)
    for yy, count in zip(y, counts):
        ax_a.text(count + 10, yy, f"{int(count)}/{total}", va="center", fontsize=6.4)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlim(0, 720)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("catchments rejected at p <= 1/N")
    add_panel_title(ax_a, "Adversarial null hierarchy", "each rung preserves more structure")
    ax_a.grid(True, axis="x", color=COL["grid"], lw=0.55)

    panel_label(ax_b, "b")
    sens = pd.read_csv(TABLES / "camels_673_collapse_sensitivity.csv")
    # Support both historical and current column names.
    row_col = "n_bins" if "n_bins" in sens.columns else sens.columns[0]
    col_col = "min_stations" if "min_stations" in sens.columns else sens.columns[1]
    val_col = "variance_reduction" if "variance_reduction" in sens.columns else "variance_reduction_vs_raw"
    heat = sens.pivot(index=row_col, columns=col_col, values=val_col).sort_index()
    im = ax_b.imshow(100 * heat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", origin="lower")
    ax_b.set_xticks(np.arange(len(heat.columns)))
    ax_b.set_xticklabels([str(c) for c in heat.columns])
    ax_b.set_yticks(np.arange(len(heat.index)))
    ax_b.set_yticklabels([str(i) for i in heat.index])
    ax_b.set_xlabel("minimum stations per bin")
    ax_b.set_ylabel("number of bins")
    add_panel_title(ax_b, "Collapse-score sensitivity", "positive in all tested binning/threshold settings")
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.set_label("reduction [%]")

    panel_label(ax_c, "c")
    logit = pd.read_csv(TABLES / "camels_673_phase_rejection_logit.csv").sort_values("odds_ratio_per_sd")
    y = np.arange(len(logit))
    x = logit["odds_ratio_per_sd"].to_numpy(dtype=float)
    lo = logit["ci95_low"].to_numpy(dtype=float)
    hi = logit["ci95_high"].to_numpy(dtype=float)
    colors = [COL["noise"] if q >= 0.05 else COL["signal"] for q in logit["q_bh"]]
    ax_c.hlines(y, lo, hi, color=colors, lw=2.4, alpha=0.55)
    ax_c.scatter(x, y, s=35, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax_c.axvline(1, color="#AAB0B8", ls="--", lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([p.replace("_", " ") for p in logit["predictor"]])
    ax_c.set_xscale("log")
    ax_c.set_xlabel("odds ratio per s.d.")
    add_panel_title(ax_c, "No phase-rejection predictor survives FDR", "mechanism remains bounded and hypothesis-generating")
    ax_c.grid(True, axis="x", color=COL["grid"], lw=0.55)

    panel_label(ax_d, "d")
    validation = pd.read_csv(TABLES / "variance_reduction_validation_summary.csv").iloc[0]
    cv = pd.read_csv(TABLES / "camels_manuscript_supplements_cross_validation.csv").iloc[0]
    items = [
        ("Phase-null\nrejection", float(validation["rejection_rate"]), "higher means PSD amplitude is insufficient"),
        ("DE incremental\nR2", float(cv["r2_de_incremental_over_additive"]) * 100, "near zero: not yet predictive"),
        ("Shuffle p95\nR2", float(cv["shuffled_p95_r2"]) * 100, "null reference for predictive test"),
    ]
    x_pos = np.arange(len(items))
    vals = [item[1] for item in items]
    colors = [COL["event"], COL["signal"], COL["noise"]]
    ax_d.bar(x_pos, vals, color=colors, alpha=0.75, width=0.58)
    for xx, vv, _ in zip(x_pos, vals, items):
        ax_d.text(xx, vv + 1.0, f"{vv:.2g}", ha="center", fontsize=6.4)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([item[0] for item in items])
    ax_d.set_ylabel("reported value [%]")
    add_panel_title(ax_d, "Specificity is strong; prediction still weak", "a strict claim gate keeps interpretation bounded")
    ax_d.set_ylim(-1, max(vals) + 8)
    ax_d.grid(True, axis="y", color=COL["grid"], lw=0.55)

    save_pub(fig, "nature_r11_fig3_specificity_sensitivity_nulls")


def fig4_portability_boundary(cross: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 5.35))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    us = pd.read_csv(TABLES / "camels_673_attributed_diagnostics.csv")
    gb = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv")
    eq = pd.read_csv(TABLES / "earthquake_regions_2000_20260528_summary.csv")
    landslide = pd.read_csv(TABLES / "landslide_glc_trigger_nulls_summary.csv")
    groups = [
        ("US rivers", us["mean_beta_de_window"], COL["water"]),
        ("GB rivers", gb["mean_beta_de_window"], COL["gb"]),
        ("Earthquake\ncatalogues", eq["mean_beta_de_window"], COL["event"]),
        ("Landslide\ntriggers", landslide["observed_mean_beta_de_window"], COL["landslide"]),
    ]
    for i, (label, values, color) in enumerate(groups):
        arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        parts = ax_a.violinplot(arr, positions=[i], widths=0.68, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
        q = np.nanpercentile(arr, [10, 50, 90])
        ax_a.vlines(i, q[0], q[2], color=color, lw=3.0, alpha=0.60)
        ax_a.scatter(i, q[1], color=color, edgecolor="white", linewidth=0.7, s=38, zorder=3)
    ax_a.set_xticks(range(len(groups)))
    ax_a.set_xticklabels([g[0] for g in groups])
    ax_a.set_ylabel(r"mean beta in $0.5\leq De\leq2$")
    add_panel_title(ax_a, "Cross-process boundary", "rivers separate from event catalogues; landslides remain exploratory")
    ax_a.grid(True, axis="y", color=COL["grid"], lw=0.55)

    panel_label(ax_b, "b")
    order = [
        "linear_reservoir",
        "nonlinear_convex_storage",
        "seasonal_forcing_control",
        "nonlinear_concave_storage",
    ]
    labels = ["linear", "convex\nstorage", "seasonal\nforcing", "concave\nstorage"]
    work = synthetic.set_index("scenario_group").loc[order].reset_index()
    ratio = work["de_weighted_beta_variance"] / work["raw_weighted_beta_variance"]
    vals = np.log10(ratio.to_numpy(dtype=float))
    x = np.arange(len(work))
    colors = [COL["water"], COL["signal"], COL["landslide"], COL["storage"]]
    ax_b.bar(x, vals, color=colors, alpha=0.78, width=0.62)
    ax_b.axhline(0, color="#AAB0B8", lw=0.8)
    for xx, vv, rr in zip(x, vals, ratio):
        label_y = vv + 0.07 if vv >= 0 else 0.035
        ax_b.text(xx, label_y, f"{rr:.2g}x", ha="center", va="bottom", fontsize=6.3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel(r"$\log_{10}(\mathrm{Var}_{De}/\mathrm{Var}_{raw})$")
    add_panel_title(ax_b, "Synthetic reservoirs are a falsification test", "De helps some storage processes and fails for others")
    ax_b.grid(True, axis="y", color=COL["grid"], lw=0.55)

    panel_label(ax_c, "c")
    ax_c.set_axis_off()
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    rows = [
        ("Supported", "dispersion reduction in two river archives", COL["water"]),
        ("Supported", "BFI-over-area memory association", COL["storage"]),
        ("Supported", "event catalogues fail river-like De test", COL["event"]),
        ("Partial", "reservoir models show process-specific plausibility", COL["landslide"]),
        ("Not claimed", "universal Earth-system collapse law", COL["noise"]),
        ("Not claimed", "groundwater causality from BFI alone", COL["noise"]),
    ]
    for i, (status, statement, color) in enumerate(rows):
        y = 0.88 - i * 0.135
        ax_c.add_patch(patches.Rectangle((0.02, y - 0.052), 0.94, 0.10, fc="#FFFFFF", ec="#E2E6EA", lw=0.5))
        ax_c.add_patch(patches.Circle((0.07, y), 0.026, fc=color, ec="white", lw=0.5))
        ax_c.text(0.12, y, status, fontsize=6.3, color=color, fontweight="bold", va="center")
        ax_c.text(0.34, y, statement, fontsize=6.4, color=COL["ink"], va="center")
    add_panel_title(ax_c, "Claim boundary ledger", "what the current evidence can and cannot support")

    panel_label(ax_d, "d")
    y = np.arange(len(cross))
    vals = 100 * cross["rho_tau_bfi"].to_numpy(dtype=float)
    vals_area = 100 * cross["rho_tau_log_area"].to_numpy(dtype=float)
    ax_d.scatter(vals, y + 0.13, s=70, color=COL["storage"], edgecolor="white", linewidth=0.8, label="BFI")
    ax_d.scatter(vals_area, y - 0.13, s=70, color=COL["noise"], edgecolor="white", linewidth=0.8, label="log area")
    for xx, yy in zip(vals, y + 0.13):
        ax_d.text(xx + 2.0, yy, f"{xx/100:.2f}", va="center", fontsize=6.4, color=COL["storage"])
    for xx, yy in zip(vals_area, y - 0.13):
        ax_d.text(xx + 2.0, yy, f"{xx/100:.2f}", va="center", fontsize=6.4, color=COL["noise"])
    ax_d.axvline(0, color="#AAB0B8", lw=0.8)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(cross["archive"])
    ax_d.set_xlabel(r"Spearman rho with $\tau_{acf}$ [% scale]")
    add_panel_title(ax_d, "Memory tracks storage proxy across archives", "mechanistic compatibility, not standalone causality")
    ax_d.set_xlim(-5, 92)
    ax_d.legend(loc="lower right")
    ax_d.grid(True, axis="x", color=COL["grid"], lw=0.55)

    save_pub(fig, "nature_r11_fig4_portability_boundary_process")


def write_figure_contract(cross: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    us = float(cross.loc[cross["archive"] == "CAMELS-US", "de_variance_reduction_vs_raw"].iloc[0])
    gb = float(cross.loc[cross["archive"] == "CAMELS-GB v2", "de_variance_reduction_vs_raw"].iloc[0])
    syn = synthetic.set_index("scenario_group")
    linear = float(syn.loc["linear_reservoir", "variance_reduction_vs_raw"])
    convex = float(syn.loc["nonlinear_convex_storage", "variance_reduction_vs_raw"])
    concave = float(syn.loc["nonlinear_concave_storage", "variance_reduction_vs_raw"])
    seasonal = float(syn.loc["seasonal_forcing_control", "variance_reduction_vs_raw"])
    text = f"""# R11 Main Figure Suite Contract

Date: 2026-06-02

## Core conclusion

The manuscript should argue that measured output memory defines a bounded
storage-memory spectral coordinate: De normalization reduces river spectral
dispersion in CAMELS-US and CAMELS-GB, passes specificity controls, and fails
where continuous storage-filtered output is absent or where synthetic process
structure violates the coordinate's assumptions.

## Figure jobs

| Figure | Output files | Main job |
| --- | --- | --- |
| Fig. 1 | `reports/figures/nature_r11_fig1_mechanism_spectral_coordinate.*` | Converts the story from a scalar-beta comparison into a mechanism-led beta(De) coordinate hypothesis. |
| Fig. 2 | `reports/figures/nature_r11_fig2_reduction_uncertainty.*` | Shows CAMELS-US ({us:.1%}) and CAMELS-GB ({gb:.1%}) dispersion reduction, uncertainty intervals, random-tau failure and alternative-coordinate underperformance. |
| Fig. 3 | `reports/figures/nature_r11_fig3_specificity_sensitivity_nulls.*` | Shows the null hierarchy, binning sensitivity, phase-predictor limits and strict claim discipline. |
| Fig. 4 | `reports/figures/nature_r11_fig4_portability_boundary_process.*` | Shows boundary conditions across rivers, event catalogues, landslides and synthetic reservoirs. |

## R11 synthetic process-control numbers

| Scenario | Variance-reduction interpretation |
| --- | --- |
| Linear reservoir | {linear:.1%} reduction; supports mechanistic plausibility. |
| Nonlinear convex storage | {convex:.1%} reduction; supports conditional process compatibility. |
| Nonlinear concave storage | {concave:.1%} reduction; De worsens dispersion, so the coordinate is not automatic. |
| Seasonal forcing control | {seasonal:.1%} reduction; De worsens dispersion under strong seasonal forcing. |

## Claim boundaries

- The figures support storage-compatible spectral organization and bounded
  portability, not a universal Earth-system collapse law.
- Synthetic reservoirs are artifact/process controls, not proof that real
  catchments follow one storage-discharge equation.
- LLM/model-panel outputs are advisory audit records and are not scientific
  evidence panels.
"""
    (NOTES / "r11_main_figure_suite_contract.md").write_text(text, encoding="utf-8")
    print("Saved reports/manuscript_notes/r11_main_figure_suite_contract.md")


def main() -> None:
    bins = load_bin_stats()
    cross = load_cross_archive()
    synthetic = load_synthetic_metrics()
    fig1_mechanism(bins)
    fig2_reduction_uncertainty(bins, cross)
    fig3_specificity_nulls()
    fig4_portability_boundary(cross, synthetic)
    write_figure_contract(cross, synthetic)


if __name__ == "__main__":
    main()
