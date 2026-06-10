"""Build the R35 five-archive Fig. 2 and its source-data tables.

This figure replaces the older two-temperate-archive Fig. 2.  It uses only
already-derived, submission-facing evidence tables:

* R26 four-archive gauge-bootstrap uncertainty
* R27 CAMELS-DK lowland stress-test metrics
* shuffled-memory null draws for each archive
* R27 storage-state coordinate-sensitivity checks

The goal is to put portability and boundary evidence in the same main figure
without implying that CAMELS-DK is a fifth positive replication.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT_FIGURES = ROOT / "reports" / "figures"
MAIN_FIGURES = ROOT / "figures"
SOURCE_DATA = ROOT / "source_data"

OUT_STEM = "fig2_five_archive_alignment_boundary"

ARCHIVE_ORDER = [
    "CAMELS-US",
    "CAMELS-GB v2",
    "CAMELS-BR v1.2",
    "CAMELS-AUS v2",
    "CAMELS-DK lowland",
]

RANDOM_TAU_FILES = {
    "CAMELS-US": TABLES / "camels_673_random_tau_normalization_null.csv",
    "CAMELS-GB v2": TABLES / "camels_gb_v2_replication_random_tau_null.csv",
    "CAMELS-BR v1.2": TABLES / "r23_camels_br_third_archive_random_tau_null.csv",
    "CAMELS-AUS v2": TABLES / "r25_camels_aus_fourth_archive_random_tau_null.csv",
    "CAMELS-DK lowland": TABLES
    / "r27_camels_dk_groundwater_storage_validation_random_tau_null.csv",
}

PALETTE = {
    "positive": "#2166ac",
    "weak": "#67a9cf",
    "boundary": "#b2182b",
    "null": "#f4a582",
    "grid": "#d7d7d7",
    "text": "#232323",
    "muted": "#6d6d6d",
    "panel_bg": "#f8f8f6",
    "green": "#1b7837",
}


def pct(x: float) -> float:
    return 100.0 * float(x)


def random_reduction_column(df: pd.DataFrame) -> str:
    for col in (
        "variance_reduction_vs_matched_raw",
        "variance_reduction_vs_raw",
        "variance_reduction_fraction",
    ):
        if col in df.columns:
            return col
    raise KeyError(f"No random-tau reduction column in {df.columns.tolist()}")


def load_archive_reductions() -> pd.DataFrame:
    r26 = pd.read_csv(TABLES / "r26_four_archive_uncertainty_summary.csv")
    r26 = r26.rename(
        columns={
            "observed_reduction_fraction": "observed_fraction",
            "bootstrap_median_reduction_fraction": "bootstrap_median_fraction",
            "bootstrap_ci_low_fraction": "ci_low_fraction",
            "bootstrap_ci_high_fraction": "ci_high_fraction",
            "bootstrap_probability_positive": "probability_positive",
        }
    )
    r26 = r26[
        [
            "archive",
            "n_gauges",
            "observed_fraction",
            "bootstrap_median_fraction",
            "ci_low_fraction",
            "ci_high_fraction",
            "probability_positive",
        ]
    ]
    r26["claim_class"] = np.where(
        r26["ci_low_fraction"].astype(float) > 0,
        "positive cross-archive alignment",
        "weak positive archive-level effect",
    )

    dk = pd.read_csv(TABLES / "r27_camels_dk_groundwater_storage_validation_archive_metrics.csv").iloc[0]
    dk_subsample = pd.read_csv(TABLES / "r27_camels_dk_groundwater_storage_validation_subsample_stability.csv")
    dk_probability_positive = float((dk_subsample["variance_reduction_fraction"].astype(float) > 0).mean())
    dk_row = pd.DataFrame(
        [
            {
                "archive": "CAMELS-DK lowland",
                "n_gauges": int(dk["n_gauges"]),
                "observed_fraction": float(dk["variance_reduction_fraction"]),
                "bootstrap_median_fraction": float(dk["subsample_median_reduction_fraction"]),
                "ci_low_fraction": float(dk["subsample_ci95_low_reduction_fraction"]),
                "ci_high_fraction": float(dk["subsample_ci95_high_reduction_fraction"]),
                "probability_positive": dk_probability_positive,
                "claim_class": "lowland boundary; not a positive replication",
            }
        ]
    )
    out = pd.concat([r26, dk_row], ignore_index=True)
    out["archive"] = pd.Categorical(out["archive"], ARCHIVE_ORDER, ordered=True)
    out = out.sort_values("archive").reset_index(drop=True)
    return out


def load_random_tau_summary(observed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    obs_map = observed.set_index("archive")["observed_fraction"].to_dict()
    for archive, path in RANDOM_TAU_FILES.items():
        df = pd.read_csv(path)
        col = random_reduction_column(df)
        values = df[col].dropna().astype(float)
        obs = float(obs_map[archive])
        rows.append(
            {
                "archive": archive,
                "n_draws": int(values.size),
                "observed_fraction": obs,
                "random_tau_median_fraction": float(values.median()),
                "random_tau_ci_low_fraction": float(values.quantile(0.05)),
                "random_tau_ci_high_fraction": float(values.quantile(0.95)),
                "random_tau_max_fraction": float(values.max()),
                "empirical_p_random_ge_observed": float(((values >= obs).sum() + 1) / (values.size + 1)),
                "random_tau_column": col,
            }
        )
    out = pd.DataFrame(rows)
    out["archive"] = pd.Categorical(out["archive"], ARCHIVE_ORDER, ordered=True)
    return out.sort_values("archive").reset_index(drop=True)


def load_coordinate_sensitivity() -> pd.DataFrame:
    sens = pd.read_csv(TABLES / "r27_camels_dk_groundwater_storage_validation_coordinate_sensitivity.csv")
    sens = sens.rename(
        columns={
            "variance_reduction_fraction": "reduction_fraction",
            "coordinate": "coordinate_key",
        }
    )
    keep_order = [
        "observed discharge tau",
        "DK-model soil-water tau",
        "DK-model phreatic-depth tau",
        "DK-model runoff tau",
        "DK-model deep-groundwater-head tau",
    ]
    sens["label"] = pd.Categorical(sens["label"], keep_order, ordered=True)
    return sens.sort_values("label").reset_index(drop=True)


def write_source_tables(
    observed: pd.DataFrame, random_tau: pd.DataFrame, sensitivity: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)

    observed_out = observed.copy()
    for col in [
        "observed_fraction",
        "bootstrap_median_fraction",
        "ci_low_fraction",
        "ci_high_fraction",
        "probability_positive",
    ]:
        observed_out[col.replace("_fraction", "_percent")] = observed_out[col].astype(float) * 100

    random_out = random_tau.copy()
    for col in [
        "observed_fraction",
        "random_tau_median_fraction",
        "random_tau_ci_low_fraction",
        "random_tau_ci_high_fraction",
        "random_tau_max_fraction",
    ]:
        random_out[col.replace("_fraction", "_percent")] = random_out[col].astype(float) * 100

    sensitivity_out = sensitivity.copy()
    sensitivity_out["reduction_percent"] = sensitivity_out["reduction_fraction"].astype(float) * 100

    ledger = pd.DataFrame(
        [
            {
                "evidence_layer": "four primary archives",
                "result": "US, GB and Brazil have fully positive bootstrap intervals; Australia is weak-positive with a lower CI crossing zero",
                "claim_allowed": "bounded portability of a diagnostic coordinate",
                "claim_not_allowed": "universal collapse constant",
            },
            {
                "evidence_layer": "CAMELS-DK lowland archive",
                "result": "observed-discharge De increases beta variance by about 5 percent",
                "claim_allowed": "lowland boundary case",
                "claim_not_allowed": "fifth positive replication",
            },
            {
                "evidence_layer": "shuffled-memory null",
                "result": "random tau assignments are negative in all five archives",
                "claim_allowed": "measured-memory specificity under this null",
                "claim_not_allowed": "excess over matched AR(1)",
            },
            {
                "evidence_layer": "DK storage-state coordinates",
                "result": "soil-water memory is small-positive; phreatic, runoff and deep-groundwater memories are negative",
                "claim_allowed": "storage-compatible boundary evidence",
                "claim_not_allowed": "monotone groundwater-memory causality",
            },
        ]
    )

    fig2_source = pd.concat(
        [
            observed_out.assign(panel="a_archive_observed_bootstrap"),
            random_out.assign(panel="b_random_tau_null"),
            sensitivity_out.assign(panel="c_dk_coordinate_sensitivity"),
            ledger.assign(panel="d_evidence_ledger"),
        ],
        ignore_index=True,
        sort=False,
    )
    fig2_source.to_csv(SOURCE_DATA / "r35_fig2_five_archive_alignment_source.csv", index=False)
    observed_out.to_csv(TABLES / "r35_five_archive_observed_bootstrap_summary.csv", index=False)
    random_out.to_csv(TABLES / "r35_five_archive_random_tau_summary.csv", index=False)
    sensitivity_out.to_csv(TABLES / "r35_dk_coordinate_sensitivity_fig2.csv", index=False)
    ledger.to_csv(TABLES / "r35_fig2_evidence_ledger.csv", index=False)

    filter_rows = [
        {
            "archive": "CAMELS-US",
            "analysis_ready_gauges": 673,
            "record_length_rule": "daily discharge archive records accepted by the CAMELS-US pipeline; full-archive spectral-window filters applied",
            "missingness_rule": "finite anomaly series after day-of-year de-seasonalization and project gap filters",
            "gap_interpolation_rule": "short gaps only; exact accepted gauges recorded in derived gauge-summary/source-data tables",
            "spectral_window_rule": "finite tau_acf, PSD and local-slope windows; at least 50 catchments per main log-frequency bin",
            "source_table": "camels_673_collapse_metrics.csv; camels_673_* source data",
        },
        {
            "archive": "CAMELS-GB v2",
            "analysis_ready_gauges": 664,
            "record_length_rule": "daily discharge archive records accepted by the CAMELS-GB v2 replication pipeline",
            "missingness_rule": "archive-specific filter audit retained in camels_gb_v2_replication_filter_audit.csv",
            "gap_interpolation_rule": "short gaps only under the replication filter audit",
            "spectral_window_rule": "finite tau_acf, PSD and local-slope windows; at least 50 catchments per main log-frequency bin",
            "source_table": "camels_gb_v2_replication_filter_audit.csv; camels_gb_v2_replication_collapse_metrics.csv",
        },
        {
            "archive": "CAMELS-BR v1.2",
            "analysis_ready_gauges": 893,
            "record_length_rule": "selected streamflow records require at least 20 years",
            "missingness_rule": "no more than 30 percent missing values",
            "gap_interpolation_rule": "no gap requiring more than 30 days of linear interpolation",
            "spectral_window_rule": "24-bin weighted beta-variance metric with accepted finite local-slope windows",
            "source_table": "r23_camels_br_third_archive_gauge_summary.csv; r23_camels_br_third_archive_archive_metrics.csv",
        },
        {
            "archive": "CAMELS-AUS v2",
            "analysis_ready_gauges": 560,
            "record_length_rule": "stations trimmed to first and last valid daily observation; at least 20 years required",
            "missingness_rule": "negative sentinels treated as missing; no more than 30 percent missing values",
            "gap_interpolation_rule": "no gap requiring more than 30 days of linear interpolation",
            "spectral_window_rule": "24-bin weighted beta-variance metric with accepted finite local-slope windows",
            "source_table": "r25_camels_aus_fourth_archive_gauge_summary.csv; r25_camels_aus_fourth_archive_archive_metrics.csv",
        },
        {
            "archive": "CAMELS-DK gauged lowland",
            "analysis_ready_gauges": 220,
            "record_length_rule": "public gauged-catchment dynamics from the 304 observed-runoff CAMELS-DK catchments",
            "missingness_rule": "same daily anomaly, interpolation, coverage and spectral pipeline used for the other archives",
            "gap_interpolation_rule": "short gaps only under the common large-sample archive filters",
            "spectral_window_rule": "finite observed-discharge tau_acf, PSD and local-slope windows; storage-state coordinates require finite state tau",
            "source_table": "r27_camels_dk_groundwater_storage_validation_gauge_summary.csv; r27_* source data",
        },
    ]
    filter_audit = pd.DataFrame(filter_rows)
    filter_audit.to_csv(TABLES / "r35_archive_filter_audit_summary.csv", index=False)
    filter_audit.to_csv(SOURCE_DATA / "r35_archive_filter_audit_summary.csv", index=False)

    return fig2_source, filter_audit


def style_axes(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9e9e9e")
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.text(
        -0.075,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold", pad=7)


def draw_figure(observed: pd.DataFrame, random_tau: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.titlecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(12.8, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel a: observed/boundary archive-level reductions.
    y = np.arange(len(observed))
    med = observed["bootstrap_median_fraction"].astype(float).map(pct).to_numpy()
    lo = observed["ci_low_fraction"].astype(float).map(pct).to_numpy()
    hi = observed["ci_high_fraction"].astype(float).map(pct).to_numpy()
    colors = []
    for _, row in observed.iterrows():
        if row["archive"] == "CAMELS-DK lowland":
            colors.append(PALETTE["boundary"])
        elif float(row["ci_low_fraction"]) <= 0:
            colors.append(PALETTE["weak"])
        else:
            colors.append(PALETTE["positive"])
    ax_a.axvline(0, color="#444444", linewidth=1.1)
    ax_a.barh(y, med, color=colors, height=0.55, zorder=3)
    ax_a.errorbar(
        med,
        y,
        xerr=np.vstack([med - lo, hi - med]),
        fmt="none",
        ecolor="#2b2b2b",
        capsize=3,
        linewidth=1.1,
        zorder=4,
    )
    for idx, row in observed.iterrows():
        x = med[idx]
        label = f"n={int(row['n_gauges'])}; P+={float(row['probability_positive']):.2f}"
        ax_a.text(max(x, 0) + 1.2, idx, label, va="center", fontsize=8.2, color=PALETTE["muted"])
    ax_a.set_yticks(y, observed["archive"].astype(str))
    ax_a.invert_yaxis()
    ax_a.set_xlim(-14, 33)
    ax_a.set_xlabel("Weighted beta-variance reduction (%)")
    style_axes(ax_a, "a", "Observed De alignment: four primary archives plus DK boundary")

    # Panel b: measured memory versus shuffled-memory null.
    y2 = np.arange(len(random_tau))
    obs = random_tau["observed_fraction"].astype(float).map(pct).to_numpy()
    null_med = random_tau["random_tau_median_fraction"].astype(float).map(pct).to_numpy()
    null_lo = random_tau["random_tau_ci_low_fraction"].astype(float).map(pct).to_numpy()
    null_hi = random_tau["random_tau_ci_high_fraction"].astype(float).map(pct).to_numpy()
    ax_b.axvline(0, color="#444444", linewidth=1.1)
    ax_b.errorbar(
        null_med,
        y2 + 0.16,
        xerr=np.vstack([null_med - null_lo, null_hi - null_med]),
        fmt="o",
        color=PALETTE["null"],
        ecolor=PALETTE["null"],
        markersize=5,
        capsize=3,
        label="random tau, median and 5-95%",
        zorder=3,
    )
    ax_b.scatter(obs, y2 - 0.16, s=64, marker="D", color=PALETTE["positive"], label="measured tau", zorder=4)
    for idx, row in random_tau.iterrows():
        ax_b.text(
            -47,
            idx,
            f"p={float(row['empirical_p_random_ge_observed']):.3f}",
            va="center",
            fontsize=8.2,
            color=PALETTE["muted"],
        )
    ax_b.set_yticks(y2, random_tau["archive"].astype(str))
    ax_b.invert_yaxis()
    ax_b.set_xlim(-50, 32)
    ax_b.set_xlabel("Reduction under measured or shuffled tau (%)")
    ax_b.text(0.58, 0.07, "blue diamond: measured tau", transform=ax_b.transAxes, fontsize=8.2, color=PALETTE["positive"])
    ax_b.text(0.58, 0.015, "orange dot: random tau", transform=ax_b.transAxes, fontsize=8.2, color=PALETTE["null"])
    style_axes(ax_b, "b", "Shuffled tau nulls fail in all five archives")

    # Panel c: DK coordinate sensitivity.
    labels = sensitivity["label"].astype(str).str.replace("DK-model ", "", regex=False).str.replace(" tau", "", regex=False)
    vals = sensitivity["reduction_fraction"].astype(float).map(pct).to_numpy()
    y3 = np.arange(len(sensitivity))
    ccols = [
        PALETTE["boundary"] if v < 0 else PALETTE["green"]
        for v in vals
    ]
    ax_c.axvline(0, color="#444444", linewidth=1.1)
    ax_c.barh(y3, vals, color=ccols, height=0.55, zorder=3)
    for idx, value in enumerate(vals):
        ha = "left" if value >= 0 else "right"
        off = 0.9 if value >= 0 else -0.9
        ax_c.text(value + off, idx, f"{value:.1f}%", va="center", ha=ha, fontsize=8.6, color=PALETTE["text"])
    ax_c.set_yticks(y3, labels)
    ax_c.invert_yaxis()
    ax_c.set_xlim(-43, 10)
    ax_c.set_xlabel("DK coordinate-sensitivity reduction (%)")
    style_axes(ax_c, "c", "Lowland storage coordinates expose the boundary")

    # Panel d: compact evidence ledger.
    ax_d.axis("off")
    ax_d.set_facecolor(PALETTE["panel_bg"])
    ax_d.text(0.0, 1.03, "d", fontsize=13, fontweight="bold", transform=ax_d.transAxes)
    ax_d.text(
        0.08,
        1.02,
        "Interpretation boundary for Fig. 2",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["text"],
        transform=ax_d.transAxes,
        va="top",
    )
    ledger_rows = [
        ("Portability", "US/GB/BR positive; AUS weak-positive", "benchmark claim"),
        ("Boundary", "DK observed-discharge De is negative", "no universal collapse"),
        ("Specificity", "random tau is negative in all archives", "not arbitrary scaling"),
        ("Mechanism", "soil-water small-positive; groundwater states negative", "no direct causality"),
        ("Null floor", "matched AR(1) handled in Fig. 4", "separates artifact risk"),
    ]
    y0 = 0.86
    for idx, (tag, result, boundary) in enumerate(ledger_rows):
        yloc = y0 - idx * 0.17
        color = PALETTE["positive"] if idx in (0, 2) else PALETTE["boundary"] if idx in (1, 3) else "#4d4d4d"
        ax_d.add_patch(
            plt.Rectangle((0.02, yloc - 0.065), 0.25, 0.105, color=color, transform=ax_d.transAxes, clip_on=False)
        )
        ax_d.text(0.145, yloc - 0.012, tag, ha="center", va="center", fontsize=8.0, color="white", fontweight="bold", transform=ax_d.transAxes)
        ax_d.text(0.31, yloc + 0.018, result, ha="left", va="center", fontsize=8.6, color=PALETTE["text"], transform=ax_d.transAxes)
        ax_d.text(0.31, yloc - 0.035, boundary, ha="left", va="center", fontsize=8.2, color=PALETTE["muted"], transform=ax_d.transAxes)
    ax_d.text(
        0.02,
        0.03,
        "Figure uses archive-level gauge-bootstrap reductions for portability; the stricter gauge-matched AR(1) null is reported separately in Fig. 4.",
        fontsize=8.1,
        color=PALETTE["muted"],
        wrap=True,
        transform=ax_d.transAxes,
    )

    for out_dir in (REPORT_FIGURES, MAIN_FIGURES):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{OUT_STEM}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{OUT_STEM}.svg", bbox_inches="tight")
        fig.savefig(out_dir / f"{OUT_STEM}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Keep backward-compatible copies until the manuscript include is switched.
    for ext in (".pdf", ".svg", ".png"):
        shutil.copy2(MAIN_FIGURES / f"{OUT_STEM}{ext}", MAIN_FIGURES / f"fig2_reduction_uncertainty{ext}")


def main() -> None:
    observed = load_archive_reductions()
    random_tau = load_random_tau_summary(observed)
    sensitivity = load_coordinate_sensitivity()
    fig2_source, filter_audit = write_source_tables(observed, random_tau, sensitivity)
    draw_figure(observed, random_tau, sensitivity)
    print(f"Wrote figure {MAIN_FIGURES / (OUT_STEM + '.pdf')}")
    print(f"Wrote source rows: {len(fig2_source)}")
    print(f"Wrote filter-audit rows: {len(filter_audit)}")


if __name__ == "__main__":
    main()
