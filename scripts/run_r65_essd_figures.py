"""Regenerate the ESSD-facing figures from released source-data tables.

The script reads only derived, non-sensitive CSV files under ``source_data``
and writes deterministic PNG/SVG outputs under ``figures/essd``. It does not
read or redistribute raw CAMELS-family data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "source_data"
FIGURES = ROOT / "figures" / "essd"


def read_table(name: str, required: set[str]) -> pd.DataFrame:
    path = TABLES / name
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    return frame


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{stem}.{suffix}", dpi=300)
    plt.close(figure)


def load_obs() -> pd.DataFrame:
    frame = read_table(
        "r62_decision_poc_gb_obs_curves.csv",
        {"gauge_id", "de", "beta", "obs_tau_days"},
    )
    frame["gauge_id"] = frame["gauge_id"].astype(str)
    return frame


def load_metrics() -> pd.DataFrame:
    frame = read_table(
        "r39_open_model_intercomparison_metrics.csv",
        {"gauge_id", "model_type", "eval_nse", "beta_de_median_abs_distance"},
    )
    frame["gauge_id"] = frame["gauge_id"].astype(str)
    return frame


def fig01(obs: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    archives = ["CAMELS-US", "CAMELS-GB v2", "CAMELS-BR v1.2", "CAMELS-AUS v2", "CAMELS-DK"]
    gauges = [673, 664, 893, 560, 220]
    tau_medians = [8.7, 9.8, 38.2, 8.6, 34.6]

    axis = axes[0]
    axis.axis("off")
    table = axis.table(
        cellText=[[archive, str(n), f"{tau:.1f}"] for archive, n, tau in zip(archives, gauges, tau_medians)],
        colLabels=["Archive", "Gauges", "median tau_acf (d)"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    axis.set_title("a) Dataset coverage", pad=20)

    axis = axes[1]
    examples = obs.dropna(subset=["de"]).sort_values("obs_tau_days")
    gauge_ids = examples["gauge_id"].unique()
    selected = gauge_ids[[int(len(gauge_ids) * fraction) for fraction in (0.05, 0.35, 0.65, 0.95)]]
    for gauge_id in selected:
        group = examples.loc[examples["gauge_id"] == gauge_id]
        tau = float(group["obs_tau_days"].iloc[0])
        axis.plot(group["de"], group["beta"], "o", ms=2.5, label=f"gauge {gauge_id}, tau={tau:.1f} d")
    axis.set_xscale("log")
    axis.set_xlabel("De = tau_acf x f (dimensionless)")
    axis.set_ylabel("local spectral slope beta")
    axis.set_title("b) Example observed beta(De) curves (CAMELS-GB v2)")
    axis.legend(fontsize=7)
    save_figure(figure, "fig01_dataset_overview")


def fig02() -> None:
    summary = read_table(
        "final_extreme_hardening_cluster_bootstrap_summary.csv",
        {"archive", "observed_percent", "cluster_ci_low_percent", "cluster_ci_high_percent"},
    )
    order = ["CAMELS-US", "CAMELS-GB v2", "CAMELS-BR v1.2", "CAMELS-AUS v2", "CAMELS-DK lowland"]
    labels = ["US", "GB", "BR", "AUS", "DK"]
    selected = summary.set_index("archive").loc[order]
    values = selected["observed_percent"].to_numpy(float)
    low = selected["cluster_ci_low_percent"].to_numpy(float)
    high = selected["cluster_ci_high_percent"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    positions = np.arange(len(order))
    axis.errorbar(positions, values, yerr=[values - low, high - values], fmt="o", capsize=5, color="steelblue")
    axis.axhline(0, color="black", lw=0.8)
    for position, value in zip(positions, values):
        axis.annotate(f"{value:.1f}%", (position, value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("weighted beta-variance reduction (%)")
    axis.set_title("Dispersion reduction on De axis, region-cluster bootstrap 95% CI")
    save_figure(figure, "fig02_dispersion_intervals")


def fig03() -> None:
    archive_source = read_table(
        "r35_fig2_five_archive_alignment_source.csv",
        {"archive", "panel", "random_tau_median_percent"},
    )
    analytic = read_table(
        "r37_gauge_matched_analytic_nulls_summary.csv",
        {"archive", "family", "null_reduction"},
    )
    matched = read_table(
        "r33_null_calibrated_ensemble_summary.csv",
        {"archive", "observed_de_reduction_percent", "matched_ar1_null_reduction_percent"},
    )

    def row_value(frame: pd.DataFrame, archive: str, column: str, **filters: str) -> float:
        subset = frame.loc[frame["archive"] == archive]
        for key, value in filters.items():
            subset = subset.loc[subset[key] == value]
        if len(subset) != 1:
            raise ValueError(f"Expected one row for {archive}, {filters}; found {len(subset)}")
        return float(subset.iloc[0][column])

    categories = ["shuffled memory", "observed", "two-timescale AR", "matched AR(1)"]
    values: dict[str, list[float]] = {}
    for archive, label in (("CAMELS-US", "CAMELS-US"), ("CAMELS-GB", "CAMELS-GB v2")):
        source_archive = "CAMELS-GB v2" if archive == "CAMELS-GB" else archive
        values[label] = [
            row_value(archive_source, source_archive, "random_tau_median_percent", panel="b_random_tau_null"),
            row_value(matched, archive, "observed_de_reduction_percent"),
            100.0 * row_value(analytic, archive, "null_reduction", family="analytic_two_timescale_ar"),
            row_value(matched, archive, "matched_ar1_null_reduction_percent"),
        ]

    positions = np.arange(len(categories))
    width = 0.38
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.bar(positions - width / 2, values["CAMELS-US"], width, label="CAMELS-US", color="steelblue")
    axis.bar(positions + width / 2, values["CAMELS-GB v2"], width, label="CAMELS-GB v2", color="darkorange")
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(positions, categories, fontsize=8)
    axis.set_ylabel("weighted beta-variance reduction (%)")
    axis.set_title("Null-calibration ladder (observed below matched AR(1))")
    axis.legend(fontsize=8)
    save_figure(figure, "fig03_null_calibration_ladder")


def fig04(metrics: pd.DataFrame) -> None:
    ranks = read_table(
        "r39_open_model_intercomparison_rank_disagreement.csv",
        {"gauge_id", "best_nse_is_best_beta"},
    )
    models = ["rrmpg_gr4j", "rrmpg_hbvedu", "global_lstm", "seasonal_ar1_null", "seasonal_climatology"]
    labels = ["GR4J", "HBV-Edu", "LSTM", "seasonal AR(1)", "seasonal climatology"]
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for model, label in zip(models, labels):
        subset = metrics.loc[metrics["model_type"] == model].dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        axis.scatter(subset["eval_nse"], subset["beta_de_median_abs_distance"], s=5, alpha=0.4, label=label)
    agreement = float(ranks["best_nse_is_best_beta"].astype(bool).mean())
    axis.annotate(
        f"best-by-NSE equals best-by-beta(De) distance\nin {agreement:.1%} of gauges (n={len(ranks)})",
        xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=8,
    )
    axis.set_xlabel("held-out NSE")
    axis.set_ylabel("beta(De) curve distance")
    axis.set_title("Hydrograph skill versus spectral distance (CAMELS-GB v2)")
    axis.legend(fontsize=7, markerscale=2)
    save_figure(figure, "fig04_model_layer")


def supplementary_fig_s1() -> None:
    gauges = read_table(
        "r62_decision_poc_gb_ind3_gauge_variants.csv",
        {"disagree_beta_de", "delta_tau_err_beta_de"},
    )
    deltas = gauges.loc[gauges["disagree_beta_de"].astype(bool), "delta_tau_err_beta_de"].dropna().to_numpy(float)
    x_values = np.sort(deltas)
    y_values = np.arange(1, len(x_values) + 1) / len(x_values)
    median = float(np.median(x_values))
    fraction_lower = float(np.mean(x_values < 0))

    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.step(x_values, y_values, where="post", color="steelblue", lw=1.6)
    axis.axvline(median, color="crimson", lw=1.2, label=f"median {median:+.3f}")
    axis.axvline(0, color="black", lw=0.9, ls="--")
    axis.axvspan(-1.2, 0, color="lightgreen", alpha=0.25)
    axis.annotate(
        f"fraction below zero = {fraction_lower:.2f}\nreference threshold = 0.55",
        xy=(0.02, 0.95), xycoords="axes fraction", va="top", fontsize=8,
    )
    axis.set_xlabel("timescale-error difference: beta(De) selection minus NSE selection (log10 days)")
    axis.set_ylabel(f"ECDF (n={len(x_values)} disagreement gauges)")
    axis.set_title("Time-stamped decision test: distribution of the decision difference")
    axis.legend(fontsize=8)
    save_figure(figure, "supp_figS1_decision_delta_ecdf")


def main() -> None:
    observations = load_obs()
    metrics = load_metrics()
    fig01(observations)
    fig02()
    fig03()
    fig04(metrics)
    supplementary_fig_s1()
    print(f"ESSD figures written to {FIGURES}")


if __name__ == "__main__":
    main()
