from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.signal.de import beta_vs_de
from edb.signal.slopes import local_loglog_slope
from run_r20_gauge_matched_surrogate_ladder import archive_dispersion


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SECONDS_PER_DAY = 86_400.0
OUT = "final_all_archive_ar1_floor"

ARCHIVES = [
    ("CAMELS-US", "camels_673_beta_curve_points.csv", "tau_acf_days"),
    ("CAMELS-GB v2", "camels_gb_v2_replication_beta_curve_points.csv", "tau_acf_days"),
    ("CAMELS-BR v1.2", "r23_camels_br_third_archive_curves.csv", "tau_days"),
    ("CAMELS-AUS v2", "r25_camels_aus_fourth_archive_curves.csv", "tau_days"),
    ("CAMELS-DK", "r27_camels_dk_groundwater_storage_validation_curves.csv", "tau_days"),
]
MIN_UNITS = {
    "CAMELS-US": 50,
    "CAMELS-GB v2": 45,
    "CAMELS-BR v1.2": 50,
    "CAMELS-AUS v2": 45,
    "CAMELS-DK": 30,
}


def ar1_psd_cpd(frequency_cpd: np.ndarray, tau_days: float) -> np.ndarray:
    phi = float(np.exp(-1.0 / max(float(tau_days), 0.2)))
    phi = float(np.clip(phi, -0.995, 0.995))
    omega = 2.0 * np.pi * np.clip(frequency_cpd, 1e-8, 0.499999)
    return 1.0 / np.maximum(1.0 + phi * phi - 2.0 * phi * np.cos(omega), 1e-12)


def ar1_beta_for_gauge(group: pd.DataFrame, tau_col: str) -> pd.DataFrame:
    g = group.sort_values("frequency_cpd").copy()
    if "gauge_id" not in g.columns:
        g["gauge_id"] = group.name
    tau = float(g[tau_col].iloc[0])
    if not math.isfinite(tau) or tau <= 0:
        return pd.DataFrame()
    psd = ar1_psd_cpd(g["frequency_cpd"].to_numpy(dtype=float), tau)
    beta = local_loglog_slope(g["frequency"].to_numpy(dtype=float), psd, window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau * SECONDS_PER_DAY)
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["tau_acf_days"] = tau
    curve["gauge_id"] = str(g["gauge_id"].iloc[0])
    return curve[["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]]


def reduction_from_metrics(metrics: dict) -> float:
    if "reduction_vs_raw" in metrics:
        return float(100.0 * metrics["reduction_vs_raw"])
    raw = float(metrics["raw_axis_weighted_variance"])
    de = float(metrics["de_axis_weighted_variance"])
    if not np.isfinite(raw) or raw <= 0 or not np.isfinite(de):
        return np.nan
    return float(100.0 * (1.0 - de / raw))


def summarize_archive(label: str, filename: str, tau_col: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(TABLE_DIR / filename)
    if tau_col not in df.columns and "tau_acf_days" in df.columns:
        tau_col = "tau_acf_days"
    required = ["gauge_id", "frequency", "frequency_cpd", "de", "beta", tau_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")
    df = df[required].dropna().copy()
    df["gauge_id"] = df["gauge_id"].astype(str)
    if tau_col != "tau_acf_days":
        df = df.rename(columns={tau_col: "tau_acf_days"})
        tau_col = "tau_acf_days"
    ar1 = (
        df.groupby("gauge_id", group_keys=False)
        .apply(ar1_beta_for_gauge, tau_col=tau_col)
        .reset_index(drop=True)
    )
    min_units = MIN_UNITS[label]
    observed_metrics = archive_dispersion(df, min_units=min_units)
    ar1_metrics = archive_dispersion(ar1, min_units=min_units)
    observed_reduction = reduction_from_metrics(observed_metrics)
    ar1_reduction = reduction_from_metrics(ar1_metrics)
    null_adjusted = observed_reduction - ar1_reduction
    gauge_count = int(ar1["gauge_id"].nunique())
    tau_median = float(ar1.groupby("gauge_id")[tau_col].first().median())
    summary = {
        "archive": label,
        "gauge_count": gauge_count,
        "tau_median_days": tau_median,
        "observed_reduction_pct": observed_reduction,
        "analytic_ar1_reduction_pct": ar1_reduction,
        "null_adjusted_reduction_pct": null_adjusted,
        "observed_raw_bins": int(observed_metrics["raw_bins_used"]),
        "observed_de_bins": int(observed_metrics["de_bins_used"]),
        "ar1_raw_bins": int(ar1_metrics["raw_bins_used"]),
        "ar1_de_bins": int(ar1_metrics["de_bins_used"]),
        "source_curve_file": filename,
        "floor_type": "deterministic analytic AR(1) using gauge tau, observed frequency support and manuscript slope estimator",
    }
    keep = ar1[["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]].copy()
    keep = keep.rename(columns={"beta": "analytic_ar1_beta"})
    keep.insert(0, "archive", label)
    return keep, summary


def plot_summary(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
        }
    )
    colors = {
        "observed": "#1f77b4",
        "ar1": "#d62728",
        "adjusted": "#2ca02c",
        "zero": "#555555",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    y = np.arange(len(summary))
    h = 0.34
    ax.barh(y - h / 2, summary["observed_reduction_pct"], height=h, color=colors["observed"], label="Observed")
    ax.barh(y + h / 2, summary["analytic_ar1_reduction_pct"], height=h, color=colors["ar1"], label="Analytic AR(1)")
    ax.axvline(0, color=colors["zero"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["archive"])
    ax.set_xlabel("Dispersion reduction (%)")
    ax.set_title("a  Observed alignment versus AR(1) floor")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.bar(summary["archive"], summary["null_adjusted_reduction_pct"], color=colors["adjusted"])
    ax.axhline(0, color=colors["zero"], lw=0.8)
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylabel("Observed - AR(1) (%)")
    ax.set_title("b  Null-adjusted score")

    ax = axes[1, 0]
    ax.scatter(summary["tau_median_days"], summary["analytic_ar1_reduction_pct"], s=65, color=colors["ar1"])
    for _, row in summary.iterrows():
        ax.annotate(row["archive"].replace("CAMELS-", ""), (row["tau_median_days"], row["analytic_ar1_reduction_pct"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Median output memory (days)")
    ax.set_ylabel("Analytic AR(1) reduction (%)")
    ax.set_title("c  Floor scales with memory")

    ax = axes[1, 1]
    values = summary[["observed_raw_bins", "observed_de_bins", "ar1_raw_bins", "ar1_de_bins"]].to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["archive"])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["obs raw", "obs De", "AR1 raw", "AR1 De"], rotation=35, ha="right")
    ax.set_title("d  Eligible support bins")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{int(values[i, j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="bin count")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{OUT}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    curves = []
    summaries = []
    for archive in ARCHIVES:
        curve, summary = summarize_archive(*archive)
        curves.append(curve)
        summaries.append(summary)
    curve_df = pd.concat(curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    curve_df.to_csv(TABLE_DIR / f"{OUT}_curves.csv", index=False)
    summary_df.to_csv(TABLE_DIR / f"{OUT}_summary.csv", index=False)
    plot_summary(summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
