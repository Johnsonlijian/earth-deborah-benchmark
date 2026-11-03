"""R65 ESSD figures 1-4 from registered tables.

Sources: reports/tables/r62_decision_poc_gb_obs_curves.csv,
r39_open_model_intercomparison_metrics.csv,
r39_open_model_intercomparison_rank_disagreement.csv. Archive-level numbers
for fig02/fig03 are display values from the R60 frozen manuscript Table 1 and
the R20/R31/R37 verified records (see comments); no new computation.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
OUT = "r65_essd"


def load_obs() -> pd.DataFrame:
    c = pd.read_csv(TABLES / "r62_decision_poc_gb_obs_curves.csv")
    c["gauge_id"] = c["gauge_id"].astype(str)
    return c


def load_metrics() -> pd.DataFrame:
    m = pd.read_csv(TABLES / "r39_open_model_intercomparison_metrics.csv")
    m["gauge_id"] = m["gauge_id"].astype(str)
    return m


def fig01(obs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    archives = ["CAMELS-US", "CAMELS-GB v2", "CAMELS-BR v1.2", "CAMELS-AUS v2", "CAMELS-DK"]
    gauges = [673, 664, 893, 560, 220]
    tau_med = [8.7, 9.8, 38.2, 8.6, 34.6]  # GB from R62 obs-curve run; others fact bank
    ax = axes[0]
    ax.axis("off")
    tbl = ax.table(
        cellText=[[a, f"{g}", f"{t}"] for a, g, t in zip(archives, gauges, tau_med)],
        colLabels=["Archive", "Gauges", "tau_acf median (d)"],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title("a) Dataset coverage", pad=20)

    ax = axes[1]
    ex = obs.dropna(subset=["de"]).sort_values("obs_tau_days")
    gids = ex["gauge_id"].unique()
    pick = gids[[int(len(gids) * f) for f in (0.05, 0.35, 0.65, 0.95)]]
    for gid in pick:
        g = ex[ex["gauge_id"] == gid]
        tau = float(g["obs_tau_days"].iloc[0])
        ax.plot(g["de"], g["beta"], "o", ms=2.5,
                label=f"gauge {gid}, tau={tau:.1f} d")
    ax.set_xscale("log")
    ax.set_xlabel("De = tau_acf x f (dimensionless); raw axis = De/tau per gauge")
    ax.set_ylabel("local spectral slope beta")
    ax.set_title("b) Example observed beta(De) curves (CAMELS-GB v2)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{OUT}_fig01_dataset_overview.{ext}", dpi=300)
    plt.close(fig)


def fig02() -> None:
    # R60 Table 1: observed cluster-bootstrap medians and 95% intervals
    med = np.array([24.0, 14.7, 11.8, 7.7, -4.7])
    lo = np.array([15.5, 5.6, 6.2, -0.7, -20.2])
    hi = np.array([34.9, 27.3, 20.6, 27.6, 11.1])
    names = ["US", "GB", "BR", "AUS", "DK"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.arange(5)
    ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt="o", capsize=5, color="steelblue")
    ax.axhline(0, color="k", lw=0.8)
    for x, m in zip(xs, med):
        ax.annotate(f"{m:.1f}%", (x, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("weighted beta-variance reduction (%)")
    ax.set_title("Dispersion reduction on De axis, region-cluster bootstrap 95% CI")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{OUT}_fig02_dispersion_intervals.{ext}", dpi=300)
    plt.close(fig)


def fig03() -> None:
    # US/GB null-calibration ladder (R60 Table 1; R37/R31 verified records)
    cats = ["shuffled memory", "observed", "two-timescale AR", "matched AR(1)"]
    us = [-21.7, 24.0, 43.3, 69.0]
    gb = [-14.2, 14.7, 40.5, 65.7]
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, us, w, label="CAMELS-US", color="steelblue")
    ax.bar(x + w / 2, gb, w, label="CAMELS-GB v2", color="darkorange")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("weighted beta-variance reduction (%)")
    ax.set_title("Null-calibration ladder (observed below matched AR(1))")
    ax.annotate(
        "per-gauge AR(1) residual tightening:\nUS -0.417, GB -0.996 (medians)",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="dimgray",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{OUT}_fig03_null_calibration_ladder.{ext}", dpi=300)
    plt.close(fig)


def fig04(metrics: pd.DataFrame, ranks: pd.DataFrame) -> None:
    models = ["rrmpg_gr4j", "rrmpg_hbvedu", "global_lstm", "seasonal_ar1_null", "seasonal_climatology"]
    labels = ["GR4J", "HBV-Edu", "LSTM", "seasonal AR(1)", "seasonal clim."]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for mt, lab in zip(models, labels):
        sub = metrics[metrics["model_type"] == mt].dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        ax.scatter(sub["eval_nse"], sub["beta_de_median_abs_distance"], s=5, alpha=0.4, label=lab)
    frac = float(np.mean(~ranks["best_nse_is_best_beta"].astype(bool)))
    ax.annotate(
        f"best-by-NSE == best-by-beta(De) distance\nin {1 - frac:.1%} of gauges ({len(ranks)} gauges)",
        xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=8,
    )
    ax.set_xlabel("held-out NSE")
    ax.set_ylabel("beta(De) curve distance")
    ax.set_title("Hydrograph skill vs spectral distance (CAMELS-GB v2)")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{OUT}_fig04_model_layer.{ext}", dpi=300)
    plt.close(fig)


def fig05() -> None:
    """Decision-test distribution (pre-registered negative result)."""
    g = pd.read_csv(TABLES / "r62_decision_poc_gb_ind3_gauge_variants.csv")
    dd = g[g["disagree_beta_de"].astype(bool)]["delta_tau_err_beta_de"].dropna()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.sort(dd.to_numpy())
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, where="post", color="steelblue", lw=1.6)
    med = float(np.median(xs))
    ax.axvline(med, color="crimson", lw=1.2, label=f"median {med:+.3f}")
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.axvspan(-1.2, 0, color="lightgreen", alpha=0.25)
    ax.annotate(
        f"fraction with negative delta = {float(np.mean(xs < 0)):.2f}\n"
        f"(pre-registered threshold 0.55)",
        xy=(0.02, 0.95), xycoords="axes fraction", va="top", fontsize=8,
    )
    ax.set_xlabel("delta tau-error: beta(De)-best minus NSE-best (log10 days)")
    ax.set_ylabel("ECDF (n = %d disagreement gauges)" % len(xs))
    ax.set_title("Pre-registered decision test: distribution of the decision delta")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{OUT}_fig05_decision_test_distribution.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    obs = load_obs()
    metrics = load_metrics()
    ranks = pd.read_csv(TABLES / "r39_open_model_intercomparison_rank_disagreement.csv")
    ranks["gauge_id"] = ranks["gauge_id"].astype(str)
    fig01(obs)
    fig02()
    fig03()
    fig04(metrics, ranks)
    fig05()
    print(f"[r65] figures written to {FIGURES}")


if __name__ == "__main__":
    main()
