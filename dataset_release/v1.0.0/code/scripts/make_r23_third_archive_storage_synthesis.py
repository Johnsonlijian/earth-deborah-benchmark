"""Make the R23 third-archive and independent-storage synthesis figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
OUT = "r23_third_archive_storage_synthesis"


def main() -> None:
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
    br = pd.read_csv(TABLES / "r23_camels_br_third_archive_gauge_summary.csv")
    br_bins = pd.read_csv(TABLES / "r23_camels_br_third_archive_bin_stats.csv")
    br_metrics = pd.read_csv(TABLES / "r23_camels_br_third_archive_archive_metrics.csv").iloc[0]
    storage = pd.read_csv(TABLES / "r23_cross_archive_storage_proxy_validation.csv")
    gw = pd.read_csv(TABLES / "r23_groundwater_storage_validation_well_summary.csv")

    fig = plt.figure(figsize=(7.35, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    tau = pd.to_numeric(br["tau_acf_days"], errors="coerce")
    sc = ax0.scatter(
        br["gauge_lon"],
        br["gauge_lat"],
        c=np.log10(tau),
        s=8,
        cmap="viridis",
        alpha=0.82,
        linewidths=0,
    )
    ax0.set_xlabel("longitude")
    ax0.set_ylabel("latitude")
    ax0.set_title("a  CAMELS-BR third archive", loc="left", fontsize=8.8, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax0, fraction=0.04, pad=0.02)
    cb.set_label(r"$\log_{10}\tau_{\mathrm{acf}}$ [d]")
    ax0.text(
        0.02,
        0.03,
        f"n={int(br_metrics['n_gauges'])}, median tau={br_metrics['median_tau_acf_days']:.1f} d",
        transform=ax0.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.88, "pad": 3},
    )
    ax0.grid(True, color="#E4E9EF", linewidth=0.45)

    raw = br_bins[br_bins["axis"] == "frequency_cpd"].copy()
    de = br_bins[br_bins["axis"] == "de"].copy()
    ax1.plot(raw["bin"] + 1, raw["beta_variance"], color="#667085", lw=1.3, marker="o", ms=2.2, label="raw frequency")
    ax1.plot(de["bin"] + 1, de["beta_variance"], color="#C4513F", lw=1.5, marker="o", ms=2.2, label=r"$\mathrm{De}$")
    ax1.set_xlabel("ordered coordinate bin")
    ax1.set_ylabel(r"cross-gauge $\beta$ variance")
    ax1.set_title("b  Tropical replication of De tightening", loc="left", fontsize=8.8, fontweight="bold")
    ax1.text(
        0.03,
        0.90,
        f"weighted reduction={100 * br_metrics['variance_reduction_fraction']:.1f}%",
        transform=ax1.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.88, "pad": 3},
    )
    ax1.legend(loc="lower left", fontsize=7)
    ax1.grid(True, color="#E4E9EF", linewidth=0.45)

    independent = storage[storage["proxy_type"].str.contains("independent", na=False)].copy()
    keep = [
        "CAMELS-US / geol permeability",
        "CAMELS-US / geol porosity",
        "CAMELS-BR / geol permeability",
        "CAMELS-BR / geol porosity",
        "CAMELS-BR / water-table depth",
        "CAMELS-BR / bedrock depth",
        "CAMELS-BR / sand fraction",
        "CAMELS-BR / clay fraction",
    ]
    independent["label"] = independent["archive"] + " / " + independent["proxy"]
    independent = independent[independent["label"].isin(keep)].copy()
    independent = independent.sort_values("partial_rank_rho_logtau")
    y = np.arange(len(independent))
    colors = ["#C4513F" if v > 0 else "#2F6FA8" for v in independent["partial_rank_rho_logtau"]]
    ax2.axvline(0, color="#667085", lw=0.8)
    ax2.barh(y, independent["partial_rank_rho_logtau"], color=colors, alpha=0.88)
    ax2.set_yticks(y, independent["label"], fontsize=6.4)
    ax2.set_xlabel(r"partial rank rho with $\log_{10}\tau_{\mathrm{acf}}$")
    ax2.set_title("c  Independent storage-proxy support", loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, axis="x", color="#E4E9EF", linewidth=0.45)

    gw = gw[np.isfinite(pd.to_numeric(gw["tau_gw_months"], errors="coerce"))].copy()
    aquifers = gw["aquifer"].fillna("unknown")
    common = aquifers.value_counts().head(6).index
    gw_plot = gw[gw["aquifer"].isin(common)].copy()
    order = gw_plot.groupby("aquifer")["tau_gw_months"].median().sort_values().index
    data = [gw_plot.loc[gw_plot["aquifer"] == a, "tau_gw_months"].dropna() for a in order]
    ax3.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.62,
        boxprops={"facecolor": "#A5C8E1", "edgecolor": "#2F6FA8"},
        medianprops={"color": "#C4513F", "lw": 1.2},
    )
    ax3.set_yticks(range(1, len(order) + 1), order, fontsize=6.4)
    ax3.set_xscale("log")
    ax3.set_xlabel("groundwater-level memory [months]")
    ax3.set_title("d  Independent GB well-level memory", loc="left", fontsize=8.8, fontweight="bold")
    ax3.text(
        0.03,
        0.92,
        f"n={len(gw)}, median={gw['tau_gw_months'].median():.1f} months",
        transform=ax3.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.88, "pad": 3},
    )
    ax3.grid(True, axis="x", color="#E4E9EF", linewidth=0.45)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
