"""Create an R25 four-archive synthesis figure and summary table."""

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
OUT = "r25_four_archive_portability_synthesis"


def archive_summary() -> pd.DataFrame:
    us_metrics = pd.read_csv(TABLES / "camels_673_collapse_metrics.csv")
    gb_metrics = pd.read_csv(TABLES / "camels_gb_v2_replication_collapse_metrics.csv")
    br_metrics = pd.read_csv(TABLES / "r23_camels_br_third_archive_archive_metrics.csv").iloc[0]
    aus_metrics = pd.read_csv(TABLES / "r25_camels_aus_fourth_archive_archive_metrics.csv").iloc[0]
    us_sum = pd.read_csv(TABLES / "camels_673_attributed_diagnostics.csv")
    gb_sum = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv")
    br_sum = pd.read_csv(TABLES / "r23_camels_br_third_archive_gauge_summary.csv")
    aus_sum = pd.read_csv(TABLES / "r25_camels_aus_fourth_archive_gauge_summary.csv")
    rows = [
        {
            "archive": "CAMELS-US",
            "region": "United States",
            "n_gauges": int(us_sum["gauge_id"].nunique()),
            "median_tau_acf_days": float(us_sum["tau_acf_days"].median()),
            "median_aridity": float(us_sum["aridity"].median()),
            "variance_reduction_fraction": float(us_metrics.loc[us_metrics["axis"] == "variance_reduction", "weighted_variance"].iloc[0]),
            "evidence_class": "temperate-to-arid benchmark origin",
        },
        {
            "archive": "CAMELS-GB v2",
            "region": "Great Britain",
            "n_gauges": int(gb_sum["gauge_id"].nunique()),
            "median_tau_acf_days": float(gb_sum["tau_acf_days"].median()),
            "median_aridity": float(gb_sum["aridity"].median()),
            "variance_reduction_fraction": float(gb_metrics.loc[gb_metrics["axis"] == "de_normalized", "variance_reduction_vs_raw"].iloc[0]),
            "evidence_class": "temperate replication",
        },
        {
            "archive": "CAMELS-BR v1.2",
            "region": "Brazil",
            "n_gauges": int(br_metrics["n_gauges"]),
            "median_tau_acf_days": float(br_metrics["median_tau_acf_days"]),
            "median_aridity": float(br_sum["aridity"].median()),
            "variance_reduction_fraction": float(br_metrics["variance_reduction_fraction"]),
            "evidence_class": "tropical replication",
        },
        {
            "archive": "CAMELS-AUS v2",
            "region": "Australia",
            "n_gauges": int(aus_metrics["n_gauges"]),
            "median_tau_acf_days": float(aus_metrics["median_tau_acf_days"]),
            "median_aridity": float(aus_metrics["median_aridity"]),
            "variance_reduction_fraction": float(aus_metrics["variance_reduction_fraction"]),
            "evidence_class": "dry-continent boundary replication",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"{OUT}_archive_summary.csv", index=False)
    return out


def proxy_summary() -> pd.DataFrame:
    cross = pd.read_csv(TABLES / "r23_cross_archive_storage_proxy_validation.csv")
    aus = pd.read_csv(TABLES / "r25_camels_aus_fourth_archive_storage_proxy_associations.csv")
    rows = []
    for _, r in cross.iterrows():
        if "independent" not in str(r["proxy_type"]):
            continue
        if r["proxy"] in {"geol porosity", "geol permeability", "water-table depth", "bedrock depth", "sand fraction", "clay fraction"}:
            rows.append(
                {
                    "archive": r["archive"],
                    "proxy": r["proxy"],
                    "partial_rank_rho_logtau": r["partial_rank_rho_logtau"],
                    "partial_q_value": r["partial_q_value_bh_within_archive"],
                    "proxy_group": "independent geology/soil",
                }
            )
    aus_keep = {
        "ksat": "soil saturated hydraulic conductivity",
        "solpawhc": "soil plant-available water holding capacity",
        "clayb": "subsoil clay fraction",
        "mrvbf_prop_8": "valley-bottom flatness class 8",
    }
    for _, r in aus.iterrows():
        if r["attribute"] in aus_keep:
            rows.append(
                {
                    "archive": "CAMELS-AUS v2",
                    "proxy": aus_keep[r["attribute"]],
                    "partial_rank_rho_logtau": r["partial_rank_rho_logtau"],
                    "partial_q_value": r["partial_q_value_bh"],
                    "proxy_group": r["attribute_type"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"{OUT}_proxy_summary.csv", index=False)
    return out


def plot(summary: pd.DataFrame, proxy: pd.DataFrame) -> None:
    strata = pd.read_csv(TABLES / "r25_camels_aus_fourth_archive_strata_metrics.csv")
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
        "CAMELS-US": "#2F6FA8",
        "CAMELS-GB v2": "#6A7FDB",
        "CAMELS-BR v1.2": "#2B9C77",
        "CAMELS-AUS v2": "#C4513F",
    }
    fig = plt.figure(figsize=(7.55, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ordered = summary.sort_values("variance_reduction_fraction", ascending=False)
    y = np.arange(len(ordered))
    ax0.barh(y, ordered["variance_reduction_fraction"] * 100, color=[colors[a] for a in ordered["archive"]], alpha=0.9)
    ax0.set_yticks(y, ordered["archive"])
    ax0.invert_yaxis()
    ax0.set_xlabel("De variance reduction [%]")
    ax0.set_title("a  Four-archive portability", loc="left", fontsize=8.8, fontweight="bold")
    ax0.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)
    for yi, r in zip(y, ordered.itertuples(index=False)):
        ax0.text(r.variance_reduction_fraction * 100 + 0.5, yi, f"n={r.n_gauges}", va="center", fontsize=6)

    sc = ax1.scatter(
        summary["median_aridity"],
        summary["median_tau_acf_days"],
        s=np.sqrt(summary["n_gauges"]) * 12,
        c=summary["variance_reduction_fraction"] * 100,
        cmap="viridis",
        edgecolor="#111827",
        linewidth=0.4,
    )
    for _, r in summary.iterrows():
        label = r["archive"].replace("CAMELS-", "")
        dx = -0.13 if "AUS" in label else 0.02
        dy = 0.9 if "BR" in label else 0.0
        ax1.text(r["median_aridity"] + dx, r["median_tau_acf_days"] + dy, label, fontsize=6.2)
    ax1.set_xlabel("archive median aridity")
    ax1.set_ylabel(r"median $\tau_{\mathrm{acf}}$ [d]")
    ax1.set_title("b  Effect size varies with hydroclimatic regime", loc="left", fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0.52, 1.52)
    ax1.set_ylim(6.5, max(summary["median_tau_acf_days"]) + 6.0)
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)
    fig.colorbar(sc, ax=ax1, shrink=0.82, pad=0.02, label="reduction [%]")

    plot_strata = strata[strata["stratifier"].isin(["aridity", "zero_flow_fraction"])].copy()
    plot_strata["label"] = plot_strata["stratum"].str.replace("_", " ")
    plot_strata["label"] = plot_strata["stratifier"].map({"aridity": "aridity", "zero_flow_fraction": "zero flow"}) + ": " + plot_strata["label"]
    plot_strata = plot_strata.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(plot_strata))
    ax2.axvline(0, color="#667085", lw=0.9)
    ax2.barh(yy, plot_strata["variance_reduction_fraction"] * 100, color=["#2F6FA8" if s == "aridity" else "#C4513F" for s in plot_strata["stratifier"]], alpha=0.86)
    ax2.set_yticks(yy, plot_strata["label"])
    ax2.set_xlabel("AUS De reduction [%]")
    ax2.set_title("c  Dry/intermittent strata define a boundary", loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)
    for yi, r in zip(yy, plot_strata.itertuples(index=False)):
        ax2.text(r.variance_reduction_fraction * 100, yi, f"  n={r.n_gauges}", va="center", fontsize=5.8)

    p = proxy.copy()
    p["label"] = p["archive"].str.replace("CAMELS-", "") + ": " + p["proxy"]
    p = p.reindex(p["partial_rank_rho_logtau"].abs().sort_values(ascending=False).index).head(12)
    p = p.sort_values("partial_rank_rho_logtau")
    yy = np.arange(len(p))
    ax3.axvline(0, color="#667085", lw=0.9)
    ax3.barh(yy, p["partial_rank_rho_logtau"], color=[colors.get(a, "#7B8794") for a in p["archive"]], alpha=0.86)
    ax3.set_yticks(yy, p["label"])
    ax3.set_xlabel(r"partial rank rho with $\log_{10}\tau_{\mathrm{acf}}$")
    ax3.set_title("d  Independent storage-proxy evidence is convergent but not causal", loc="left", fontsize=8.8, fontweight="bold")
    ax3.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)
    for yi, r in zip(yy, p.itertuples(index=False)):
        ax3.text(r.partial_rank_rho_logtau, yi, f"  q={r.partial_q_value:.1g}", va="center", fontsize=5.8)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    summary = archive_summary()
    proxy = proxy_summary()
    plot(summary, proxy)
    print(summary.to_string(index=False))
    print(proxy.sort_values("partial_q_value").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
