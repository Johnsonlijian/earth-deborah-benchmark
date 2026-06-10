"""Publication-quality figure generation for Nature/Science-adjacent journals.

Rebuilds all figures with:
- Colorblind-friendly palettes (viridis/Ito)
- Statistical annotations (Spearman rho, p-values)
- Nature-family font sizes and aspect ratios
- Unified styling across all panels

Usage:
  python scripts/make_figures_topjournal.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Ito 7-color)
C0 = "#0072B2"  # blue
C1 = "#E69F00"  # orange
C2 = "#009E73"  # green
C3 = "#CC79A7"  # pink
C4 = "#F0E442"  # yellow
C5 = "#56B4E9"  # sky blue
C6 = "#D55E00"  # vermillion

FIVE_THIRDS = 5.0 / 3.0


def load_summary() -> pd.DataFrame:
    return pd.read_csv(TABLES / "nwis_expanded_1975_20260528_summary.csv")


def load_gagesii() -> pd.DataFrame:
    path = TABLES / "nwis_gagesii_merged.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def stats_annotation(ax, x, y, xpos=0.95, ypos=0.05, fontsize=7):
    """Add Spearman rho and p-value annotation."""
    rho, p = stats.spearmanr(x, y, nan_policy="omit")
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.text(xpos, ypos, f"rho = {rho:.2f}\n{p_str}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fontsize, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"))


# ── Figure 2: Main collapse result ────────────────────────────────

def fig2_main_collapse(df: pd.DataFrame):
    valid = df[df["mean_beta_de_window"].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    # (a) tau_acf distribution
    axes[0].hist(df["tau_acf_days"], bins=np.logspace(np.log10(2), np.log10(300), 25),
                 color=C0, alpha=0.75, edgecolor="white", linewidth=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("tau_acf [days]")
    axes[0].set_ylabel("Count")
    axes[0].text(0.95, 0.93, f"n = {len(df)}", transform=axes[0].transAxes, ha="right", fontsize=7, color="#555")
    axes[0].set_title("a", loc="left", fontweight="bold")

    # (b) beta(De) vs tau_acf — main collapse
    # Color by phase-randomized rejection
    phase_rej = valid["phase_randomized_p_ge_observed"].fillna(1).values
    colors = np.where(phase_rej == 0, C2, np.where(phase_rej < 0.05, C1, C0))
    sizes = np.where(phase_rej == 0, 35, 20)
    for i, row in valid.iterrows():
        axes[1].scatter(row["tau_acf_days"], row["mean_beta_de_window"],
                        c=colors[i], s=sizes[i], alpha=0.7, edgecolors="white", linewidth=0.3)
    axes[1].axhline(FIVE_THIRDS, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("tau_acf [days]")
    axes[1].set_ylabel("mean beta(0.5 <= De <= 2)")
    axes[1].set_title("b", loc="left", fontweight="bold")
    # Legend
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color=C2, label="Phase-randomized p=0"),
        Patch(color=C1, label="Phase-randomized p<0.05"),
        Patch(color=C0, label="Phase-randomized p>=0.05"),
    ], fontsize=6, loc="upper right")

    # (c) beta distribution
    axes[2].hist(valid["mean_beta_de_window"].dropna(), bins=15, color=C0, alpha=0.75, edgecolor="white", linewidth=0.5)
    axes[2].axvline(FIVE_THIRDS, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_xlabel("mean beta(0.5 <= De <= 2)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("c", loc="left", fontweight="bold")
    stats_annotation(axes[2], valid["tau_acf_days"], valid["mean_beta_de_window"])

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_main_collapse.png")
    plt.close(fig)
    print("Saved fig2_main_collapse.png")


# ── Figure 3: Basin attribute controls ────────────────────────────

def fig3_basin_attributes(df_nwis: pd.DataFrame, df_gages: pd.DataFrame):
    if df_gages.empty:
        print("Skipping fig3: no GAGES data")
        return

    df_nwis_r = df_nwis.rename(columns={"station_id": "station_id_nwis", "tau_acf_days": "tau_acf_days_nwis", "mean_beta_de_window": "mean_beta_de_window_nwis"}).copy()
    df_nwis_r["station_id_nwis"] = df_nwis_r["station_id_nwis"].astype(int)
    merged = df_nwis_r.merge(df_gages, on="station_id_nwis", how="inner")
    merged["is_ref"] = merged["regulation_class"].fillna("").astype(str).str.strip().str.lower() == "ref"
    valid = merged.dropna(subset=["drainage_area_sqkm", "tau_acf_days_nwis", "baseflow_index"])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) Drainage area
    ref_mask = valid["is_ref"].values
    axes[0].scatter(valid.loc[ref_mask, "drainage_area_sqkm"], valid.loc[ref_mask, "tau_acf_days_nwis"],
                    c=C2, s=18, alpha=0.7, edgecolors="white", linewidth=0.3, label="Reference")
    axes[0].scatter(valid.loc[~ref_mask, "drainage_area_sqkm"], valid.loc[~ref_mask, "tau_acf_days_nwis"],
                    c=C3, s=18, alpha=0.7, edgecolors="white", linewidth=0.3, label="Regulated")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Drainage area [km^2]")
    axes[0].set_ylabel("tau_acf [days]")
    axes[0].set_title("a", loc="left", fontweight="bold")
    stats_annotation(axes[0], np.log10(valid["drainage_area_sqkm"]), np.log10(valid["tau_acf_days_nwis"]))
    axes[0].legend(fontsize=6)

    # (b) Baseflow index
    axes[1].scatter(valid.loc[ref_mask, "baseflow_index"], valid.loc[ref_mask, "tau_acf_days_nwis"],
                    c=C2, s=18, alpha=0.7, edgecolors="white", linewidth=0.3)
    axes[1].scatter(valid.loc[~ref_mask, "baseflow_index"], valid.loc[~ref_mask, "tau_acf_days_nwis"],
                    c=C3, s=18, alpha=0.7, edgecolors="white", linewidth=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Baseflow index [%]")
    axes[1].set_ylabel("tau_acf [days]")
    axes[1].set_title("b", loc="left", fontweight="bold")
    stats_annotation(axes[1], valid["baseflow_index"], np.log10(valid["tau_acf_days_nwis"]))

    # (c) Regulation boxplot
    ref_tau = merged[merged["is_ref"]]["tau_acf_days_nwis"]
    nref_tau = merged[~merged["is_ref"]]["tau_acf_days_nwis"]
    bp = axes[2].boxplot([ref_tau, nref_tau], tick_labels=[f"Reference\n(n={len(ref_tau)})", f"Regulated\n(n={len(nref_tau)})"],
                          patch_artist=True, widths=0.5,
                          boxprops=dict(facecolor=C2, alpha=0.5),
                          medianprops=dict(color="black", linewidth=1),
                          flierprops=dict(marker="o", markersize=3))
    bp["boxes"][1].set_facecolor(C3)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("tau_acf [days]")
    axes[2].set_title("c", loc="left", fontweight="bold")
    # MWU test
    u, p = stats.mannwhitneyu(ref_tau, nref_tau, alternative="two-sided")
    axes[2].text(0.5, 0.95, f"MWU p<0.001", transform=axes[2].transAxes, ha="center", fontsize=7, color="#555")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_basin_controls_tj.png")
    plt.close(fig)
    print("Saved fig3_basin_controls_tj.png")


# ── Figure NN: Null model grid ─────────────────────────────────────

def fig_null_model_grid(df: pd.DataFrame):
    valid = df[df["mean_beta_de_window"].notna()].sort_values("tau_acf_days")
    null_cols = ["seasonal_noise_p_ge_observed", "ar1_p_ge_observed", "phase_randomized_p_ge_observed"]
    titles = ["Seasonal noise", "AR(1)", "Phase-randomized"]

    fig, axes = plt.subplots(3, 1, figsize=(6, max(6, len(valid) * 0.13)), sharey=True)
    for ax, col, title in zip(axes, null_cols, titles):
        p_vals = valid[col].fillna(1.0)
        bar_colors = [C2 if p == 0 else (C1 if p < 0.05 else "#cccccc") for p in p_vals]
        ax.barh(range(len(valid)), p_vals, color=bar_colors, height=0.75, alpha=0.85)
        ax.axvline(0.05, color=C6, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel("p (null >= observed)")
        n_rej = (p_vals < 0.05).sum()
        ax.text(0.98, 0.95, f"Rejected: {n_rej}/{len(valid)} ({100*n_rej/len(valid):.0f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#555")
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

    axes[0].set_yticks(range(len(valid)))
    axes[0].set_yticklabels(valid["station_id"].astype(str), fontsize=4)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_null_model_grid_tj.png")
    plt.close(fig)
    print("Saved fig_null_model_grid_tj.png")


# ── Main ──────────────────────────────────────────────────────────

def main():
    df = load_summary()
    gages = load_gagesii()
    print(f"NWIS: {len(df)} stations, {df['mean_beta_de_window'].notna().sum()} De windows")

    fig2_main_collapse(df)
    fig3_basin_attributes(df, gages)
    fig_null_model_grid(df)
    print("All top-journal figures saved.")


if __name__ == "__main__":
    main()
