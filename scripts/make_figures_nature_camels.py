"""Publication-style figures for the CAMELS Deborah-number manuscript.

The script builds a first submission-grade figure set from the 673-station
CAMELS diagnostics and the cross-process control tables. It deliberately uses
only data-derived plots and hand-drawn vector schematics in matplotlib; no
generative imagery is used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


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
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)


COL = {
    "water": "#2B7BB9",
    "storage": "#2F855A",
    "storage_light": "#D8EEE0",
    "signal": "#C93434",
    "signal_light": "#F8D4D2",
    "event": "#9B4762",
    "neutral": "#6E7781",
    "pale": "#EEF1F4",
    "gold": "#D79A20",
    "ink": "#1F2328",
}


def save_pub(fig: plt.Figure, stem: str) -> None:
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(FIGURES / f"{stem}{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/svg/pdf")


def load_camels() -> pd.DataFrame:
    path = TABLES / "camels_673_attributed_diagnostics.csv"
    if not path.exists():
        path = TABLES / "camels_674_n15_summary.csv"
    df = pd.read_csv(path, dtype={"gauge_id": str})
    if "baseflow_index" not in df.columns:
        hydro = pd.read_csv(
            ROOT / "data" / "external" / "camels" / "camels_attributes_v2.0" / "camels_hydro.txt",
            sep=";",
            dtype={"gauge_id": str},
        )
        df = df.merge(hydro[["gauge_id", "baseflow_index", "slope_fdc"]], on="gauge_id", how="left")
    return df


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
    )


def spearman_text(x: pd.Series, y: pd.Series) -> str:
    sub = pd.concat([x, y], axis=1).dropna()
    rho, p = stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
    p_txt = "p<0.001" if p < 0.001 else f"p={p:.2g}"
    return f"rho={rho:.3f}\n{p_txt}\nn={len(sub)}"


def fig1_conceptual_model() -> None:
    """Schematic-led composite: storage-filtered flux versus event catalog."""

    fig = plt.figure(figsize=(7.2, 4.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[1.05, 0.75], wspace=0.18, hspace=0.22)
    ax_flux = fig.add_subplot(gs[0, 0])
    ax_event = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[1, 0])
    ax_fail = fig.add_subplot(gs[1, 1])

    for ax in [ax_flux, ax_event]:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Storage-filtered continuous flux.
    panel_label(ax_flux, "a")
    ax_flux.text(0.04, 0.95, "Storage-filtered continuous flux", fontsize=10, fontweight="bold", color=COL["ink"])
    for cx, cy, r in [(0.14, 0.66, 0.08), (0.22, 0.68, 0.09), (0.28, 0.61, 0.08)]:
        ax_flux.add_patch(patches.Circle((cx, cy), r, fc="#BDE4F4", ec="#5B9ABC", lw=1.2, alpha=0.9))
    ax_flux.text(0.12, 0.82, "input P(t)", color=COL["water"], fontsize=8, ha="center")
    box = patches.FancyBboxPatch(
        (0.42, 0.46),
        0.45,
        0.28,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        fc=COL["storage_light"],
        ec=COL["storage"],
        lw=1.6,
    )
    ax_flux.add_patch(box)
    ax_flux.text(0.645, 0.65, "catchment storage", ha="center", color=COL["storage"], fontsize=9, fontweight="bold")
    ax_flux.text(0.645, 0.56, "soil + groundwater + channels", ha="center", color=COL["storage"], fontsize=7)
    ax_flux.annotate("", xy=(0.42, 0.61), xytext=(0.28, 0.67), arrowprops=dict(arrowstyle="->", lw=1.7, color=COL["water"]))
    ax_flux.annotate("", xy=(0.92, 0.60), xytext=(0.87, 0.60), arrowprops=dict(arrowstyle="->", lw=1.7, color=COL["water"]))
    ax_flux.text(0.91, 0.66, "Q(t)", color=COL["water"], fontsize=8, ha="center")
    ax_flux.add_patch(
        patches.FancyBboxPatch(
            (0.52, 0.22),
            0.28,
            0.12,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            fc=COL["signal_light"],
            ec=COL["signal"],
            lw=1.2,
        )
    )
    ax_flux.text(0.66, 0.285, "memory time\n tau_acf", ha="center", va="center", color=COL["signal"], fontsize=8, fontweight="bold")
    ax_flux.annotate("", xy=(0.66, 0.46), xytext=(0.66, 0.34), arrowprops=dict(arrowstyle="->", lw=1.5, color=COL["signal"]))

    # Event catalog.
    panel_label(ax_event, "b")
    ax_event.text(0.04, 0.95, "Clustered event catalog", fontsize=10, fontweight="bold", color=COL["ink"])
    rng = np.random.default_rng(7)
    centers = np.array([[0.18, 0.52], [0.47, 0.73], [0.77, 0.45]])
    for center in centers:
        pts = center + rng.normal(0, [0.045, 0.085], size=(16, 2))
        ax_event.scatter(pts[:, 0], pts[:, 1], s=18, c=COL["event"], alpha=0.72, edgecolor="white", linewidth=0.3)
    ax_event.plot([0.08, 0.88], [0.84, 0.18], ls="--", lw=1.4, color="#B8BDC3")
    ax_event.text(0.63, 0.82, "triggering + thresholds\nbut no continuous storage", color=COL["event"], fontsize=8, ha="center")
    ax_event.text(0.42, 0.22, "no physical tau\nno De collapse", color=COL["neutral"], fontsize=8, fontweight="bold", ha="center")

    # Shared beta(De) curve.
    de = np.logspace(-1.2, 1.2, 250)
    beta = 0.55 + 2.15 * (de**1.25 / (1 + de**1.25)) + 0.18 * np.exp(-(np.log10(de) / 0.35) ** 2)
    ax_curve.plot(de, beta, lw=2.2, color=COL["signal"])
    ax_curve.axvline(1, lw=0.9, ls="--", color="#9AA0A6")
    ax_curve.text(1.08, 2.15, "De=1", fontsize=7, color="#555")
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("Deborah number, De = tau_acf f")
    ax_curve.set_ylabel("local slope beta")
    ax_curve.set_title("Shared beta(De) form for flux systems", loc="left", fontsize=8)
    ax_curve.set_ylim(0.4, 3.05)

    # Event failure mini-panel.
    de2 = np.logspace(-1.2, 1.2, 80)
    for offset in np.linspace(-0.18, 0.2, 9):
        ax_fail.plot(de2, 0.35 + offset + 0.12 * np.sin(np.log(de2) * 1.5), color=COL["event"], alpha=0.18, lw=1)
    ax_fail.axvline(1, lw=0.9, ls="--", color="#9AA0A6")
    ax_fail.axhspan(1.0, 2.9, fc=COL["water"], alpha=0.08, lw=0)
    ax_fail.text(0.1, 2.25, "river range", color=COL["water"], fontsize=7)
    ax_fail.text(0.55, 0.58, "earthquake subsets\nremain <0.7", color=COL["event"], fontsize=7, ha="center")
    ax_fail.set_xscale("log")
    ax_fail.set_xlabel("arbitrary De")
    ax_fail.set_ylabel("local slope beta")
    ax_fail.set_title("Event catalogs fail the test", loc="left", fontsize=8)
    ax_fail.set_ylim(-0.2, 3.05)

    save_pub(fig, "nature_fig1_conceptual_model_v2")


def fig2_camels_controls() -> None:
    df = load_camels()
    valid_de = df[df["n_beta_de_window"].fillna(0) > 0].copy()

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    bins = np.logspace(np.log10(df["tau_acf_days"].min()), np.log10(df["tau_acf_days"].max()), 32)
    ax_a.hist(df["tau_acf_days"], bins=bins, color=COL["water"], alpha=0.78, edgecolor="white", linewidth=0.4)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("storage-memory time tau_acf (days)")
    ax_a.set_ylabel("catchments")
    ax_a.text(0.96, 0.88, f"n={len(df)}\nmedian={df['tau_acf_days'].median():.1f} d", ha="right", va="top", transform=ax_a.transAxes, color=COL["ink"])
    panel_label(ax_a, "a")

    ax_b.scatter(df["baseflow_index"], df["tau_acf_days"], s=14, c=COL["storage"], alpha=0.52, edgecolor="white", linewidth=0.25)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("baseflow index")
    ax_b.set_ylabel("tau_acf (days)")
    ax_b.text(0.04, 0.96, spearman_text(df["baseflow_index"], df["tau_acf_days"]), transform=ax_b.transAxes, va="top", color=COL["ink"])
    panel_label(ax_b, "b")

    area = df["drainage_area_km2"]
    ax_c.scatter(area, df["tau_acf_days"], s=14, c="#7C8794", alpha=0.42, edgecolor="white", linewidth=0.25)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("drainage area (km2)")
    ax_c.set_ylabel("tau_acf (days)")
    ax_c.text(0.04, 0.96, spearman_text(np.log10(area), df["tau_acf_days"]), transform=ax_c.transAxes, va="top", color=COL["ink"])
    panel_label(ax_c, "c")

    rec = df.dropna(subset=["tau_recession_days", "tau_acf_days"])
    ax_d.scatter(rec["tau_recession_days"], rec["tau_acf_days"], s=14, c=COL["gold"], alpha=0.55, edgecolor="white", linewidth=0.25)
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("recession time tau_rec (days)")
    ax_d.set_ylabel("tau_acf (days)")
    ax_d.text(0.04, 0.96, spearman_text(rec["tau_recession_days"], rec["tau_acf_days"]), transform=ax_d.transAxes, va="top", color=COL["ink"])
    panel_label(ax_d, "d")

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.grid(True, which="major", color="#E6E8EB", lw=0.6, zorder=0)
    save_pub(fig, "nature_fig2_camels_memory_controls")


def fig3_nulls_and_phase_predictors() -> None:
    nulls = pd.read_csv(TABLES / "camels_673_null_rejection_counts.csv")
    logit = pd.read_csv(TABLES / "camels_673_phase_rejection_logit.csv").sort_values("p_value").head(8)

    fig = plt.figure(figsize=(7.2, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    labels = nulls["null_model"].tolist()
    y = np.arange(len(labels))
    total = 673
    strong = nulls["p_zero_count"].to_numpy()
    edge = nulls["p_le_1_over_n_count"].to_numpy() - strong
    ax_a.barh(y, strong / total, color=COL["signal"], label="no surrogate >= observed")
    ax_a.barh(y, edge / total, left=strong / total, color=COL["signal_light"], edgecolor=COL["signal"], linewidth=0.4, label="at p<1/N resolution")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 1.03)
    ax_a.set_xlabel("fraction of 673 catchments")
    for yi, s, e in zip(y, strong, strong + edge):
        ax_a.text(e / total + 0.02, yi, f"{int(e)}/673", va="center", color=COL["ink"])
    ax_a.legend(loc="lower right")
    ax_a.set_title("Hierarchical null-model exclusions", loc="left")
    panel_label(ax_a, "a")

    # Odds-ratio forest plot for phase rejection predictors.
    logit = logit.iloc[::-1].copy()
    y2 = np.arange(len(logit))
    ax_b.axvline(1, color="#9AA0A6", lw=0.8, ls="--")
    ax_b.hlines(y2, logit["ci95_low"], logit["ci95_high"], color=COL["neutral"], lw=1.2)
    colors = np.where(logit["q_bh"] < 0.05, COL["signal"], COL["water"])
    ax_b.scatter(logit["odds_ratio_per_sd"], y2, s=28, c=colors, zorder=3, edgecolor="white", linewidth=0.4)
    ax_b.set_xscale("log")
    ax_b.set_yticks(y2)
    ax_b.set_yticklabels(logit["predictor"].str.replace("_", " "))
    ax_b.set_ylim(-1.0, len(logit) - 0.35)
    ax_b.set_xlabel("odds ratio per SD for phase rejection")
    ax_b.set_title("Exploratory predictors of phase-rejection", loc="left")
    ax_b.text(
        0.02,
        0.03,
        "No predictor survives BH q<0.05;\nmechanism remains bounded.",
        transform=ax_b.transAxes,
        color=COL["neutral"],
        fontsize=6.5,
        va="bottom",
    )
    panel_label(ax_b, "b")
    save_pub(fig, "nature_fig3_nulls_phase_predictors")


def fig4_cross_process_and_controls() -> None:
    camels = load_camels()
    river = camels.loc[camels["n_beta_de_window"].fillna(0) > 0, "mean_beta_de_window"].dropna().to_numpy()
    eq = pd.read_csv(TABLES / "earthquake_regions_2000_20260528_summary.csv")
    eq_vals = eq.loc[eq["n_beta_de_window"].fillna(0) > 0, "mean_beta_de_window"].dropna().to_numpy()
    land = pd.read_csv(TABLES / "landslide_glc_trigger_nulls_summary.csv")
    land_vals = land.loc[land["observed_n_beta_de_window"].fillna(0) > 0, "observed_mean_beta_de_window"].dropna().to_numpy()
    ctrl = pd.read_csv(TABLES / "control_random_psd_summary.csv")
    colored = ctrl.loc[(ctrl["kind"] == "colored_noise") & (ctrl["n_beta_de_window"].fillna(0) > 0), "mean_beta_de_window"].dropna().to_numpy()
    random = ctrl.loc[(ctrl["kind"] == "random_psd") & (ctrl["n_beta_de_window"].fillna(0) > 0), "mean_beta_de_window"].dropna().to_numpy()

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    groups = [river, eq_vals, land_vals]
    labels = [f"river\nn={len(river)}", f"earthquake\nn={len(eq_vals)}", f"landslide\nn={len(land_vals)}"]
    colors = [COL["water"], COL["event"], "#D98C7C"]
    bp = ax_a.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.5, edgecolor=COL["ink"], linewidth=0.8)
    for median in bp["medians"]:
        median.set(color=COL["ink"], linewidth=1.2)
    rng = np.random.default_rng(12)
    for i, (vals, color) in enumerate(zip(groups, colors), start=1):
        if len(vals) > 180:
            vals = rng.choice(vals, 180, replace=False)
        ax_a.scatter(rng.normal(i, 0.045, len(vals)), vals, s=8, color=color, alpha=0.28, edgecolor="none")
    ax_a.axhline(5 / 3, color="#9AA0A6", lw=0.8, ls="--")
    ax_a.set_ylabel("mean beta in 0.5 <= De <= 2")
    ax_a.set_title("Cross-process falsification", loc="left")
    ax_a.text(1.75, 0.75, f"max earthquake={np.nanmax(eq_vals):.3f}", color=COL["event"], fontsize=6.5)
    panel_label(ax_a, "a")

    groups2 = [river, colored, random]
    labels2 = [f"river\nn={len(river)}", f"colored noise\nn={len(colored)}", f"random PSD\nn={len(random)}"]
    colors2 = [COL["water"], COL["gold"], "#ADB5BD"]
    bp2 = ax_b.boxplot(groups2, tick_labels=labels2, patch_artist=True, widths=0.55, showfliers=False)
    for box, color in zip(bp2["boxes"], colors2):
        box.set(facecolor=color, alpha=0.5, edgecolor=COL["ink"], linewidth=0.8)
    for median in bp2["medians"]:
        median.set(color=COL["ink"], linewidth=1.2)
    ax_b.axhline(5 / 3, color="#9AA0A6", lw=0.8, ls="--")
    ax_b.set_ylabel("mean beta in 0.5 <= De <= 2")
    ax_b.set_title("Non-physical spectra do not mimic rivers", loc="left")
    panel_label(ax_b, "b")

    for ax in [ax_a, ax_b]:
        ax.grid(True, axis="y", color="#E6E8EB", lw=0.6)
    save_pub(fig, "nature_fig4_cross_process_controls")


def main() -> None:
    fig1_conceptual_model()
    fig2_camels_controls()
    fig3_nulls_and_phase_predictors()
    fig4_cross_process_and_controls()


if __name__ == "__main__":
    main()
