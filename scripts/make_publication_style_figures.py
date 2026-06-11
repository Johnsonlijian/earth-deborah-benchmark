from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patheffects
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from scipy.interpolate import PchipInterpolator

HALO = [patheffects.withStroke(linewidth=1.8, foreground="white")]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_figures.src.publication_style import (
    ARCHIVE_COLORS,
    MODEL_COLORS,
    PALETTE,
    apply_style,
    clean_axis,
    panel_label,
    save_figure,
)


SOURCE = ROOT / "submission_hess" / "journal_upload_R47" / "source_data"
OUT = ROOT / "paper_figures" / "publication_style_r51"
LATEX_FIG = ROOT / "submission_hess" / "latex" / "figures"
AUDIT = ROOT / "paper_figures" / "reviews" / "R51_publication_style_figure_audit.md"

# truncate viridis so the long-memory end stays readable on white
TAU_CMAP = matplotlib.colors.ListedColormap(
    plt.get_cmap("viridis")(np.linspace(0.0, 0.88, 256)), name="viridis_trunc"
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE / name)


def pct(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return x * 100.0


def arrow(ax, xy1, xy2, color=PALETTE["gray"], lw=0.9, mutation=8):
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            mutation_scale=mutation,
            lw=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def binned_median_curve(x: np.ndarray, y: np.ndarray, gauges: np.ndarray,
                        n_bins: int = 26, min_gauges: int = 50):
    """Median y in log-spaced x bins, kept only where >=min_gauges distinct gauges contribute
    (same support contract as the manuscript variance metric)."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y, gauges = x[mask], y[mask], gauges[mask]
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    idx = np.digitize(x, edges) - 1
    centers, medians = [], []
    for i in range(n_bins):
        sel = idx == i
        if sel.any() and len(np.unique(gauges[sel])) >= min_gauges:
            centers.append(np.sqrt(edges[i] * edges[i + 1]))
            medians.append(np.median(y[sel]))
    return np.asarray(centers), np.asarray(medians)


def fig1_mechanism() -> Path:
    curves = read_csv("r19_ar1_conditioned_residual_curves.csv")
    us = curves[curves["archive"] == "CAMELS-US"].copy()
    n_gauges = us["gauge_id"].nunique()
    gauge_tau = us.groupby("gauge_id")["tau_acf_days"].first().sort_values()
    n_show = 70
    show_ids = gauge_tau.iloc[np.linspace(0, len(gauge_tau) - 1, n_show).astype(int)].index
    norm = LogNorm(vmin=gauge_tau.min(), vmax=gauge_tau.max())

    fig = plt.figure(figsize=(7.4, 5.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.18, 1, 1], height_ratios=[1.3, 0.7])

    # --- a: catchment cutaway (the single retained mechanism schematic) ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.02)
    ax.set_title("Output-memory mechanism (schematic)", loc="left", pad=4)

    xf = np.linspace(0.03, 0.97, 300)
    surface = PchipInterpolator(
        [0.03, 0.18, 0.36, 0.55, 0.74, 0.97],
        [0.870, 0.930, 0.800, 0.640, 0.520, 0.450],
    )(xf)
    soil_b = surface - PchipInterpolator(
        [0.03, 0.45, 0.97], [0.110, 0.130, 0.085]
    )(xf)
    shallow_b = soil_b - PchipInterpolator(
        [0.03, 0.45, 0.97], [0.105, 0.125, 0.080]
    )(xf)
    deep_floor = np.full_like(xf, 0.05)

    def band(top, bottom, face, edge, lw=0.6):
        coords = list(zip(xf, top)) + list(zip(xf[::-1], bottom[::-1]))
        ax.add_patch(Polygon(coords, closed=True, facecolor=face, edgecolor=edge, lw=lw))

    band(surface, soil_b, "#e7eedd", "#93a48b")
    band(soil_b, shallow_b, "#e6d6b8", "#b09465")
    band(shallow_b, deep_floor, "#dbe7ef", "#8aa6b8")

    def path_arrow(xy1, xy2, color, rad, lw=0.95, mutation=8):
        ax.add_patch(
            FancyArrowPatch(
                xy1, xy2, arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}",
                mutation_scale=mutation, lw=lw, color=color, shrinkA=1, shrinkB=1,
            )
        )

    def schem_label(x, y, text, color, ha="center", size=6.3):
        ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center",
                path_effects=HALO, zorder=6)

    # precipitation forcing over the ridge
    for x0 in [0.08, 0.18, 0.28]:
        y0 = np.interp(x0, xf, surface)
        arrow(ax, (x0, y0 + 0.115), (x0 + 0.010, y0 + 0.022), color=PALETTE["sky"], lw=0.7, mutation=7)
    schem_label(0.38, 1.005, "precipitation $P(t)$", PALETTE["gray"], ha="left", size=6.4)

    # river from mid-slope to the outlet at the right edge
    rx = np.linspace(0.46, 0.975, 120)
    ry = np.interp(rx, xf, surface) - 0.010
    ax.plot(rx, ry, color=PALETTE["blue"], lw=2.0, solid_capstyle="round", zorder=4)

    # fast surface runoff hugging the hillslope
    path_arrow((0.24, 0.860), (0.50, 0.690), PALETTE["blue"], rad=0.20, lw=1.0)
    schem_label(0.345, 0.815, "fast runoff", PALETTE["blue"])

    # interflow through the soil band
    path_arrow((0.27, 0.700), (0.63, 0.530), PALETTE["orange"], rad=-0.16, lw=0.95)
    schem_label(0.435, 0.585, "interflow", PALETTE["orange"])

    # recharge into the saturated zone (kept quiet)
    for x0 in [0.26, 0.42]:
        y_top = np.interp(x0, xf, soil_b) - 0.02
        path_arrow((x0, y_top), (x0 + 0.02, y_top - 0.10), PALETTE["green"], rad=0.10, lw=0.65, mutation=6)

    # storage buffering: dashed reservoir in the saturated band
    theta = np.linspace(0, 2 * np.pi, 120)
    ax.plot(0.33 + 0.115 * np.cos(theta), 0.310 + 0.045 * np.sin(theta),
            color=PALETTE["green"], lw=0.7, ls="--")
    schem_label(0.33, 0.225, "storage buffering", PALETTE["green"], size=6.1)

    # slow baseflow through the saturated zone to the outlet
    path_arrow((0.13, 0.380), (0.93, 0.330), "#4a7fa5", rad=-0.16, lw=1.05)
    schem_label(0.66, 0.125, "slow baseflow", "#4a7fa5")

    # outlet annotation
    y_out = float(np.interp(0.975, xf, surface)) - 0.010
    arrow(ax, (0.93, y_out), (0.995, y_out), color=PALETTE["blue"], lw=1.4, mutation=9)
    schem_label(0.995, y_out + 0.085, "$Q(t)$", PALETTE["blue"], ha="right", size=8.5)
    ax.texts[-1].set_fontweight("bold")
    schem_label(0.995, y_out - 0.075, r"output memory $\tau_{\mathrm{acf}}$", PALETTE["dark"], ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.03, 1.05)

    # --- b: real per-gauge slope curves on the raw frequency axis ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    ax_b.set_title("Raw frequency: curves disperse", loc="left", pad=4)
    for gid in show_ids:
        g = us[us["gauge_id"] == gid]
        ax_b.plot(g["frequency_cpd"], g["beta"], lw=0.5, alpha=0.38,
                  color=TAU_CMAP(norm(g["tau_acf_days"].iloc[0])))
    cx, cm = binned_median_curve(us["frequency_cpd"].to_numpy(), us["beta"].to_numpy(),
                                 us["gauge_id"].to_numpy())
    ax_b.plot(cx, cm, color=PALETTE["dark"], lw=1.4)
    ax_b.set_xscale("log")
    ax_b.set_ylim(-0.1, 3.6)
    ax_b.set_xlabel(r"Frequency $f$ (d$^{-1}$)")
    ax_b.set_ylabel(r"Local spectral slope $\beta$")
    ax_b.text(0.03, 0.96, f"CAMELS-US, $n$ = {n_gauges} gauges\n({n_show} shown)",
              transform=ax_b.transAxes, fontsize=6.2, color=PALETTE["gray"], va="top")
    clean_axis(ax_b)

    # --- c: the same curves on the memory-normalized axis ---
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c")
    ax_c.set_title("Memory coordinate: curves align", loc="left", pad=4)
    for gid in show_ids:
        g = us[us["gauge_id"] == gid]
        ax_c.plot(g["de"], g["beta"], lw=0.5, alpha=0.38,
                  color=TAU_CMAP(norm(g["tau_acf_days"].iloc[0])))
    cx, cm = binned_median_curve(us["de"].to_numpy(), us["beta"].to_numpy(),
                                 us["gauge_id"].to_numpy())
    ax_c.plot(cx, cm, color=PALETTE["dark"], lw=1.4)
    ax_c.axvline(1, color=PALETTE["gray"], lw=0.7, ls="--")
    ax_c.set_xscale("log")
    ax_c.set_ylim(-0.1, 3.6)
    ax_c.set_xlabel(r"$\mathrm{De}=\tau_{\mathrm{acf}} f$")
    ax_c.set_ylabel(r"Local spectral slope $\beta$")
    ax_c.text(0.03, 0.96, "weighted $\\beta$-variance\nreduction: 23.7%",
              transform=ax_c.transAxes, fontsize=6.2, color=PALETTE["gray"], va="top")
    clean_axis(ax_c)

    sm = plt.cm.ScalarMappable(cmap=TAU_CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=[ax_b, ax_c], fraction=0.035, pad=0.02, aspect=24)
    cbar.set_label(r"$\tau_{\mathrm{acf}}$ (days)")
    cbar.ax.tick_params(labelsize=6)

    # --- d: restrained inference pipeline plus claim boundary, full width ---
    ax = fig.add_subplot(gs[1, :])
    panel_label(ax, "d", x=-0.012)
    ax.set_axis_off()
    ax.set_title("Inference pipeline and claim boundary", loc="left", pad=4)
    steps = [
        ("$Q(t)$", "outlet discharge"),
        (r"$\tau_{\mathrm{acf}}$", "output memory"),
        (r"$\beta(f)$", "local spectral slope"),
        (r"$\mathrm{De}=\tau_{\mathrm{acf}}f$", "cross-gauge alignment"),
        ("null floor", "matched AR(1)"),
        ("model audit", "separable skill"),
    ]
    xs = np.linspace(0.05, 0.95, len(steps))
    y = 0.74
    for i, ((head, sub), x) in enumerate(zip(steps, xs)):
        ax.text(x, y, head, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=PALETTE["dark"])
        ax.text(x, y - 0.13, sub, ha="center", va="top", fontsize=6.3, color=PALETTE["gray"])
        if i < len(xs) - 1:
            arrow(ax, (x + 0.052, y), (xs[i + 1] - 0.052, y), color=PALETTE["gray"], lw=0.7, mutation=6)

    ax.plot([0.03, 0.97], [0.36, 0.36], color=PALETTE["light"], lw=0.8)
    ax.text(0.03, 0.225, "Supported", ha="left", va="center", fontsize=6.9,
            fontweight="bold", color=PALETTE["dark"])
    ax.text(0.155, 0.225, "cross-archive coordinate utility; model-output diagnostic beyond NSE/KGE",
            ha="left", va="center", fontsize=6.7, color=PALETTE["gray"])
    ax.text(0.03, 0.07, "Not supported", ha="left", va="center", fontsize=6.9,
            fontweight="bold", color=PALETTE["dark"])
    ax.text(0.155, 0.07, "groundwater residence time; tracer causality; excess over the matched AR(1) floor",
            ha="left", va="center", fontsize=6.7, color=PALETTE["gray"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = OUT / "fig01_mechanism_scientific_schematic_r51"
    save_figure(fig, out)
    return out.with_suffix(".pdf")


def fig2_archives() -> Path:
    df = read_csv("r35_fig2_five_archive_alignment_source.csv")
    cluster = read_csv("final_extreme_hardening_cluster_bootstrap_summary.csv")
    a = df[df["panel"] == "a_archive_observed_bootstrap"].copy()
    b = df[df["panel"] == "b_random_tau_null"].copy()
    c = df[df["panel"] == "c_dk_coordinate_sensitivity"].copy()
    order = ["CAMELS-US", "CAMELS-GB v2", "CAMELS-BR v1.2", "CAMELS-AUS v2", "CAMELS-DK lowland"]
    a["archive"] = pd.Categorical(a["archive"], order, ordered=True)
    a = a.sort_values("archive")
    b["archive"] = pd.Categorical(b["archive"], order, ordered=True)
    b = b.sort_values("archive")

    fig = plt.figure(figsize=(7.4, 5.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1], height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    y = np.arange(len(a))
    colors = [ARCHIVE_COLORS[x] for x in a["archive"]]
    ax.errorbar(
        a["bootstrap_median_percent"],
        y,
        xerr=[a["bootstrap_median_percent"] - a["ci_low_percent"], a["ci_high_percent"] - a["bootstrap_median_percent"]],
        fmt="o",
        color=PALETTE["dark"],
        ecolor=PALETTE["gray"],
        elinewidth=0.9,
        capsize=2,
        ms=0,
        zorder=1,
    )
    ax.scatter(a["observed_percent"], y, s=28, c=colors, edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    labels = [f"{arch} ($n$={int(n)})" for arch, n in zip(a["archive"].astype(str), a["n_gauges"])]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Reduction in cross-catchment $\beta$ dispersion (%)")
    ax.set_title("Archive effect with bootstrap intervals", loc="left")
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    x = np.arange(len(b))
    width = 0.36
    ax.bar(x - width / 2, b["observed_percent"], width, color=[ARCHIVE_COLORS[v] for v in b["archive"]])
    ax.bar(x + width / 2, b["random_tau_median_percent"], width, color="#b9c0cb")
    ax.errorbar(
        x + width / 2,
        b["random_tau_median_percent"],
        yerr=[
            b["random_tau_median_percent"] - b["random_tau_ci_low_percent"],
            b["random_tau_ci_high_percent"] - b["random_tau_median_percent"],
        ],
        fmt="none",
        ecolor=PALETTE["dark"],
        elinewidth=0.7,
        capsize=1.5,
    )
    ax.axhline(0, color=PALETTE["gray"], lw=0.8)
    first_obs = float(b["observed_percent"].iloc[0])
    first_null = float(b["random_tau_ci_low_percent"].iloc[0])
    ax.annotate("observed", (x[0] - width / 2, first_obs), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=6.4, color=PALETTE["dark"])
    ax.annotate("shuffled $\\tau$", (x[0] + width / 2, first_null), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top", fontsize=6.4, color=PALETTE["gray"])
    ax.set_xticks(x, [s.replace("CAMELS-", "").replace(" lowland", "") for s in b["archive"]], rotation=30, ha="right")
    ax.set_ylim(-46, 30)
    ax.set_ylabel("Reduction (%)")
    ax.set_title("Measured memory vs shuffled-memory null", loc="left")
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    cc = cluster.copy()
    cc["archive"] = pd.Categorical(cc["archive"], order, ordered=True)
    cc = cc.sort_values("archive")
    yy = np.arange(len(cc))
    ax.errorbar(
        cc["cluster_median_percent"],
        yy,
        xerr=[cc["cluster_median_percent"] - cc["cluster_ci_low_percent"], cc["cluster_ci_high_percent"] - cc["cluster_median_percent"]],
        fmt="o",
        color=PALETTE["dark"],
        ecolor=PALETTE["gray"],
        capsize=2,
        lw=0.9,
    )
    ax.scatter(cc["observed_percent"], yy, s=22, c=[ARCHIVE_COLORS[v] for v in cc["archive"]], zorder=3)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    labels = [f"{arch} ($n$={int(n)})" for arch, n in zip(cc["archive"].astype(str), cc["n_gauges"])]
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Region-cluster reduction (%)")
    ax.set_title("Spatial-dependence hardening", loc="left")
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    labels = c["label"].str.replace(" tau", "", regex=False).str.replace("DK-model ", "", regex=False)
    vals = c["reduction_percent"]
    colors = [PALETTE["blue"] if v > 0 else PALETTE["red"] for v in vals]
    ax.barh(np.arange(len(c)), vals, color=colors, alpha=0.88)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_yticks(np.arange(len(c)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reduction (%)")
    ax.set_title("CAMELS-DK coordinate sensitivity", loc="left")
    clean_axis(ax, grid=True)

    out = OUT / "fig02_archives_publication_r51"
    save_figure(fig, out)
    return out.with_suffix(".pdf")


def fig3_robustness() -> Path:
    rob = read_csv("r35_fig4_robustness_support_source.csv")
    cluster = read_csv("final_extreme_hardening_cluster_bootstrap_summary.csv")
    forcing = read_csv("final_extreme_hardening_forcing_control_summary.csv")
    support = read_csv("final_extreme_hardening_slow_memory_support_audit.csv")

    fig = plt.figure(figsize=(7.4, 6.0), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    a = rob[rob["panel"] == "a_temporal_crossfit"].dropna(subset=["reduction_percent"]).copy()
    aa = pd.concat(
        [
            a[(a["analysis_group"] == "all") & (a["method"].isin(["same_segment_tau", "crossfit_tau"]))],
            a[(a["analysis_group"].isin(["block_1", "block_2", "block_3", "block_4"])) & (a["method"] == "crossfit_tau")],
        ]
    )
    labels = []
    for _, row in aa.iterrows():
        if row["analysis_group"] == "all":
            labels.append("same" if row["method"] == "same_segment_tau" else "cross-fit")
        else:
            labels.append(row["analysis_group"].replace("block_", "B"))
    bar_colors = [PALETTE["blue"] if lab in ("same", "cross-fit") else "#7fa8cc" for lab in labels]
    ax.bar(np.arange(len(aa)), aa["reduction_percent"], color=bar_colors)
    ax.axhline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_xticks(np.arange(len(aa)), labels)
    ax.set_ylabel("Reduction (%)")
    ax.set_title("Temporal cross-fitting", loc="left")
    clean_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    b = rob[rob["panel"] == "b_estimator_sensitivity"].dropna(subset=["reduction_percent"])
    piv = b.pivot_table(index="slope_window_decades", columns="nperseg", values="reduction_percent", aggfunc="mean").sort_index()
    vmin = max(0, np.nanmin(piv.values) - 2)
    vmax = np.nanmax(piv.values) + 2
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                frac = (v - vmin) / (vmax - vmin)
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.3,
                        color="white" if frac > 0.62 else PALETTE["dark"])
    ax.set_xticks(np.arange(len(piv.columns)), [int(v) for v in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)), [f"{v:.1f}" for v in piv.index])
    ax.set_xlabel("Welch segment length")
    ax.set_ylabel("Slope window (decades)")
    ax.set_title("Estimator perturbation", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Reduction (%)")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    c = rob[rob["panel"] == "c_support_matched_bins"].dropna(subset=["reduction_percent"]).copy()
    c["label"] = (
        c["archive"].astype(str).str.replace("CAMELS-", "", regex=False)
        + " · "
        + c["support_contract"].replace({"standard_axis_specific_support": "standard", "same_point_central_support": "same-point"}).astype(str)
        + " · "
        + c["binning"].replace({"log_spaced": "log", "equal_count": "equal"}).astype(str)
    )
    yy = np.arange(len(c))
    ax.barh(yy, c["reduction_percent"], color=[ARCHIVE_COLORS.get(v, PALETTE["gray"]) for v in c["archive"]], alpha=0.88)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_yticks(yy, c["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Reduction (%)")
    ax.set_title("Support and binning sensitivity", loc="left")
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    order = ["CAMELS-US", "CAMELS-GB v2", "CAMELS-BR v1.2", "CAMELS-AUS v2", "CAMELS-DK lowland"]
    cluster["archive"] = pd.Categorical(cluster["archive"], order, ordered=True)
    cluster = cluster.sort_values("archive")
    y = np.arange(len(cluster))
    ax.errorbar(
        cluster["cluster_median_percent"],
        y,
        xerr=[cluster["cluster_median_percent"] - cluster["cluster_ci_low_percent"], cluster["cluster_ci_high_percent"] - cluster["cluster_median_percent"]],
        fmt="o",
        color=PALETTE["dark"],
        ecolor=PALETTE["gray"],
        capsize=2,
    )
    ax.axvline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_yticks(y, cluster["archive"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Cluster-bootstrap reduction (%)")
    ax.set_title("Region-cluster bootstrap", loc="left")
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "e")
    f = forcing[forcing["analysis"] != "tau_q_tau_p_spearman"].copy()
    label_map = {
        "precipitation_beta_on_precipitation_memory": "$P$ spectra on $P$ memory",
        "q_minus_p_beta_contrast_on_q_memory": "$Q-P$ contrast on $Q$ memory",
        "q_minus_p_beta_contrast_on_p_memory": "$Q-P$ contrast on $P$ memory",
    }
    labels = [label_map.get(v, v.replace("_", " ")) for v in f["analysis"]]
    colors = [PALETTE["blue"] if v > 0 else PALETTE["red"] for v in f["variance_reduction_percent"]]
    ax.barh(np.arange(len(f)), f["variance_reduction_percent"], color=colors, alpha=0.88)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_yticks(np.arange(len(f)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reduction (%)")
    ax.set_title("GB forcing-side control", loc="left")
    spearman = forcing[forcing["analysis"] == "tau_q_tau_p_spearman"]
    if not spearman.empty:
        rho = spearman["variance_reduction_fraction"].iloc[0]
        ax.text(0.97, 0.96, "$\\tau_Q$ vs $\\tau_P$ non-equivalence:\n"
                rf"Spearman $\rho$ = {rho:.2f} ($p<10^{{-39}}$)",
                transform=ax.transAxes, fontsize=6.2, color=PALETTE["gray"],
                ha="right", va="top", linespacing=1.3, path_effects=HALO)
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[2, 1])
    panel_label(ax, "f")
    s = support[(support["audit_type"] == "frequency_support") & (support["archive"].isin(["CAMELS-US", "CAMELS-GB v2"]))].copy()
    for archive, g in s.groupby("archive"):
        g = g.sort_values("nperseg")
        short = archive.replace("CAMELS-", "").replace(" v2", "")
        color = ARCHIVE_COLORS.get(archive, PALETTE["gray"])
        ax.plot(g["nperseg"], pct(g["de1_resolvable_fraction"]), marker="o", ms=3, color=color, label=f"{short}, De=1")
        ax.plot(g["nperseg"], pct(g["de05_resolvable_fraction"]), marker="s", ms=3, color=color, ls="--", label=f"{short}, De=0.5")
    ax.set_xscale("log", base=2)
    ticks = sorted(s["nperseg"].unique())
    ax.set_xticks(ticks, [str(int(t)) for t in ticks])
    ax.set_ylim(90, 101)
    ax.set_xlabel("Welch segment length")
    ax.set_ylabel("Resolvable gauges (%)")
    ax.set_title("Slow-memory frequency support", loc="left")
    ax.legend(ncols=2, loc="lower right", columnspacing=1.0, handlelength=1.6)
    clean_axis(ax)

    out = OUT / "fig03_robustness_publication_r51"
    save_figure(fig, out)
    return out.with_suffix(".pdf")


def fig4_artifact_floor() -> Path:
    ar1 = read_csv("r33_matched_ar1_ensemble_summary.csv")
    draws = read_csv("r33_matched_ar1_ensemble_draw_metrics.csv")
    proc = read_csv("r37_gauge_matched_analytic_nulls_summary.csv")
    ar1["archive"] = ar1["archive"].replace({"CAMELS-GB": "CAMELS-GB v2"})
    draws["archive"] = draws["archive"].replace({"CAMELS-GB": "CAMELS-GB v2"})
    ar1 = ar1.sort_values("archive", key=lambda s: s.map({"CAMELS-US": 0, "CAMELS-GB v2": 1}))

    fig = plt.figure(figsize=(7.4, 5.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    x = np.arange(len(ar1))
    w = 0.34
    ax.bar(x - w / 2, pct(ar1["observed_reduction"]), w,
           color=[ARCHIVE_COLORS.get(v, PALETTE["gray"]) for v in ar1["archive"]])
    ax.bar(x + w / 2, pct(ar1["null_median_reduction"]), w, color="#c4cad4")
    ax.errorbar(x + w / 2, pct(ar1["null_median_reduction"]),
                yerr=[pct(ar1["null_median_reduction"] - ar1["null_p05_reduction"]),
                      pct(ar1["null_p95_reduction"] - ar1["null_median_reduction"])],
                fmt="none", ecolor=PALETTE["dark"], lw=0.8, capsize=2)
    for xi, (obs, nul) in enumerate(zip(pct(ar1["observed_reduction"]), pct(ar1["null_median_reduction"]))):
        ax.annotate("observed", (xi - w / 2, obs), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6.2, color=PALETTE["dark"])
        ax.annotate("matched\nAR(1)", (xi + w / 2, nul), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.2, color=PALETTE["gray"], linespacing=1.1)
    ax.set_xticks(x, ar1["archive"])
    ax.set_ylim(0, 86)
    ax.set_ylabel("Reduction (%)")
    ax.set_title("Strict stochastic artifact floor", loc="left")
    clean_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    rng = np.random.default_rng(42)
    rows = list(ar1["archive"])
    for yi, archive in enumerate(rows):
        d = draws[(draws["archive"] == archive) & (draws["family"] == "matched_ar1")]
        vals = pct(d["reduction_vs_raw"].to_numpy())
        jitter = rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(vals, np.full(len(vals), yi) + jitter, s=5, color="#9aa3ad", alpha=0.6, lw=0)
        row = ar1[ar1["archive"] == archive].iloc[0]
        obs = float(pct(row["observed_reduction"]))
        null_med = float(pct(row["null_median_reduction"]))
        ax.plot([obs, null_med], [yi, yi], color=PALETTE["gray"], lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.scatter([obs], [yi], marker="D", s=34, color=ARCHIVE_COLORS.get(archive, PALETTE["dark"]),
                   edgecolor="white", lw=0.5, zorder=4)
        gap = float(pct(row["observed_minus_null_median"]))
        ax.annotate(f"{gap:+.1f} pp", ((obs + null_med) / 2, yi),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=6.4, color=PALETTE["dark"])
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Reduction (%)")
    ax.set_title("Observed vs 100-draw matched AR(1) ensemble", loc="left")
    handles = [
        Line2D([], [], marker="D", ls="", ms=5, color=PALETTE["dark"], label="observed"),
        Line2D([], [], marker="o", ls="", ms=3.5, color="#9aa3ad", label="null draws"),
    ]
    ax.legend(handles=handles, loc="lower left", handletextpad=0.4)
    clean_axis(ax, grid=True)

    ax = fig.add_subplot(gs[1, :])
    panel_label(ax, "c", x=-0.045)
    p = proc[proc["family"] != "observed"].copy()
    p["archive"] = p["archive"].replace({"CAMELS-GB": "CAMELS-GB v2"})
    family_order = ["analytic_ar1", "analytic_two_timescale_ar", "analytic_arfima_d0p15", "analytic_arfima_d0p28", "analytic_arfima_d0p40"]
    fam_labels = ["AR(1)", "two-timescale AR", "ARFIMA d=0.15", "ARFIMA d=0.28", "ARFIMA d=0.40"]
    obs_vals = proc[proc["family"] == "observed"].copy()
    obs_vals["archive"] = obs_vals["archive"].replace({"CAMELS-GB": "CAMELS-GB v2"})
    for archive, g in p.groupby("archive"):
        color = ARCHIVE_COLORS.get(archive, PALETTE["gray"])
        g = g.set_index("family").reindex(family_order)
        ax.plot(np.arange(len(family_order)), pct(g["null_reduction"]), marker="o", ms=3.5,
                color=color, label=archive)
        obs = obs_vals[obs_vals["archive"] == archive]
        if not obs.empty:
            ov = float(pct(obs["observed_reduction"].iloc[0]))
            ax.axhline(ov, color=color, lw=0.7, ls=(0, (5, 3)), alpha=0.85)
            ax.annotate(f"observed {archive.replace('CAMELS-', '')}: {ov:.1f}%",
                        (len(family_order) - 1.02, ov), xytext=(0, 2), textcoords="offset points",
                        ha="right", fontsize=6.2, color=color)
    ax.axhline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_xticks(np.arange(len(family_order)), fam_labels)
    ax.set_ylabel("Analytic-null reduction (%)")
    ax.set_title("Process-family boundary for the same pipeline", loc="left")
    ax.legend(loc="lower left", ncols=2)
    clean_axis(ax)

    out = OUT / "fig04_artifact_floor_publication_r51"
    save_figure(fig, out)
    return out.with_suffix(".pdf")


def fig5_models() -> Path:
    summary = read_csv("r39_open_model_intercomparison_summary.csv")
    metrics = read_csv("r39_open_model_intercomparison_metrics.csv")
    ranks = read_csv("r39_open_model_intercomparison_rank_disagreement.csv")

    fig = plt.figure(figsize=(7.4, 5.4), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    short = {
        "rrmpg_gr4j": "GR4J",
        "rrmpg_hbvedu": "HBV-Edu",
        "global_lstm": "LSTM",
        "seasonal_climatology": "climatology",
        "seasonal_ar1_null": "seasonal AR(1)",
        "lagged_q_lower_bound": "lagged Q",
    }

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    plot_metrics = metrics.dropna(subset=["eval_nse", "beta_de_median_abs_distance"]).copy()
    for mt, g in plot_metrics.groupby("model_type"):
        ax.scatter(g["eval_nse"], g["beta_de_median_abs_distance"], s=6, alpha=0.20,
                   color=MODEL_COLORS.get(mt, PALETTE["gray"]), label=short.get(mt, mt), lw=0)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.axhline(1, color=PALETTE["gray"], lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("Held-out NSE (symlog)")
    ax.set_ylabel(r"$\beta(\mathrm{De})$ distance")
    ax.set_title("Hydrograph and spectral-memory axes", loc="left")
    leg = ax.legend(markerscale=2.2, ncols=2, loc="lower left", columnspacing=0.9,
                    handletextpad=0.3, borderaxespad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(0.9)
    clean_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    order = summary.sort_values("median_beta_de_distance")["model_type"].tolist()
    data = [plot_metrics.loc[plot_metrics["model_type"] == mt, "beta_de_median_abs_distance"].dropna().clip(lower=1e-3).values for mt in order]
    bp = ax.boxplot(data, tick_labels=[short.get(mt, mt).replace(" ", "\n") for mt in order], patch_artist=True, showfliers=False)
    for patch, mt in zip(bp["boxes"], order):
        patch.set_facecolor(MODEL_COLORS.get(mt, PALETTE["gray"]))
        patch.set_alpha(0.55)
        patch.set_edgecolor(PALETTE["dark"])
    for med in bp["medians"]:
        med.set_color(PALETTE["dark"])
        med.set_linewidth(1.0)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\beta(\mathrm{De})$ distance")
    ax.set_title("Distribution across valid gauges", loc="left")
    ax.tick_params(axis="x", rotation=0)
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    common = pd.crosstab(ranks["best_nse_model"], ranks["best_beta_model"])
    all_models = order
    common = common.reindex(index=all_models, columns=all_models, fill_value=0)
    im = ax.imshow(common.values, cmap="Blues")
    ax.set_xticks(np.arange(len(all_models)), [short.get(m, m) for m in all_models], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(all_models)), [short.get(m, m) for m in all_models])
    ax.set_xlabel("Best by spectral-memory distance")
    ax.set_ylabel("Best by NSE")
    overlap = int(ranks["best_nse_is_best_beta"].sum())
    ax.set_title(f"Rank disagreement\nsame best: {overlap}/{len(ranks)} gauges", loc="left")
    vmax = common.values.max()
    for i in range(common.shape[0]):
        for j in range(common.shape[1]):
            v = common.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.62 * vmax else PALETTE["dark"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("# gauges")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    s = summary.copy()
    ax.scatter(s["median_eval_nse"], s["median_beta_de_distance"], s=42,
               color=[MODEL_COLORS.get(mt, PALETTE["gray"]) for mt in s["model_type"]],
               edgecolor="white", lw=0.5, zorder=3)
    offsets = {
        "rrmpg_gr4j": (-7, 5, "right"),
        "rrmpg_hbvedu": (7, 5, "left"),
        "global_lstm": (7, -9, "left"),
        "seasonal_climatology": (7, 4, "left"),
        "seasonal_ar1_null": (7, 4, "left"),
        "lagged_q_lower_bound": (-7, 4, "right"),
    }
    for _, row in s.iterrows():
        dx, dy, ha = offsets.get(row["model_type"], (7, 4, "left"))
        ax.annotate(short.get(row["model_type"], row["model_label"]),
                    (row["median_eval_nse"], row["median_beta_de_distance"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.6)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Median NSE")
    ax.set_ylabel(r"Median $\beta(\mathrm{De})$ distance")
    ax.set_title("Model-summary trade-off", loc="left")
    clean_axis(ax)

    out = OUT / "fig05_model_diagnostic_publication_r51"
    save_figure(fig, out)
    return out.with_suffix(".pdf")


def copy_to_latex(paths: dict[str, Path]) -> dict[str, Path]:
    # Journal-facing filenames carry no internal round suffix; the _r51 copies
    # are kept alongside for internal provenance.
    mapping = {
        "fig01": "fig01_mechanism_evidence.pdf",
        "fig02": "fig02_five_archive_alignment_boundary.pdf",
        "fig03": "fig03_robustness_support_checks.pdf",
        "fig04": "fig04_artifact_floor_boundary.pdf",
        "fig05": "fig05_multimodel_benchmark.pdf",
    }
    copied = {}
    for key, src in paths.items():
        dst = LATEX_FIG / mapping[key]
        shutil.copy2(src, dst)
        shutil.copy2(src, LATEX_FIG / dst.name.replace(".pdf", "_r51.pdf"))
        copied[key] = dst
    return copied


def write_audit(copied: dict[str, Path]) -> None:
    rows = [
        ("Fig. 1", "A: mechanism/model schematic + real-data coordinate demonstration", "Yes (only schematic panel: 1a)",
         "Panel a redrawn as a restrained catchment cutaway with direct pathway labels. Panels b/c replaced the previous synthetic placeholder curves with real per-gauge CAMELS-US local-slope curves (r19 source table) on raw-frequency and De axes, fixing a figure/caption integrity gap. Panel d replaced the numbered-badge evidence ladder and colored claim table with a plain typographic pipeline plus claim-boundary lines."),
        ("Fig. 2", "B: data/statistical", "No",
         "Forest plots and bar charts retained; gauge counts added to axis labels, shuffled-tau null now carries bootstrap intervals and direct labels instead of a legend overlapping data, diverging bars switched from green/red to blue/red."),
        ("Fig. 3", "B/D: robustness data and workflow audit", "No",
         "Heatmap cells annotated with values; forcing-control panel removed the Spearman-rho row that was previously plotted as a percent reduction (unit error) and reports it as text; slow-memory panel uses explicit segment-length ticks and compact archive/De legend."),
        ("Fig. 4", "B: null-calibration result", "No",
         "Text-card interpretation-boundary panel removed. New panel b shows the full 100-draw matched AR(1) null ensemble against the observed reduction with the null-adjusted gap annotated; process-family panel spans the bottom row with observed reference lines per archive. Interpretation boundary moved to the caption."),
        ("Fig. 5", "B: model intercomparison data", "No",
         "Scatter, boxplot, confusion matrix and trade-off panels retained; point labels in panel d switched to collision-free offset annotations; heatmap text contrast and legend density improved."),
        ("Fig. S1--S10", "B/C/D supplementary diagnostics", "No",
         "Already ordinary analysis-figure style; retained unchanged as non-schematic support figures."),
    ]
    lines = [
        "# R51 publication-style figure audit",
        "",
        "Purpose: correct the R48/R50 card/dashboard/poster drift. Only Fig. 1a remains a mechanism/model schematic; all other main-text content is restrained publication-style quantitative graphics.",
        "",
        "R51.2 refinement (2026-06-10): second-pass uplift after the initial de-carding.",
        "Key integrity fixes in this pass:",
        "1. Fig. 1b/c previously showed synthetic illustrative curves while the caption claimed registered source data; they now plot real per-gauge CAMELS-US beta curves from `r19_ar1_conditioned_residual_curves.csv` (673 gauges; 70 drawn, binned median over all gauges).",
        "2. Fig. 3e previously plotted the tau_Q-vs-tau_P Spearman correlation (-0.48) on the percent-reduction axis as if it were a -48% variance reduction; the correlation is now reported as text and the axis carries only variance reductions.",
        "3. Fig. 1d and Fig. 4d infographic elements (numbered colored badges, colored claim table, text-card panel) were removed; claim boundaries now live in plain typography (Fig. 1d) and the captions.",
        "",
        "## Audit table",
        "| Figure | Type | 3D/model schematic retained? | Action |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines.extend(["", "## Exported main-text files", "| Figure | LaTeX PDF | SVG source | PNG preview |", "|---|---|---|---|"])
    for key, dst in copied.items():
        stem = {
            "fig01": "fig01_mechanism_scientific_schematic_r51",
            "fig02": "fig02_archives_publication_r51",
            "fig03": "fig03_robustness_publication_r51",
            "fig04": "fig04_artifact_floor_publication_r51",
            "fig05": "fig05_model_diagnostic_publication_r51",
        }[key]
        src_stem = OUT / stem
        lines.append(
            f"| {key} | `{dst.relative_to(ROOT).as_posix()}` | `{src_stem.with_suffix('.svg').relative_to(ROOT).as_posix()}` | `{src_stem.with_suffix('.png').relative_to(ROOT).as_posix()}` |"
        )
    lines.extend(
        [
            "",
            "## Removed visual grammar",
            "- Removed card-style panel containers, rounded dashboard blocks, heavy title banners, drop shadows and poster-level framing (R51.0).",
            "- Removed numbered colored evidence-ladder badges and colored SUPPORTED/BOUNDED claim table from Fig. 1d (R51.2).",
            "- Removed the Fig. 4d text-card panel; replaced with the registered 100-draw null-ensemble data panel (R51.2).",
            "- Kept white backgrounds, thin axes, direct data encodings, source-data values, uncertainty intervals and explicit null boundaries.",
            "",
            "## Unified style",
            "- Style module: `paper_figures/src/publication_style.py`.",
            "- Generator: `scripts/make_r51_publication_style_figures.py`.",
            "- Backend: Python/matplotlib with editable SVG text and Type 42 PDF fonts.",
            "- Real-data sources added in R51.2: `r19_ar1_conditioned_residual_curves.csv` (Fig. 1b/c), `r33_matched_ar1_ensemble_draw_metrics.csv` (Fig. 4b).",
        ]
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "fig01": fig1_mechanism(),
        "fig02": fig2_archives(),
        "fig03": fig3_robustness(),
        "fig04": fig4_artifact_floor(),
        "fig05": fig5_models(),
    }
    copied = copy_to_latex(paths)
    write_audit(copied)
    print("r51_publication_style_figures=ready")
    for key, path in copied.items():
        print(f"{key}={path}")
    print(f"audit={AUDIT}")


if __name__ == "__main__":
    main()
