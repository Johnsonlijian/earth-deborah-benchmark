"""Build the R35 robustness/support Fig. 4 from traceable source-data tables."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "source_data"
REPORT_FIGURES = ROOT / "reports" / "figures"
MAIN_FIGURES = ROOT / "figures"
OUT_STEM = "fig4_robustness_support_checks"

COL = {
    "blue": "#2166ac",
    "lightblue": "#67a9cf",
    "red": "#b2182b",
    "orange": "#f4a582",
    "green": "#1b7837",
    "gray": "#6d6d6d",
    "grid": "#d7d7d7",
    "bg": "#f8f8f6",
    "text": "#232323",
}


def pct(x: float) -> float:
    return 100.0 * float(x)


def style(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_facecolor(COL["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#a8a8a8")
    ax.tick_params(labelsize=8.5, colors=COL["text"])
    ax.grid(axis="y", color=COL["grid"], linewidth=0.7, zorder=0)
    ax.text(-0.09, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold", pad=7)


def load_crossfit() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DATA / "camels_673_blocked_crossfit_memory_coordinate_metrics.csv")
    df = df[df["method"].isin(["same_segment_tau", "crossfit_tau"])].copy()
    df["reduction_percent"] = df["variance_reduction_vs_raw"].astype(float) * 100
    return df


def load_estimator() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DATA / "camels_673_psd_estimator_robustness_metrics.csv")
    out = df[df["axis"] == "variance_reduction"].copy()
    out["reduction_percent"] = out["weighted_beta_variance"].astype(float) * 100
    return out


def load_support() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DATA / "r20_support_matched_coordinate_sensitivity.csv")
    df = df.copy()
    df["reduction_percent"] = df["reduction_vs_raw"].astype(float) * 100
    df["archive"] = df["archive"].replace({"CAMELS-GB": "CAMELS-GB v2"})
    return df


def load_process_controls() -> pd.DataFrame:
    synthetic = pd.read_csv(SOURCE_DATA / "synthetic_reservoir_process_control_metrics.csv")
    fourier = pd.read_csv(SOURCE_DATA / "fourier_pair_circularity_baseline_metrics.csv")
    rows = []
    for _, row in synthetic.iterrows():
        rows.append(
            {
                "scenario": row["scenario_group"],
                "class": "process simulation",
                "reduction_percent": pct(row["variance_reduction_vs_raw"]),
            }
        )
    for _, row in fourier.iterrows():
        rows.append(
            {
                "scenario": row["scenario_group"],
                "class": "stochastic control",
                "reduction_percent": pct(row["variance_reduction_vs_raw"]),
            }
        )
    return pd.DataFrame(rows)


def write_source(crossfit: pd.DataFrame, estimator: pd.DataFrame, support: pd.DataFrame, controls: pd.DataFrame) -> None:
    out = pd.concat(
        [
            crossfit.assign(panel="a_temporal_crossfit"),
            estimator.assign(panel="b_estimator_sensitivity"),
            support.assign(panel="c_support_matched_bins"),
            controls.assign(panel="d_process_and_null_controls"),
        ],
        ignore_index=True,
        sort=False,
    )
    out.to_csv(SOURCE_DATA / "r35_fig4_robustness_support_source.csv", index=False)


def draw(crossfit: pd.DataFrame, estimator: pd.DataFrame, support: pd.DataFrame, controls: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.2, 7.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a: cross-fit by temporal block.
    block_order = ["all", "block_1", "block_2", "block_3", "block_4"]
    method_order = ["same_segment_tau", "crossfit_tau"]
    pivot = crossfit.pivot(index="analysis_group", columns="method", values="reduction_percent").reindex(block_order)
    x = np.arange(len(block_order))
    width = 0.34
    ax_a.axhline(0, color="#333333", linewidth=1)
    ax_a.bar(x - width / 2, pivot[method_order[0]], width, color=COL["lightblue"], label="same segment", zorder=3)
    ax_a.bar(x + width / 2, pivot[method_order[1]], width, color=COL["blue"], label="cross-fit", zorder=3)
    ax_a.set_xticks(x, ["all", "B1", "B2", "B3", "B4"])
    ax_a.set_ylabel("Reduction (%)")
    ax_a.set_ylim(0, 24)
    ax_a.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax_a, "a", "Temporal cross-fitting remains positive")

    # b: estimator heatmap.
    heat = estimator.pivot(index="slope_window_decades", columns="nperseg", values="reduction_percent").sort_index()
    im = ax_b.imshow(heat.to_numpy(), cmap="Blues", vmin=18, vmax=30, aspect="auto")
    ax_b.set_xticks(np.arange(len(heat.columns)), [str(int(c)) for c in heat.columns])
    ax_b.set_yticks(np.arange(len(heat.index)), [f"{v:.1f}" for v in heat.index])
    ax_b.set_xlabel("Welch segment")
    ax_b.set_ylabel("Slope window (decades)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax_b.text(j, i, f"{heat.to_numpy()[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#183a5a")
    fig.colorbar(im, ax=ax_b, fraction=0.045, pad=0.02, label="Reduction (%)")
    ax_b.grid(False)
    style(ax_b, "b", "Estimator choices preserve the sign")

    # c: support and binning checks.
    support_order = [
        ("standard_axis_specific_support", "log_spaced"),
        ("same_point_central_support", "log_spaced"),
        ("standard_axis_specific_support", "equal_count"),
        ("same_point_central_support", "equal_count"),
    ]
    labels = ["standard/log", "same-point/log", "standard/equal", "same-point/equal"]
    y = np.arange(len(labels))
    for offset, archive, color in [(-0.16, "CAMELS-US", COL["blue"]), (0.16, "CAMELS-GB v2", COL["orange"])]:
        vals = []
        for support_contract, binning in support_order:
            row = support[(support["archive"] == archive) & (support["support_contract"] == support_contract) & (support["binning"] == binning)]
            vals.append(float(row["reduction_percent"].iloc[0]))
        ax_c.scatter(vals, y + offset, s=58, color=color, label=archive, zorder=4)
        for val, yy in zip(vals, y + offset):
            ax_c.plot([0, val], [yy, yy], color=color, linewidth=2, alpha=0.55, zorder=3)
    ax_c.axvline(0, color="#333333", linewidth=1)
    ax_c.set_yticks(y, labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Reduction (%)")
    ax_c.set_xlim(0, 33)
    ax_c.text(0.76, 0.88, "CAMELS-US", transform=ax_c.transAxes, fontsize=8.2, color=COL["blue"])
    ax_c.text(0.76, 0.82, "CAMELS-GB v2", transform=ax_c.transAxes, fontsize=8.2, color=COL["orange"])
    style(ax_c, "c", "Support and equal-count checks stay positive")

    # d: process/null decision ledger with exact values.
    ax_d.axis("off")
    ax_d.text(-0.04, 1.08, "d", transform=ax_d.transAxes, fontsize=12, fontweight="bold")
    ax_d.text(0.04, 1.07, "Process and null controls define the boundary", transform=ax_d.transAxes, fontsize=10.2, fontweight="bold")
    selected = [
        ("linear_reservoir", "linear storage", "supports", COL["green"]),
        ("nonlinear_convex_storage", "convex storage", "supports", COL["green"]),
        ("ar1_fourier_pair", "same-series AR(1)", "artifact can align", COL["red"]),
        ("arma_positive_ma", "ARMA +MA", "worsens", COL["gray"]),
        ("seasonal_ar1_control", "seasonal AR(1)", "worsens", COL["gray"]),
        ("nonlinear_concave_storage", "concave storage", "worsens strongly", COL["gray"]),
    ]
    c_map = controls.set_index("scenario")["reduction_percent"].to_dict()
    y0 = 0.86
    for i, (key, label, interp, color) in enumerate(selected):
        value = c_map[key]
        yy = y0 - i * 0.135
        ax_d.add_patch(plt.Rectangle((0.04, yy - 0.045), 0.23, 0.075, color=color, transform=ax_d.transAxes))
        ax_d.text(0.155, yy - 0.008, f"{value:+.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold", transform=ax_d.transAxes)
        ax_d.text(0.32, yy + 0.008, label, ha="left", va="center", fontsize=8.8, color=COL["text"], transform=ax_d.transAxes)
        ax_d.text(0.32, yy - 0.038, interp, ha="left", va="center", fontsize=8.1, color=COL["gray"], transform=ax_d.transAxes)
    ax_d.text(
        0.04,
        0.03,
        "Fig. 4 supports coordinate robustness. The hard matched-AR(1) artifact floor is the separate formal gate in Fig. 5.",
        fontsize=8.1,
        color=COL["gray"],
        transform=ax_d.transAxes,
        wrap=True,
    )

    for out_dir in (REPORT_FIGURES, MAIN_FIGURES):
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            kwargs = {"dpi": 320} if ext == "png" else {}
            fig.savefig(out_dir / f"{OUT_STEM}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    for ext in (".pdf", ".svg", ".png"):
        shutil.copy2(MAIN_FIGURES / f"{OUT_STEM}{ext}", MAIN_FIGURES / f"fig4_specificity_sensitivity{ext}")


def main() -> None:
    crossfit = load_crossfit()
    estimator = load_estimator()
    support = load_support()
    controls = load_process_controls()
    write_source(crossfit, estimator, support, controls)
    draw(crossfit, estimator, support, controls)
    print(f"Wrote {MAIN_FIGURES / (OUT_STEM + '.pdf')}")


if __name__ == "__main__":
    main()
