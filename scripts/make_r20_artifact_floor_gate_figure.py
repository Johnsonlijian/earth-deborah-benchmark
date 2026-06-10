"""Create the support/null artifact-floor boundary figure."""

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
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r20_artifact_floor_gate"


def main() -> None:
    support = pd.read_csv(TABLES / "r20_support_matched_coordinate_sensitivity.csv")
    nulls = pd.read_csv(TABLES / "r20_gauge_matched_surrogate_ladder_summary.csv")

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
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 0.9])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    support_order = [
        ("standard_axis_specific_support", "log_spaced"),
        ("same_point_central_support", "log_spaced"),
        ("standard_axis_specific_support", "equal_count"),
        ("same_point_central_support", "equal_count"),
    ]
    colors = {
        "CAMELS-US": "#4477AA",
        "CAMELS-GB": "#55A47C",
    }
    x = np.arange(len(support_order))
    width = 0.36
    for offset, archive in [(-0.5, "CAMELS-US"), (0.5, "CAMELS-GB")]:
        vals = []
        for support_contract, binning in support_order:
            row = support[
                (support["archive"] == archive)
                & (support["support_contract"] == support_contract)
                & (support["binning"] == binning)
            ].iloc[0]
            vals.append(100 * float(row["reduction_vs_raw"]))
        xpos = x + offset * width
        ax0.bar(xpos, vals, width=width, color=colors[archive], alpha=0.82, label=archive)
        for xx, yy in zip(xpos, vals):
            ax0.text(xx, yy + 0.6, f"{yy:.1f}%", ha="center", va="bottom", fontsize=6.2)
    ax0.axhline(0, color="#7B8794", lw=0.8)
    ax0.set_ylabel("dispersion reduction [%]")
    ax0.set_xticks(x)
    ax0.set_xticklabels(
        [
            "axis-specific\nlog bins",
            "same points\nlog bins",
            "axis-specific\nequal-count",
            "same points\nequal-count",
        ],
        fontsize=6.4,
    )
    ax0.set_title("a  Coordinate-support gate: De reductions persist under stricter point/bin contracts", loc="left", fontsize=9, fontweight="bold")
    ax0.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)
    ax0.legend(loc="upper left", fontsize=6)

    archives = ["CAMELS-US", "CAMELS-GB"]
    xpos = np.arange(len(archives))
    obs = []
    matched = []
    for archive in archives:
        row = nulls[(nulls["archive"] == archive) & (nulls["family"] == "matched_ar1")].iloc[0]
        obs.append(100 * float(row["observed_reduction"]))
        matched.append(100 * float(row["null_median_reduction"]))
    ax1.bar(xpos - 0.18, obs, width=0.36, color="#4477AA", alpha=0.86, label="observed")
    ax1.bar(xpos + 0.18, matched, width=0.36, color="#C4513F", alpha=0.78, label="gauge-matched AR(1)")
    for xx, yy in zip(xpos - 0.18, obs):
        ax1.text(xx, yy + 1.2, f"{yy:.1f}%", ha="center", fontsize=6.2)
    for xx, yy in zip(xpos + 0.18, matched):
        ax1.text(xx, yy + 1.2, f"{yy:.1f}%", ha="center", fontsize=6.2)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(archives)
    ax1.set_ylabel("dispersion reduction [%]")
    ax1.set_title("b  Matched-AR(1) floor: stochastic autocorrelation can align more strongly", loc="left", fontsize=9, fontweight="bold")
    ax1.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)
    ax1.legend(loc="upper left", fontsize=6)

    ax2.set_axis_off()
    ax2.text(0.00, 0.92, "Artifact-floor decision", fontsize=10, fontweight="bold")
    verdict = [
        ("Pass", "Not caused by losing high-dispersion points or changing bin occupancy."),
        ("Fail", "Observed reduction does not exceed a gauge-matched AR(1) null."),
        ("Use", "Frame as a benchmark for spectral-memory structure, not mechanism proof."),
    ]
    y = 0.74
    verdict_colors = {"Pass": "#2E8B57", "Fail": "#C4513F", "Use": "#28313B"}
    for label, text in verdict:
        ax2.text(0.02, y, label, color=verdict_colors[label], fontweight="bold", fontsize=8.5)
        ax2.text(0.20, y, text, color="#28313B", fontsize=7.2, wrap=True)
        y -= 0.22

    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{OUT}{suffix}", **kwargs)
    plt.close(fig)

    lines = [
        "# R20 Artifact-Floor Gate Figure",
        "",
        "This figure combines the support-matched coordinate check and the",
        "gauge-matched AR(1) null. It should be interpreted as a stricter",
        "reviewer-facing gate: support artifacts are less likely, but the observed",
        "effect does not exceed a matched AR(1) stochastic expectation.",
        "",
        "Outputs:",
        "",
        f"- `reports/figures/{OUT}.png`",
        f"- `reports/figures/{OUT}.svg`",
        f"- `reports/figures/{OUT}.pdf`",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(FIGURES / f"{OUT}.pdf")


if __name__ == "__main__":
    main()
