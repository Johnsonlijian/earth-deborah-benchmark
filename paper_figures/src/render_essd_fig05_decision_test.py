"""Render the ESSD Figure 5 bounded decision-test diagnostics.

This script only reads derived, release-eligible tables. It intentionally
reports the time-stamped decision-test null result and does not infer a causal
mechanism. The public analysis-design transcription is documented in
``DECISION_TEST_PROTOCOL_PUBLIC_RECORD.md``.
Run from any location with:

    python paper_figures/src/render_essd_fig05_decision_test.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "source_data" / "r39_open_model_intercomparison_metrics.csv"
SUMMARY_PATH = ROOT / "source_data" / "r62_decision_poc_gb_variant_summary.csv"
OUTPUT_DIR = ROOT / "figures" / "essd"

MODEL_ORDER = [
    "rrmpg_gr4j",
    "rrmpg_hbvedu",
    "global_lstm",
    "seasonal_climatology",
    "seasonal_ar1_null",
]
MODEL_LABELS = {
    "rrmpg_gr4j": "GR4J",
    "rrmpg_hbvedu": "HBV",
    "global_lstm": "LSTM",
    "seasonal_climatology": "Seasonal climatology",
    "seasonal_ar1_null": "Seasonal AR(1) null",
}
MODEL_COLOURS = {
    "rrmpg_gr4j": "#0072B2",
    "rrmpg_hbvedu": "#E69F00",
    "global_lstm": "#009E73",
    "seasonal_climatology": "#CC79A7",
    "seasonal_ar1_null": "#7A7A7A",
}
VARIANT_ORDER = ["beta_de", "beta_raw", "scalar", "de_const", "de_shuff"]
VARIANT_LABELS = {
    "beta_de": r"$\beta(De)$",
    "beta_raw": r"raw $\beta$",
    "scalar": "scalar",
    "de_const": r"constant $De$",
    "de_shuff": r"shuffled $De$",
}
VARIANT_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#7A7A7A"]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    """Fail early if a released table no longer matches the figure contract."""
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.17,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def write_provenance(outputs: list[Path]) -> None:
    """Write a human-readable checksum and source trace for the rendered figure."""
    provenance = OUTPUT_DIR / "fig05_provenance.md"
    rows = [
        "# Provenance — ESSD Figure 5",
        "",
        "This figure is a deterministic matplotlib rendering of release-eligible",
        "derived tables. It contains no generative or illustrative visual component.",
        "",
        "## Build command",
        "",
        "```text",
        "python paper_figures/src/render_essd_fig05_decision_test.py",
        "```",
        "",
        "## Input checksums",
        "",
        f"- `{METRICS_PATH.relative_to(ROOT).as_posix()}`: `{sha256(METRICS_PATH)}`",
        f"- `{SUMMARY_PATH.relative_to(ROOT).as_posix()}`: `{sha256(SUMMARY_PATH)}`",
        "",
        "## Output checksums",
        "",
    ]
    rows.extend(
        f"- `{output.relative_to(ROOT).as_posix()}`: `{sha256(output)}`"
        for output in outputs
    )
    rows.extend(
        [
            "",
            "## Scope boundary",
            "",
            "Panel (a) is descriptive. Panels (b) and (c) report the",
            "time-stamped disagreement-subset comparison and preserve its null",
            "result; they do not validate a general model-selection rule.",
            "",
        ]
    )
    provenance.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    require_columns(
        metrics,
        {"model_type", "eval_nse", "beta_de_median_abs_distance"},
        METRICS_PATH,
    )
    require_columns(
        summary,
        {
            "model_set",
            "variant",
            "subset",
            "median_delta_tau_err",
            "bootstrap_ci_lo",
            "bootstrap_ci_hi",
            "frac_variant_lower_tau_err",
        },
        SUMMARY_PATH,
    )

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(7.5, 2.3),
            gridspec_kw={"width_ratios": [1.14, 1.0, 1.0], "wspace": 0.58},
        )
        axis_a, axis_b, axis_c = axes

        # (a) Descriptive association in the released cross-model product.
        for model in MODEL_ORDER:
            subset = metrics.loc[metrics["model_type"] == model]
            if subset.empty:
                raise ValueError(f"Expected model {model!r} is absent from {METRICS_PATH.name}")
            axis_a.scatter(
                subset["eval_nse"],
                subset["beta_de_median_abs_distance"],
                s=9,
                alpha=0.68,
                linewidths=0,
                color=MODEL_COLOURS[model],
                label=MODEL_LABELS[model],
            )
        finite = metrics[["beta_de_median_abs_distance", "eval_nse"]].dropna()
        slope, intercept = np.polyfit(
            finite["eval_nse"], finite["beta_de_median_abs_distance"], deg=1
        )
        x_values = np.linspace(
            finite["eval_nse"].min(),
            finite["eval_nse"].max(),
            200,
        )
        axis_a.plot(
            x_values,
            slope * x_values + intercept,
            color="#303030",
            lw=0.85,
            ls="--",
            label="descriptive trend",
        )
        axis_a.set_xlabel("Held-out NSE")
        axis_a.set_ylabel(r"Median absolute $\beta(De)$ distance")
        axis_a.set_title("Released cross-model diagnostics", pad=5)
        axis_a.grid(axis="y", color="#D9D9D9", lw=0.45, zorder=0)
        axis_a.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            edgecolor="#C8C8C8",
            borderpad=0.28,
            handletextpad=0.35,
            labelspacing=0.25,
            markerscale=1.0,
        )
        add_panel_label(axis_a, "a")

        # (b) and (c) Pre-registered disagreement-subset test.
        decision = summary.loc[
            (summary["model_set"] == "ind3")
            & (summary["subset"] == "disagreement")
            & (summary["variant"].isin(VARIANT_ORDER))
        ].copy()
        decision["variant"] = pd.Categorical(
            decision["variant"], categories=VARIANT_ORDER, ordered=True
        )
        decision = decision.sort_values("variant")
        if len(decision) != len(VARIANT_ORDER):
            present = decision["variant"].astype(str).tolist()
            raise ValueError(
                "The decision-test table does not contain exactly the expected "
                f"variants. Present: {present}"
            )
        x_positions = np.arange(len(decision))
        labels = [VARIANT_LABELS[str(value)] for value in decision["variant"]]

        point = decision["median_delta_tau_err"].to_numpy(dtype=float)
        low = decision["bootstrap_ci_lo"].to_numpy(dtype=float)
        high = decision["bootstrap_ci_hi"].to_numpy(dtype=float)
        y_error = np.vstack((point - low, high - point))
        axis_b.axhline(0.0, color="#303030", lw=0.75, zorder=1)
        axis_b.errorbar(
            x_positions,
            point,
            yerr=y_error,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )
        axis_b.scatter(
            x_positions,
            point,
            s=26,
            c=VARIANT_COLOURS,
            edgecolors="#202020",
            linewidths=0.35,
            zorder=4,
        )
        axis_b.set_xticks(x_positions, labels, rotation=27, ha="right")
        axis_b.set_ylabel(r"Median $\Delta$ timescale error")
        axis_b.set_title("Distance choice − NSE choice\n(95% bootstrap interval)", pad=4)
        axis_b.grid(axis="y", color="#D9D9D9", lw=0.45, zorder=0)
        add_panel_label(axis_b, "b")

        fractions = decision["frac_variant_lower_tau_err"].to_numpy(dtype=float)
        for threshold, colour, linestyle in [
            (0.50, "#303030", "-"),
            (0.55, "#7A7A7A", "--"),
            (0.60, "#A0A0A0", ":"),
        ]:
            axis_c.axhline(
                threshold, color=colour, lw=0.7, ls=linestyle, zorder=1
            )
        axis_c.scatter(
            x_positions,
            fractions,
            s=28,
            c=VARIANT_COLOURS,
            edgecolors="#202020",
            linewidths=0.35,
            zorder=3,
        )
        axis_c.set_xticks(x_positions, labels, rotation=27, ha="right")
        axis_c.set_ylim(0.0, 1.0)
        axis_c.set_ylabel("Fraction with lower\n timescale error")
        axis_c.set_title("Disagreement-subset outcome", pad=5)
        axis_c.grid(axis="y", color="#D9D9D9", lw=0.45, zorder=0)
        add_panel_label(axis_c, "c")

        for axis in axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(length=2.6, pad=2)

        figure.subplots_adjust(left=0.065, right=0.995, bottom=0.29, top=0.84)
        base = OUTPUT_DIR / "fig05_decision_test"
        output_png = base.with_suffix(".png")
        output_pdf = base.with_suffix(".pdf")
        output_svg = base.with_suffix(".svg")
        figure.savefig(output_png, dpi=300)
        figure.savefig(output_pdf)
        figure.savefig(output_svg)
        plt.close(figure)

    write_provenance([output_png, output_pdf, output_svg])
    print(f"Wrote {output_png.relative_to(ROOT)}")
    print(f"Wrote {output_pdf.relative_to(ROOT)}")
    print(f"Wrote {output_svg.relative_to(ROOT)}")
    print(f"Wrote {(OUTPUT_DIR / 'fig05_provenance.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
