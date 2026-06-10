"""R33 multi-draw null-calibrated benchmark summary.

This script consolidates the formal gauge-matched AR(1) ensemble with the
deterministic gauge-level residual boundary. It supersedes the R32 table when a
multi-draw R33 ensemble is available, while retaining the same core columns so
the main Fig. 5 script and source-data index can read either version.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
LATEX_SOURCE = ROOT / "source_data"


def pct(x: float) -> float:
    return float(100.0 * x)


def _load_draw_percentile(draw_metrics_path: Path) -> dict[str, float]:
    draws = pd.read_csv(draw_metrics_path)
    out: dict[str, float] = {}
    for archive, group in draws[draws["family"].eq("matched_ar1")].groupby("archive"):
        observed = draws[(draws["archive"].eq(archive)) & (draws["family"].eq("observed"))]
        if observed.empty:
            continue
        obs = float(observed["reduction_vs_raw"].iloc[0])
        null = group["reduction_vs_raw"].astype(float).to_numpy()
        out[str(archive)] = float(100.0 * np.mean(null <= obs))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-prefix", default="r33_matched_ar1_ensemble")
    parser.add_argument("--output", default="r33_null_calibrated_ensemble_summary.csv")
    args = parser.parse_args()

    ensemble_summary = TABLES / f"{args.ensemble_prefix}_summary.csv"
    draw_metrics = TABLES / f"{args.ensemble_prefix}_draw_metrics.csv"
    boundary_path = TABLES / "r31_gauge_level_ar1_boundary_summary.csv"

    if not ensemble_summary.exists():
        raise FileNotFoundError(f"Missing ensemble summary: {ensemble_summary}")
    if not draw_metrics.exists():
        raise FileNotFoundError(f"Missing draw metrics: {draw_metrics}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Missing R31 boundary summary: {boundary_path}")

    ensemble = pd.read_csv(ensemble_summary)
    ensemble = ensemble[ensemble["family"].eq("matched_ar1")].copy()
    boundary = pd.read_csv(boundary_path)
    percentile = _load_draw_percentile(draw_metrics)

    rows: list[dict[str, object]] = []
    for _, row in ensemble.sort_values("archive").iterrows():
        archive = str(row["archive"])
        match = boundary[boundary["archive"].eq(archive)]
        if match.empty:
            raise ValueError(f"Missing R31 boundary row for {archive}")
        b = match.iloc[0]
        observed = float(row["observed_reduction"])
        null_median = float(row["null_median_reduction"])
        null_p05 = float(row["null_p05_reduction"])
        null_p95 = float(row["null_p95_reduction"])
        excess = float(row["observed_minus_null_median"])
        ratio = observed / null_median if np.isfinite(null_median) and null_median != 0 else np.nan
        median_residual = float(b["median_reduction"])
        positive_fraction = float(b["positive_fraction"])

        if excess > 0 and median_residual > 0 and positive_fraction > 0.5:
            verdict = "passes matched-ar1 excess gate"
        elif observed > 0:
            verdict = "coordinate effect positive; multi-draw matched-ar1 mechanism-excess gate fails"
        else:
            verdict = "coordinate effect not positive"

        rows.append(
            {
                "archive": archive,
                "observed_de_reduction_percent": pct(observed),
                "matched_ar1_null_reduction_percent": pct(null_median),
                "matched_ar1_null_p05_percent": pct(null_p05),
                "matched_ar1_null_p95_percent": pct(null_p95),
                "null_adjusted_excess_percent": pct(excess),
                "observed_to_matched_ar1_ratio": ratio,
                "matched_ar1_null_draws": int(row["n_null_draws"]),
                "null_draws_ge_observed": int(row["null_draws_ge_observed"]),
                "empirical_p_null_ge_observed": float(row["empirical_p_ge_observed"]),
                "observed_percentile_against_null_percent": percentile.get(archive, np.nan),
                "per_gauge_ar1_residual_median_percent": pct(median_residual),
                "per_gauge_ar1_residual_ci_low_percent": pct(float(b["median_reduction_ci_low"])),
                "per_gauge_ar1_residual_ci_high_percent": pct(float(b["median_reduction_ci_high"])),
                "per_gauge_positive_fraction_percent": pct(positive_fraction),
                "per_gauge_positive_fraction_ci_low_percent": pct(float(b["positive_fraction_ci_low"])),
                "per_gauge_positive_fraction_ci_high_percent": pct(float(b["positive_fraction_ci_high"])),
                "interpretation": verdict,
            }
        )

    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)
    out_path = TABLES / args.output
    out.to_csv(out_path, index=False)
    shutil.copy2(out_path, LATEX_SOURCE / args.output)

    note_path = ROOT / "reports" / "manuscript_notes" / args.output.replace(".csv", ".md")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# R33 multi-draw matched-AR(1) benchmark summary",
        "",
        f"Ensemble prefix: `{args.ensemble_prefix}`.",
        "",
        "Interpretation rule: a positive observed coordinate effect is retained as a benchmark-use result, but an observed-minus-null excess below zero rejects an excess-over-gauge-matched-AR(1) mechanism claim.",
        "",
        out.to_markdown(index=False),
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out.to_string(index=False))
    print(f"Wrote {out_path}")
    print(f"Copied {LATEX_SOURCE / args.output}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
