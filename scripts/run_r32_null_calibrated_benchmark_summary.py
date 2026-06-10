"""R32 null-calibrated benchmark score table.

This round consolidates the two strongest AR(1) artifact-floor diagnostics
already computed in R20 and R31 into one reviewer-facing source-data table.
It does not create new stochastic draws; it records what the current evidence
can support without overstating mechanism.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
LATEX_SOURCE = ROOT / "source_data"
OUT = "r32_null_calibrated_benchmark_summary.csv"


def pct(x: float) -> float:
    return float(100.0 * x)


def main() -> None:
    ladder = pd.read_csv(TABLES / "r20_gauge_matched_surrogate_ladder_summary.csv")
    ladder = ladder[ladder["family"].eq("matched_ar1")].copy()
    boundary = pd.read_csv(TABLES / "r31_gauge_level_ar1_boundary_summary.csv")

    rows: list[dict[str, object]] = []
    for _, row in ladder.sort_values("archive").iterrows():
        archive = str(row["archive"])
        match = boundary[boundary["archive"].eq(archive)]
        if match.empty:
            raise ValueError(f"Missing R31 boundary row for {archive}")
        b = match.iloc[0]
        observed = float(row["observed_reduction"])
        null_median = float(row["null_median_reduction"])
        excess = float(row["observed_minus_null_median"])
        ratio = observed / null_median if np.isfinite(null_median) and null_median != 0 else np.nan
        median_residual = float(b["median_reduction"])
        positive_fraction = float(b["positive_fraction"])

        if excess > 0 and median_residual > 0 and positive_fraction > 0.5:
            verdict = "passes matched-ar1 excess gate"
        elif observed > 0:
            verdict = "coordinate effect positive; matched-ar1 mechanism-excess gate fails"
        else:
            verdict = "coordinate effect not positive"

        rows.append(
            {
                "archive": archive,
                "observed_de_reduction_percent": pct(observed),
                "matched_ar1_null_reduction_percent": pct(null_median),
                "null_adjusted_excess_percent": pct(excess),
                "observed_to_matched_ar1_ratio": ratio,
                "matched_ar1_null_draws": int(row["n_null_draws"]),
                "empirical_p_null_ge_observed": float(row["empirical_p_ge_observed"]),
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
    out_path = TABLES / OUT
    out.to_csv(out_path, index=False)
    shutil.copy2(out_path, LATEX_SOURCE / OUT)
    print(out.to_string(index=False))
    print(f"Wrote {out_path}")
    print(f"Copied {LATEX_SOURCE / OUT}")


if __name__ == "__main__":
    main()
