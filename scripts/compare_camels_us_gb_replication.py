"""Create a compact US/GB replication comparison table for manuscript review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OUT = TABLES / "camels_us_gb_replication_comparison.csv"


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    work = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 5:
        return np.nan, np.nan, int(len(work))
    rho, p_value = stats.spearmanr(work["x"], work["y"])
    return float(rho), float(p_value), int(len(work))


def random_p(null_values: pd.Series, observed: float) -> float:
    values = null_values.dropna().to_numpy(dtype=float)
    return float((np.sum(values >= observed) + 1) / (len(values) + 1))


def main() -> None:
    us_summary = pd.read_csv(TABLES / "camels_673_n30_parallel_summary.csv", dtype={"gauge_id": str})
    us_metrics = pd.read_csv(TABLES / "camels_673_alternative_normalization_metrics.csv")
    us_random = pd.read_csv(TABLES / "camels_673_random_tau_normalization_null.csv")
    gb_summary = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv", dtype={"gauge_id": str})
    gb_metrics = pd.read_csv(TABLES / "camels_gb_v2_replication_collapse_metrics.csv")
    gb_random = pd.read_csv(TABLES / "camels_gb_v2_replication_random_tau_null.csv")

    us_de = float(us_metrics.loc[us_metrics["method"] == "tau_acf", "variance_reduction_vs_matched_raw"].iloc[0])
    us_const = float(us_metrics.loc[us_metrics["method"] == "constant_tau", "variance_reduction_vs_matched_raw"].iloc[0])
    gb_de = float(gb_metrics.loc[gb_metrics["axis"] == "de_normalized", "variance_reduction_vs_raw"].iloc[0])
    gb_const = float(gb_metrics.loc[gb_metrics["axis"] == "constant_tau", "variance_reduction_vs_raw"].iloc[0])

    us_bfi_rho, us_bfi_p, us_bfi_n = spearman(us_summary["tau_acf_days"], us_summary["baseflow_index"])
    us_area_rho, us_area_p, us_area_n = spearman(us_summary["tau_acf_days"], np.log10(us_summary["drainage_area_km2"]))
    gb_bfi_rho, gb_bfi_p, gb_bfi_n = spearman(gb_summary["tau_acf_days"], gb_summary["baseflow_index"])
    gb_area_rho, gb_area_p, gb_area_n = spearman(gb_summary["tau_acf_days"], np.log10(gb_summary["drainage_area_km2"]))

    rows = [
        {
            "archive": "CAMELS-US",
            "collapse_metric_definition": "weighted cross-catchment variance of station-median local beta across log-spaced bins",
            "n_stations": int(us_summary["tau_acf_days"].notna().sum()),
            "median_tau_acf_days": float(us_summary["tau_acf_days"].median()),
            "tau_acf_iqr_low_days": float(us_summary["tau_acf_days"].quantile(0.25)),
            "tau_acf_iqr_high_days": float(us_summary["tau_acf_days"].quantile(0.75)),
            "de_variance_reduction_vs_raw": us_de,
            "constant_tau_reduction_vs_raw": us_const,
            "random_tau_median_reduction": float(us_random["variance_reduction_vs_matched_raw"].median()),
            "random_tau_95th_reduction": float(us_random["variance_reduction_vs_matched_raw"].quantile(0.95)),
            "random_tau_empirical_p_ge_observed": random_p(us_random["variance_reduction_vs_matched_raw"], us_de),
            "random_tau_p_note": "300 shuffled tau assignments; p=(count>=observed+1)/(300+1)",
            "rho_tau_bfi": us_bfi_rho,
            "p_tau_bfi": us_bfi_p,
            "n_tau_bfi": us_bfi_n,
            "rho_tau_log_area": us_area_rho,
            "p_tau_log_area": us_area_p,
            "n_tau_log_area": us_area_n,
        },
        {
            "archive": "CAMELS-GB v2",
            "collapse_metric_definition": "weighted cross-catchment variance of station-median local beta across log-spaced bins",
            "n_stations": int(gb_summary["tau_acf_days"].notna().sum()),
            "median_tau_acf_days": float(gb_summary["tau_acf_days"].median()),
            "tau_acf_iqr_low_days": float(gb_summary["tau_acf_days"].quantile(0.25)),
            "tau_acf_iqr_high_days": float(gb_summary["tau_acf_days"].quantile(0.75)),
            "de_variance_reduction_vs_raw": gb_de,
            "constant_tau_reduction_vs_raw": gb_const,
            "random_tau_median_reduction": float(gb_random["variance_reduction_vs_raw"].median()),
            "random_tau_95th_reduction": float(gb_random["variance_reduction_vs_raw"].quantile(0.95)),
            "random_tau_empirical_p_ge_observed": random_p(gb_random["variance_reduction_vs_raw"], gb_de),
            "random_tau_p_note": "300 shuffled tau assignments; p=(count>=observed+1)/(300+1)",
            "rho_tau_bfi": gb_bfi_rho,
            "p_tau_bfi": gb_bfi_p,
            "n_tau_bfi": gb_bfi_n,
            "rho_tau_log_area": gb_area_rho,
            "p_tau_log_area": gb_area_p,
            "n_tau_log_area": gb_area_n,
        },
    ]
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
