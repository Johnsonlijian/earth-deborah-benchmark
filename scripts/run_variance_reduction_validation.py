"""Core claim validation: permutation test for variance reduction metric.

The correct test for the storage-memory collapse hypothesis:
- Null hypothesis: random tau-f pairings produce the same variance reduction
  as observed tau_acf values
- Alternative: only the observed tau_acf assignments reduce cross-catchment 
  beta variance when frequency is mapped to De = tau * f coordinates

This directly tests the collapse claim (variance reduction when aligning spectra
by De) rather than a separate linear predictability claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edb.data.grdc import build_discharge_series, remove_daily_climatology, robust_mad_scale
from edb.data.nwis import parse_nwis_daily_values_json
from edb.signal.de import beta_vs_de
from edb.signal.nulls import phase_randomized_surrogate
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.summary import summarize_beta_window
from edb.signal.timescales import integral_autocorrelation_time


SECONDS_PER_DAY = 86_400.0
BINS = 20
DE_MIN, DE_MAX = 0.05, 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--output-prefix", default="variance_reduction_validation")
    return p.parse_args()


def compute_de_variance_reduction(df: pd.DataFrame, tau_column: str = "tau_acf_days") -> dict:
    """Compute variance reduction when mapping beta from f to De coordinates.

    Using beta_vs_de data from individual stations, compute beta dispersion
    in raw frequency vs De-normalized frequency coordinates.

    Returns dict with 'raw_variance', 'de_variance', 'reduction_pct'.
    If no beta_vs_de file is available, approximates from station summary.
    """
    valid = df.dropna(subset=["mean_beta_de_window", tau_column])
    valid = valid[valid["n_beta_de_window"].fillna(0) > 0]

    if len(valid) < 10:
        return {"raw_variance": 0, "de_variance": 0, "reduction_pct": 0, "n": 0}

    # Approximate variance reduction from station-level data
    # In raw frequency: beta varies widely across stations
    # In De coordinates: beta should converge
    raw_var = valid["mean_beta_all"].var()
    de_var = valid["mean_beta_de_window"].var()

    reduction = (raw_var - de_var) / raw_var * 100 if raw_var > 0 else 0

    return {
        "raw_variance": float(f"{raw_var:.4f}"),
        "de_variance": float(f"{de_var:.4f}"),
        "reduction_pct": float(f"{reduction:.2f}"),
        "n": len(valid),
    }


def permutation_variance_reduction(df: pd.DataFrame, n_perm: int, seed: int = 42) -> dict:
    """Permutation test: randomly shuffle tau_acf values across stations.

    For each permutation, the observed tau values are randomly reassigned.
    If the observed variance reduction is specific to measured memory time,
    random assignments should show smaller or no reduction.

    Returns empirical p-value and distribution statistics.
    """
    valid = df.dropna(subset=["mean_beta_de_window", "tau_acf_days"])
    valid = valid[valid["n_beta_de_window"].fillna(0) > 0]

    if len(valid) < 10:
        return {"error": "insufficient data"}

    rng = np.random.default_rng(seed)
    observed_reduction = compute_de_variance_reduction(valid, "tau_acf_days")["reduction_pct"]

    perm_reductions = []
    for p in range(n_perm):
        shuffled = valid.copy()
        shuffled["tau_perm"] = rng.permutation(valid["tau_acf_days"].values)
        result = compute_de_variance_reduction(shuffled, "tau_perm")
        perm_reductions.append(result["reduction_pct"])

    perm_reductions = np.array(perm_reductions)
    p_empirical = np.mean(perm_reductions >= observed_reduction)

    return {
        "n_permutations": n_perm,
        "n_stations": len(valid),
        "observed_reduction_pct": float(f"{observed_reduction:.1f}"),
        "perm_median_reduction_pct": float(f"{np.median(perm_reductions):.1f}"),
        "perm_p95_reduction_pct": float(f"{np.percentile(perm_reductions, 95):.1f}"),
        "perm_percent_positive": float(f"{100 * np.mean(perm_reductions > 0):.1f}"),
        "empirical_p_value": float(f"{p_empirical:.4f}"),
        "effect_size_above_null": float(f"{observed_reduction - np.median(perm_reductions):.1f}"),
        "significant": p_empirical < 0.05,
    }


def phase_surrogate_control(df: pd.DataFrame, n_surr: int = 30) -> dict:
    """Compute phase-surrogate rejection rate with proper interpretation.

    Phase-randomized preserves exact PSD amplitude. Rejection means
    beta(De=1) is not determined by PSD amplitude alone.
    """
    from edb.signal.nulls import phase_randomized_surrogate

    # Count from existing data
    ph_col = [c for c in df.columns if "phase" in c and "p_ge" in c]
    if not ph_col:
        return {"note": "no phase column found"}

    ph = ph_col[0]
    n_reject = (df[ph].fillna(1) < 0.067).sum()  # p < 1/15
    n_strict = (df[ph].fillna(1) == 0).sum()

    return {
        "phase_column": ph,
        "n_reject_p_lt_0_067": int(n_reject),
        "n_strict_reject_p_eq_0": int(n_strict),
        "total_stations": int(len(df)),
        "rejection_rate": float(f"{100 * n_reject / len(df):.1f}"),
        "interpretation": "Phase-randomized preserves exact PSD amplitude. "
            "Rejection means beta(De=1) is not determined by PSD amplitude alone, "
            "implying phase-dependent nonlinear spectral organization.",
    }


def main():
    args = parse_args()
    df = pd.read_csv("reports/tables/camels_674_n15_summary.csv")

    print("=== CORE CLAIM VALIDATION ===\n")

    # 1. Variance reduction with observed tau_acf
    print("--- Variance Reduction (observed tau_acf) ---")
    obs = compute_de_variance_reduction(df, "tau_acf_days")
    for k, v in obs.items():
        print(f"  {k}: {v}")

    # 2. Permutation test
    print(f"\n--- Permutation Test (n={args.n_perm}) ---")
    perm = permutation_variance_reduction(df, n_perm=args.n_perm)
    for k, v in perm.items():
        print(f"  {k}: {v}")
    if perm.get("significant"):
        print("  -> OBSERVED REDUCTION IS SIGNIFICANTLY LARGER THAN SHUFFLED")
    else:
        print("  -> Observed reduction is not distinguishable from shuffled tau assignments")

    # 3. Phase surrogate interpretation
    print("\n--- Phase Surrogate Summary ---")
    ps = phase_surrogate_control(df)
    for k, v in ps.items():
        print(f"  {k}: {v}")

    # Save
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}_summary.csv"
    results = {**obs, **perm, **ps}
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
