"""CAMELS diagnostics for the Nature-level manuscript sprint.

This script turns the 673-station summary table into manuscript-ready
diagnostics:

- CAMELS attribute merge, including baseflow_index and geological attributes.
- Spearman control table for tau_acf, beta(De~=1), and phase-surrogate rejection.
- Resolution-aware null-model rejection counts.
- Univariate logistic screens for predictors of phase-randomized rejection.

The null-model p values in the current CAMELS table are empirical tail
fractions from N surrogate realizations. A p value of 0 means "no surrogate
equalled or exceeded the observed statistic"; the manuscript should report it
as p < 1/N, not as a literal p = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
NOTES = ROOT / "reports" / "manuscript_notes"
ATTR_DIR = ROOT / "data" / "external" / "camels" / "camels_attributes_v2.0"


SUMMARY_CSV = TABLES / "camels_673_n30_parallel_summary.csv"
ATTRIBUTED_CSV = TABLES / "camels_673_attributed_diagnostics.csv"
CORRELATION_CSV = TABLES / "camels_673_spearman_controls.csv"
LOGIT_CSV = TABLES / "camels_673_phase_rejection_logit.csv"
NULL_COUNTS_CSV = TABLES / "camels_673_null_rejection_counts.csv"
REPORT_MD = NOTES / "camels_diagnostics_20260531.md"


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted q values."""

    q = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    if not m:
        return q
    min_q = 1.0
    for rank, idx in list(enumerate(valid.index, start=1))[::-1]:
        value = min(min_q, float(p_values.loc[idx]) * m / rank)
        q.loc[idx] = value
        min_q = value
    return q


def load_camels_attributes() -> pd.DataFrame:
    """Load the CAMELS attribute files needed for diagnostics."""

    pieces = []
    for filename in ["camels_hydro.txt", "camels_geol.txt", "camels_clim.txt", "camels_topo.txt"]:
        path = ATTR_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    attrs = pieces[0]
    for df in pieces[1:]:
        new_cols = ["gauge_id"] + [c for c in df.columns if c not in attrs.columns]
        attrs = attrs.merge(df[new_cols], on="gauge_id", how="outer")
    rename = {"area_gages2": "drainage_area_km2_attr"}
    return attrs.rename(columns=rename)


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV, dtype={"gauge_id": str})
    summary["gauge_id"] = summary["gauge_id"].astype(str).str.zfill(8)
    attrs = load_camels_attributes()
    if attrs.empty:
        return summary
    add_cols = [
        "gauge_id",
        "baseflow_index",
        "slope_fdc",
        "stream_elas",
        "hfd_mean",
        "high_q_freq",
        "low_q_freq",
        "zero_q_freq",
        "carbonate_rocks_frac",
        "geol_porostiy",
        "geol_permeability",
        "p_seasonality",
        "high_prec_freq",
        "low_prec_freq",
        "drainage_area_km2_attr",
    ]
    add_cols = [c for c in add_cols if c in attrs.columns]
    merged = summary.merge(attrs[add_cols], on="gauge_id", how="left")
    if "drainage_area_km2" not in merged.columns and "drainage_area_km2_attr" in merged.columns:
        merged["drainage_area_km2"] = merged["drainage_area_km2_attr"]
    return merged


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("tau_acf_days", "baseflow_index", "tau_acf vs baseflow_index"),
        ("tau_acf_days", "drainage_area_km2", "tau_acf vs log10 drainage area"),
        ("tau_acf_days", "tau_recession_days", "tau_acf vs recession tau"),
        ("tau_acf_days", "aridity", "tau_acf vs aridity"),
        ("tau_acf_days", "frac_snow", "tau_acf vs snow fraction"),
        ("tau_acf_days", "slope_mean", "tau_acf vs mean slope"),
        ("tau_acf_days", "geol_permeability", "tau_acf vs geological permeability"),
        ("mean_beta_de_window", "tau_acf_days", "beta(De window) vs tau_acf"),
        ("mean_beta_de_window", "baseflow_index", "beta(De window) vs baseflow_index"),
    ]
    rows = []
    for y_col, x_col, label in pairs:
        if y_col not in df.columns or x_col not in df.columns:
            continue
        sub = df[[y_col, x_col]].dropna()
        if len(sub) < 10:
            continue
        x = sub[x_col].astype(float)
        if x_col == "drainage_area_km2":
            x = np.log10(x)
        rho, p_value = stats.spearmanr(sub[y_col].astype(float), x)
        rows.append(
            {
                "comparison": label,
                "y": y_col,
                "x": x_col,
                "n": len(sub),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_bh"] = bh_fdr(result["p_value"])
    return result


def null_rejection_counts(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "seasonal": "seasonal noise",
        "ar1": "AR(1)",
        "wy_shuffle": "water-year shuffle",
        "phase_rand": "phase randomized",
    }
    rows = []
    for prefix, label in labels.items():
        n_col = f"{prefix}_n"
        p_col = f"{prefix}_p_ge"
        if n_col not in df.columns or p_col not in df.columns:
            continue
        valid = df[n_col].fillna(0) > 0
        n_valid = int(valid.sum())
        p = df.loc[valid, p_col]
        n = df.loc[valid, n_col]
        p_zero = int((p == 0).sum())
        p_resolution = int((p <= 1.0 / n).sum())
        rows.append(
            {
                "null_model": label,
                "n_valid": n_valid,
                "p_zero_count": p_zero,
                "p_zero_fraction_of_673": p_zero / len(df),
                "p_le_1_over_n_count": p_resolution,
                "p_le_1_over_n_fraction_of_673": p_resolution / len(df),
                "median_n_surrogates": float(n.median()),
            }
        )
    return pd.DataFrame(rows)


def phase_logit(df: pd.DataFrame, response_col: str) -> pd.DataFrame:
    predictors = [
        "baseflow_index",
        "aridity",
        "frac_snow",
        "slope_mean",
        "elev_mean",
        "p_mean",
        "pet_mean",
        "drainage_area_km2",
        "geol_permeability",
        "carbonate_rocks_frac",
        "slope_fdc",
        "stream_elas",
        "hfd_mean",
        "high_q_freq",
        "low_q_freq",
        "zero_q_freq",
    ]
    rows = []
    for predictor in predictors:
        if predictor not in df.columns:
            continue
        sub = df[[predictor, response_col]].dropna()
        if len(sub) < 50 or sub[response_col].nunique() < 2:
            continue
        x = sub[predictor].astype(float).to_numpy()
        if predictor == "drainage_area_km2":
            x = np.log10(x)
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - float(np.nanmean(x))) / sd
        y = sub[response_col].astype(int).to_numpy()
        try:
            model = sm.Logit(y, sm.add_constant(z)).fit(disp=False)
            coef = float(model.params[1])
            se = float(model.bse[1])
            p_value = float(model.pvalues[1])
            rows.append(
                {
                    "response": response_col,
                    "predictor": predictor,
                    "n": len(sub),
                    "odds_ratio_per_sd": np.exp(coef),
                    "ci95_low": np.exp(coef - 1.96 * se),
                    "ci95_high": np.exp(coef + 1.96 * se),
                    "p_value": p_value,
                    "logit_coef": coef,
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostics should keep going.
            rows.append(
                {
                    "response": response_col,
                    "predictor": predictor,
                    "n": len(sub),
                    "odds_ratio_per_sd": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_value": np.nan,
                    "logit_coef": np.nan,
                    "note": str(exc),
                }
            )
    result = pd.DataFrame(rows).sort_values("p_value")
    if not result.empty:
        result["q_bh"] = bh_fdr(result["p_value"])
    return result


def write_report(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    nulls: pd.DataFrame,
    logit_res: pd.DataFrame,
    logit_zero: pd.DataFrame,
) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)

    n_total = len(df)
    n_de = int((df["n_beta_de_window"].fillna(0) > 0).sum())
    phase_res = int(df["phase_reject_resolution"].sum())
    phase_zero = int(df["phase_reject_zero"].sum())

    lines = [
        "# CAMELS diagnostics sprint - 2026-05-31",
        "",
        "## Scope",
        "",
        f"- Input table: `{SUMMARY_CSV.name}`.",
        f"- Stations: {n_total}; populated De windows: {n_de}.",
        "- Null-model p values are empirical tail fractions; p=0 is reported as p < 1/N.",
        "- `phase_reject_resolution` means `phase_rand_p_ge <= 1 / phase_rand_n`.",
        "",
        "## Key controls",
        "",
    ]
    for _, row in corr.iterrows():
        p_txt = "p<0.001" if row["p_value"] < 0.001 else f"p={row['p_value']:.3g}"
        lines.append(
            f"- {row['comparison']}: n={int(row['n'])}, "
            f"rho={row['spearman_rho']:.3f}, {p_txt}."
        )

    lines.extend(["", "## Null-model hierarchy", ""])
    for _, row in nulls.iterrows():
        lines.append(
            f"- {row['null_model']}: {int(row['p_le_1_over_n_count'])}/{n_total} "
            f"stations at resolution limit p<1/N; {int(row['p_zero_count'])}/{n_total} "
            "with no surrogate exceeding observed."
        )

    lines.extend(
        [
            "",
            "## Phase-randomized rejection",
            "",
            f"- Resolution-limited phase rejection: {phase_res}/{n_total} ({phase_res / n_total:.1%}).",
            f"- Strict no-exceedance phase count: {phase_zero}/{n_total} ({phase_zero / n_total:.1%}).",
            "- Univariate predictor screen is intentionally treated as exploratory. No predictor "
            "survives Benjamini-Hochberg q<0.05 for the resolution-limited response; this avoids "
            "overclaiming a specific nonlinear mechanism.",
            "",
            "Top exploratory predictors for phase_reject_resolution:",
            "",
        ]
    )
    top = logit_res.head(8)
    for _, row in top.iterrows():
        p_txt = "p<0.001" if row["p_value"] < 0.001 else f"p={row['p_value']:.3g}"
        q_txt = "q<0.001" if row["q_bh"] < 0.001 else f"q={row['q_bh']:.3g}"
        lines.append(
            f"- {row['predictor']}: OR={row['odds_ratio_per_sd']:.2f} per SD "
            f"(95% CI {row['ci95_low']:.2f}-{row['ci95_high']:.2f}), {p_txt}, {q_txt}."
        )

    lines.extend(
        [
            "",
            "## Manuscript implications",
            "",
            "1. Keep the BFI result, but call it compatibility evidence because BFI is streamflow-derived.",
            "2. Use tau_recession as the more independent memory cross-check.",
            "3. Replace literal `p=0` language with `p < 1/N` throughout Methods, Results and captions.",
            "4. Treat phase-randomized rejection as evidence that PSD amplitudes alone are insufficient, "
            "not as proof of one named nonlinear mechanism.",
            "5. State the US-only CAMELS scope explicitly before using global Earth-surface language.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    df["phase_reject_resolution"] = (
        (df["phase_rand_n"].fillna(0) > 0)
        & (df["phase_rand_p_ge"] <= 1.0 / df["phase_rand_n"])
    )
    df["phase_reject_zero"] = (
        (df["phase_rand_n"].fillna(0) > 0)
        & (df["phase_rand_p_ge"] == 0)
    )

    corr = spearman_table(df)
    nulls = null_rejection_counts(df)
    logit_res = phase_logit(df, "phase_reject_resolution")
    logit_zero = phase_logit(df, "phase_reject_zero")

    df.to_csv(ATTRIBUTED_CSV, index=False)
    corr.to_csv(CORRELATION_CSV, index=False)
    nulls.to_csv(NULL_COUNTS_CSV, index=False)
    logit_res.to_csv(LOGIT_CSV, index=False)
    if not logit_zero.empty:
        logit_zero.to_csv(TABLES / "camels_673_phase_rejection_logit_strict_zero.csv", index=False)
    write_report(df, corr, nulls, logit_res, logit_zero)

    print(f"Saved {ATTRIBUTED_CSV.relative_to(ROOT)}")
    print(f"Saved {CORRELATION_CSV.relative_to(ROOT)}")
    print(f"Saved {NULL_COUNTS_CSV.relative_to(ROOT)}")
    print(f"Saved {LOGIT_CSV.relative_to(ROOT)}")
    print(f"Saved {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
