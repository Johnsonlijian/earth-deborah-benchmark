"""Partial-correlation checks for CAMELS tau_acf versus baseflow index."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
INPUT = TABLES / "camels_673_attributed_diagnostics.csv"
OUT_CSV = TABLES / "camels_673_bfi_partial_controls.csv"
OUT_MD = NOTES / "bfi_partial_controls_20260531.md"


def rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="average", na_option="keep")


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict[str, float | str | int]:
    cols = [x, y, *controls]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < max(10, len(controls) + 3):
        return {
            "model": "+".join(controls) if controls else "unadjusted",
            "controls": ", ".join(controls),
            "n": len(work),
            "partial_rho": np.nan,
            "p_value": np.nan,
            "r2_x": np.nan,
            "r2_y": np.nan,
        }
    for col in cols:
        work[col] = rank_series(work[col])
    if not controls:
        rho, p = stats.pearsonr(work[x], work[y])
        return {
            "model": "unadjusted",
            "controls": "",
            "n": len(work),
            "partial_rho": float(rho),
            "p_value": float(p),
            "r2_x": 0.0,
            "r2_y": 0.0,
        }
    X = work[controls].to_numpy(dtype=float)
    x_vals = work[x].to_numpy(dtype=float)
    y_vals = work[y].to_numpy(dtype=float)
    reg_x = LinearRegression().fit(X, x_vals)
    reg_y = LinearRegression().fit(X, y_vals)
    x_res = x_vals - reg_x.predict(X)
    y_res = y_vals - reg_y.predict(X)
    rho, p = stats.pearsonr(x_res, y_res)
    return {
        "model": "+".join(controls),
        "controls": ", ".join(controls),
        "n": len(work),
        "partial_rho": float(rho),
        "p_value": float(p),
        "r2_x": float(reg_x.score(X, x_vals)),
        "r2_y": float(reg_y.score(X, y_vals)),
    }


def build_analysis_frame() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df["log_tau_acf_days"] = np.log10(df["tau_acf_days"].where(df["tau_acf_days"] > 0))
    df["log_area"] = np.log10(df["drainage_area_km2"].where(df["drainage_area_km2"] > 0))
    # CAMELS geol_permeability is already log10 permeability.
    df["geol_permeability_log10"] = df["geol_permeability"]
    df["log_q_mean"] = np.log10(df["q_mean"].where(df["q_mean"] > 0))
    return df


def run_controls(df: pd.DataFrame) -> pd.DataFrame:
    control_sets = [
        ("unadjusted", []),
        ("area", ["log_area"]),
        ("climate", ["p_mean", "pet_mean", "aridity", "frac_snow", "p_seasonality"]),
        ("topography", ["elev_mean", "slope_mean", "log_area"]),
        ("hydroclimate", ["p_mean", "pet_mean", "aridity", "frac_snow", "runoff_ratio", "log_q_mean"]),
        ("geology", ["carbonate_rocks_frac", "geol_porostiy", "geol_permeability_log10"]),
        (
            "combined_core",
            ["log_area", "aridity", "frac_snow", "elev_mean", "slope_mean", "runoff_ratio"],
        ),
        (
            "combined_extended",
            [
                "log_area",
                "p_mean",
                "pet_mean",
                "aridity",
                "frac_snow",
                "elev_mean",
                "slope_mean",
                "runoff_ratio",
                "p_seasonality",
            ],
        ),
    ]
    rows = []
    for label, controls in control_sets:
        result = partial_spearman(df, "log_tau_acf_days", "baseflow_index", controls)
        result["model"] = label
        rows.append(result)
    return pd.DataFrame(rows)


def permutation_pvalue(df: pd.DataFrame, controls: list[str], n_perm: int = 2000, seed: int = 20260531) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    obs = partial_spearman(df, "log_tau_acf_days", "baseflow_index", controls)
    cols = ["log_tau_acf_days", "baseflow_index", *controls]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    obs_abs = abs(float(obs["partial_rho"]))
    perm = []
    for _ in range(n_perm):
        shuffled = work.copy()
        shuffled["baseflow_index"] = rng.permutation(shuffled["baseflow_index"].to_numpy())
        pr = partial_spearman(shuffled, "log_tau_acf_days", "baseflow_index", controls)
        perm.append(abs(float(pr["partial_rho"])))
    arr = np.asarray(perm, dtype=float)
    return {
        "observed_partial_rho": float(obs["partial_rho"]),
        "permutation_p_two_sided": float((np.sum(arr >= obs_abs) + 1) / (len(arr) + 1)),
        "n_perm": n_perm,
    }


def plot_results(results: pd.DataFrame) -> None:
    plot = results.copy()
    plot["label"] = plot["model"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.1), constrained_layout=True)
    y = np.arange(len(plot))
    colors = ["#1f6f78" if v >= 0.5 else "#c4513f" for v in plot["partial_rho"]]
    ax.barh(y, plot["partial_rho"], color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.axvline(0.5, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("partial Spearman rho (rank residual)")
    ax.set_title("BFI-memory association after covariate controls", loc="left", fontsize=9)
    ax.grid(True, axis="x", color="#e5e2dc", lw=0.6)
    out = FIGURES / "nature_extended_bfi_partial_controls"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write_report(results: pd.DataFrame, perm: dict[str, float]) -> None:
    core = results.loc[results["model"] == "combined_core"].iloc[0]
    ext = results.loc[results["model"] == "combined_extended"].iloc[0]
    lines = [
        "# BFI Partial-Control Analysis",
        "",
        "Date: 2026-05-31",
        "",
        "## Method",
        "",
        "Partial Spearman checks were implemented as rank-residual correlations. log10(tau_acf) and baseflow index were rank transformed, residualized against covariate groups using linear regression, and then correlated. This does not remove the shared streamflow-derived nature of BFI, but it tests whether the association is only a proxy for area, climate or topography.",
        "",
        "## Result",
        "",
        f"- Unadjusted rank correlation: rho = {results.loc[results['model'] == 'unadjusted', 'partial_rho'].iloc[0]:.3f}.",
        f"- Combined core controls (area, aridity, snow fraction, elevation, slope, runoff ratio): partial rho = {core['partial_rho']:.3f}, p = {core['p_value']:.2e}, n = {int(core['n'])}.",
        f"- Combined extended controls (area, precipitation, PET, aridity, snow fraction, elevation, slope, runoff ratio, precipitation seasonality): partial rho = {ext['partial_rho']:.3f}, p = {ext['p_value']:.2e}, n = {int(ext['n'])}.",
        f"- Permutation check for combined core controls: two-sided p = {perm['permutation_p_two_sided']:.4f} over {int(perm['n_perm'])} permutations.",
        "",
        "## Interpretation",
        "",
        "The BFI-memory association remains strong after broad covariate control. The manuscript can say that the association is not explained away by drainage area, simple climate gradients, topography or runoff ratio. It still cannot say BFI is a fully independent causal proof of groundwater storage, because BFI is streamflow-derived.",
        "",
        "## Outputs",
        "",
        "- `reports/tables/camels_673_bfi_partial_controls.csv`",
        "- `reports/figures/nature_extended_bfi_partial_controls.png`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


def main() -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = build_analysis_frame()
    results = run_controls(df)
    perm = permutation_pvalue(
        df,
        ["log_area", "aridity", "frac_snow", "elev_mean", "slope_mean", "runoff_ratio"],
        n_perm=2000,
    )
    results.to_csv(OUT_CSV, index=False)
    plot_results(results)
    write_report(results, perm)


if __name__ == "__main__":
    main()
