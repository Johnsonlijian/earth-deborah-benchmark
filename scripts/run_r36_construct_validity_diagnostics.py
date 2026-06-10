"""R36 construct-validity diagnostics for De=tau_acf*f.

The diagnostics consolidate two reviewer-facing checks:

1. frequency--tau support: because De multiplies a frequency grid by each
   gauge's output-memory time, an apparent collapse must not be described as
   independent causal evidence;
2. seasonality controls: deterministic annual structure should not be allowed
   to masquerade as catchment memory.

The output is a compact source-data table and manuscript note, not a new claim
of direct storage or tracer causality.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX_SOURCE = ROOT / "source_data"
PREFIX = "r36_construct_validity_diagnostics"


ARCHIVE_CURVES = {
    "CAMELS-US": "camels_673_beta_curve_points.csv",
    "CAMELS-GB": "camels_gb_v2_replication_beta_curve_points.csv",
    "CAMELS-BR": "r23_camels_br_third_archive_curves.csv",
    "CAMELS-AUS": "r25_camels_aus_fourth_archive_curves.csv",
    "CAMELS-DK": "r27_camels_dk_groundwater_storage_validation_curves.csv",
}


def safe_spearman(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    work = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 3:
        return np.nan
    return float(work[x_col].corr(work[y_col], method="spearman"))


def archive_coordinate_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for archive, filename in ARCHIVE_CURVES.items():
        path = TABLES / filename
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if "tau_acf_days" not in data.columns and "tau_days" in data.columns:
            data["tau_acf_days"] = data["tau_days"]
        required = {"gauge_id", "tau_acf_days", "frequency_cpd", "de", "beta"}
        if not required.issubset(data.columns):
            continue
        work = data[list(required)].replace([np.inf, -np.inf], np.nan).dropna()
        work = work[(work["tau_acf_days"] > 0) & (work["frequency_cpd"] > 0) & (work["de"] > 0)]
        if work.empty:
            continue
        work["log10_tau"] = np.log10(work["tau_acf_days"])
        work["log10_frequency"] = np.log10(work["frequency_cpd"])
        work["log10_de"] = np.log10(work["de"])
        de_window = work[(work["de"] >= 0.5) & (work["de"] <= 2.0)].copy()
        rows.append(
            {
                "diagnostic_family": "coordinate_support",
                "archive": archive,
                "n_gauges": int(work["gauge_id"].nunique()),
                "n_curve_points": int(len(work)),
                "spearman_log_frequency_log_tau": safe_spearman(work, "log10_frequency", "log10_tau"),
                "spearman_beta_log_tau": safe_spearman(work, "beta", "log10_tau"),
                "spearman_beta_log_frequency": safe_spearman(work, "beta", "log10_frequency"),
                "spearman_beta_log_de": safe_spearman(work, "beta", "log10_de"),
                "de_window_points": int(len(de_window)),
                "de_window_frequency_iqr_log10": float(
                    np.nanpercentile(de_window["log10_frequency"], 75)
                    - np.nanpercentile(de_window["log10_frequency"], 25)
                )
                if not de_window.empty
                else np.nan,
                "de_window_tau_iqr_log10": float(
                    np.nanpercentile(de_window["log10_tau"], 75) - np.nanpercentile(de_window["log10_tau"], 25)
                )
                if not de_window.empty
                else np.nan,
                "interpretation": "frequency and tau support for De bins; not independent causal proof",
            }
        )
    return rows


def lookup_metric(path: Path, group: str) -> float:
    if not path.exists():
        return np.nan
    data = pd.read_csv(path)
    if "scenario_group" not in data.columns or "variance_reduction_vs_raw" not in data.columns:
        return np.nan
    hit = data[data["scenario_group"] == group]
    if hit.empty:
        return np.nan
    return float(hit["variance_reduction_vs_raw"].iloc[0])


def seasonality_rows() -> list[dict[str, float | int | str]]:
    fourier = TABLES / "fourier_pair_circularity_baseline_metrics.csv"
    synthetic = TABLES / "synthetic_reservoir_process_control_metrics.csv"
    r36 = TABLES / "r36_multiscale_process_family_nulls_metrics.csv"
    rows = [
        {
            "diagnostic_family": "seasonality_control",
            "archive": "synthetic_fourier_pair",
            "n_gauges": np.nan,
            "n_curve_points": np.nan,
            "spearman_log_frequency_log_tau": np.nan,
            "spearman_beta_log_tau": np.nan,
            "spearman_beta_log_frequency": np.nan,
            "spearman_beta_log_de": np.nan,
            "de_window_points": np.nan,
            "de_window_frequency_iqr_log10": np.nan,
            "de_window_tau_iqr_log10": np.nan,
            "variance_reduction_vs_raw": lookup_metric(fourier, "seasonal_ar1_control"),
            "comparison_reduction_vs_raw": lookup_metric(fourier, "ar1_fourier_pair"),
            "interpretation": "seasonal AR(1) without anomaly removal worsens De dispersion",
        },
        {
            "diagnostic_family": "seasonality_control",
            "archive": "synthetic_storage_release",
            "n_gauges": np.nan,
            "n_curve_points": np.nan,
            "spearman_log_frequency_log_tau": np.nan,
            "spearman_beta_log_tau": np.nan,
            "spearman_beta_log_frequency": np.nan,
            "spearman_beta_log_de": np.nan,
            "de_window_points": np.nan,
            "de_window_frequency_iqr_log10": np.nan,
            "de_window_tau_iqr_log10": np.nan,
            "variance_reduction_vs_raw": lookup_metric(synthetic, "seasonal_forcing_control"),
            "comparison_reduction_vs_raw": lookup_metric(synthetic, "linear_reservoir"),
            "interpretation": "strong seasonal forcing in storage simulation worsens De dispersion",
        },
        {
            "diagnostic_family": "seasonality_control",
            "archive": "r36_dayofyear_anomaly",
            "n_gauges": np.nan,
            "n_curve_points": np.nan,
            "spearman_log_frequency_log_tau": np.nan,
            "spearman_beta_log_tau": np.nan,
            "spearman_beta_log_frequency": np.nan,
            "spearman_beta_log_de": np.nan,
            "de_window_points": np.nan,
            "de_window_frequency_iqr_log10": np.nan,
            "de_window_tau_iqr_log10": np.nan,
            "variance_reduction_vs_raw": lookup_metric(r36, "seasonal_two_timescale_control"),
            "comparison_reduction_vs_raw": lookup_metric(r36, "two_timescale_ar_mix"),
            "interpretation": "annual component is added before day-of-year anomaly removal; result remains a boundary control",
        },
    ]
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)
    rows = archive_coordinate_rows() + seasonality_rows()
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"{PREFIX}.csv", index=False)
    shutil.copy2(TABLES / f"{PREFIX}.csv", LATEX_SOURCE / f"{PREFIX}.csv")
    lines = [
        "# R36 Construct-Validity Diagnostics",
        "",
        "Date: 2026-06-05",
        "",
        "These checks consolidate De coordinate-support and seasonality controls.",
        "They are meant to calibrate claim strength rather than prove direct",
        "storage or tracer causality.",
        "",
        "## Table",
        "",
        out.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Safe use in manuscript",
        "",
        "- Treat De as an output-memory coordinate whose frequency--tau support is",
        "  explicitly audited.",
        "- Do not describe De alignment as independent causal evidence because tau",
        "  and PSD are same-series summaries.",
        "- Seasonal controls show that annual forcing is not a sufficient positive",
        "  explanation; it can worsen De dispersion or act only as a boundary.",
    ]
    (NOTES / f"{PREFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
