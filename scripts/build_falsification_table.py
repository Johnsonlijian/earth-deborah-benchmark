"""Build a cross-process falsification table from benchmark summary outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_BETA = 5.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports/tables")
    parser.add_argument("--output", default="reports/tables/falsification_table.csv")
    parser.add_argument("--tolerance", type=float, default=0.25)
    return parser.parse_args()


def status_from_beta(beta: float, n_window: int, tolerance: float) -> tuple[str, str]:
    """Classify one beta-window row against the 5/3 benchmark."""

    if n_window <= 0 or not np.isfinite(beta):
        return "no_test", "No finite local beta estimates in 0.5<=De<=2."
    distance = abs(beta - TARGET_BETA)
    if distance <= tolerance:
        return "near_5_3", f"Mean beta is within {tolerance:g} of 5/3."
    if beta < TARGET_BETA - tolerance:
        return "below_5_3", f"Mean beta is {distance:.3g} below 5/3."
    return "above_5_3", f"Mean beta is {distance:.3g} above 5/3."


def add_standard_rows(rows: list[dict[str, object]], path: Path, df: pd.DataFrame, tolerance: float) -> None:
    """Append rows from standard summary tables."""

    if "mean_beta_de_window" not in df.columns:
        return
    for _, item in df.iterrows():
        beta = pd.to_numeric(item.get("mean_beta_de_window"), errors="coerce")
        n_window = int(pd.to_numeric(item.get("n_beta_de_window", 0), errors="coerce") or 0)
        status, reason = status_from_beta(float(beta), n_window, tolerance)
        rows.append(
            {
                "source_table": path.name,
                "process": item.get("process", infer_process(path.name)),
                "subset": subset_label(item),
                "variable": item.get("variable", infer_variable(path.name)),
                "tau_days": item.get("tau_days", item.get("tau_acf_days")),
                "n_observations": item.get("n_events", item.get("n_records", item.get("n_months", ""))),
                "mean_beta_de_window": beta,
                "distance_to_5_3": abs(float(beta) - TARGET_BETA) if np.isfinite(beta) else np.nan,
                "n_beta_de_window": n_window,
                "null_context": "",
                "falsification_status": status,
                "reason": reason,
            }
        )


def add_landslide_null_rows(rows: list[dict[str, object]], path: Path, df: pd.DataFrame, tolerance: float) -> None:
    """Append trigger/null rows using observed and null columns."""

    required = {"observed_mean_beta_de_window", "observed_n_beta_de_window"}
    if not required.issubset(df.columns):
        return
    for _, item in df.iterrows():
        beta = pd.to_numeric(item.get("observed_mean_beta_de_window"), errors="coerce")
        n_window = int(pd.to_numeric(item.get("observed_n_beta_de_window", 0), errors="coerce") or 0)
        status, reason = status_from_beta(float(beta), n_window, tolerance)
        null_bits: list[str] = []
        for null_name in ["seasonal_noise", "phase_randomized"]:
            p95 = pd.to_numeric(item.get(f"{null_name}_p95"), errors="coerce")
            p_ge = pd.to_numeric(item.get(f"{null_name}_p_ge_observed"), errors="coerce")
            if np.isfinite(p95):
                null_bits.append(f"{null_name}_p95={p95:.3g}")
            if np.isfinite(p_ge):
                null_bits.append(f"{null_name}_p_ge_observed={p_ge:.3g}")
        rows.append(
            {
                "source_table": path.name,
                "process": "landslide",
                "subset": item.get("group", ""),
                "variable": "monthly_count",
                "tau_days": item.get("tau_days"),
                "n_observations": item.get("n_events", item.get("n_months", "")),
                "mean_beta_de_window": beta,
                "distance_to_5_3": abs(float(beta) - TARGET_BETA) if np.isfinite(beta) else np.nan,
                "n_beta_de_window": n_window,
                "null_context": "; ".join(null_bits),
                "falsification_status": status,
                "reason": reason,
            }
        )


def infer_process(filename: str) -> str:
    """Infer process name from a summary filename."""

    if "earthquake" in filename:
        return "earthquake"
    if "landslide" in filename:
        return "landslide"
    if "river" in filename or "nwis" in filename:
        return "river"
    if "glacier" in filename:
        return "glacier"
    if "synthetic" in filename:
        return "synthetic"
    return "unknown"


def infer_variable(filename: str) -> str:
    """Infer variable name from a summary filename."""

    if "nwis" in filename:
        return "nwis_discharge_anomaly"
    if "river" in filename:
        return "discharge_anomaly"
    if "glacier" in filename:
        return "velocity_anomaly"
    return ""


def subset_label(item: pd.Series) -> str:
    """Build a compact subset label from common summary columns."""

    parts: list[str] = []
    for column in ["region", "group", "station_id", "depth_class", "minmagnitude", "resample_rule", "input_kind"]:
        value = item.get(column)
        if pd.notna(value):
            parts.append(f"{column}={value}")
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    rows: list[dict[str, object]] = []
    for path in sorted(reports_dir.glob("*summary.csv")):
        if path.name == Path(args.output).name:
            continue
        df = pd.read_csv(path, dtype=str)
        add_landslide_null_rows(rows, path, df, args.tolerance)
        add_standard_rows(rows, path, df, args.tolerance)

    if not rows:
        raise SystemExit("No benchmark summary rows found")
    out = pd.DataFrame(rows).sort_values(["process", "source_table", "subset", "tau_days"], na_position="last")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved {output}")
    print(out["falsification_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
