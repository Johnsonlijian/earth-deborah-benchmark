"""Glacier velocity time-series helpers."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "glacier_id",
    "point_id",
    "time",
    "velocity_myr",
    "velocity_error_myr",
    "source",
]


def read_glacier_velocity_csv(source: str | Path | IO[str]) -> pd.DataFrame:
    """Read a glacier velocity time-series CSV."""

    return pd.read_csv(source)


def parse_glacier_velocity_dataframe(
    df: pd.DataFrame,
    glacier_id_col: str = "glacier_id",
    time_col: str = "time",
    velocity_col: str = "velocity_myr",
    point_id_col: str | None = "point_id",
    velocity_error_col: str | None = "velocity_error_myr",
    source_col: str | None = "source",
) -> pd.DataFrame:
    """Parse a generic glacier velocity dataframe into a stable schema."""

    required = {glacier_id_col, time_col, velocity_col}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"glacier dataframe is missing required columns: {missing_cols}")

    out = pd.DataFrame(index=df.index)
    out["glacier_id"] = df[glacier_id_col].astype("string").str.strip()
    out["time"] = pd.to_datetime(df[time_col], errors="coerce")
    out["velocity_myr"] = pd.to_numeric(df[velocity_col], errors="coerce")
    if point_id_col is not None and point_id_col in df.columns:
        out["point_id"] = df[point_id_col].astype("string").str.strip()
    else:
        out["point_id"] = "point_001"
    if velocity_error_col is not None and velocity_error_col in df.columns:
        out["velocity_error_myr"] = pd.to_numeric(df[velocity_error_col], errors="coerce")
    else:
        out["velocity_error_myr"] = np.nan
    if source_col is not None and source_col in df.columns:
        out["source"] = df[source_col].astype("string").str.strip()
    else:
        out["source"] = pd.NA

    out = out.dropna(subset=["glacier_id", "time", "velocity_myr"]).copy()
    return out.loc[:, STANDARD_COLUMNS].sort_values(["glacier_id", "point_id", "time"]).reset_index(drop=True)


def filter_glacier_velocity(
    df: pd.DataFrame,
    glacier_id: str | None = None,
    point_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Filter glacier velocity records."""

    out = df.copy()
    if glacier_id is not None:
        out = out[out["glacier_id"].astype("string") == str(glacier_id)]
    if point_id is not None:
        out = out[out["point_id"].astype("string") == str(point_id)]
    if start_date is not None:
        out = out[out["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["time"] <= pd.Timestamp(end_date)]
    return out.reset_index(drop=True)


def build_velocity_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Build median-centered velocity anomaly for one glacier point."""

    if df.empty:
        return pd.DataFrame(columns=["time", "velocity_myr", "value"])
    if df[["glacier_id", "point_id"]].drop_duplicates().shape[0] > 1:
        raise ValueError("build_velocity_anomaly expects one glacier point")
    out = df.loc[:, ["time", "velocity_myr", "velocity_error_myr"]].copy()
    center = out["velocity_myr"].median()
    mad = (out["velocity_myr"] - center).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        scale = out["velocity_myr"].std()
    if not np.isfinite(scale) or scale == 0:
        out["value"] = 0.0
    else:
        out["value"] = (out["velocity_myr"] - center) / scale
    return out.sort_values("time").reset_index(drop=True)


def summarize_glacier_points(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize record coverage by glacier point."""

    rows: list[dict[str, object]] = []
    for (glacier_id, point_id), group in df.groupby(["glacier_id", "point_id"]):
        start = group["time"].min()
        end = group["time"].max()
        rows.append(
            {
                "glacier_id": glacier_id,
                "point_id": point_id,
                "n_records": int(group.shape[0]),
                "start_time": start,
                "end_time": end,
                "n_years": (end - start).days / 365.25 if pd.notna(start) and pd.notna(end) else np.nan,
                "median_velocity_myr": float(group["velocity_myr"].median()),
                "median_error_myr": float(group["velocity_error_myr"].median())
                if group["velocity_error_myr"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_records", "glacier_id", "point_id"], ascending=[False, True, True])
