"""GRDC-compatible river discharge loaders and anomaly builders."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "station_id",
    "station_name",
    "river_name",
    "country_code",
    "time",
    "discharge_m3s",
    "quality_flag",
]


def read_river_csv(source: str | Path | IO[str]) -> pd.DataFrame:
    """Read a river discharge CSV file."""

    return pd.read_csv(source)


def parse_river_dataframe(
    df: pd.DataFrame,
    station_id_col: str = "station_id",
    time_col: str = "time",
    discharge_col: str = "discharge_m3s",
    station_name_col: str | None = "station_name",
    river_name_col: str | None = "river_name",
    country_code_col: str | None = "country_code",
    quality_flag_col: str | None = "quality_flag",
) -> pd.DataFrame:
    """Parse a generic daily discharge dataframe into the project schema."""

    required = {station_id_col, time_col, discharge_col}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"river dataframe is missing required columns: {missing_cols}")

    out = pd.DataFrame(index=df.index)
    out["station_id"] = df[station_id_col].astype("string").str.strip()
    out["time"] = pd.to_datetime(df[time_col], errors="coerce")
    out["discharge_m3s"] = pd.to_numeric(df[discharge_col], errors="coerce")

    optional_map = {
        "station_name": station_name_col,
        "river_name": river_name_col,
        "country_code": country_code_col,
        "quality_flag": quality_flag_col,
    }
    for output_col, input_col in optional_map.items():
        if input_col is not None and input_col in df.columns:
            out[output_col] = df[input_col].astype("string").str.strip()
            out[output_col] = out[output_col].mask(out[output_col].eq(""), pd.NA)
        else:
            out[output_col] = pd.NA

    out = out.dropna(subset=["station_id", "time", "discharge_m3s"]).copy()
    out = out[out["discharge_m3s"] >= 0]
    return out.loc[:, STANDARD_COLUMNS].sort_values(["station_id", "time"]).reset_index(drop=True)


def filter_river_records(
    df: pd.DataFrame,
    station_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Filter parsed river records by station and date."""

    out = df.copy()
    if station_id is not None:
        out = out[out["station_id"].astype("string") == str(station_id)]
    if start_date is not None:
        out = out[out["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["time"] <= pd.Timestamp(end_date)]
    return out.reset_index(drop=True)


def build_discharge_series(
    df: pd.DataFrame,
    station_id: str | None = None,
    rule: str = "D",
) -> pd.DataFrame:
    """Build a regular discharge series for one station."""

    work = filter_river_records(df, station_id=station_id) if station_id is not None else df.copy()
    if work.empty:
        return pd.DataFrame(columns=["time", "discharge_m3s", "value"])
    if work["station_id"].nunique() > 1:
        raise ValueError("build_discharge_series expects one station; pass station_id or pre-filter the dataframe")

    series = (
        work.sort_values("time")
        .set_index("time")["discharge_m3s"]
        .resample(rule)
        .mean()
        .rename("discharge_m3s")
        .reset_index()
    )
    series["value"] = series["discharge_m3s"].astype(float)
    return series


def remove_daily_climatology(series: pd.DataFrame, time_col: str = "time", value_col: str = "value") -> pd.DataFrame:
    """Subtract a day-of-year median climatology from a daily discharge series."""

    out = series.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    day = out[time_col].dt.dayofyear
    climatology = out.groupby(day)[value_col].transform("median")
    out["seasonal_climatology"] = climatology
    out["value"] = out[value_col] - climatology
    return out


def robust_mad_scale(series: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Scale a dataframe value column by median absolute deviation."""

    out = series.copy()
    values = pd.to_numeric(out[value_col], errors="coerce").to_numpy(float)
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale == 0:
        out[value_col] = 0.0
    else:
        out[value_col] = (values - center) / scale
    return out


def summarize_station_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize station coverage, missingness, and record span."""

    if df.empty:
        return pd.DataFrame(columns=["station_id", "n_records", "start_time", "end_time", "n_years", "missing_daily_fraction"])

    rows: list[dict[str, object]] = []
    for station_id, group in df.groupby("station_id"):
        start = group["time"].min()
        end = group["time"].max()
        expected = pd.date_range(start, end, freq="D")
        n_records = int(group["time"].nunique())
        missing_fraction = 1.0 - n_records / len(expected) if len(expected) else np.nan
        rows.append(
            {
                "station_id": station_id,
                "station_name": group["station_name"].dropna().iloc[0] if group["station_name"].notna().any() else pd.NA,
                "river_name": group["river_name"].dropna().iloc[0] if group["river_name"].notna().any() else pd.NA,
                "country_code": group["country_code"].dropna().iloc[0] if group["country_code"].notna().any() else pd.NA,
                "n_records": n_records,
                "start_time": start,
                "end_time": end,
                "n_years": (end - start).days / 365.25 if pd.notna(start) and pd.notna(end) else np.nan,
                "missing_daily_fraction": missing_fraction,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_records", "station_id"], ascending=[False, True]).reset_index(drop=True)
