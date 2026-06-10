"""NASA Global Landslide Catalog legacy CSV loader."""

from __future__ import annotations

from pathlib import Path
from io import StringIO
from typing import IO
import time

import pandas as pd
import requests

GLC_LEGACY_CSV_URL = (
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.csv"
)


STANDARD_COLUMNS = [
    "event_id",
    "event_date",
    "event_year",
    "country_name",
    "country_code",
    "admin_division_name",
    "landslide_category",
    "landslide_trigger",
    "landslide_size",
    "landslide_setting",
    "location_accuracy",
    "fatality_count",
    "injury_count",
    "longitude",
    "latitude",
    "source_name",
    "source_link",
]


def _parse_event_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed


def read_glc_csv(
    source: str | Path | IO[str] = GLC_LEGACY_CSV_URL,
    timeout: float = 60.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Read the NASA GLC legacy CSV export from a URL, path, or file-like object."""

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        headers = {"User-Agent": "Mozilla/5.0 earth-deborah-benchmark/0.1"}
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.get(source, timeout=timeout, headers=headers)
                response.raise_for_status()
                return pd.read_csv(StringIO(response.text))
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
    return pd.read_csv(source)


def parse_glc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw GLC dataframe into stable columns and dtypes."""

    missing = {"event_id", "event_date", "longitude", "latitude"}.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"GLC dataframe is missing required columns: {missing_cols}")

    out = pd.DataFrame(index=df.index)
    for column in STANDARD_COLUMNS:
        if column in df.columns:
            out[column] = df[column]
        else:
            out[column] = pd.NA

    out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").astype("Int64")
    out["event_date"] = _parse_event_dates(out["event_date"])
    out["event_year"] = out["event_date"].dt.year.astype("Int64")
    for column in ["fatality_count", "injury_count", "longitude", "latitude"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    text_columns = [
        "country_name",
        "country_code",
        "admin_division_name",
        "landslide_category",
        "landslide_trigger",
        "landslide_size",
        "landslide_setting",
        "location_accuracy",
        "source_name",
        "source_link",
    ]
    for column in text_columns:
        out[column] = out[column].astype("string").str.strip()
        out[column] = out[column].mask(out[column].eq(""), pd.NA)

    return out.sort_values(["event_date", "event_id"]).reset_index(drop=True)


def filter_glc_records(
    df: pd.DataFrame,
    start_date: str | None = "2007-01-01",
    end_date: str | None = None,
    country_code: str | None = None,
    landslide_trigger: str | None = None,
    require_coordinates: bool = True,
) -> pd.DataFrame:
    """Filter parsed GLC records for first-pass benchmark sensitivity runs."""

    out = df.copy()
    if start_date is not None:
        out = out[out["event_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["event_date"] <= pd.Timestamp(end_date)]
    if country_code is not None:
        out = out[out["country_code"].str.upper() == country_code.upper()]
    if landslide_trigger is not None:
        out = out[out["landslide_trigger"].str.lower() == landslide_trigger.lower()]
    if require_coordinates:
        out = out[out["longitude"].notna() & out["latitude"].notna()]
    return out.reset_index(drop=True)


def build_landslide_count_series(
    df: pd.DataFrame,
    time_col: str = "event_date",
    rule: str = "MS",
) -> pd.DataFrame:
    """Build a regular landslide event-count activity series."""

    if time_col not in df.columns:
        raise ValueError(f"missing time column: {time_col}")
    work = df.loc[:, [time_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    if work.empty:
        return pd.DataFrame(columns=["time", "landslide_count", "value"])

    counts = work.set_index(time_col).assign(landslide_count=1)["landslide_count"].resample(rule).sum()
    out = counts.rename("landslide_count").reset_index().rename(columns={time_col: "time"})
    out["value"] = out["landslide_count"].astype(float)
    return out


def summarize_glc_reporting(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize annual record counts for reporting-bias diagnostics."""

    required = {"event_year", "country_code", "landslide_trigger"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"GLC dataframe is missing required columns: {missing_cols}")

    return (
        df.groupby("event_year", dropna=False)
        .agg(
            n_events=("event_id", "count"),
            n_countries=("country_code", "nunique"),
            n_triggers=("landslide_trigger", "nunique"),
        )
        .reset_index()
        .sort_values("event_year")
    )
