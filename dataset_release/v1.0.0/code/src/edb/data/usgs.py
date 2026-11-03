"""USGS ANSS ComCat loader and earthquake series builders."""

from __future__ import annotations

from collections.abc import Mapping
from time import sleep
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from numpy.typing import ArrayLike

COMCAT_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _drop_none(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def build_comcat_url(
    starttime: str,
    endtime: str,
    minmagnitude: float,
    maxmagnitude: float | None = None,
    minlatitude: float | None = None,
    maxlatitude: float | None = None,
    minlongitude: float | None = None,
    maxlongitude: float | None = None,
    format: str = "geojson",
    *,
    eventtype: str | None = "earthquake",
    orderby: str = "time-asc",
    limit: int | None = 20000,
    offset: int | None = None,
) -> str:
    """Build a USGS FDSN Event query URL."""

    params = _drop_none(
        {
            "format": format,
            "starttime": starttime,
            "endtime": endtime,
            "minmagnitude": minmagnitude,
            "maxmagnitude": maxmagnitude,
            "minlatitude": minlatitude,
            "maxlatitude": maxlatitude,
            "minlongitude": minlongitude,
            "maxlongitude": maxlongitude,
            "eventtype": eventtype,
            "orderby": orderby,
            "limit": limit,
            "offset": offset,
        }
    )
    return f"{COMCAT_QUERY_URL}?{urlencode(params)}"


def fetch_comcat_events(
    starttime: str,
    endtime: str,
    minmagnitude: float,
    maxmagnitude: float | None = None,
    minlatitude: float | None = None,
    maxlatitude: float | None = None,
    minlongitude: float | None = None,
    maxlongitude: float | None = None,
    format: str = "geojson",
    *,
    timeout: float = 30.0,
    retries: int = 3,
    session: requests.Session | None = None,
    **query_kwargs: Any,
) -> dict[str, Any]:
    """Fetch earthquake events from USGS ComCat as GeoJSON."""

    if format != "geojson":
        raise ValueError("fetch_comcat_events currently supports only format='geojson'")

    url = build_comcat_url(
        starttime=starttime,
        endtime=endtime,
        minmagnitude=minmagnitude,
        maxmagnitude=maxmagnitude,
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
        format=format,
        **query_kwargs,
    )
    client = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            try:
                response = client.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0 earth-deborah-benchmark/0.1"},
                )
            except TypeError as exc:
                if "headers" not in str(exc):
                    raise
                response = client.get(url, timeout=timeout)
            response.raise_for_status()
            json_obj = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError("ComCat request failed without an exception") from last_error
    if not isinstance(json_obj, dict):
        raise ValueError("ComCat response JSON must be an object")
    return json_obj


def merge_comcat_geojson_pages(pages: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge multiple ComCat GeoJSON FeatureCollection pages."""

    features: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for page in pages:
        metadata.update(page.get("metadata") or {})
        features.extend(page.get("features") or [])

    seen: set[str] = set()
    unique_features: list[dict[str, Any]] = []
    for feature in features:
        event_id = feature.get("id")
        if event_id is None or event_id not in seen:
            unique_features.append(feature)
        if event_id is not None:
            seen.add(event_id)

    metadata["count"] = len(unique_features)
    metadata["page_count"] = len(pages)
    return {"type": "FeatureCollection", "metadata": metadata, "features": unique_features}


def fetch_comcat_events_paginated(
    starttime: str,
    endtime: str,
    minmagnitude: float,
    maxmagnitude: float | None = None,
    minlatitude: float | None = None,
    maxlatitude: float | None = None,
    minlongitude: float | None = None,
    maxlongitude: float | None = None,
    *,
    page_limit: int = 20000,
    max_pages: int = 20,
    timeout: float = 30.0,
    retries: int = 3,
    session: requests.Session | None = None,
    **query_kwargs: Any,
) -> dict[str, Any]:
    """Fetch ComCat events with explicit 1-based offset pagination."""

    if page_limit <= 0:
        raise ValueError("page_limit must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    pages: list[Mapping[str, Any]] = []
    client = session or requests.Session()
    offset = 1
    for _ in range(max_pages):
        page = fetch_comcat_events(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minmagnitude,
            maxmagnitude=maxmagnitude,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            timeout=timeout,
            retries=retries,
            session=client,
            limit=page_limit,
            offset=offset,
            **query_kwargs,
        )
        features = page.get("features") or []
        pages.append(page)
        if len(features) < page_limit:
            break
        offset += page_limit
    else:
        raise RuntimeError(f"ComCat pagination hit max_pages={max_pages}; narrow query or increase max_pages")

    return merge_comcat_geojson_pages(pages)


def parse_comcat_geojson(json_obj: Mapping[str, Any]) -> pd.DataFrame:
    """Parse a USGS ComCat GeoJSON FeatureCollection into a tidy dataframe."""

    features = json_obj.get("features", [])
    rows: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [np.nan, np.nan, np.nan]
        longitude = coordinates[0] if len(coordinates) > 0 else np.nan
        latitude = coordinates[1] if len(coordinates) > 1 else np.nan
        depth_km = coordinates[2] if len(coordinates) > 2 else np.nan

        rows.append(
            {
                "event_id": feature.get("id"),
                "time": pd.to_datetime(properties.get("time"), unit="ms", utc=True, errors="coerce"),
                "latitude": latitude,
                "longitude": longitude,
                "depth_km": depth_km,
                "magnitude": properties.get("mag"),
                "mag_type": properties.get("magType"),
                "place": properties.get("place"),
                "url": properties.get("url"),
            }
        )

    columns = ["event_id", "time", "latitude", "longitude", "depth_km", "magnitude", "mag_type", "place", "url"]
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out

    numeric_cols = ["latitude", "longitude", "depth_km", "magnitude"]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("time").reset_index(drop=True)


def earthquake_energy_joule(magnitude: ArrayLike | float, formula: str = "usgs") -> np.ndarray | float:
    """Estimate radiated earthquake energy in joules from magnitude."""

    if formula != "usgs":
        raise ValueError("only formula='usgs' is implemented")

    values = np.asarray(magnitude, dtype=float)
    energy = 10.0 ** (5.24 + 1.44 * values)
    if np.isscalar(magnitude):
        return float(energy)
    return energy


def build_event_rate_series(
    df: pd.DataFrame,
    time_col: str = "time",
    rule: str = "D",
) -> pd.DataFrame:
    """Build a regular earthquake event-count activity series."""

    if time_col not in df.columns:
        raise ValueError(f"missing time column: {time_col}")
    work = df.loc[:, [time_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    if work.empty:
        return pd.DataFrame(columns=["time", "event_count", "value"])

    counts = work.set_index(time_col).assign(event_count=1)["event_count"].resample(rule).sum()
    out = counts.rename("event_count").reset_index().rename(columns={time_col: "time"})
    out["value"] = out["event_count"].astype(float)
    return out


def build_energy_rate_series(
    df: pd.DataFrame,
    time_col: str = "time",
    magnitude_col: str = "magnitude",
    rule: str = "D",
) -> pd.DataFrame:
    """Build a regular earthquake energy-release activity series."""

    missing = {time_col, magnitude_col}.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_cols}")

    work = df.loc[:, [time_col, magnitude_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work[magnitude_col] = pd.to_numeric(work[magnitude_col], errors="coerce")
    work = work.dropna(subset=[time_col, magnitude_col]).sort_values(time_col)
    if work.empty:
        return pd.DataFrame(columns=["time", "energy_joule", "value"])

    work["energy_joule"] = earthquake_energy_joule(work[magnitude_col])
    energy = work.set_index(time_col)["energy_joule"].resample(rule).sum()
    out = energy.rename("energy_joule").reset_index().rename(columns={time_col: "time"})
    out["value"] = out["energy_joule"].astype(float)
    return out
