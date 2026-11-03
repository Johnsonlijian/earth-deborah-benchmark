"""USGS NWIS daily-value river discharge loader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode
from time import sleep

import pandas as pd
import requests


NWIS_DAILY_VALUES_URL = "https://waterservices.usgs.gov/nwis/dv/"
CFS_TO_M3S = 0.028316846592


def build_nwis_daily_values_url(
    sites: str | Sequence[str],
    start_date: str,
    end_date: str,
    parameter_cd: str = "00060",
    stat_cd: str = "00003",
    format: str = "json",
    site_status: str = "all",
) -> str:
    """Build a USGS NWIS daily-values service URL."""

    site_arg = ",".join(sites) if not isinstance(sites, str) else sites
    params = {
        "format": format,
        "sites": site_arg,
        "parameterCd": parameter_cd,
        "statCd": stat_cd,
        "startDT": start_date,
        "endDT": end_date,
        "siteStatus": site_status,
    }
    return f"{NWIS_DAILY_VALUES_URL}?{urlencode(params)}"


def fetch_nwis_daily_values(
    sites: str | Sequence[str],
    start_date: str,
    end_date: str,
    parameter_cd: str = "00060",
    stat_cd: str = "00003",
    *,
    timeout: float = 120.0,
    retries: int = 3,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch USGS NWIS daily values as WaterML JSON."""

    url = build_nwis_daily_values_url(
        sites=sites,
        start_date=start_date,
        end_date=end_date,
        parameter_cd=parameter_cd,
        stat_cd=stat_cd,
    )
    client = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 earth-deborah-benchmark/0.1"},
            )
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError("NWIS request failed without an exception") from last_error
    if not isinstance(payload, dict):
        raise ValueError("NWIS response JSON must be an object")
    return payload


def _first_value(items: object, default: object = None) -> object:
    """Return the first list item value from a WaterML-ish field."""

    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, Mapping) and "value" in first:
            return first.get("value")
        return first
    return default


def parse_nwis_daily_values_json(json_obj: Mapping[str, Any]) -> pd.DataFrame:
    """Parse NWIS daily discharge JSON into the project river schema."""

    value = json_obj.get("value") if isinstance(json_obj, Mapping) else {}
    time_series = (value or {}).get("timeSeries", []) if isinstance(value, Mapping) else []
    rows: list[dict[str, object]] = []
    for series in time_series:
        source_info = series.get("sourceInfo") or {}
        variable = series.get("variable") or {}
        site_id = _first_value(source_info.get("siteCode"), default="")
        site_name = source_info.get("siteName")
        variable_name = variable.get("variableName") or "Discharge"
        unit_code = (variable.get("unit") or {}).get("unitCode")
        quality_values = series.get("values") or []
        for value_block in quality_values:
            for point in value_block.get("value") or []:
                raw_value = pd.to_numeric(point.get("value"), errors="coerce")
                if pd.isna(raw_value):
                    continue
                discharge = float(raw_value)
                if unit_code in {"ft3/s", "cfs"}:
                    discharge *= CFS_TO_M3S
                rows.append(
                    {
                        "station_id": str(site_id).strip(),
                        "station_name": site_name,
                        "river_name": variable_name,
                        "country_code": "US",
                        "time": pd.to_datetime(point.get("dateTime"), utc=True, errors="coerce"),
                        "discharge_m3s": discharge,
                        "quality_flag": ",".join(point.get("qualifiers") or []),
                        "source_unit": unit_code,
                    }
                )

    columns = [
        "station_id",
        "station_name",
        "river_name",
        "country_code",
        "time",
        "discharge_m3s",
        "quality_flag",
        "source_unit",
    ]
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out = out.dropna(subset=["station_id", "time", "discharge_m3s"])
    out = out[out["station_id"].astype(str).str.len() > 0]
    out = out[out["discharge_m3s"] >= 0]
    return out.sort_values(["station_id", "time"]).reset_index(drop=True)
