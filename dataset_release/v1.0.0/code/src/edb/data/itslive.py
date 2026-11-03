"""ITS_LIVE STAC metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import time

import pandas as pd
import requests

ITS_LIVE_STAC_URL = "https://stac.itslive.cloud"


HEADERS = {"User-Agent": "Mozilla/5.0 earth-deborah-benchmark/0.1"}


def fetch_stac_json(url: str, timeout: float = 30.0, retries: int = 3) -> dict[str, Any]:
    """Fetch a JSON document from the ITS_LIVE STAC API."""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"failed to fetch {url}")
    json_obj = response.json()
    if not isinstance(json_obj, dict):
        raise ValueError("STAC response must be a JSON object")
    return json_obj


def fetch_itslive_collections(stac_url: str = ITS_LIVE_STAC_URL, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch the ITS_LIVE STAC collections document."""

    return fetch_stac_json(f"{stac_url.rstrip('/')}/collections", timeout=timeout)


def search_itslive_items(
    collection: str,
    bbox: list[float] | None = None,
    datetime_range: str | None = None,
    limit: int = 100,
    stac_url: str = ITS_LIVE_STAC_URL,
    timeout: float = 60.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Search ITS_LIVE STAC items with a bounded POST query."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    payload: dict[str, Any] = {"collections": [collection], "limit": limit}
    if bbox is not None:
        payload["bbox"] = bbox
    if datetime_range is not None:
        payload["datetime"] = datetime_range

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{stac_url.rstrip('/')}/search",
                json=payload,
                timeout=timeout,
                headers=HEADERS,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("failed to search ITS_LIVE items")
    json_obj = response.json()
    if not isinstance(json_obj, dict):
        raise ValueError("STAC search response must be a JSON object")
    return json_obj


def summarize_stac_items(items_geojson: Mapping[str, Any]) -> pd.DataFrame:
    """Summarize STAC features into a compact dataframe."""

    rows: list[dict[str, Any]] = []
    for feature in items_geojson.get("features", []) or []:
        properties = feature.get("properties") or {}
        assets = feature.get("assets") or {}
        links = feature.get("links") or []
        rows.append(
            {
                "item_id": feature.get("id"),
                "collection": feature.get("collection"),
                "datetime": properties.get("datetime"),
                "start_datetime": properties.get("start_datetime"),
                "end_datetime": properties.get("end_datetime"),
                "bbox": feature.get("bbox"),
                "asset_keys": ",".join(sorted(assets.keys())),
                "self_href": next((link.get("href") for link in links if link.get("rel") == "self"), None),
            }
        )
    return pd.DataFrame(rows)
