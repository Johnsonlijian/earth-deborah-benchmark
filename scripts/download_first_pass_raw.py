"""Download/cache first-pass raw inputs and metadata into ignored data/raw."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from edb.data.itslive import fetch_itslive_collections, search_itslive_items, summarize_stac_items
from edb.data.landslide_glc import GLC_LEGACY_CSV_URL
from edb.data.usgs import fetch_comcat_events_paginated


def file_sha256(path: Path) -> str:
    """Return SHA256 for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def download_url(path: Path, url: str, timeout: float = 120.0) -> None:
    """Download a URL to a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 earth-deborah-benchmark/0.1"})
    response.raise_for_status()
    path.write_bytes(response.content)


def manifest_row(dataset: str, path: Path, source_url: str, status: str, notes: str) -> dict[str, object]:
    """Build one manifest row."""

    return {
        "dataset": dataset,
        "path": str(path),
        "source_url": source_url,
        "status": status,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--reports-dir", default="reports/tables")
    parser.add_argument("--starttime", default="2000-01-01")
    parser.add_argument("--endtime", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--usgs-minmagnitude", type=float, default=5.0)
    parser.add_argument("--skip-usgs", action="store_true")
    parser.add_argument("--skip-glc", action="store_true")
    parser.add_argument("--skip-itslive", action="store_true")
    parser.add_argument("--itslive-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    if not args.skip_usgs:
        usgs_path = raw_dir / "usgs" / f"comcat_global_m{args.usgs_minmagnitude:g}_{args.starttime}_{args.endtime}.geojson"
        payload = fetch_comcat_events_paginated(
            starttime=args.starttime,
            endtime=args.endtime,
            minmagnitude=args.usgs_minmagnitude,
            page_limit=20000,
            max_pages=20,
            timeout=180.0,
            retries=4,
        )
        write_json(usgs_path, payload)
        rows.append(
            manifest_row(
                "usgs_comcat",
                usgs_path,
                "https://earthquake.usgs.gov/fdsnws/event/1/",
                "downloaded_raw_cache",
                f"GeoJSON FeatureCollection; features={len(payload.get('features', []))}",
            )
        )

    if not args.skip_glc:
        glc_path = raw_dir / "nasa_glc" / "Global_Landslide_Catalog_Export_rows.csv"
        download_url(glc_path, GLC_LEGACY_CSV_URL)
        rows.append(
            manifest_row(
                "nasa_glc",
                glc_path,
                GLC_LEGACY_CSV_URL,
                "downloaded_raw_cache",
                "NASA legacy CSV export; reporting-biased catalog",
            )
        )

    if not args.skip_itslive:
        its_dir = raw_dir / "its_live"
        collections = fetch_itslive_collections()
        collections_path = its_dir / "collections.json"
        write_json(collections_path, collections)
        rows.append(
            manifest_row(
                "its_live_collections",
                collections_path,
                "https://stac.itslive.cloud/collections",
                "downloaded_metadata_cache",
                "STAC collections metadata only",
            )
        )
        for collection in ["itslive-cubes", "itslive-granules"]:
            search_payload = search_itslive_items(collection=collection, limit=args.itslive_limit)
            search_path = its_dir / f"{collection}_search_limit{args.itslive_limit}.geojson"
            write_json(search_path, search_payload)
            rows.append(
                manifest_row(
                    f"its_live_{collection}",
                    search_path,
                    "https://stac.itslive.cloud/search",
                    "downloaded_metadata_cache",
                    f"STAC item search metadata only; returned={len(search_payload.get('features', []))}",
                )
            )
            summary = summarize_stac_items(search_payload)
            summary.to_csv(reports_dir / f"{collection}_metadata_summary.csv", index=False)

    manifest = pd.DataFrame(rows)
    manifest_path = reports_dir / "raw_download_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved {manifest_path}")
    if rows:
        print(manifest.loc[:, ["dataset", "status", "size_bytes", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
