"""Download CAMELS-GB v2 daily discharge files from the official CEH store.

The downloader is intentionally small and resumable. Raw CSV files stay under
``data/external`` and are excluded from public reproducibility packages.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

DATASET_ROOT = (
    "https://catalogue.ceh.ac.uk/datastore/eidchub/"
    "9a46d428-958f-4ac1-86eb-94eee70c0955/"
)
DAILY_DIR = DATASET_ROOT + "Catchment_Timeseries/hydro-meteorological/daily/"
ATTR_DIR = DATASET_ROOT + "Catchment_Attributes/"
USER_AGENT = "earth-deborah-benchmark/0.1 (+https://github.com/Johnsonlijian/)"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


@dataclass(frozen=True)
class DownloadItem:
    gauge_id: str
    url: str
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/external/camels_gb_v2")
    parser.add_argument("--limit", type=int, default=0, help="Download first N daily CSVs; 0 means all.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--download-attributes", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download complete files.")
    return parser.parse_args()


def session() -> requests.Session:
    out = requests.Session()
    out.headers.update({"User-Agent": USER_AGENT})
    return out


def discover_daily_files(http: requests.Session) -> list[DownloadItem]:
    response = http.get(DAILY_DIR, timeout=60)
    response.raise_for_status()
    parser = LinkParser()
    parser.feed(response.text)
    items: list[DownloadItem] = []
    pattern = re.compile(r"timeseries_(?P<gid>[0-9]+)_19701001-20220930\.csv$")
    for href in parser.links:
        if not href.endswith(".csv"):
            continue
        url = urljoin(DAILY_DIR, href)
        filename = Path(url).name
        match = pattern.search(filename)
        if not match:
            continue
        items.append(DownloadItem(gauge_id=match.group("gid"), url=url, filename=filename))
    return sorted(items, key=lambda item: int(item.gauge_id))


def remote_size(http: requests.Session, url: str, timeout: float) -> int | None:
    try:
        response = http.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException:
        return None
    size = response.headers.get("Content-Length")
    return int(size) if size and size.isdigit() else None


def is_complete(path: Path, expected_size: int | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if expected_size is None:
        return True
    return path.stat().st_size == expected_size


def download_one(item: DownloadItem, out_dir: Path, timeout: float, force: bool) -> dict[str, str | int]:
    http = session()
    target = out_dir / item.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(http, item.url, timeout=timeout)
    if not force and is_complete(target, expected):
        return {
            "gauge_id": item.gauge_id,
            "filename": item.filename,
            "url": item.url,
            "status": "already_complete",
            "bytes": target.stat().st_size,
            "expected_bytes": expected or "",
        }

    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    bytes_written = 0
    try:
        with http.get(item.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)
        if expected is not None and bytes_written != expected:
            raise RuntimeError(f"incomplete download: {bytes_written} != {expected}")
        tmp_path.replace(target)
        status = "downloaded"
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return {
        "gauge_id": item.gauge_id,
        "filename": item.filename,
        "url": item.url,
        "status": status,
        "bytes": target.stat().st_size,
        "expected_bytes": expected or "",
    }


def download_attributes(out_dir: Path, timeout: float, force: bool) -> list[dict[str, str | int]]:
    names = [
        "camels_gb_v2_hydrologic_attributes.csv",
        "camels_gb_v2_topographic_attributes.csv",
        "camels_gb_v2_climatic_attributes.csv",
    ]
    attr_dir = out_dir / "attributes"
    items = [DownloadItem(gauge_id="attributes", url=urljoin(ATTR_DIR, name), filename=name) for name in names]
    rows: list[dict[str, str | int]] = []
    for item in items:
        rows.append(download_one(item, attr_dir, timeout=timeout, force=force))
    return rows


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["gauge_id", "filename", "url", "status", "bytes", "expected_bytes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    daily_dir = out_dir / "hydromet_daily"
    http = session()
    daily_items = discover_daily_files(http)
    if args.limit > 0:
        daily_items = daily_items[: args.limit]
    print(f"Discovered {len(daily_items)} CAMELS-GB daily files selected for download.")

    rows: list[dict[str, str | int]] = []
    if args.download_attributes:
        rows.extend(download_attributes(out_dir, timeout=args.timeout, force=args.force))
        print("Attribute downloads complete.")

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(download_one, item, daily_dir, args.timeout, args.force): item
            for item in daily_items
        }
        for idx, future in enumerate(cf.as_completed(future_map), start=1):
            item = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "gauge_id": item.gauge_id,
                    "filename": item.filename,
                    "url": item.url,
                    "status": f"failed: {exc}",
                    "bytes": 0,
                    "expected_bytes": "",
                }
            rows.append(row)
            if idx % 25 == 0 or idx == len(daily_items):
                ok = sum(str(r["status"]) in {"downloaded", "already_complete"} for r in rows)
                print(f"  {idx}/{len(daily_items)} finished ({ok} usable)")

    manifest = out_dir / "download_manifest_camels_gb_v2.csv"
    write_manifest(manifest, rows)
    print(f"Manifest written: {manifest}")


if __name__ == "__main__":
    main()
