"""Join GAGES-II basin attributes with NWIS memory-scaling results.

Downloads GAGES-II basin characteristics, extracts key attributes for stations
in the NWIS memory-scaling summary, and produces a merged table with tau_acf,
beta(De), drainage area, baseflow index, precipitation, temperature, snow fraction,
and regulation class.

Usage:
  python scripts/join_gagesii_attributes.py
  python scripts/join_gagesii_attributes.py --summary-csv reports/tables/nwis_expanded_*.csv
"""

from __future__ import annotations

import argparse
import io
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# GAGES-II download URL (USGS ScienceBase)
GAGESII_BASINCHAR_URL = (
    "https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a"
    "?name=basinchar_and_report_sept_2011.zip"
)

# Columns to extract from GAGES-II basin characteristics
GAGESII_COLUMNS: dict[str, str] = {
    "STAID": "station_id",
    "DRAIN_SQKM": "drainage_area_sqkm",
    "BFI_AVE": "baseflow_index",
    "PPTAVG_SITE": "precip_avg_mm",
    "T_AVG_SITE": "temp_avg_degC",
    "ELEV_MEAN_M_BASIN": "elev_mean_m",
    "SNOW_PCT_PRECIP": "snow_frac_pct",
    "PCT_IRRIG_AG": "pct_irrigated",
    "PCT_IMPERV": "pct_impervious",
    "PCT_FOREST": "pct_forest",
    "CLASS": "regulation_class",
    "AGECOOP_CODE": "ecoregion_code",
    "LAT_GAGE": "lat",
    "LNG_GAGE": "lng",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default="reports/tables/nwis_expanded_1975_20260528_summary.csv",
    )
    parser.add_argument(
        "--gagesii-cache",
        default="data/external/gagesii/basinchar_and_report_sept_2011.zip",
    )
    parser.add_argument("--output-csv", default="reports/tables/nwis_gagesii_merged.csv")
    return parser.parse_args()


def download_gagesii(cache_path: Path) -> Path:
    """Download GAGES-II basin characteristics zip if not cached."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 1_000_000:
        print(f"GAGES-II cached ({cache_path.stat().st_size:,} bytes)")
        return cache_path

    print(f"Downloading GAGES-II from {GAGESII_BASINCHAR_URL[:80]}...")
    import os
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(var, None)

    resp = requests.get(GAGESII_BASINCHAR_URL, timeout=300, stream=True,
                        proxies={}, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    total = 0
    with open(cache_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    print(f"Downloaded {total:,} bytes")
    return cache_path


def extract_gagesii(zip_path: Path) -> pd.DataFrame:
    """Extract basin characteristics from GAGES-II zip (handles nested csv zip or xlsx)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Check for nested CSV zip (GAGES-II v2 structure)
        csv_zip_name = None
        for n in names:
            if "csv" in n.lower() and n.lower().endswith(".zip"):
                csv_zip_name = n
                break

        if csv_zip_name:
            print(f"Found nested CSV zip: {csv_zip_name}")
            with zf.open(csv_zip_name) as inner_f:
                inner_bytes = BytesIO(inner_f.read())
            with zipfile.ZipFile(inner_bytes) as inner_zf:
                inner_names = inner_zf.namelist()

                # GAGES-II uses .txt files (tab-separated), not .csv
                txt_files = [n for n in inner_names if n.endswith(".txt")]
                if not txt_files:
                    csv_names = [n for n in inner_names if n.endswith(".csv")]
                    if not csv_names:
                        raise ValueError(f"No CSV/txt in nested zip. Files: {inner_names[:20]}")
                    target = csv_names[0]
                    print(f"Extracting {target} from nested CSV zip...")
                    with inner_zf.open(target) as f:
                        raw = io.TextIOWrapper(f, encoding="latin-1")
                        return pd.read_csv(raw, low_memory=False)

                # Build combined dataframe from multiple .txt files
                # Core files: basinid (with STAID), bas_classif, climate, hydro
                print(f"Found {len(txt_files)} txt files in nested zip. Building combined table...")
                combined = None
                for txt_name in sorted(txt_files):
                    # Skip AKHIPR (Alaska/Hawaii/PR) — our stations are all CONUS
                    if "conterm" not in txt_name.lower():
                        continue
                    with inner_zf.open(txt_name) as f:
                        raw = io.TextIOWrapper(f, encoding="latin-1")
                        content = raw.read()
                        sep = "," if content.count(",") > content.count("\t") else "\t"
                        df = pd.read_csv(io.StringIO(content), sep=sep, low_memory=False)
                    if "STAID" not in df.columns:
                        continue
                    if combined is None:
                        combined = df
                    else:
                        new_cols = [c for c in df.columns if c not in combined.columns]
                        if new_cols:
                            combined = combined.merge(df[["STAID"] + new_cols], on="STAID", how="left")
                    print(f"  Merged {txt_name} -> {len(combined)} stations, {len(combined.columns)} columns")
                if combined is not None:
                    return combined

                raise ValueError(f"No valid basin data found")

        # Check for xlsx (GAGES-II v1 structure)
        xlsx_names = [n for n in names if n.lower().endswith(".xlsx") and "conterm" in n.lower()]
        if not xlsx_names:
            xlsx_names = [n for n in names if n.lower().endswith(".xlsx") and "basin" not in n.lower()]
        if xlsx_names:
            target = xlsx_names[0]
            print(f"Extracting {target} from GAGES-II zip (xlsx)...")
            with zf.open(target) as f:
                return pd.read_excel(io.BytesIO(f.read()))

        # Fallback: look for any CSV
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        if csv_names:
            target = csv_names[0]
            print(f"Extracting {target} from GAGES-II zip...")
            with zf.open(target) as f:
                raw = io.TextIOWrapper(f, encoding="latin-1")
                return pd.read_csv(raw)

        raise ValueError(f"No extractable basin data found. Files: {names[:20]}")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    nwis = pd.read_csv(summary_path, dtype={"station_id": str})
    print(f"Loaded {len(nwis)} NWIS stations from {summary_path.name}")

    cache_path = Path(args.gagesii_cache)
    if not cache_path.exists():
        download_gagesii(cache_path)

    gagesii = extract_gagesii(cache_path)
    gagesii["STAID"] = gagesii["STAID"].astype(str).str.strip().str.zfill(8)
    nwis["station_id_8digit"] = nwis["station_id"].astype(str).str.strip().str.zfill(8)

    # Merge
    available_cols = [c for c in GAGESII_COLUMNS if c in gagesii.columns]
    gagesii_subset = gagesii[available_cols].copy()
    gagesii_subset.columns = [GAGESII_COLUMNS[c] for c in available_cols]

    merged = nwis.merge(
        gagesii_subset,
        left_on="station_id_8digit",
        right_on="station_id",
        how="left",
        suffixes=("_nwis", "_gagesii"),
    )

    # Clean up duplicate station_id column
    if "station_id_nwis" in merged.columns:
        merged = merged.drop(columns=["station_id_gagesii"])
    merged = merged.drop(columns=["station_id_8digit"], errors="ignore")

    n_matched = merged["drainage_area_sqkm"].notna().sum()
    print(f"Merged {n_matched}/{len(merged)} stations with GAGES-II attributes")

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

    # Print key correlations (if enough data)
    if n_matched >= 5:
        numeric_cols = ["tau_acf_days", "mean_beta_de_window", "drainage_area_sqkm",
                        "baseflow_index", "precip_avg_mm", "temp_avg_degC",
                        "elev_mean_m", "snow_frac_pct"]
        available_numeric = [c for c in numeric_cols if c in merged.columns and merged[c].notna().sum() >= 5]
        if len(available_numeric) >= 2:
            corr = merged[available_numeric].corr(method="spearman")
            print("\n--- Key Spearman correlations ---")
            print(corr[["tau_acf_days", "mean_beta_de_window"]].dropna(how="all").to_string())

        # Regulation class comparison
        if "regulation_class" in merged.columns:
            ref = merged[merged["regulation_class"].astype(str).str.upper().str.contains("REF", na=False)]
            non_ref = merged[~merged.index.isin(ref.index)]
            print(f"\n--- Regulation comparison ---")
            for label, subset in [("Reference", ref), ("Non-reference", non_ref)]:
                if len(subset) >= 2:
                    print(f"  {label} (n={len(subset)}): tau_acf = {subset['tau_acf_days'].median():.1f} d median, "
                          f"beta = {subset['mean_beta_de_window'].dropna().median():.2f} median")


if __name__ == "__main__":
    main()
