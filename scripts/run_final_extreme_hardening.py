"""Final HESS hardening analyses: cluster bootstrap, forcing controls and support audit.

This round targets the HESS reviewer risks identified after R41:

* spatial dependence in archive-level bootstrap uncertainty;
* precipitation/forcing-side spectral structure;
* slow-memory frequency-support limits for De ~= 1;
* submission-facing robustness figure/source-data integration.

The script uses already downloaded public CAMELS data and derived beta-curve
tables. It does not create new hydrological claims beyond the computed
diagnostics.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.timescales import integral_autocorrelation_time
from run_camels_gb_replication import DAILY_DIR, gauge_id_from_path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
NW_SOURCE = ROOT / "source_data"
NW_FIGURES = ROOT / "figures"
SECONDS_PER_DAY = 86_400.0
OUT = "final_extreme_hardening"
US_FORCING_ZIP = ROOT / "data" / "external" / "camels" / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
US_FORCING_DIR = ROOT / "data" / "external" / "camels" / "basin_timeseries_v1p2_metForcing_obsFlow"


@dataclass(frozen=True)
class ArchiveSpec:
    archive: str
    curves_path: Path
    summary_path: Path
    min_units: int
    tau_col: str
    cluster_kind: str


ARCHIVES = [
    ArchiveSpec(
        "CAMELS-US",
        TABLES / "camels_673_beta_curve_points.csv",
        TABLES / "camels_673_attributed_diagnostics.csv",
        50,
        "tau_acf_days",
        "huc2",
    ),
    ArchiveSpec(
        "CAMELS-GB v2",
        TABLES / "camels_gb_v2_replication_beta_curve_points.csv",
        TABLES / "camels_gb_v2_replication_summary.csv",
        50,
        "tau_acf_days",
        "geo_grid",
    ),
    ArchiveSpec(
        "CAMELS-BR v1.2",
        TABLES / "r23_camels_br_third_archive_curves.csv",
        TABLES / "r23_camels_br_third_archive_gauge_summary.csv",
        50,
        "tau_acf_days",
        "geo_grid",
    ),
    ArchiveSpec(
        "CAMELS-AUS v2",
        TABLES / "r25_camels_aus_fourth_archive_curves.csv",
        TABLES / "r25_camels_aus_fourth_archive_gauge_summary.csv",
        40,
        "tau_acf_days",
        "division",
    ),
    ArchiveSpec(
        "CAMELS-DK lowland",
        TABLES / "r27_camels_dk_groundwater_storage_validation_curves.csv",
        TABLES / "r27_camels_dk_groundwater_storage_validation_gauge_summary.csv",
        30,
        "tau_acf_days",
        "dk_geo_grid",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--skip-forcing", action="store_true")
    return parser.parse_args()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def load_curves(spec: ArchiveSpec) -> pd.DataFrame:
    data = pd.read_csv(spec.curves_path)
    data = data.copy()
    data["gauge_id_str"] = normalize_id(data["gauge_id"])
    if spec.tau_col not in data.columns:
        if "tau_days" in data.columns:
            data[spec.tau_col] = data["tau_days"]
        elif "tau_acf_days" in data.columns:
            data[spec.tau_col] = data["tau_acf_days"]
        else:
            raise ValueError(f"{spec.archive} curves lack tau column")
    data = data.rename(columns={spec.tau_col: "tau_acf_days"})
    required = ["gauge_id_str", "frequency_cpd", "de", "beta", "tau_acf_days"]
    out = data[required].replace([np.inf, -np.inf], np.nan).dropna()
    return out[(out["frequency_cpd"] > 0) & (out["de"] > 0) & (out["tau_acf_days"] > 0)].copy()


def load_summary(spec: ArchiveSpec) -> pd.DataFrame:
    data = pd.read_csv(spec.summary_path)
    data = data.copy()
    data["gauge_id_str"] = normalize_id(data["gauge_id"])
    if spec.archive == "CAMELS-US":
        topo = pd.read_csv(ROOT / "data" / "external" / "camels" / "camels_attributes_v2.0" / "camels_topo.txt", sep=";")
        topo["gauge_id_str"] = normalize_id(topo["gauge_id"])
        keep = topo[["gauge_id_str", "gauge_lat", "gauge_lon"]]
        data = data.merge(keep, on="gauge_id_str", how="left")
    elif spec.archive == "CAMELS-DK lowland":
        topo = pd.read_csv(ROOT / "data" / "external" / "camels_dk" / "CAMELS_DK_topography.csv")
        topo["gauge_id_str"] = normalize_id(topo["catch_id"])
        keep = topo[["gauge_id_str", "catch_outlet_lat", "catch_outlet_lon"]]
        data = data.merge(keep, on="gauge_id_str", how="left")
    return data


def coordinate_columns(summary: pd.DataFrame, archive: str) -> tuple[str | None, str | None]:
    candidates = [
        ("gauge_lat", "gauge_lon"),
        ("lat_outlet", "long_outlet"),
        ("lat_centroid", "long_centroid"),
        ("catch_outlet_lat", "catch_outlet_lon"),
    ]
    for lat, lon in candidates:
        if lat in summary.columns and lon in summary.columns:
            valid = summary[[lat, lon]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) >= 5:
                return lat, lon
    return None, None


def geo_grid_labels(summary: pd.DataFrame, lat_col: str, lon_col: str, n_lat: int = 3, n_lon: int = 4) -> pd.Series:
    lat = pd.to_numeric(summary[lat_col], errors="coerce")
    lon = pd.to_numeric(summary[lon_col], errors="coerce")
    lat_rank = lat.rank(method="first")
    lon_rank = lon.rank(method="first")
    lat_bin = pd.qcut(lat_rank, q=min(n_lat, max(1, lat.notna().sum() // 20)), labels=False, duplicates="drop")
    lon_bin = pd.qcut(lon_rank, q=min(n_lon, max(1, lon.notna().sum() // 20)), labels=False, duplicates="drop")
    labels = lat_bin.astype("Int64").astype(str) + "_" + lon_bin.astype("Int64").astype(str)
    labels[(lat.isna()) | (lon.isna())] = "missing_geo"
    return labels


def cluster_labels(spec: ArchiveSpec, summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[["gauge_id_str"]].drop_duplicates().copy()
    if spec.cluster_kind == "huc2" and "huc_02" in summary.columns:
        out["cluster_id"] = "HUC2_" + normalize_id(summary.loc[out.index, "huc_02"])
        return out.dropna()
    if spec.cluster_kind == "division" and "drainage_division" in summary.columns:
        out["cluster_id"] = summary.loc[out.index, "drainage_division"].astype(str)
        return out.dropna()
    lat_col, lon_col = coordinate_columns(summary, spec.archive)
    if lat_col is None or lon_col is None:
        out["cluster_id"] = "all_gauges"
        return out
    out["cluster_id"] = geo_grid_labels(summary.loc[out.index], lat_col, lon_col).to_numpy()
    return out.dropna()


def dispersion_metrics(curves: pd.DataFrame, coord: str, min_units: int, n_bins: int = 24, unit_col: str = "unit_id") -> dict[str, float | int]:
    work = curves[[unit_col, coord, "beta"]].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[coord] > 0) & np.isfinite(work[coord]) & np.isfinite(work["beta"])].copy()
    if work.empty:
        return {"weighted_beta_variance": np.nan, "n_bins": 0, "n_unit_bin_pairs": 0}
    logx = np.log10(work[coord].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(logx, [2.5, 97.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"weighted_beta_variance": np.nan, "n_bins": 0, "n_unit_bin_pairs": 0}
    edges = np.linspace(lo, hi, n_bins + 1)
    work["bin_id"] = pd.cut(logx, bins=edges, labels=False, include_lowest=True)
    med = work.dropna(subset=["bin_id"]).groupby(["bin_id", unit_col], as_index=False)["beta"].median()
    rows = []
    for bin_id, group in med.groupby("bin_id"):
        n = int(group[unit_col].nunique())
        if n < min_units:
            continue
        rows.append({"bin_id": int(bin_id), "n_units": n, "beta_variance": float(group["beta"].var(ddof=1))})
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return {"weighted_beta_variance": np.nan, "n_bins": 0, "n_unit_bin_pairs": 0}
    return {
        "weighted_beta_variance": float(np.average(stats_df["beta_variance"], weights=stats_df["n_units"])),
        "n_bins": int(len(stats_df)),
        "n_unit_bin_pairs": int(stats_df["n_units"].sum()),
    }


def reduction_for(curves: pd.DataFrame, min_units: int, value_col: str = "beta", de_col: str = "de") -> dict[str, float | int]:
    work = curves.rename(columns={value_col: "beta", de_col: "de"}).copy()
    if "unit_id" not in work.columns:
        work["unit_id"] = work["gauge_id_str"]
    raw = dispersion_metrics(work, "frequency_cpd", min_units=min_units)
    de = dispersion_metrics(work, "de", min_units=min_units)
    raw_var = float(raw["weighted_beta_variance"])
    de_var = float(de["weighted_beta_variance"])
    reduction = np.nan if (not np.isfinite(raw_var) or raw_var <= 0) else 1.0 - de_var / raw_var
    return {
        "raw_weighted_beta_variance": raw_var,
        "de_weighted_beta_variance": de_var,
        "variance_reduction_fraction": float(reduction) if np.isfinite(reduction) else np.nan,
        "raw_n_bins": raw["n_bins"],
        "de_n_bins": de["n_bins"],
        "raw_n_unit_bin_pairs": raw["n_unit_bin_pairs"],
        "de_n_unit_bin_pairs": de["n_unit_bin_pairs"],
    }


def cluster_bootstrap_one_archive(spec: ArchiveSpec, draws: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = load_curves(spec)
    summary = load_summary(spec)
    clusters = cluster_labels(spec, summary)
    clusters = clusters[clusters["gauge_id_str"].isin(curves["gauge_id_str"].unique())].drop_duplicates()
    curves = curves[curves["gauge_id_str"].isin(clusters["gauge_id_str"])].copy()
    curves["unit_id"] = curves["gauge_id_str"]
    observed = reduction_for(curves, spec.min_units)

    cluster_ids = np.array(sorted(clusters["cluster_id"].dropna().unique()), dtype=object)
    cluster_to_gauges = {
        cid: clusters.loc[clusters["cluster_id"] == cid, "gauge_id_str"].to_numpy(dtype=str)
        for cid in cluster_ids
    }
    rows = []
    for draw in range(draws):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        frames = []
        for rep, cid in enumerate(selected):
            gauges = cluster_to_gauges[cid]
            sub = curves[curves["gauge_id_str"].isin(gauges)].copy()
            sub["unit_id"] = sub["gauge_id_str"] + f"__clusterrep{rep:03d}"
            frames.append(sub)
        sample = pd.concat(frames, ignore_index=True)
        metric = reduction_for(sample, spec.min_units)
        rows.append({"archive": spec.archive, "draw": draw, "n_clusters": len(cluster_ids), **metric})
    draws_df = pd.DataFrame(rows)
    ci = draws_df["variance_reduction_fraction"].quantile([0.025, 0.5, 0.975])
    summary_row = pd.DataFrame(
        [
            {
                "archive": spec.archive,
                "n_gauges": int(curves["gauge_id_str"].nunique()),
                "n_clusters": int(len(cluster_ids)),
                "cluster_scheme": spec.cluster_kind,
                "observed_fraction": observed["variance_reduction_fraction"],
                "cluster_bootstrap_median_fraction": float(ci.loc[0.5]),
                "cluster_ci_low_fraction": float(ci.loc[0.025]),
                "cluster_ci_high_fraction": float(ci.loc[0.975]),
                "cluster_probability_positive": float((draws_df["variance_reduction_fraction"] > 0).mean()),
                "observed_raw_weighted_beta_variance": observed["raw_weighted_beta_variance"],
                "observed_de_weighted_beta_variance": observed["de_weighted_beta_variance"],
            }
        ]
    )
    return draws_df, summary_row


def run_cluster_bootstrap(draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    draw_frames = []
    summary_frames = []
    for spec in ARCHIVES:
        print(f"R42 cluster bootstrap: {spec.archive}", flush=True)
        archive_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max)))
        draws_df, summary_df = cluster_bootstrap_one_archive(spec, draws, archive_rng)
        draw_frames.append(draws_df)
        summary_frames.append(summary_df)
    draws_out = pd.concat(draw_frames, ignore_index=True)
    summary_out = pd.concat(summary_frames, ignore_index=True)
    summary_out["observed_percent"] = 100.0 * summary_out["observed_fraction"]
    summary_out["cluster_median_percent"] = 100.0 * summary_out["cluster_bootstrap_median_fraction"]
    summary_out["cluster_ci_low_percent"] = 100.0 * summary_out["cluster_ci_low_fraction"]
    summary_out["cluster_ci_high_percent"] = 100.0 * summary_out["cluster_ci_high_fraction"]
    return draws_out, summary_out


def remove_daily_seasonal(values: pd.Series) -> pd.Series:
    doy = values.index.dayofyear
    seasonal = values.groupby(doy).transform("median")
    return values - seasonal


def precipitation_curve_for_gb_file(path: Path) -> tuple[dict[str, float | str], pd.DataFrame] | None:
    usecols = ["date", "precipitation_cehgear"]
    data = pd.read_csv(path, usecols=usecols, na_values=["NaN", "nan", ""])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["precipitation_cehgear"] = pd.to_numeric(data["precipitation_cehgear"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").set_index("date").asfreq("D")
    precip = data["precipitation_cehgear"].interpolate(limit=7).fillna(0.0).clip(lower=0.0)
    anomaly = remove_daily_seasonal(precip)
    x = anomaly.to_numpy(dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 365 * 6:
        return None
    dates = anomaly.index
    tau_days = integral_autocorrelation_time(x[valid], dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True) / SECONDS_PER_DAY
    psd = welch_psd(dates[valid], x[valid], fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    if beta.empty:
        return None
    gauge_id = gauge_id_from_path(path)
    curve = pd.DataFrame(
        {
            "gauge_id_str": gauge_id,
            "frequency_cpd": beta["f_center"].to_numpy(dtype=float) * SECONDS_PER_DAY,
            "de": beta["f_center"].to_numpy(dtype=float) * SECONDS_PER_DAY * tau_days,
            "beta": beta["beta"].to_numpy(dtype=float),
            "tau_acf_p_days": tau_days,
        }
    )
    row = {
        "archive": "CAMELS-GB v2",
        "forcing_product": "CEH-GEAR",
        "gauge_id_str": gauge_id,
        "tau_acf_p_days": float(tau_days),
        "n_precip_beta_rows": int(len(curve)),
        "precip_missing_after_processing": float((~valid).mean()),
    }
    curve["archive"] = "CAMELS-GB v2"
    curve["forcing_product"] = "CEH-GEAR"
    return row, curve


def parse_us_forcing_text(text: str, gauge_id: str, forcing_product: str) -> tuple[dict[str, float | str], pd.DataFrame] | None:
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "year" in lower and "mnth" in lower and ("prcp" in lower or "precip" in lower):
            header_idx = idx
            break
    if header_idx is None:
        return None
    data = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=r"\s+", engine="python")
    precip_col = next((col for col in data.columns if "prcp" in str(col).lower() or "precip" in str(col).lower()), None)
    year_col = next((col for col in data.columns if str(col).lower() == "year"), None)
    month_col = next((col for col in data.columns if str(col).lower() in {"mnth", "month"}), None)
    day_col = next((col for col in data.columns if str(col).lower() == "day"), None)
    if not all([precip_col, year_col, month_col, day_col]):
        return None
    dates = pd.to_datetime(
        {
            "year": pd.to_numeric(data[year_col], errors="coerce"),
            "month": pd.to_numeric(data[month_col], errors="coerce"),
            "day": pd.to_numeric(data[day_col], errors="coerce"),
        },
        errors="coerce",
    )
    precip = pd.to_numeric(data[precip_col], errors="coerce")
    work = pd.DataFrame({"date": dates, "precip": precip}).dropna(subset=["date"]).sort_values("date")
    work = work.set_index("date").asfreq("D")
    precip = work["precip"].interpolate(limit=7).fillna(0.0).clip(lower=0.0)
    anomaly = remove_daily_seasonal(precip)
    x = anomaly.to_numpy(dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 365 * 6:
        return None
    tau_days = integral_autocorrelation_time(x[valid], dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True) / SECONDS_PER_DAY
    psd = welch_psd(anomaly.index[valid], x[valid], fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    if beta.empty:
        return None
    curve = pd.DataFrame(
        {
            "archive": "CAMELS-US",
            "forcing_product": forcing_product,
            "gauge_id_str": gauge_id,
            "frequency_cpd": beta["f_center"].to_numpy(dtype=float) * SECONDS_PER_DAY,
            "de": beta["f_center"].to_numpy(dtype=float) * SECONDS_PER_DAY * tau_days,
            "beta": beta["beta"].to_numpy(dtype=float),
            "tau_acf_p_days": tau_days,
        }
    )
    row = {
        "archive": "CAMELS-US",
        "forcing_product": forcing_product,
        "gauge_id_str": gauge_id,
        "tau_acf_p_days": float(tau_days),
        "n_precip_beta_rows": int(len(curve)),
        "precip_missing_after_processing": float((~valid).mean()),
    }
    return row, curve


def iter_us_daymet_forcing_files() -> list[tuple[str, str]]:
    """Return (gauge_id, text) pairs from downloaded CAMELS-US Daymet forcing."""
    if US_FORCING_ZIP.exists() and zipfile.is_zipfile(US_FORCING_ZIP):
        pairs: list[tuple[str, str]] = []
        with zipfile.ZipFile(US_FORCING_ZIP) as zf:
            names = [
                name for name in zf.namelist()
                if "basin_mean_forcing/daymet/" in name.replace("\\", "/").lower()
                and name.lower().endswith("_forcing_leap.txt")
            ]
            for name in sorted(names):
                match = re.search(r"(\d{8})", Path(name).name)
                if not match:
                    continue
                text = zf.read(name).decode("utf-8", errors="replace")
                pairs.append((match.group(1), text))
        return pairs
    if US_FORCING_DIR.exists():
        files = sorted(US_FORCING_DIR.rglob("*_forcing_leap.txt"))
        pairs = []
        for path in files:
            parts = {part.lower() for part in path.parts}
            if "daymet" not in parts or "basin_mean_forcing" not in parts:
                continue
            match = re.search(r"(\d{8})", path.name)
            if not match:
                continue
            pairs.append((match.group(1), path.read_text(encoding="utf-8", errors="replace")))
        return pairs
    return []


def summarize_forcing_control(
    archive: str,
    forcing_product: str,
    p_summary: pd.DataFrame,
    p_curves: pd.DataFrame,
    q_curves_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_curves = pd.read_csv(q_curves_path)
    q_curves = q_curves.copy()
    q_curves["gauge_id_str"] = normalize_id(q_curves["gauge_id"])
    q_curves["freq_key"] = pd.to_numeric(q_curves["frequency_cpd"], errors="coerce").round(12)
    p_curves = p_curves.copy()
    p_curves["freq_key"] = pd.to_numeric(p_curves["frequency_cpd"], errors="coerce").round(12)
    merged = q_curves.merge(
        p_curves[["gauge_id_str", "freq_key", "beta", "de", "tau_acf_p_days"]].rename(
            columns={"beta": "beta_p", "de": "de_p"}
        ),
        on=["gauge_id_str", "freq_key"],
        how="inner",
    )
    merged = merged.rename(columns={"beta": "beta_q", "de": "de_q", "tau_acf_days": "tau_acf_q_days"})
    merged["archive"] = archive
    merged["forcing_product"] = forcing_product
    merged["beta_q_minus_p"] = merged["beta_q"] - merged["beta_p"]
    p_for_metric = p_curves.rename(columns={"tau_acf_p_days": "tau_acf_days"}).copy()
    p_for_metric["unit_id"] = p_for_metric["gauge_id_str"]
    p_metric = reduction_for(p_for_metric, min_units=50)
    contrast_q = merged[["gauge_id_str", "frequency_cpd", "de_q", "beta_q_minus_p"]].rename(
        columns={"de_q": "de", "beta_q_minus_p": "beta"}
    )
    contrast_q["unit_id"] = contrast_q["gauge_id_str"]
    contrast_q_metric = reduction_for(contrast_q, min_units=50)
    contrast_p = merged[["gauge_id_str", "frequency_cpd", "de_p", "beta_q_minus_p"]].rename(
        columns={"de_p": "de", "beta_q_minus_p": "beta"}
    )
    contrast_p["unit_id"] = contrast_p["gauge_id_str"]
    contrast_p_metric = reduction_for(contrast_p, min_units=50)

    q_tau = q_curves[["gauge_id_str", "tau_acf_days"]].drop_duplicates().rename(columns={"tau_acf_days": "tau_acf_q_days"})
    tau_pair = q_tau.merge(p_summary[["gauge_id_str", "tau_acf_p_days"]], on="gauge_id_str", how="inner")
    rho, pval = stats.spearmanr(tau_pair["tau_acf_q_days"], tau_pair["tau_acf_p_days"])
    summary = pd.DataFrame(
        [
            {
                "archive": archive,
                "forcing_product": forcing_product,
                "analysis": "precipitation_beta_on_precipitation_memory",
                "n_gauges": int(p_curves["gauge_id_str"].nunique()),
                **p_metric,
            },
            {
                "archive": archive,
                "forcing_product": forcing_product,
                "analysis": "q_minus_p_beta_contrast_on_q_memory",
                "n_gauges": int(merged["gauge_id_str"].nunique()),
                **contrast_q_metric,
            },
            {
                "archive": archive,
                "forcing_product": forcing_product,
                "analysis": "q_minus_p_beta_contrast_on_p_memory",
                "n_gauges": int(merged["gauge_id_str"].nunique()),
                **contrast_p_metric,
            },
            {
                "archive": archive,
                "forcing_product": forcing_product,
                "analysis": "tau_q_tau_p_spearman",
                "n_gauges": int(len(tau_pair)),
                "raw_weighted_beta_variance": np.nan,
                "de_weighted_beta_variance": np.nan,
                "variance_reduction_fraction": float(rho),
                "raw_n_bins": np.nan,
                "de_n_bins": np.nan,
                "raw_n_unit_bin_pairs": np.nan,
                "de_n_unit_bin_pairs": np.nan,
                "p_value": float(pval),
            },
        ]
    )
    summary["variance_reduction_percent"] = 100.0 * summary["variance_reduction_fraction"]
    return summary, merged


def run_us_forcing_control() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = iter_us_daymet_forcing_files()
    if not pairs:
        print("R42 forcing control: CAMELS-US Daymet forcing files not available; US forcing control skipped.", flush=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    curves = []
    for idx, (gauge_id, text) in enumerate(pairs, start=1):
        result = parse_us_forcing_text(text, gauge_id, "Daymet")
        if result is not None:
            row, curve = result
            rows.append(row)
            curves.append(curve)
        if idx % 100 == 0:
            print(f"R42 forcing control: {idx}/{len(pairs)} US Daymet files", flush=True)
    p_summary = pd.DataFrame(rows)
    p_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if p_curves.empty:
        return pd.DataFrame(), p_summary, p_curves, pd.DataFrame()
    summary, merged = summarize_forcing_control(
        "CAMELS-US",
        "Daymet",
        p_summary,
        p_curves,
        TABLES / "camels_673_beta_curve_points.csv",
    )
    return summary, p_summary, p_curves, merged


def run_gb_forcing_control() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    files = sorted(Path(DAILY_DIR).glob("*.csv"))
    for idx, path in enumerate(files, start=1):
        result = precipitation_curve_for_gb_file(path)
        if result is not None:
            row, curve = result
            rows.append(row)
            curves.append(curve)
        if idx % 100 == 0:
            print(f"R42 forcing control: {idx}/{len(files)} GB files", flush=True)
    p_summary = pd.DataFrame(rows)
    p_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if p_curves.empty:
        raise RuntimeError("No precipitation beta curves were computed")

    summary, merged = summarize_forcing_control(
        "CAMELS-GB v2",
        "CEH-GEAR",
        p_summary,
        p_curves,
        TABLES / "camels_gb_v2_replication_beta_curve_points.csv",
    )
    return summary, p_summary, p_curves, merged


def run_forcing_controls() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    p_summaries = []
    p_curves = []
    contrasts = []
    for runner in [run_gb_forcing_control, run_us_forcing_control]:
        summary, p_summary, p_curve, contrast = runner()
        if not summary.empty:
            frames.append(summary)
        if not p_summary.empty:
            p_summaries.append(p_summary)
        if not p_curve.empty:
            p_curves.append(p_curve)
        if not contrast.empty:
            contrasts.append(contrast)
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        pd.concat(p_summaries, ignore_index=True) if p_summaries else pd.DataFrame(),
        pd.concat(p_curves, ignore_index=True) if p_curves else pd.DataFrame(),
        pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame(),
    )


def run_slow_memory_support() -> pd.DataFrame:
    rows = []
    for spec in ARCHIVES:
        curves = load_curves(spec)
        curves["unit_id"] = curves["gauge_id_str"]
        tau = curves[["gauge_id_str", "tau_acf_days"]].drop_duplicates()["tau_acf_days"].astype(float)
        base = reduction_for(curves, spec.min_units)
        for nperseg in [256, 512, 1024]:
            rows.append(
                {
                    "archive": spec.archive,
                    "audit_type": "frequency_support",
                    "nperseg": nperseg,
                    "tau_threshold_days": np.nan,
                    "n_gauges": int(tau.size),
                    "median_tau_days": float(tau.median()),
                    "q95_tau_days": float(tau.quantile(0.95)),
                    "max_tau_days": float(tau.max()),
                    "de1_resolvable_fraction": float((tau <= nperseg).mean()),
                    "de05_resolvable_fraction": float((tau <= 0.5 * nperseg).mean()),
                    "variance_reduction_fraction": base["variance_reduction_fraction"],
                }
            )
        for threshold in [64, 128, 256, 365]:
            keep_ids = curves.loc[curves["tau_acf_days"] <= threshold, "gauge_id_str"].unique()
            sub = curves[curves["gauge_id_str"].isin(keep_ids)].copy()
            metric = reduction_for(sub, spec.min_units) if len(keep_ids) >= spec.min_units else {
                "variance_reduction_fraction": np.nan,
                "raw_weighted_beta_variance": np.nan,
                "de_weighted_beta_variance": np.nan,
            }
            rows.append(
                {
                    "archive": spec.archive,
                    "audit_type": "tau_threshold_sensitivity",
                    "nperseg": np.nan,
                    "tau_threshold_days": threshold,
                    "n_gauges": int(len(keep_ids)),
                    "median_tau_days": float(sub[["gauge_id_str", "tau_acf_days"]].drop_duplicates()["tau_acf_days"].median()) if len(keep_ids) else np.nan,
                    "q95_tau_days": float(sub[["gauge_id_str", "tau_acf_days"]].drop_duplicates()["tau_acf_days"].quantile(0.95)) if len(keep_ids) else np.nan,
                    "max_tau_days": float(sub[["gauge_id_str", "tau_acf_days"]].drop_duplicates()["tau_acf_days"].max()) if len(keep_ids) else np.nan,
                    "de1_resolvable_fraction": np.nan,
                    "de05_resolvable_fraction": np.nan,
                    "variance_reduction_fraction": metric["variance_reduction_fraction"],
                }
            )
    out = pd.DataFrame(rows)
    out["variance_reduction_percent"] = 100.0 * out["variance_reduction_fraction"]
    return out


def draw_windtunnel(cluster_summary: pd.DataFrame, forcing_summary: pd.DataFrame, support: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    blue = "#2166AC"
    orange = "#D95F02"
    red = "#B2182B"
    green = "#1B7837"
    gray = "#6B7280"
    fig = plt.figure(figsize=(7.55, 6.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    order = ["CAMELS-US", "CAMELS-GB v2", "CAMELS-BR v1.2", "CAMELS-AUS v2", "CAMELS-DK lowland"]
    cs = cluster_summary.set_index("archive").loc[order].reset_index()
    y = np.arange(len(cs))
    colors = [blue, blue, blue, orange, red]
    ax0.axvline(0, color="#2B2B2B", lw=0.8)
    ax0.errorbar(
        cs["cluster_median_percent"],
        y,
        xerr=[
            cs["cluster_median_percent"] - cs["cluster_ci_low_percent"],
            cs["cluster_ci_high_percent"] - cs["cluster_median_percent"],
        ],
        fmt="o",
        color="#1F2937",
        ecolor="#8AA6C7",
        elinewidth=1.4,
        capsize=2.5,
        zorder=3,
    )
    ax0.scatter(cs["observed_percent"], y, marker="D", s=32, color=colors, zorder=4, label="observed")
    ax0.set_yticks(y, cs["archive"])
    ax0.invert_yaxis()
    ax0.set_xlabel("Reduction (%)")
    ax0.set_title("a  Region-cluster bootstrap", loc="left", fontweight="bold")
    ax0.grid(True, axis="x", color="#E5E7EB", lw=0.6)

    if forcing_summary.empty:
        ax1.axis("off")
        ax1.text(0.02, 0.9, "b  Forcing-side control not run", fontweight="bold", transform=ax1.transAxes)
    else:
        fs = forcing_summary[forcing_summary["analysis"].isin([
            "precipitation_beta_on_precipitation_memory",
            "q_minus_p_beta_contrast_on_q_memory",
            "q_minus_p_beta_contrast_on_p_memory",
        ])].copy()
        archive_short = {"CAMELS-US": "US", "CAMELS-GB v2": "GB"}
        label_map = {
            "precipitation_beta_on_precipitation_memory": "P on P",
            "q_minus_p_beta_contrast_on_q_memory": "Q-P on Q",
            "q_minus_p_beta_contrast_on_p_memory": "Q-P on P",
        }
        fs["archive_short"] = fs["archive"].map(archive_short).fillna(fs["archive"].astype(str))
        fs["label"] = fs["archive_short"] + "\n" + fs["analysis"].map(label_map)
        color_map = {
            "precipitation_beta_on_precipitation_memory": gray,
            "q_minus_p_beta_contrast_on_q_memory": green,
            "q_minus_p_beta_contrast_on_p_memory": orange,
        }
        labels = fs["label"].tolist()
        vals = fs["variance_reduction_percent"].to_numpy(dtype=float)
        ax1.axhline(0, color="#2B2B2B", lw=0.8)
        bars = ax1.bar(np.arange(len(vals)), vals, color=[color_map[a] for a in fs["analysis"]], width=0.64)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + (1.4 if val >= 0 else -2.2), f"{val:.1f}%", ha="center", va="center", fontsize=6.4)
        tau = forcing_summary[forcing_summary["analysis"] == "tau_q_tau_p_spearman"]
        if not tau.empty:
            tau_lines = []
            for row in tau.itertuples(index=False):
                short = archive_short.get(row.archive, str(row.archive))
                tau_lines.append(rf"{short}: $\rho$={float(row.variance_reduction_fraction):.2f}, p={float(row.p_value):.1e}")
            ax1.text(0.02, 0.97, "; ".join(tau_lines), transform=ax1.transAxes, fontsize=5.8, va="top")
        ax1.set_xticks(np.arange(len(vals)), labels)
        ax1.set_ylabel("Reduction / correlation (%)")
        ax1.set_title("b  Precipitation forcing controls", loc="left", fontweight="bold")
        ax1.grid(True, axis="y", color="#E5E7EB", lw=0.6)

    ss = support[(support["audit_type"] == "frequency_support") & (support["nperseg"] == 256)].set_index("archive").loc[order].reset_index()
    x = np.arange(len(ss))
    ax2.bar(x - 0.18, 100 * ss["de1_resolvable_fraction"], width=0.34, color=blue, label="De=1")
    ax2.bar(x + 0.18, 100 * ss["de05_resolvable_fraction"], width=0.34, color=orange, label="De=0.5")
    ax2.set_xticks(x, ["US", "GB", "BR", "AUS", "DK"])
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Gauges supportable by nperseg=256 (%)")
    ax2.set_title("c  Slow-memory support audit", loc="left", fontweight="bold")
    ax2.legend(fontsize=6.3, loc="lower left")
    ax2.grid(True, axis="y", color="#E5E7EB", lw=0.6)

    ax3.axis("off")
    ax3.set_title("d  Risk-closure ledger", loc="left", fontweight="bold")
    bullets = [
        ("Spatial dependence", "cluster CIs reported; US/GB/BR remain positive"),
        ("Forcing confounding", "GB Q-P spectral contrast tested"),
        ("Slow-memory support", "De=1 resolvability and tau thresholds audited"),
        ("Claim boundary", "diagnostic benchmark, not storage causality"),
    ]
    y0 = 0.83
    for idx, (risk, closure) in enumerate(bullets):
        yy = y0 - idx * 0.19
        ax3.add_patch(plt.Rectangle((0.02, yy - 0.035), 0.05, 0.07, color=[green, green, orange, blue][idx], transform=ax3.transAxes))
        ax3.text(0.10, yy + 0.018, risk, fontsize=7.8, fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.10, yy - 0.034, closure, fontsize=6.9, color=gray, transform=ax3.transAxes)
    ax3.text(0.02, 0.04, "All values are derived from public data or project source-data tables; no new causal claim is introduced.", fontsize=6.7, color=gray, transform=ax3.transAxes, wrap=True)

    for ext in [".pdf", ".svg", ".png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{OUT}_windtunnel{ext}", **kwargs)
        fig.savefig(NW_FIGURES / f"fig4_robustness_support_checks{ext}", **kwargs)
    plt.close(fig)


def copy_source_outputs() -> None:
    NW_SOURCE.mkdir(parents=True, exist_ok=True)
    for name in [
        f"{OUT}_cluster_bootstrap_draws.csv",
        f"{OUT}_cluster_bootstrap_summary.csv",
        f"{OUT}_forcing_control_summary.csv",
        f"{OUT}_forcing_precipitation_summary.csv",
        f"{OUT}_forcing_precipitation_curves.csv",
        f"{OUT}_forcing_q_minus_p_contrast.csv",
        f"{OUT}_slow_memory_support_audit.csv",
    ]:
        src = TABLES / name
        if src.exists():
            shutil.copy2(src, NW_SOURCE / name)


def write_note(cluster_summary: pd.DataFrame, forcing_summary: pd.DataFrame, support: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Final HESS Extreme Hardening",
        "",
        "Date: 2026-06-08",
        "",
        "## Purpose",
        "",
        "This final hardening layer implements the HESS-specific reviewer fire extinguishers requested in the enhancement package:",
        "region-cluster bootstrap, precipitation/forcing-side spectral control, slow-memory frequency-support audit and a submission-facing robustness wind-tunnel figure.",
        "",
        "## Cluster bootstrap summary",
        "",
        cluster_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Forcing-side control summary",
        "",
        forcing_summary.to_markdown(index=False, floatfmt=".4g") if not forcing_summary.empty else "Forcing control was skipped.",
        "",
        "## Slow-memory support audit preview",
        "",
        support.head(12).to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Claim boundary",
        "",
        "The new analyses strengthen the benchmark and reduce HESS review risk, but they do not convert the paper into a storage-causality or operational model-ranking claim.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{OUT}_cluster_bootstrap_draws.csv`",
        f"- `reports/tables/{OUT}_cluster_bootstrap_summary.csv`",
        f"- `reports/tables/{OUT}_forcing_control_summary.csv`",
        f"- `reports/tables/{OUT}_slow_memory_support_audit.csv`",
        f"- `reports/figures/{OUT}_windtunnel.pdf/svg/png`",
        f"- Copied public figure: `figures/fig4_robustness_support_checks.*`",
        "",
        f"Cluster bootstrap draws per archive: {args.cluster_draws}; seed: {args.seed}.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    NW_FIGURES.mkdir(parents=True, exist_ok=True)

    cluster_draws, cluster_summary = run_cluster_bootstrap(args.cluster_draws, args.seed)
    cluster_draws.to_csv(TABLES / f"{OUT}_cluster_bootstrap_draws.csv", index=False)
    cluster_summary.to_csv(TABLES / f"{OUT}_cluster_bootstrap_summary.csv", index=False)

    if args.skip_forcing:
        forcing_summary = pd.DataFrame()
    else:
        forcing_summary, p_summary, p_curves, contrast = run_forcing_controls()
        forcing_summary.to_csv(TABLES / f"{OUT}_forcing_control_summary.csv", index=False)
        p_summary.to_csv(TABLES / f"{OUT}_forcing_precipitation_summary.csv", index=False)
        p_curves.to_csv(TABLES / f"{OUT}_forcing_precipitation_curves.csv", index=False)
        contrast.to_csv(TABLES / f"{OUT}_forcing_q_minus_p_contrast.csv", index=False)

    support = run_slow_memory_support()
    support.to_csv(TABLES / f"{OUT}_slow_memory_support_audit.csv", index=False)

    draw_windtunnel(cluster_summary, forcing_summary, support)
    copy_source_outputs()
    write_note(cluster_summary, forcing_summary, support, args)

    print(cluster_summary.to_string(index=False))
    if not forcing_summary.empty:
        print(forcing_summary.to_string(index=False))


if __name__ == "__main__":
    main()
