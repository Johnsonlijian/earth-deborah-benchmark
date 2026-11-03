"""R23 independent groundwater-storage validation.

CAMELS-GB v2 includes groundwater-well level time series, but the published
well table does not provide an official one-to-one well-to-catchment mapping.
This script therefore keeps the evidence boundary explicit:

* compute independent groundwater-level memory for monthly well records;
* compare well memory with the nearest CAMELS-GB gauge memory only as a
  nearest-neighbour sensitivity analysis;
* keep cross-archive hydrogeologic and soil proxy support as the stronger
  catchment-level evidence path.
"""

from __future__ import annotations

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "camels_gb_v2"
GW_DIR = DATA / "groundwater_monthly"
ATTR_DIR = DATA / "attributes"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r23_groundwater_storage_validation"

BASE = (
    "https://catalogue.ceh.ac.uk/datastore/eidchub/"
    "9a46d428-958f-4ac1-86eb-94eee70c0955"
)
MONTHLY_URL = BASE + "/Catchment_Timeseries/groundwater/monthly/"
ATTR_URL = BASE + "/Catchment_Attributes/"


def ensure_inputs() -> None:
    GW_DIR.mkdir(parents=True, exist_ok=True)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    for filename in [
        "camels_gb_v2_groundwaterwell_attributes.csv",
        "camels_gb_v2_hydrogeology_attributes.csv",
    ]:
        target = ATTR_DIR / filename
        if (not target.exists()) or target.stat().st_size < 20:
            urllib.request.urlretrieve(ATTR_URL + filename, target)

    index_path = GW_DIR / "index.html"
    if not index_path.exists():
        urllib.request.urlretrieve(MONTHLY_URL, index_path)
    html = index_path.read_text(encoding="utf-8", errors="ignore")
    files = sorted(set(re.findall(r'href="(camels_gb_v2_groundwater_monthly_timeseries_[^"]+\.csv)"', html)))
    if not files:
        raise RuntimeError("No CAMELS-GB v2 monthly groundwater files found in index.")
    def fetch(filename: str) -> tuple[str, str]:
        target = GW_DIR / filename
        if not target.exists():
            last_error = ""
            for _ in range(3):
                try:
                    with urllib.request.urlopen(MONTHLY_URL + filename, timeout=90) as src:
                        target.write_bytes(src.read())
                    return filename, "downloaded"
                except Exception as exc:  # network robustness for public datastore
                    last_error = repr(exc)
            return filename, last_error
        return filename, "exists"

    missing = [filename for filename in files if (not (GW_DIR / filename).exists()) or (GW_DIR / filename).stat().st_size < 20]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, filename) for filename in missing]
            failures = []
            for future in as_completed(futures):
                filename, status = future.result()
                if status not in {"exists", "downloaded"}:
                    failures.append({"filename": filename, "status": status})
            if failures:
                pd.DataFrame(failures).to_csv(TABLES / f"{OUT}_download_failures.csv", index=False)


def positive_integral_acf_tau(values: pd.Series, max_lag: int = 120) -> float:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if x.notna().sum() < 60:
        return np.nan
    month_median = x.groupby(x.index.month).transform("median")
    anomalies = (x - month_median).interpolate(limit=3, limit_direction="both").dropna()
    if len(anomalies) < 60:
        return np.nan
    arr = anomalies.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 60 or np.nanstd(arr) <= 0:
        return np.nan
    arr = (arr - np.nanmean(arr)) / np.nanstd(arr)
    tau = 1.0
    for lag in range(1, min(max_lag, arr.size - 1) + 1):
        a = arr[:-lag]
        b = arr[lag:]
        if a.size < 30:
            break
        rho = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(rho) or rho <= 0:
            break
        tau += 2.0 * rho
    return tau


def summarise_wells() -> pd.DataFrame:
    attrs = pd.read_csv(ATTR_DIR / "camels_gb_v2_groundwaterwell_attributes.csv")
    attrs["gw_well_id_norm"] = attrs["gw_well_id"].astype(str).str.lower()
    rows = []
    for path in sorted(GW_DIR.glob("camels_gb_v2_groundwater_monthly_timeseries_*.csv")):
        match = re.search(r"monthly_timeseries_(.+)_\d{8}-\d{8}\.csv$", path.name)
        if not match:
            continue
        well_id = match.group(1).lower()
        try:
            data = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "date" not in data or "groundwater_level" not in data:
            continue
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).set_index("date").sort_index()
        gw = pd.to_numeric(data["groundwater_level"], errors="coerce")
        valid = int(gw.notna().sum())
        n_months = int(len(gw))
        if n_months == 0:
            continue
        tau_months = positive_integral_acf_tau(gw)
        rows.append(
            {
                "gw_well_id_norm": well_id,
                "gw_well_id": match.group(1).upper(),
                "n_months": n_months,
                "valid_months": valid,
                "missing_frac": 1.0 - valid / n_months,
                "tau_gw_months": tau_months,
            }
        )
    summary = pd.DataFrame(rows)
    return summary.merge(attrs, on=["gw_well_id_norm"], how="left", suffixes=("", "_attr"))


def nearest_gauge_join(wells: pd.DataFrame) -> pd.DataFrame:
    gauges = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv", dtype={"gauge_id": str})
    needed = [
        "gauge_id",
        "gauge_name",
        "gauge_easting",
        "gauge_northing",
        "gauge_lat",
        "gauge_lon",
        "tau_acf_days",
        "baseflow_index",
        "area",
        "aridity",
        "p_seasonality",
    ]
    gauges = gauges[[c for c in needed if c in gauges.columns]].dropna(subset=["gauge_easting", "gauge_northing", "tau_acf_days"])
    gx = gauges["gauge_easting"].to_numpy(dtype=float)
    gy = gauges["gauge_northing"].to_numpy(dtype=float)
    rows = []
    for _, row in wells.iterrows():
        wx = pd.to_numeric(pd.Series([row.get("gw_well_easting")]), errors="coerce").iloc[0]
        wy = pd.to_numeric(pd.Series([row.get("gw_well_northing")]), errors="coerce").iloc[0]
        if not np.isfinite(wx) or not np.isfinite(wy):
            rows.append(row.to_dict())
            continue
        dist_km = np.sqrt((gx - wx) ** 2 + (gy - wy) ** 2) / 1000.0
        idx = int(np.nanargmin(dist_km))
        merged = row.to_dict()
        for col, val in gauges.iloc[idx].items():
            merged[f"nearest_{col}"] = val
        merged["nearest_gauge_distance_km"] = float(dist_km[idx])
        rows.append(merged)
    return pd.DataFrame(rows)


def threshold_sensitivity(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [15, 25, 40, 60, 80, 120]:
        subset = joined[
            (joined["nearest_gauge_distance_km"] <= threshold)
            & np.isfinite(joined["tau_gw_months"])
            & np.isfinite(pd.to_numeric(joined["nearest_tau_acf_days"], errors="coerce"))
        ].copy()
        if len(subset) >= 8:
            rho, pval = stats.spearmanr(np.log10(subset["tau_gw_months"]), np.log10(subset["nearest_tau_acf_days"]))
        else:
            rho, pval = np.nan, np.nan
        rows.append(
            {
                "distance_threshold_km": threshold,
                "n_wells": int(len(subset)),
                "spearman_rho_log_gw_tau_vs_log_nearest_q_tau": rho,
                "p_value": pval,
            }
        )
    return pd.DataFrame(rows)


def hydrogeology_associations() -> pd.DataFrame:
    gb = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv", dtype={"gauge_id": str})
    hydro = pd.read_csv(ATTR_DIR / "camels_gb_v2_hydrogeology_attributes.csv", dtype={"gauge_id": str})
    work = gb[["gauge_id", "tau_acf_days"]].merge(hydro, on="gauge_id", how="left")
    work["log_tau"] = np.log10(pd.to_numeric(work["tau_acf_days"], errors="coerce"))
    labels = {
        "intergranular high productivity": "inter_high_perc",
        "intergranular moderate productivity": "inter_mod_perc",
        "fracture high productivity": "frac_high_perc",
        "fracture moderate productivity": "frac_mod_perc",
        "no groundwater aquifer": "no_gw_perc",
        "low productivity aquifer": "low_nsig_perc",
    }
    rows = []
    for label, col in labels.items():
        if col not in work:
            continue
        pair = work[["log_tau", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 50:
            continue
        rho, pval = stats.spearmanr(pair[col], pair["log_tau"])
        rows.append(
            {
                "proxy": label,
                "column": col,
                "n": int(len(pair)),
                "spearman_rho_logtau": float(rho),
                "p_value": float(pval),
            }
        )
    return pd.DataFrame(rows)


def plot(joined: pd.DataFrame, sensitivity: pd.DataFrame, hydro: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    usable = joined[np.isfinite(joined["tau_gw_months"])].copy()
    fig = plt.figure(figsize=(7.35, 5.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    aquifers = usable["aquifer"].fillna("unknown")
    common = aquifers.value_counts().head(6).index
    plot_data = usable[usable["aquifer"].isin(common)].copy()
    order = plot_data.groupby("aquifer")["tau_gw_months"].median().sort_values().index
    data = [plot_data.loc[plot_data["aquifer"] == a, "tau_gw_months"].dropna() for a in order]
    ax0.boxplot(data, vert=False, patch_artist=True, widths=0.62, boxprops={"facecolor": "#A5C8E1", "edgecolor": "#2F6FA8"}, medianprops={"color": "#C4513F", "lw": 1.2})
    ax0.set_yticks(range(1, len(order) + 1), order)
    ax0.set_xscale("log")
    ax0.set_xlabel("groundwater-level memory, tau [months]")
    ax0.set_title("a  Independent groundwater memory by aquifer", loc="left", fontsize=8.8, fontweight="bold")
    ax0.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    sc = ax1.scatter(
        usable["gw_well_easting"],
        usable["gw_well_northing"],
        c=np.log10(usable["tau_gw_months"]),
        s=34,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.3,
    )
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("British National Grid easting")
    ax1.set_ylabel("northing")
    ax1.set_title("b  CAMELS-GB well locations", loc="left", fontsize=8.8, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label(r"$\log_{10}$ tau [months]")

    paired = usable[(usable["nearest_gauge_distance_km"] <= 60) & np.isfinite(pd.to_numeric(usable["nearest_tau_acf_days"], errors="coerce"))].copy()
    ax2.scatter(
        np.log10(paired["tau_gw_months"]),
        np.log10(paired["nearest_tau_acf_days"]),
        c=paired["nearest_gauge_distance_km"],
        cmap="magma_r",
        s=34,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.9,
    )
    if len(paired) >= 8:
        rho, pval = stats.spearmanr(np.log10(paired["tau_gw_months"]), np.log10(paired["nearest_tau_acf_days"]))
        title = f"c  Nearest-gauge check (rho={rho:.2f}, n={len(paired)})"
    else:
        title = f"c  Nearest-gauge check (n={len(paired)})"
    ax2.set_xlabel(r"$\log_{10}$ groundwater tau [months]")
    ax2.set_ylabel(r"$\log_{10}$ nearest discharge tau [days]")
    ax2.set_title(title, loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, color="#E4E9EF", linewidth=0.55)

    hydro_sorted = hydro.sort_values("spearman_rho_logtau")
    y = np.arange(len(hydro_sorted))
    ax3.axvline(0, color="#667085", lw=0.8)
    colors = ["#C4513F" if v > 0 else "#2F6FA8" for v in hydro_sorted["spearman_rho_logtau"]]
    ax3.barh(y, hydro_sorted["spearman_rho_logtau"], color=colors, alpha=0.86, height=0.62)
    ax3.set_yticks(y, hydro_sorted["proxy"], fontsize=6.4)
    ax3.set_xlabel(r"Spearman rho with $\log_{10}$ discharge tau")
    ax3.set_ylabel("")
    ax3.set_title("d  Catchment hydrogeology proxies", loc="left", fontsize=8.8, fontweight="bold")
    ax3.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(joined: pd.DataFrame, sensitivity: pd.DataFrame, hydro: pd.DataFrame) -> None:
    usable = joined[np.isfinite(joined["tau_gw_months"])].copy()
    largest = sensitivity.sort_values("distance_threshold_km", ascending=False).iloc[0]
    close60 = sensitivity[sensitivity["distance_threshold_km"] == 60].iloc[0]
    strongest_hydro = hydro.reindex(hydro["spearman_rho_logtau"].abs().sort_values(ascending=False).index).head(3)
    lines = [
        "# R23 Independent Groundwater Storage Validation",
        "",
        f"Usable CAMELS-GB monthly groundwater wells: {len(usable)}.",
        f"Median groundwater-level memory: {usable['tau_gw_months'].median():.2f} months.",
        "",
        "Nearest-gauge sensitivity is explicitly downgraded because the CAMELS-GB",
        "groundwater-well table does not provide an official one-to-one well-to-catchment",
        "mapping.",
        "",
        f"At 60 km, n={int(close60['n_wells'])}, Spearman rho={close60['spearman_rho_log_gw_tau_vs_log_nearest_q_tau']:.3f}, p={close60['p_value']:.3g}.",
        f"Largest threshold tested ({int(largest['distance_threshold_km'])} km): n={int(largest['n_wells'])}, rho={largest['spearman_rho_log_gw_tau_vs_log_nearest_q_tau']:.3f}.",
        "",
        "Strongest catchment-level CAMELS-GB hydrogeology associations with discharge memory:",
    ]
    for _, row in strongest_hydro.iterrows():
        lines.append(f"- {row['proxy']}: rho={row['spearman_rho_logtau']:.3f}, p={row['p_value']:.3g}, n={int(row['n'])}.")
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    ensure_inputs()
    wells = summarise_wells()
    joined = nearest_gauge_join(wells)
    sensitivity = threshold_sensitivity(joined)
    hydro = hydrogeology_associations()
    joined.to_csv(TABLES / f"{OUT}_well_summary.csv", index=False)
    sensitivity.to_csv(TABLES / f"{OUT}_nearest_gauge_sensitivity.csv", index=False)
    hydro.to_csv(TABLES / f"{OUT}_hydrogeology_associations.csv", index=False)
    plot(joined, sensitivity, hydro)
    write_note(joined, sensitivity, hydro)


if __name__ == "__main__":
    main()
