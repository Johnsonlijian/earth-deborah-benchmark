"""R24 official-boundary groundwater-to-catchment validation.

This upgrades the R23 nearest-gauge groundwater check by using the official
CAMELS-GB v2 catchment boundary polygons. Because catchments are nested, a
groundwater well may fall inside several downstream catchments. The primary
contract assigns each well to the smallest containing catchment; the full
many-to-many containment table is retained as a boundary diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "camels_gb_v2"
BOUNDARIES = DATA / "boundaries" / "camels_gb_v2_catchment_boundaries.shp"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r24_groundwater_boundary_validation"


def load_inputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    wells_path = TABLES / "r23_groundwater_storage_validation_well_summary.csv"
    if not wells_path.exists():
        raise FileNotFoundError("Run R23 groundwater validation before R24 boundary validation.")
    wells = pd.read_csv(wells_path)
    wells = wells[np.isfinite(pd.to_numeric(wells["tau_gw_months"], errors="coerce"))].copy()
    wells["gw_well_easting"] = pd.to_numeric(wells["gw_well_easting"], errors="coerce")
    wells["gw_well_northing"] = pd.to_numeric(wells["gw_well_northing"], errors="coerce")
    wells = wells.dropna(subset=["gw_well_easting", "gw_well_northing", "tau_gw_months"])

    boundaries = gpd.read_file(BOUNDARIES)
    boundaries["gauge_id"] = boundaries["ID_STRING"].astype(str)
    boundaries["boundary_area_km2"] = pd.to_numeric(boundaries["area"], errors="coerce")

    gauges = pd.read_csv(TABLES / "camels_gb_v2_replication_summary.csv", dtype={"gauge_id": str})
    keep = [
        "gauge_id",
        "gauge_name",
        "tau_acf_days",
        "baseflow_index",
        "area",
        "aridity",
        "p_seasonality",
        "gauge_easting",
        "gauge_northing",
    ]
    gauges = gauges[[c for c in keep if c in gauges.columns]].copy()
    return wells, boundaries, gauges


def containment_tables(wells: pd.DataFrame, boundaries: gpd.GeoDataFrame, gauges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    boundary_cols = ["gauge_id", "boundary_area_km2", "geometry"]
    for _, well in wells.iterrows():
        point = Point(float(well["gw_well_easting"]), float(well["gw_well_northing"]))
        containing = boundaries.loc[boundaries.geometry.covers(point), boundary_cols].copy()
        if containing.empty:
            rows.append(
                {
                    "gw_well_id": well["gw_well_id"],
                    "gw_well_name": well.get("gw_well_name"),
                    "tau_gw_months": well["tau_gw_months"],
                    "aquifer": well.get("aquifer"),
                    "contained_gauge_id": np.nan,
                    "boundary_area_km2": np.nan,
                    "containment_rank_smallest_area": np.nan,
                    "n_containing_catchments": 0,
                }
            )
            continue
        containing = containing.sort_values("boundary_area_km2", ascending=True).reset_index(drop=True)
        n_containing = len(containing)
        for rank, (_, catch) in enumerate(containing.iterrows(), start=1):
            rows.append(
                {
                    "gw_well_id": well["gw_well_id"],
                    "gw_well_name": well.get("gw_well_name"),
                    "tau_gw_months": well["tau_gw_months"],
                    "aquifer": well.get("aquifer"),
                    "gw_well_easting": well["gw_well_easting"],
                    "gw_well_northing": well["gw_well_northing"],
                    "contained_gauge_id": catch["gauge_id"],
                    "boundary_area_km2": catch["boundary_area_km2"],
                    "containment_rank_smallest_area": rank,
                    "n_containing_catchments": n_containing,
                }
            )
    all_map = pd.DataFrame(rows)
    all_map = all_map.merge(gauges, left_on="contained_gauge_id", right_on="gauge_id", how="left", suffixes=("", "_gauge"))
    primary = all_map[all_map["containment_rank_smallest_area"] == 1].copy()
    catchment_summary = (
        primary.dropna(subset=["contained_gauge_id"])
        .groupby("contained_gauge_id", as_index=False)
        .agg(
            n_wells=("gw_well_id", "nunique"),
            median_tau_gw_months=("tau_gw_months", "median"),
            mean_tau_gw_months=("tau_gw_months", "mean"),
            aquifers=("aquifer", lambda x: "; ".join(sorted(set(str(v) for v in x.dropna()))[:4])),
            boundary_area_km2=("boundary_area_km2", "first"),
            tau_acf_days=("tau_acf_days", "first"),
            baseflow_index=("baseflow_index", "first"),
            aridity=("aridity", "first"),
            p_seasonality=("p_seasonality", "first"),
        )
    )
    catchment_summary["log_tau_gw_months"] = np.log10(catchment_summary["median_tau_gw_months"])
    catchment_summary["log_tau_acf_days"] = np.log10(catchment_summary["tau_acf_days"])
    return all_map, primary, catchment_summary


def association_summary(primary: pd.DataFrame, catchments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    well_pairs = primary.dropna(subset=["tau_gw_months", "tau_acf_days"]).copy()
    if len(well_pairs) >= 8:
        rho, pval = stats.spearmanr(np.log10(well_pairs["tau_gw_months"]), np.log10(well_pairs["tau_acf_days"]))
        rows.append(
            {
                "contract": "well_level_smallest_containing_catchment",
                "n": int(len(well_pairs)),
                "n_catchments": int(well_pairs["contained_gauge_id"].nunique()),
                "spearman_rho_log_gw_tau_vs_log_discharge_tau": float(rho),
                "p_value": float(pval),
            }
        )
    catch_pairs = catchments.dropna(subset=["median_tau_gw_months", "tau_acf_days"]).copy()
    if len(catch_pairs) >= 8:
        rho, pval = stats.spearmanr(catch_pairs["log_tau_gw_months"], catch_pairs["log_tau_acf_days"])
        rows.append(
            {
                "contract": "catchment_median_wells_smallest_containing",
                "n": int(len(catch_pairs)),
                "n_catchments": int(len(catch_pairs)),
                "spearman_rho_log_gw_tau_vs_log_discharge_tau": float(rho),
                "p_value": float(pval),
            }
        )
    if len(catch_pairs) >= 8 and "baseflow_index" in catch_pairs:
        rho, pval = stats.spearmanr(catch_pairs["median_tau_gw_months"], catch_pairs["baseflow_index"], nan_policy="omit")
        rows.append(
            {
                "contract": "catchment_median_groundwater_tau_vs_bfi",
                "n": int(catch_pairs[["median_tau_gw_months", "baseflow_index"]].dropna().shape[0]),
                "n_catchments": int(catch_pairs[["median_tau_gw_months", "baseflow_index"]].dropna().shape[0]),
                "spearman_rho_log_gw_tau_vs_log_discharge_tau": float(rho),
                "p_value": float(pval),
            }
        )
    return pd.DataFrame(rows)


def plot(boundaries: gpd.GeoDataFrame, primary: pd.DataFrame, catchments: pd.DataFrame, assoc: pd.DataFrame) -> None:
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
    fig = plt.figure(figsize=(7.35, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    boundaries.boundary.plot(ax=ax0, color="#D0D5DD", linewidth=0.18)
    matched = primary.dropna(subset=["contained_gauge_id"]).copy()
    sc = ax0.scatter(
        matched["gw_well_easting"],
        matched["gw_well_northing"],
        c=np.log10(matched["tau_gw_months"]),
        s=28,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.25,
    )
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("a  Official-boundary well containment", loc="left", fontsize=8.8, fontweight="bold")
    ax0.set_xlabel("British National Grid easting")
    ax0.set_ylabel("northing")
    cb = fig.colorbar(sc, ax=ax0, fraction=0.04, pad=0.02)
    cb.set_label(r"$\log_{10}$ groundwater tau [months]")

    x = np.log10(matched["tau_gw_months"])
    y = np.log10(pd.to_numeric(matched["tau_acf_days"], errors="coerce"))
    valid = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    rho, pval = stats.spearmanr(valid["x"], valid["y"]) if len(valid) >= 8 else (np.nan, np.nan)
    ax1.scatter(x, y, c=matched["boundary_area_km2"], cmap="magma_r", s=36, edgecolor="white", linewidth=0.25, alpha=0.88)
    ax1.set_xlabel(r"$\log_{10}$ groundwater tau [months]")
    ax1.set_ylabel(r"$\log_{10}$ discharge tau [days]")
    ax1.set_title(f"b  Smallest containing catchment (rho={rho:.2f})", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.5)

    containment_counts = primary["n_containing_catchments"].dropna()
    ax2.hist(containment_counts, bins=np.arange(0.5, containment_counts.max() + 1.5, 1), color="#2F6FA8", alpha=0.84)
    ax2.set_xlabel("number of containing CAMELS-GB catchments")
    ax2.set_ylabel("wells")
    ax2.set_title("c  Nested-catchment mapping boundary", loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, axis="y", color="#E4E9EF", linewidth=0.5)

    if not catchments.empty:
        cvalid = catchments.dropna(subset=["log_tau_gw_months", "log_tau_acf_days"]).copy()
        rho_c, p_c = stats.spearmanr(cvalid["log_tau_gw_months"], cvalid["log_tau_acf_days"]) if len(cvalid) >= 8 else (np.nan, np.nan)
        sizes = 20 + 14 * np.sqrt(cvalid["n_wells"].clip(lower=1))
        ax3.scatter(cvalid["log_tau_gw_months"], cvalid["log_tau_acf_days"], s=sizes, color="#59A14F", edgecolor="white", linewidth=0.25, alpha=0.9)
        ax3.set_title(f"d  Catchment-median wells (rho={rho_c:.2f})", loc="left", fontsize=8.8, fontweight="bold")
    ax3.set_xlabel(r"$\log_{10}$ median groundwater tau [months]")
    ax3.set_ylabel(r"$\log_{10}$ discharge tau [days]")
    ax3.grid(True, color="#E4E9EF", linewidth=0.5)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(primary: pd.DataFrame, catchments: pd.DataFrame, assoc: pd.DataFrame) -> None:
    matched = primary.dropna(subset=["contained_gauge_id"])
    lines = [
        "# R24 Official-Boundary Groundwater Validation",
        "",
        f"Usable wells with official CAMELS-GB boundary containment: {matched['gw_well_id'].nunique()} of {primary['gw_well_id'].nunique()}.",
        f"Smallest-containing catchments with at least one well: {matched['contained_gauge_id'].nunique()}.",
        f"Median number of containing nested catchments per well: {matched['n_containing_catchments'].median():.1f}.",
        "",
        "Association summary:",
    ]
    for _, row in assoc.iterrows():
        lines.append(
            f"- {row['contract']}: n={int(row['n'])}, catchments={int(row['n_catchments'])}, "
            f"rho={row['spearman_rho_log_gw_tau_vs_log_discharge_tau']:.3f}, p={row['p_value']:.3g}."
        )
    lines.extend(
        [
            "",
            "Boundary: this is an official geometry-derived containment check,",
            "not hydrogeologic proof that each well controls the discharge response.",
            "Nested catchments are handled by a smallest-containing-catchment primary contract",
            "and a retained all-containment table.",
        ]
    )
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    wells, boundaries, gauges = load_inputs()
    all_map, primary, catchments = containment_tables(wells, boundaries, gauges)
    assoc = association_summary(primary, catchments)
    all_map.to_csv(TABLES / f"{OUT}_all_containment.csv", index=False)
    primary.to_csv(TABLES / f"{OUT}_smallest_containing_wells.csv", index=False)
    catchments.to_csv(TABLES / f"{OUT}_catchment_summary.csv", index=False)
    assoc.to_csv(TABLES / f"{OUT}_association_summary.csv", index=False)
    plot(boundaries, primary, catchments, assoc)
    write_note(primary, catchments, assoc)


if __name__ == "__main__":
    main()
