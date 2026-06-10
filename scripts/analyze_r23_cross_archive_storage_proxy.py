"""R23 cross-archive hydrogeologic storage-proxy validation.

This analysis separates streamflow-derived support (BFI) from independent
catchment-property support. It compares CAMELS-US and CAMELS-BR associations
between log10(tau_acf) and hydrogeologic/soil proxies, using simple Spearman
correlations and partial rank correlations with core size/climate controls.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import run_r23_camels_br_third_archive as br

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r23_cross_archive_storage_proxy_validation"


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return pd.Series(q, index=pvals.index)
    order = np.argsort(p[mask])
    ranked = p[mask][order]
    m = ranked.size
    vals = ranked * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.clip(vals, 0.0, 1.0)
    q_masked = np.empty_like(ranked)
    q_masked[order] = vals
    q[mask] = q_masked
    return pd.Series(q, index=pvals.index)


def rank_residual(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    data = pd.concat([y.rename("y"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return pd.Series(dtype=float)
    yr = data["y"].rank().to_numpy(dtype=float)
    xr = data.drop(columns=["y"]).rank().to_numpy(dtype=float)
    xr = np.column_stack([np.ones(len(data)), xr])
    beta, *_ = np.linalg.lstsq(xr, yr, rcond=None)
    return pd.Series(yr - xr @ beta, index=data.index)


def association_rows(data: pd.DataFrame, archive: str, attr_map: dict[str, str], controls: list[str]) -> pd.DataFrame:
    work = data.copy().replace([np.inf, -np.inf], np.nan)
    work["log_tau"] = np.log10(pd.to_numeric(work["tau_acf_days"], errors="coerce"))
    rows = []
    for label, col in attr_map.items():
        if col not in work.columns:
            continue
        y = work["log_tau"]
        x = pd.to_numeric(work[col], errors="coerce")
        pair = pd.concat([y.rename("log_tau"), x.rename("x")], axis=1).dropna()
        if len(pair) < 50:
            continue
        rho, pval = stats.spearmanr(pair["x"], pair["log_tau"])
        control_cols = [c for c in controls if c != col and c in work.columns]
        common = pd.concat([y.rename("log_tau"), x.rename("x"), work[control_cols]], axis=1).dropna()
        if len(common) >= 50 and control_cols:
            yres = rank_residual(common["log_tau"], common[control_cols])
            xres = rank_residual(common["x"], common[control_cols])
            aligned = pd.concat([yres.rename("y"), xres.rename("x")], axis=1).dropna()
            prho, pp = stats.spearmanr(aligned["x"], aligned["y"]) if len(aligned) >= 50 else (np.nan, np.nan)
        else:
            prho, pp = np.nan, np.nan
        rows.append(
            {
                "archive": archive,
                "proxy": label,
                "column": col,
                "n": int(len(pair)),
                "spearman_rho_logtau": float(rho),
                "p_value": float(pval),
                "partial_rank_rho_logtau": float(prho),
                "partial_p_value": float(pp),
                "proxy_type": "streamflow-derived" if "BFI" in label else "independent hydrogeologic/soil proxy",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh_within_archive"] = out.groupby("archive")["p_value"].transform(bh_qvalues)
        out["partial_q_value_bh_within_archive"] = out.groupby("archive")["partial_p_value"].transform(bh_qvalues)
    return out


def load_us() -> pd.DataFrame:
    us = pd.read_csv(TABLES / "camels_673_attributed_diagnostics.csv", dtype={"gauge_id": str})
    us["log_area"] = np.log10(pd.to_numeric(us["drainage_area_km2"], errors="coerce"))
    return us


def load_br() -> pd.DataFrame:
    summary = pd.read_csv(TABLES / "r23_camels_br_third_archive_gauge_summary.csv", dtype={"gauge_id": str})
    attrs = br.load_attributes()
    return summary[["gauge_id", "tau_acf_days"]].merge(attrs, on="gauge_id", how="left")


def plot(table: pd.DataFrame, us: pd.DataFrame, brdata: pd.DataFrame) -> None:
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
    fig = plt.figure(figsize=(7.35, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    shared = table[table["proxy"].isin(["BFI", "geol permeability", "geol porosity", "carbonate rocks"])].copy()
    shared["label"] = shared["archive"] + " / " + shared["proxy"]
    shared = shared.sort_values("partial_rank_rho_logtau")
    y = np.arange(len(shared))
    ax0.axvline(0, color="#667085", lw=0.9)
    colors = ["#C4513F" if t == "streamflow-derived" else "#2F6FA8" for t in shared["proxy_type"]]
    ax0.barh(y, shared["partial_rank_rho_logtau"], color=colors, alpha=0.86)
    ax0.set_yticks(y, shared["label"])
    ax0.set_xlabel("partial rank rho with log tau")
    ax0.set_title("a  Shared storage proxies across US and Brazil", loc="left", fontsize=8.8, fontweight="bold")
    ax0.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    br_only = table[(table["archive"] == "CAMELS-BR") & table["proxy"].isin(["water-table depth", "bedrock depth", "sand fraction", "clay fraction"])].copy()
    br_only = br_only.sort_values("partial_rank_rho_logtau")
    y = np.arange(len(br_only))
    ax1.axvline(0, color="#667085", lw=0.9)
    ax1.barh(y, br_only["partial_rank_rho_logtau"], color=["#C4513F" if v > 0 else "#2F6FA8" for v in br_only["partial_rank_rho_logtau"]], alpha=0.86)
    ax1.set_yticks(y, br_only["proxy"])
    ax1.set_xlabel("partial rank rho with log tau")
    ax1.set_title("b  Brazil soil-depth proxies add independent support", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    usx = pd.to_numeric(us["geol_permeability_x"], errors="coerce")
    usy = np.log10(pd.to_numeric(us["tau_acf_days"], errors="coerce"))
    ax2.scatter(usx, usy, s=12, color="#2F6FA8", alpha=0.58, linewidths=0)
    rho, p = stats.spearmanr(usx.dropna(), usy.loc[usx.dropna().index])
    ax2.set_xlabel("CAMELS-US geol permeability")
    ax2.set_ylabel(r"$\log_{10}\tau_{\mathrm{acf}}$ [d]")
    ax2.set_title(f"c  US permeability proxy (rho={rho:.2f})", loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, color="#E4E9EF", linewidth=0.55)

    bx = pd.to_numeric(brdata["water_table_depth"], errors="coerce")
    by = np.log10(pd.to_numeric(brdata["tau_acf_days"], errors="coerce"))
    ax3.scatter(bx, by, s=12, color="#59A14F", alpha=0.58, linewidths=0)
    valid = pd.concat([bx.rename("x"), by.rename("y")], axis=1).dropna()
    rho, p = stats.spearmanr(valid["x"], valid["y"])
    ax3.set_xlabel("CAMELS-BR water-table depth")
    ax3.set_ylabel(r"$\log_{10}\tau_{\mathrm{acf}}$ [d]")
    ax3.set_title(f"d  Brazil water-table proxy (rho={rho:.2f})", loc="left", fontsize=8.8, fontweight="bold")
    ax3.grid(True, color="#E4E9EF", linewidth=0.55)

    for suffix, kwargs in [(".png", {"dpi": 300}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    us = load_us()
    brdata = load_br()
    us_table = association_rows(
        us,
        "CAMELS-US",
        {
            "BFI": "baseflow_index_x",
            "geol porosity": "geol_porostiy_x",
            "geol permeability": "geol_permeability_x",
            "carbonate rocks": "carbonate_rocks_frac_x",
        },
        ["log_area", "aridity", "frac_snow", "p_seasonality"],
    )
    brdata["log_area"] = np.log10(pd.to_numeric(brdata["area"], errors="coerce"))
    br_table = association_rows(
        brdata,
        "CAMELS-BR",
        {
            "BFI": "baseflow_index",
            "geol porosity": "geol_porosity",
            "geol permeability": "geol_permeability",
            "carbonate rocks": "carb_rocks_perc",
            "water-table depth": "water_table_depth",
            "bedrock depth": "bedrock_depth",
            "sand fraction": "sand_perc",
            "clay fraction": "clay_perc",
        },
        ["log_area", "aridity", "p_seasonality"],
    )
    table = pd.concat([us_table, br_table], ignore_index=True)
    table.to_csv(TABLES / f"{OUT}.csv", index=False)
    plot(table, us, brdata)
    lines = [
        "# R23 Cross-Archive Storage Proxy Validation",
        "",
        "This table separates streamflow-derived BFI support from independent",
        "hydrogeologic/soil proxy support in CAMELS-US and CAMELS-BR.",
        "",
        table.to_markdown(index=False, floatfmt=".4g"),
        "",
        "Safe interpretation: independent catchment-property proxies show",
        "directional storage compatibility, especially permeability/porosity and",
        "soil/water-table controls in CAMELS-BR. This is still observational and",
        "does not replace direct tracer or groundwater-level validation.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
