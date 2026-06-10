"""R28 tracer and stream-chemistry boundary screen.

This round addresses the direct-tracer gap without overstating it.  It adds
two public evidence layers:

1. CAMELS-Chem, a CAMELS-US-linked stream-chemistry archive.  This is a
   large-sample passive-tracer/solute compatibility test, not tracer transit
   time causality.
2. Plynlimon isotope/chloride data from Knapp et al. (2019).  This is a direct
   tracer case-study anchor, but it is not linked to the large CAMELS
   De-reduction archive.

The script writes source tables, a manuscript note and a supplementary figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r28_tracer_compatibility_validation"

CAMELS_CHEM = DATA / "camels_chem" / "Camels_chem_1980_2018.csv"
CAMELS_CHEM_NAMES = DATA / "camels_chem" / "Gauge_and_region_names.csv"
CAMELS_US_SUMMARY = TABLES / "camels_673_n30_parallel_summary.csv"
PLYN_7H = DATA / "plynlimon_tracer" / "supplement" / "Plynlimon_isotopes_and_chloride_7hourly_2007_2009.txt"
PLYN_WEEKLY = DATA / "plynlimon_tracer" / "supplement" / "Plynlimon_isotopes_and_chloride_weekly_2004_2009.txt"

CONSERVATIVE_WEATHERING = ["cl", "na", "k", "mg", "si", "so4", "ca", "hco3", "alk"]
ALL_SOLUTES = CONSERVATIVE_WEATHERING + ["doc", "toc", "no3", "tdn"]
PLYN_VARS = [
    ("delta_18O", "delta18O"),
    ("delta_2H", "delta2H"),
    ("Cl mg/l", "chloride"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-span-years", type=float, default=5.0)
    parser.add_argument("--min-constituents", type=int, default=3)
    return parser.parse_args()


def zfill_gauge(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str).str.zfill(8)


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return pd.Series(q, index=pvals.index)
    order = np.argsort(p[ok])
    vals = p[ok][order]
    ranks = np.arange(1, len(vals) + 1)
    adjusted = vals * len(vals) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    ok_idx = np.where(ok)[0][order]
    q[ok_idx] = adjusted
    return pd.Series(q, index=pvals.index)


def rank_residual(y: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    frame = pd.concat([y, controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 10:
        return np.array([])
    ranked = frame.rank(method="average")
    yr = ranked.iloc[:, 0].to_numpy(dtype=float)
    x = ranked.iloc[:, 1:].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, yr, rcond=None)
    return yr - x @ beta


def partial_spearman(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> tuple[float, float, int]:
    frame = pd.concat([x.rename("x"), y.rename("y"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 10:
        return np.nan, np.nan, int(len(frame))
    rx = rank_residual(frame["x"], frame.iloc[:, 2:])
    ry = rank_residual(frame["y"], frame.iloc[:, 2:])
    if len(rx) != len(ry) or len(rx) < 10:
        return np.nan, np.nan, int(len(frame))
    rho, p = stats.spearmanr(rx, ry)
    return float(rho), float(p), int(len(frame))


def load_camels_chem() -> tuple[pd.DataFrame, pd.DataFrame]:
    chem = pd.read_csv(CAMELS_CHEM, low_memory=False)
    chem["gauge_id8"] = zfill_gauge(chem["gauge_id"])
    chem["sample_date"] = pd.to_datetime(chem["sample_start_dt"], errors="coerce")
    for col in ALL_SOLUTES + ["q_inst", "q_15", "q_derived", "q_daily"]:
        if col in chem.columns:
            chem[col] = pd.to_numeric(chem[col], errors="coerce")
    q = chem.get("q_inst", pd.Series(np.nan, index=chem.index)).where(chem.get("q_inst", pd.Series(np.nan, index=chem.index)) > 0)
    for col in ["q_15", "q_derived", "q_daily"]:
        if col in chem.columns:
            q = q.fillna(chem[col].where(chem[col] > 0))
    chem["q_use"] = q
    names = pd.read_csv(CAMELS_CHEM_NAMES)
    names["gauge_id8"] = zfill_gauge(names["gauge_id"])
    return chem, names


def camels_chem_pair_metrics(args: argparse.Namespace) -> pd.DataFrame:
    chem, names = load_camels_chem()
    rows: list[dict] = []
    for gauge_id, group in chem.groupby("gauge_id8", dropna=True):
        for constituent in ALL_SOLUTES:
            if constituent not in group.columns:
                continue
            data = group[["sample_date", "q_use", constituent]].dropna().copy()
            data = data[(data["q_use"] > 0) & (data[constituent] > 0)]
            if len(data) < args.min_samples:
                continue
            span_years = (data["sample_date"].max() - data["sample_date"].min()).days / 365.25
            if span_years < args.min_span_years:
                continue
            logq = np.log(data["q_use"].to_numpy(dtype=float))
            logc = np.log(data[constituent].to_numpy(dtype=float))
            if not (np.isfinite(logq).all() and np.isfinite(logc).all()):
                continue
            sd_logq = float(np.std(logq, ddof=1))
            sd_logc = float(np.std(logc, ddof=1))
            if sd_logq <= 0 or sd_logc <= 0:
                continue
            slope, intercept, r_value, p_value, stderr = stats.linregress(logq, logc)
            rho, rho_p = stats.spearmanr(logq, logc)
            rows.append(
                {
                    "gauge_id": gauge_id,
                    "constituent": constituent,
                    "constituent_group": "conservative_weathering"
                    if constituent in CONSERVATIVE_WEATHERING
                    else "reactive_organic_nutrient",
                    "n_samples": int(len(data)),
                    "span_years": float(span_years),
                    "first_date": data["sample_date"].min().date().isoformat(),
                    "last_date": data["sample_date"].max().date().isoformat(),
                    "sd_log_concentration": sd_logc,
                    "sd_log_discharge": sd_logq,
                    "chem_variability_ratio": float(sd_logc / sd_logq),
                    "inverse_chem_damping": float(sd_logc / sd_logq),
                    "cq_slope": float(slope),
                    "abs_cq_slope": float(abs(slope)),
                    "cq_r2": float(r_value**2),
                    "cq_spearman": float(rho),
                    "cq_spearman_p": float(rho_p),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.merge(names[["gauge_id8", "gauge_name", "region_name"]], left_on="gauge_id", right_on="gauge_id8", how="left")
    pairs = pairs.drop(columns=["gauge_id8"])
    return pairs


def camels_chem_gauge_metrics(pairs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    summary = pd.read_csv(CAMELS_US_SUMMARY)
    summary["gauge_id"] = zfill_gauge(summary["gauge_id"])
    summary["log_area"] = np.log10(summary["drainage_area_km2"])
    rows: list[pd.DataFrame] = []
    for group_name, constituents in {
        "conservative_weathering": CONSERVATIVE_WEATHERING,
        "all_solutes": ALL_SOLUTES,
    }.items():
        work = pairs[pairs["constituent"].isin(constituents)].copy()
        if work.empty:
            continue
        agg = (
            work.groupby("gauge_id")
            .agg(
                n_constituents=("constituent", "nunique"),
                median_samples=("n_samples", "median"),
                max_span_years=("span_years", "max"),
                median_inverse_chem_damping=("inverse_chem_damping", "median"),
                median_abs_cq_slope=("abs_cq_slope", "median"),
                median_cq_spearman=("cq_spearman", "median"),
                median_cq_r2=("cq_r2", "median"),
            )
            .reset_index()
        )
        agg["aggregation"] = group_name
        joined = agg.merge(
            summary[
                [
                    "gauge_id",
                    "tau_acf_days",
                    "drainage_area_km2",
                    "log_area",
                    "p_mean",
                    "pet_mean",
                    "aridity",
                    "gauge_name",
                ]
            ],
            on="gauge_id",
            how="inner",
            suffixes=("_chem", "_camels"),
        )
        joined = joined[joined["n_constituents"] >= args.min_constituents].copy()
        rows.append(joined)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def camels_chem_associations(gauges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for aggregation, group in gauges.groupby("aggregation"):
        controls = group[["log_area", "p_mean", "aridity"]]
        for metric, label in [
            ("median_inverse_chem_damping", "median sd(log C) / sd(log Q)"),
            ("median_abs_cq_slope", "median |log C-log Q slope|"),
            ("median_cq_spearman", "median Spearman(C,Q)"),
            ("median_cq_r2", "median C-Q R2"),
        ]:
            valid = group[["tau_acf_days", metric, "log_area", "p_mean", "aridity"]].dropna()
            if len(valid) < 10:
                continue
            rho, p = stats.spearmanr(valid["tau_acf_days"], valid[metric])
            prho, pp, n_partial = partial_spearman(valid["tau_acf_days"], valid[metric], valid[["log_area", "p_mean", "aridity"]])
            rows.append(
                {
                    "aggregation": aggregation,
                    "metric": metric,
                    "metric_label": label,
                    "n_gauges": int(len(valid)),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "partial_controls": "rank(log area, mean precipitation, aridity)",
                    "partial_rho": prho,
                    "partial_p": pp,
                    "partial_n": n_partial,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["partial_q_bh"] = bh_qvalues(out["partial_p"])
    return out


def read_plynlimon(path: Path, cadence: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    data["datetime"] = pd.to_datetime(data["Date_Time yyyy.mm.dd HH:MM"], format="%Y.%m.%d %H:%M", errors="coerce")
    data["cadence"] = cadence
    for raw, _ in PLYN_VARS:
        data[raw] = pd.to_numeric(data[raw], errors="coerce")
    for col in ["water flux (mm/hr)", "Stream flow (Cumecs)", "Rainfall (mm)"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def plynlimon_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.concat([read_plynlimon(PLYN_7H, "7-hourly"), read_plynlimon(PLYN_WEEKLY, "weekly")], ignore_index=True)
    rows: list[dict] = []
    flow_rows: list[dict] = []
    for cadence, group in data.groupby("cadence"):
        inputs = group[group["Type"].str.lower() == "input"]
        streams = group[group["Type"].str.lower() == "stream"]
        for raw, label in PLYN_VARS:
            input_values = inputs[raw].dropna().to_numpy(dtype=float)
            if len(input_values) < 10:
                continue
            input_sd = float(np.std(input_values, ddof=1))
            input_iqr = float(np.quantile(input_values, 0.75) - np.quantile(input_values, 0.25))
            for site, sg in streams.groupby("Site"):
                stream_values = sg[raw].dropna().to_numpy(dtype=float)
                if len(stream_values) < 10:
                    continue
                stream_sd = float(np.std(stream_values, ddof=1))
                stream_iqr = float(np.quantile(stream_values, 0.75) - np.quantile(stream_values, 0.25))
                rows.append(
                    {
                        "dataset": "Plynlimon isotope/chloride",
                        "cadence": cadence,
                        "stream_site": site,
                        "variable": label,
                        "input_site": ",".join(sorted(inputs["Site"].dropna().astype(str).unique())),
                        "n_input": int(len(input_values)),
                        "n_stream": int(len(stream_values)),
                        "input_sd": input_sd,
                        "stream_sd": stream_sd,
                        "stream_input_sd_ratio": float(stream_sd / input_sd) if input_sd > 0 else np.nan,
                        "input_iqr": input_iqr,
                        "stream_iqr": stream_iqr,
                        "stream_input_iqr_ratio": float(stream_iqr / input_iqr) if input_iqr > 0 else np.nan,
                    }
                )
        for site, sg in streams.groupby("Site"):
            q = sg[["datetime", "Stream flow (Cumecs)"]].dropna()
            q = q[q["Stream flow (Cumecs)"] > 0]
            if len(q) < 30:
                continue
            logq = np.log(q["Stream flow (Cumecs)"].to_numpy(dtype=float))
            # Irregular case-study memory proxy: Spearman lag-1 over observed sampling sequence.
            lag_rho, lag_p = stats.spearmanr(logq[:-1], logq[1:]) if len(logq) > 2 else (np.nan, np.nan)
            flow_rows.append(
                {
                    "cadence": cadence,
                    "stream_site": site,
                    "n_flow_samples": int(len(q)),
                    "span_years": float((q["datetime"].max() - q["datetime"].min()).days / 365.25),
                    "median_flow_m3s": float(np.median(q["Stream flow (Cumecs)"])),
                    "sd_log_flow": float(np.std(logq, ddof=1)),
                    "sampling_sequence_lag1_spearman": float(lag_rho),
                    "lag1_p": float(lag_p),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(flow_rows)


def evidence_ladder(associations: pd.DataFrame, plyn: pd.DataFrame) -> pd.DataFrame:
    all_row = associations[
        (associations["aggregation"] == "all_solutes") & (associations["metric"] == "median_inverse_chem_damping")
    ]
    cons_row = associations[
        (associations["aggregation"] == "conservative_weathering")
        & (associations["metric"] == "median_inverse_chem_damping")
    ]
    all_summary = ""
    if not all_row.empty:
        r = all_row.iloc[0]
        all_summary = f"partial rho={r['partial_rho']:.3f}, q={r['partial_q_bh']:.2g}, n={int(r['partial_n'])}"
    cons_summary = ""
    if not cons_row.empty:
        r = cons_row.iloc[0]
        cons_summary = f"partial rho={r['partial_rho']:.3f}, q={r['partial_q_bh']:.2g}, n={int(r['partial_n'])}"
    isotope = plyn[plyn["variable"].isin(["delta18O", "delta2H"])]
    isotope_summary = ""
    if not isotope.empty:
        isotope_summary = (
            f"median stream/input SD ratio={isotope['stream_input_sd_ratio'].median():.3f} "
            f"across {isotope['stream_site'].nunique()} stream sites"
        )
    return pd.DataFrame(
        [
            {
                "evidence_level": "CAMELS-Chem all-solute passive-tracer screen",
                "status": "mixed_boundary",
                "score": 1.8,
                "claim_boundary": "large-sample stream chemistry is CAMELS-linked but not transit-time causality",
                "support": all_summary,
            },
            {
                "evidence_level": "CAMELS-Chem conservative/weathering solutes",
                "status": "counter_to_simple_damping",
                "score": 1.4,
                "claim_boundary": "positive inverse-damping association is not the expected simple storage-damping proof",
                "support": cons_summary,
            },
            {
                "evidence_level": "Plynlimon stable-isotope/chloride case study",
                "status": "direct_tracer_anchor",
                "score": 2.4,
                "claim_boundary": "direct tracer damping exists in a classic catchment data set but is not linked to the CAMELS De archive",
                "support": isotope_summary,
            },
            {
                "evidence_level": "CAMELS-CH-Chem stable-isotope expansion",
                "status": "candidate_deferred",
                "score": 1.0,
                "claim_boundary": "download candidate identified; full archive not yet integrated",
                "support": "Zenodo README verified; 185 MB archive download timed out and requires resume",
            },
            {
                "evidence_level": "direct tracer causality for De coordinate",
                "status": "not_closed",
                "score": 0.8,
                "claim_boundary": "not claimed in the manuscript",
                "support": "no large-sample linked tracer transit-time data have been integrated",
            },
        ]
    )


def make_figure(
    pairs: pd.DataFrame,
    gauges: pd.DataFrame,
    associations: pd.DataFrame,
    plyn: pd.DataFrame,
    ladder: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.8, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.2], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # a. Provenance counts.
    prov_labels = ["rows", "C-Q\npairs", "matched\ngauges", "tracer\nsites"]
    prov_vals = [pd.read_csv(CAMELS_CHEM, usecols=["gauge_id"]).shape[0], len(pairs), gauges["gauge_id"].nunique(), plyn["stream_site"].nunique()]
    colors = ["#3b6ea8", "#63a6a0", "#e0a63b", "#9656a1"]
    ax_a.bar(range(len(prov_vals)), prov_vals, color=colors, edgecolor="#222", linewidth=0.6)
    ax_a.set_yscale("log")
    ax_a.set_xticks(range(len(prov_vals)), prov_labels)
    ax_a.set_ylabel("count (log scale)")
    ax_a.set_title("a  added tracer-facing data")
    for x, v in enumerate(prov_vals):
        ax_a.text(x, v * 1.18, f"{v:,}", ha="center", va="bottom", fontsize=7.2)
    ax_a.spines[["top", "right"]].set_visible(False)

    # b. Constituent coverage.
    cov = (
        pairs.groupby("constituent")
        .agg(n_gauges=("gauge_id", "nunique"), median_samples=("n_samples", "median"))
        .reset_index()
        .sort_values("n_gauges")
    )
    ax_b.barh(cov["constituent"], cov["n_gauges"], color="#5d8cc0", edgecolor="#1f3657", linewidth=0.5)
    for y, (_, row) in enumerate(cov.iterrows()):
        ax_b.text(row["n_gauges"] + 1, y, f"n~{row['median_samples']:.0f}", va="center", fontsize=6.8)
    ax_b.set_xlabel("gauges with usable C-Q metric")
    ax_b.set_title("b  CAMELS-Chem coverage")
    ax_b.spines[["top", "right"]].set_visible(False)

    # c. Tau vs all-solute inverse damping.
    all_g = gauges[gauges["aggregation"] == "all_solutes"].copy()
    sc = ax_c.scatter(
        all_g["tau_acf_days"],
        all_g["median_inverse_chem_damping"],
        c=all_g["aridity"],
        s=24 + 4 * all_g["n_constituents"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.88,
    )
    ax_c.set_xscale("log")
    ax_c.set_xlabel("streamflow memory tau (days)")
    ax_c.set_ylabel("median sd(log C) / sd(log Q)")
    ax_c.set_title("c  passive-tracer screen is mixed")
    row = associations[(associations["aggregation"] == "all_solutes") & (associations["metric"] == "median_inverse_chem_damping")]
    if not row.empty:
        r = row.iloc[0]
        ax_c.text(
            0.03,
            0.97,
            f"partial rho={r['partial_rho']:.2f}\nq={r['partial_q_bh']:.1e}",
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbb", alpha=0.92),
        )
    cb = fig.colorbar(sc, ax=ax_c, shrink=0.85)
    cb.set_label("aridity")
    ax_c.spines[["top", "right"]].set_visible(False)

    # d. Conservative solute association table as signed effect bars.
    eff = associations[associations["metric"].isin(["median_inverse_chem_damping", "median_abs_cq_slope"])].copy()
    eff["label"] = eff["aggregation"].str.replace("_", "\n") + "\n" + eff["metric"].map(
        {
            "median_inverse_chem_damping": "variability ratio",
            "median_abs_cq_slope": "|C-Q slope|",
        }
    )
    eff = eff.sort_values(["aggregation", "metric"])
    ax_d.axvline(0, color="#444", linewidth=0.8)
    ax_d.barh(eff["label"], eff["partial_rho"], color=np.where(eff["partial_q_bh"] < 0.05, "#b94743", "#9aa6b2"))
    for y, (_, row) in enumerate(eff.iterrows()):
        ax_d.text(
            row["partial_rho"] + (0.015 if row["partial_rho"] >= 0 else -0.015),
            y,
            f"q={row['partial_q_bh']:.2g}",
            ha="left" if row["partial_rho"] >= 0 else "right",
            va="center",
            fontsize=6.8,
        )
    ax_d.set_xlabel("partial rank rho with tau")
    ax_d.set_title("d  no clean conservative-solute proof")
    ax_d.spines[["top", "right"]].set_visible(False)

    # e. Plynlimon tracer damping. Keep this readable because it is a small
    # direct tracer case-study anchor rather than a large-sample De gate.
    iso = plyn.copy().sort_values(["variable", "cadence", "stream_site"])
    var_order = ["delta18O", "delta2H", "chloride"]
    xbase = {v: i for i, v in enumerate(var_order)}
    marker = {"7-hourly": "o", "weekly": "s"}
    colors_e = {"UHF": "#377eb8", "LHF": "#4daf4a", "TAN": "#984ea3"}
    ax_e.axhline(1, color="#444", linestyle="--", linewidth=0.8)
    for _, row in iso.iterrows():
        jitter = {"UHF": -0.13, "LHF": 0.0, "TAN": 0.13}.get(row["stream_site"], 0.0)
        ax_e.scatter(
            xbase[row["variable"]] + jitter,
            row["stream_input_sd_ratio"],
            s=62,
            marker=marker.get(row["cadence"], "o"),
            color=colors_e.get(row["stream_site"], "#777"),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    med = iso.groupby("variable")["stream_input_sd_ratio"].median()
    ax_e.set_xticks(range(len(var_order)), [r"$\delta^{18}$O", r"$\delta^2$H", "chloride"])
    ax_e.set_ylim(0, max(1.05, float(iso["stream_input_sd_ratio"].max()) * 1.25))
    ax_e.set_ylabel("stream/input SD ratio")
    ax_e.set_title("e  direct tracer case-study damping")
    ax_e.text(0.03, 0.88, "stable isotope median=0.127", transform=ax_e.transAxes, ha="left", va="top", fontsize=7.3)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", label="7-hourly UHF", markerfacecolor=colors_e["UHF"], markeredgecolor="white", markersize=6),
        plt.Line2D([0], [0], marker="s", color="none", label="weekly LHF", markerfacecolor=colors_e["LHF"], markeredgecolor="white", markersize=6),
        plt.Line2D([0], [0], marker="s", color="none", label="weekly TAN", markerfacecolor=colors_e["TAN"], markeredgecolor="white", markersize=6),
    ]
    ax_e.legend(handles=legend_handles, loc="center right", frameon=False, handletextpad=0.25)
    ax_e.spines[["top", "right"]].set_visible(False)

    # f. Evidence ladder.
    ax_f.axis("off")
    y = 0.96
    status_color = {
        "direct_tracer_anchor": "#2f855a",
        "mixed_boundary": "#b7791f",
        "counter_to_simple_damping": "#c05621",
        "candidate_deferred": "#718096",
        "not_closed": "#c53030",
    }
    ax_f.text(0.0, y, "f  claim decision", fontweight="bold", fontsize=10, transform=ax_f.transAxes)
    y -= 0.1
    short = {
        "CAMELS-Chem all-solute passive-tracer screen": "CAMELS-linked chemistry: mixed boundary",
        "CAMELS-Chem conservative/weathering solutes": "Conservative solutes: not simple proof",
        "Plynlimon stable-isotope/chloride case study": "Plynlimon: direct tracer anchor",
        "CAMELS-CH-Chem stable-isotope expansion": "CAMELS-CH-Chem: candidate only",
        "direct tracer causality for De coordinate": "Direct De tracer causality: not closed",
    }
    for _, row in ladder.iterrows():
        color = status_color.get(row["status"], "#555")
        ax_f.add_patch(
            plt.Rectangle((0.0, y - 0.035), 0.03, 0.03, transform=ax_f.transAxes, facecolor=color, edgecolor="none")
        )
        text = f"{short.get(row['evidence_level'], row['evidence_level'])}\n{row['claim_boundary']}"
        ax_f.text(0.045, y, text, transform=ax_f.transAxes, va="center", fontsize=7.15, wrap=True)
        y -= 0.17

    fig.suptitle("Tracer-facing boundary screen: mixed evidence, not closed causality", x=0.02, ha="left", fontsize=12)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"{OUT}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def write_note(
    pairs: pd.DataFrame,
    gauges: pd.DataFrame,
    associations: pd.DataFrame,
    plyn: pd.DataFrame,
    flow: pd.DataFrame,
    ladder: pd.DataFrame,
) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    top_assoc = associations.sort_values("partial_p").head(8)
    note = [
        "# R28 tracer-facing boundary screen",
        "",
        "## What was added",
        "",
        "- Downloaded CAMELS-Chem lightweight flat files from HydroShare resource DOI `10.4211/hs.841f5e85085c423f889ac809c1bed4ac`.",
        "- Parsed CAMELS-Chem stream chemistry and matched usable gauges to the existing CAMELS-US streamflow-memory summary.",
        "- Downloaded the Knapp et al. (2019) Plynlimon supplement from HESS and parsed 7-hourly and weekly isotope/chloride files.",
        "- Tested whether tracer-facing data close the direct tracer causality gap.",
        "",
        "## CAMELS-Chem headline results",
        "",
        f"- Usable gauge-constituent C-Q pairs: {len(pairs)} across {pairs['gauge_id'].nunique()} gauges.",
        f"- Matched CAMELS-US gauges after requiring at least three constituents: {gauges['gauge_id'].nunique()}.",
        "- Metric: `inverse_chem_damping = sd(log concentration) / sd(log discharge)`; lower values would indicate stronger concentration damping relative to discharge variability.",
        "",
        "| aggregation | metric | n | rho | p | partial rho | partial q |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in associations.iterrows():
        note.append(
            f"| {row['aggregation']} | {row['metric']} | {int(row['n_gauges'])} | "
            f"{row['spearman_rho']:.3f} | {row['spearman_p']:.2g} | "
            f"{row['partial_rho']:.3f} | {row['partial_q_bh']:.2g} |"
        )
    note.extend(
        [
            "",
            "## Plynlimon tracer anchor",
            "",
            f"- Plynlimon stream sites parsed: {plyn['stream_site'].nunique()} ({', '.join(sorted(plyn['stream_site'].unique()))}).",
            f"- Median stable-isotope stream/input SD ratio: {plyn[plyn['variable'].isin(['delta18O', 'delta2H'])]['stream_input_sd_ratio'].median():.3f}.",
            "- This is direct tracer damping evidence, but it is a small case-study anchor and is not linked to the large CAMELS De-reduction archive.",
            "",
            "## Claim decision",
            "",
            "| evidence_level | status | score | claim_boundary | support |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for _, row in ladder.iterrows():
        note.append(
            f"| {row['evidence_level']} | {row['status']} | {row['score']} | "
            f"{row['claim_boundary']} | {row['support']} |"
        )
    note.extend(
        [
            "",
            "## Boundary",
            "",
            "- R28 upgrades the manuscript from `no tracer data integrated` to `tracer-facing boundary tested with CAMELS-Chem and direct Plynlimon tracer anchor`.",
            "- It still does not close direct tracer causality for the De coordinate because the strongest large-sample CAMELS-linked evidence is stream chemistry rather than transit-time observation, and the direct isotope case is not linked to the main archive.",
            "- The CAMELS-Chem inverse-damping association is positive, so it is interpreted as a boundary against a simple monotone concentration-damping mechanism, not as direct storage proof.",
        ]
    )
    (NOTES / f"{OUT}.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    pairs = camels_chem_pair_metrics(args)
    gauges = camels_chem_gauge_metrics(pairs, args)
    associations = camels_chem_associations(gauges)
    plyn, flow = plynlimon_metrics()
    ladder = evidence_ladder(associations, plyn)

    pairs.to_csv(TABLES / f"{OUT}_camels_chem_pair_metrics.csv", index=False)
    gauges.to_csv(TABLES / f"{OUT}_camels_chem_gauge_metrics.csv", index=False)
    associations.to_csv(TABLES / f"{OUT}_camels_chem_associations.csv", index=False)
    plyn.to_csv(TABLES / f"{OUT}_plynlimon_tracer_damping.csv", index=False)
    flow.to_csv(TABLES / f"{OUT}_plynlimon_flow_memory_proxy.csv", index=False)
    ladder.to_csv(TABLES / f"{OUT}_evidence_ladder.csv", index=False)
    make_figure(pairs, gauges, associations, plyn, ladder)
    write_note(pairs, gauges, associations, plyn, flow, ladder)

    print(f"wrote R28 tables/figures for {len(pairs)} CAMELS-Chem pairs and {len(plyn)} Plynlimon rows")


if __name__ == "__main__":
    main()
