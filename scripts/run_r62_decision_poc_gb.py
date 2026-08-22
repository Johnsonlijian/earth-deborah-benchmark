"""CAMELS-GB decision test for the released model-layer products.

The analysis design was frozen before the full run on 2026-08-21; its public
transcription and provenance boundary are recorded in
``DECISION_TEST_PROTOCOL_PUBLIC_RECORD.md``. The test asks whether, on gauges
where best-by-NSE and best-by-beta(De)-distance disagree, the
beta(De)-preferred model has a lower held-out timescale error. Ablations use
raw-frequency distance, a global scalar slope, a constant-tau equivalence
check, and shuffled per-series tau values.

This script does not rerun the hydrological models. Reproducing the analysis
from scratch requires the dense R39 curve table and the official CAMELS-GB v2
daily files; neither raw third-party data nor the dense curve table is stored
in this Git repository.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import run_r21_hydrological_model_benchmark as r21  # noqa: E402
from run_camels_gb_replication import DAILY_DIR, gauge_id_from_path  # noqa: E402

SOURCE_DATA = ROOT / "source_data"
REPORT_TABLES = ROOT / "reports" / "tables"
TABLES = SOURCE_DATA
FIGURES = ROOT / "figures" / "essd"
OUT = "r62_decision_poc_gb"

INDEPENDENT_MODELS = [
    "rrmpg_gr4j",
    "rrmpg_hbvedu",
    "global_lstm",
    "seasonal_climatology",
    "seasonal_ar1_null",
]

MODEL_LABELS = {
    "rrmpg_gr4j": "GR4J",
    "rrmpg_hbvedu": "HBV-Edu",
    "global_lstm": "LSTM",
    "seasonal_climatology": "seasonal clim.",
    "seasonal_ar1_null": "seasonal AR(1)",
}

SCALAR_DE_LO, SCALAR_DE_HI = 0.5, 2.0


def norm_gauge(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def find_table(name: str) -> Path:
    """Resolve a released table first, then a regenerated dense output."""
    for directory in (SOURCE_DATA, REPORT_TABLES):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {name}. Check source_data or regenerate it under reports/tables."
    )


def load_metrics() -> pd.DataFrame:
    m = pd.read_csv(find_table("r39_open_model_intercomparison_metrics.csv"))
    m["gauge_id"] = m["gauge_id"].map(norm_gauge)
    return m


def load_curves() -> pd.DataFrame:
    c = pd.read_csv(find_table("r39_open_model_intercomparison_curves.csv"))
    c["gauge_id"] = c["gauge_id"].map(norm_gauge)
    return c


def best_by(group: pd.DataFrame, value_col: str, higher_better: bool) -> pd.Series:
    g = group.dropna(subset=["eval_nse", value_col, "abs_log10_tau_error"])
    if len(g) < 4:
        return None
    if higher_better:
        nse_best = g.loc[g["eval_nse"].idxmax()]
    else:
        nse_best = g.loc[g[value_col].idxmin()]
    return nse_best


def paired_delta_stats(deltas: np.ndarray, seed: int = 62, n_boot: int = 5000) -> dict:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    out: dict = {"n": int(d.size)}
    if d.size < 5:
        out.update({"median": np.nan, "mean": np.nan, "wilcoxon_p": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan, "frac_lower": np.nan})
        return out
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(n_boot)])
    try:
        wp = stats.wilcoxon(d)[1]
    except ValueError:
        wp = np.nan
    out.update(
        {
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "wilcoxon_p": float(wp),
            "ci_lo": float(np.percentile(meds, 2.5)),
            "ci_hi": float(np.percentile(meds, 97.5)),
            "frac_lower": float(np.mean(d < 0)),
        }
    )
    return out


def obs_curve_for_gauge(gauge_id: str) -> tuple[float, pd.DataFrame] | None:
    files = sorted(DAILY_DIR.glob(f"*{gauge_id}*.csv"))
    if not files:
        return None
    try:
        data = r21.load_one(files[0])
    except Exception:
        return None
    q = data["discharge_spec"]
    train, eval_idx = r21.train_eval_split(q)
    if eval_idx.size < 365 * 6:
        return None
    try:
        tau_days, curve = r21.beta_curve(data.index[eval_idx], q.to_numpy(dtype=float)[eval_idx], "observed")
    except Exception:
        return None
    return float(tau_days), curve


def variant_distance(
    obs: pd.DataFrame,
    sim: pd.DataFrame,
    variant: str,
    tau_new: float | None,
    tau_sim: float | None = None,
) -> float:
    o = obs[["series", "frequency_cpd", "beta"]].copy()
    s = sim[["series", "frequency_cpd", "beta"]].copy()
    if variant == "scalar":
        o2 = o.copy()
        s2 = s.copy()
        o2["de"] = obs["de"].to_numpy()
        s2["de"] = sim["de"].to_numpy()
        ow = o2[(o2["de"] >= SCALAR_DE_LO) & (o2["de"] <= SCALAR_DE_HI) & o2["beta"].notna()]
        sw = s2[(s2["de"] >= SCALAR_DE_LO) & (s2["de"] <= SCALAR_DE_HI) & s2["beta"].notna()]
        if ow.empty or sw.empty:
            return np.nan
        return float(abs(np.mean(sw["beta"]) - np.mean(ow["beta"])))
    if variant == "de_const" and tau_new is not None and tau_new > 0:
        # Both series share one constant tau: mathematically raw-equivalent
        # under the R21 distance (equivalence check, not an ablation).
        o["de_v"] = tau_new * o["frequency_cpd"]
        s["de_v"] = tau_new * s["frequency_cpd"]
        return float(r21.curve_distance(o, s, "de_v"))
    if variant == "de_shuff" and tau_new is not None and tau_new > 0 and tau_sim is not None and tau_sim > 0:
        # Per-series mismatched taus: obs and sim each get a different random
        # gauge's measured tau (pre-registered v1.2 design).
        o["de_v"] = tau_new * o["frequency_cpd"]
        s["de_v"] = tau_sim * s["frequency_cpd"]
        return float(r21.curve_distance(o, s, "de_v"))
    raise ValueError(variant)


INDEPENDENT_RR_MODELS = [
    "rrmpg_gr4j",
    "rrmpg_hbvedu",
    "global_lstm",
]


def run_gauge_analysis(
    metrics: pd.DataFrame,
    gauge_list: list,
    obs_curves: dict,
    curves: pd.DataFrame,
    model_set: list,
    min_models: int,
    tau_c: float,
    seed: int,
) -> pd.DataFrame:
    """Per-gauge best-model selection per variant + tau-error decision delta."""
    work = metrics[metrics["model_type"].isin(model_set)].copy()
    rng = np.random.default_rng(seed)
    key_list = list(obs_curves.keys())
    tau_pool = np.array([obs_curves[k][0] for k in key_list], dtype=float)
    idx_obs = rng.integers(0, len(key_list), size=len(key_list))
    shuff_obs_tau = {key_list[i]: float(tau_pool[idx_obs[i]]) for i in range(len(key_list))}
    idx_sim = rng.integers(0, len(key_list), size=(len(key_list), len(model_set)))
    shuff_sim_tau = {
        key_list[i]: {mt: float(tau_pool[idx_sim[i, j]]) for j, mt in enumerate(model_set)}
        for i in range(len(key_list))
    }
    rows = []
    for gid in gauge_list:
        if gid not in obs_curves:
            continue
        g = work[work["gauge_id"] == gid].dropna(subset=["eval_nse", "beta_de_median_abs_distance", "abs_log10_tau_error"])
        if len(g) < min_models:
            continue
        obs = obs_curves[gid][1]
        nse_best = g.loc[g["eval_nse"].idxmax()]
        base = {
            "gauge_id": gid,
            "n_models": int(len(g)),
            "best_nse_model": nse_best["model_type"],
            "best_nse_eval_nse": float(nse_best["eval_nse"]),
            "tau_err_nse": float(nse_best["abs_log10_tau_error"]),
        }
        variants = {
            "beta_de": (g, "beta_de_median_abs_distance", None),
            "beta_raw": (g, "beta_raw_median_abs_distance", None),
            "scalar": None,
            "de_const": None,
            "de_shuff": None,
        }
        for variant in ["scalar", "de_const", "de_shuff"]:
            dists = {}
            for _, row in g.iterrows():
                sim = curves[(curves["gauge_id"] == gid) & (curves["model_type"] == row["model_type"])]
                if sim.empty:
                    dists[row["model_type"]] = np.nan
                    continue
                tau_new = {"scalar": None, "de_const": tau_c, "de_shuff": shuff_obs_tau[gid]}[variant]
                tau_sim = shuff_sim_tau[gid][row["model_type"]] if variant == "de_shuff" else None
                dists[row["model_type"]] = variant_distance(obs, sim, variant, tau_new, tau_sim)
            variants[variant] = (pd.Series(dists), None, None)
        for variant, payload in variants.items():
            if payload is None:
                continue
            if variant in ("beta_de", "beta_raw"):
                gv, col, _ = payload
                gg = g.dropna(subset=[col])
                if gg.empty or len(gg) < min_models:
                    continue
                best = gg.loc[gg[col].idxmin()]
                dist_best = float(best[col])
                model_best = best["model_type"]
                tau_err_best = float(best["abs_log10_tau_error"])
            else:
                sd, _, _ = payload
                sd = sd.dropna()
                if sd.empty or len(sd) < min_models:
                    continue
                model_best = sd.idxmin()
                dist_best = float(sd.min())
                best = g[g["model_type"] == model_best].iloc[0]
                tau_err_best = float(best["abs_log10_tau_error"])
            base[f"best_{variant}_model"] = model_best
            base[f"dist_{variant}_best"] = dist_best
            base[f"disagree_{variant}"] = bool(model_best != base["best_nse_model"])
            base[f"tau_err_{variant}_best"] = tau_err_best
            base[f"delta_tau_err_{variant}"] = tau_err_best - base["tau_err_nse"]
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_variants(gdf: pd.DataFrame, model_set_label: str, seed: int, n_boot: int) -> pd.DataFrame:
    summaries = []
    for variant in ["beta_de", "beta_raw", "scalar", "de_const", "de_shuff"]:
        col_d = f"delta_tau_err_{variant}"
        if col_d not in gdf.columns:
            continue
        for subset, mask in [("all", pd.Series(True, index=gdf.index)),
                             ("disagreement", gdf[f"disagree_{variant}"].astype(bool))]:
            dd = gdf.loc[mask, col_d].to_numpy(dtype=float)
            st = paired_delta_stats(dd, seed=seed, n_boot=n_boot)
            summaries.append(
                {
                    "model_set": model_set_label,
                    "variant": variant,
                    "subset": subset,
                    "n_gauges": st["n"],
                    "median_delta_tau_err": st["median"],
                    "mean_delta_tau_err": st["mean"],
                    "wilcoxon_p": st["wilcoxon_p"],
                    "bootstrap_ci_lo": st["ci_lo"],
                    "bootstrap_ci_hi": st["ci_hi"],
                    "frac_variant_lower_tau_err": st["frac_lower"],
                }
            )
    return pd.DataFrame(summaries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test gauge cap")
    ap.add_argument("--seed", type=int, default=62)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--reuse-obs-cache", action="store_true", help="reuse saved observed curves")
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    curves = load_curves()
    work_all = metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)].copy()
    gauge_list = sorted(work_all["gauge_id"].unique())
    if args.limit:
        gauge_list = gauge_list[: args.limit]

    # ---- observed curves (eval period), cached ----
    cache_path = TABLES / f"{OUT}_obs_curves.csv"
    obs_curves: dict = {}
    if args.reuse_obs_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        for gid, group in cached.groupby("gauge_id"):
            obs_curves[gid] = (float(group["obs_tau_days"].iloc[0]), group)
    else:
        for gid in gauge_list:
            res = obs_curve_for_gauge(gid)
            if res is not None:
                obs_curves[gid] = res
        if not args.limit:
            frames = []
            for gid, (t, df) in obs_curves.items():
                f = df.copy()
                f["gauge_id"] = gid
                f["obs_tau_days"] = t
                frames.append(f)
            pd.concat(frames, ignore_index=True).to_csv(cache_path, index=False)
    tau_c = float(np.median([t for t, _ in obs_curves.values()]))
    print(f"[r62] obs curves: {len(obs_curves)} gauges; constant tau = {tau_c:.2f} d")

    # ---- full 5-model set (pre-registered §1) and 3 independent models (v1.3) ----
    gdf5 = run_gauge_analysis(metrics, gauge_list, obs_curves, curves, INDEPENDENT_MODELS, 4, tau_c, args.seed)
    gdf5.to_csv(TABLES / f"{OUT}_gauge_variants.csv", index=False)
    print(f"[r62] full5 gauge table: {len(gdf5)} gauges")
    gdf3 = run_gauge_analysis(metrics, gauge_list, obs_curves, curves, INDEPENDENT_RR_MODELS, 3, tau_c, args.seed + 1)
    gdf3.to_csv(TABLES / f"{OUT}_ind3_gauge_variants.csv", index=False)
    print(f"[r62] ind3 gauge table: {len(gdf3)} gauges")

    s5 = summarize_variants(gdf5, "full5", args.seed, args.n_boot)
    s3 = summarize_variants(gdf3, "ind3", args.seed, args.n_boot)
    sdf = pd.concat([s5, s3], ignore_index=True)
    sdf.to_csv(TABLES / f"{OUT}_variant_summary.csv", index=False)
    print("[r62] variant summary:")
    print(sdf.to_string(index=False))

    # ---- threshold grid for the primary variant (both model sets) ----
    grid = []
    for label, gdf in [("full5", gdf5), ("ind3", gdf3)]:
        dd_prim = gdf["delta_tau_err_beta_de"].dropna()
        for thr in (0.50, 0.55, 0.60):
            frac = float(np.mean(dd_prim < 0))
            grid.append(
                {"model_set": label, "threshold_lower_frac": thr, "achieved": bool(frac >= thr), "frac_lower": frac}
            )
    pd.DataFrame(grid).to_csv(TABLES / f"{OUT}_threshold_grid.csv", index=False)

    # ---- sanity: compare with stored rank_disagreement (5-model rule) ----
    ranks = pd.read_csv(find_table("r39_open_model_intercomparison_rank_disagreement.csv"))
    ranks["gauge_id"] = ranks["gauge_id"].map(norm_gauge)
    check = gdf5[["gauge_id", "best_nse_model", "best_beta_de_model"]].merge(
        ranks[["gauge_id", "best_nse_model", "best_beta_model"]], on="gauge_id", suffixes=("_r62", "_r39")
    )
    agree_nse = float(np.mean(check["best_nse_model_r62"] == check["best_nse_model_r39"]))
    agree_beta = float(np.mean(check["best_beta_de_model"] == check["best_beta_model"]))
    print(f"[r62] sanity vs stored ranks: NSE-best agreement {agree_nse:.3f}, beta-best agreement {agree_beta:.3f}, n={len(check)}")
    eq = gdf5[["dist_beta_raw_best", "dist_de_const_best"]].dropna()
    if len(eq):
        print(f"[r62] de_const vs beta_raw max abs diff = {(eq['dist_beta_raw_best'] - eq['dist_de_const_best']).abs().max():.6f}")
    pd.DataFrame([{"n": len(check), "agree_best_nse": agree_nse, "agree_best_beta": agree_beta}]).to_csv(
        TABLES / f"{OUT}_rank_sanity.csv", index=False
    )

    # ---- figure ----
    make_figure(metrics, gdf5, sdf)


def make_figure(metrics: pd.DataFrame, gdf: pd.DataFrame, sdf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    m = metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)].dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
    ax = axes[0]
    for mt in INDEPENDENT_MODELS:
        sub = m[m["model_type"] == mt]
        ax.scatter(sub["eval_nse"], sub["beta_de_median_abs_distance"], s=6, alpha=0.45, label=MODEL_LABELS[mt])
    ax.set_xlabel("held-out NSE")
    ax.set_ylabel("beta(De) curve distance")
    ax.set_title("a) NSE vs spectral distance per model")
    ax.legend(fontsize=7, markerscale=2)

    ax = axes[1]
    order = ["beta_de", "beta_raw", "scalar", "de_const", "de_shuff"]
    labels = ["beta(De)", "raw f-axis", "global scalar", "const-tau (=raw)", "shuffled-tau"]
    ssel = sdf[(sdf["subset"] == "disagreement") & (sdf["model_set"] == "ind3")]
    sub = ssel.set_index("variant").reindex(order).dropna(subset=["median_delta_tau_err"])
    xs = np.arange(len(sub))
    meds = sub["median_delta_tau_err"].to_numpy()
    los = (sub["bootstrap_ci_lo"] - meds).to_numpy()
    his = (sub["bootstrap_ci_hi"] - meds).to_numpy()
    ax.errorbar(xs, meds, yerr=[np.abs(los), his], fmt="o", capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[order.index(v)] for v in sub.index], rotation=18, fontsize=8)
    ax.set_ylabel("median delta tau-error\n(variant-best minus NSE-best)")
    ax.set_title("b) Decision increment, 3 independent models\n(disagreement gauges)")
    ax.annotate("negative = variant preferred\nmodel has lower timescale error", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8)

    ax = axes[2]
    fracs = ssel.set_index("variant").reindex(order)["frac_variant_lower_tau_err"].dropna()
    ax.bar([labels[order.index(v)] for v in fracs.index], fracs.to_numpy(), color="steelblue")
    for thr in (0.50, 0.55, 0.60):
        ax.axhline(thr, color="red", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of disagreement\ngauges with lower tau-error")
    ax.set_title("c) Variant-preferred model wins on tau-error\n(3 independent models)")
    ax.tick_params(axis="x", rotation=18, labelsize=8)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"r62_decision_test_diagnostic.{ext}", dpi=300)
    plt.close(fig)
    print(f"[r62] figure saved: {FIGURES / 'r62_decision_test_diagnostic.png'}")


if __name__ == "__main__":
    main()
