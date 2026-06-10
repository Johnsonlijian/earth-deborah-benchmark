"""R38 compact GR4J-/HBV-inspired structural benchmark for beta(De).

This round hardens the Fig. 6 model-use case. It keeps the transparent R23
same-split CAMELS-GB evaluation, but adds compact GR4J-/HBV-inspired
rainfall-runoff candidates. The implementations are deliberately compact and
auditable; manuscript language must call them "inspired structural candidates"
or "transparent structural surrogates", not operational GR4J/HBV software
products or established-model suites.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import os
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import run_r21_hydrological_model_benchmark as r21
import run_r22_model_null_context as r22
import run_r23_multimodel_hydrology_benchmark as r23
from run_camels_gb_replication import DAILY_DIR, gauge_id_from_path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX = ROOT
OUT = "r38_established_model_suite"

MODEL_LABELS = {
    "gr4j_structured": "GR4J-inspired",
    "hbv_structured": "HBV-inspired",
    "established_ensemble": "GR4J/HBV-inspired ensemble",
    "single_reservoir": "single reservoir",
    "two_reservoir": "two reservoirs",
    "soil_bucket_dual_route": "soil bucket + dual route",
    "conceptual_ensemble": "conceptual ensemble",
    "seasonal_climatology": "seasonal climatology",
    "seasonal_ar1_null": "seasonal AR(1) null",
    "lagged_q_lower_bound": "lagged-Q lower bound",
}

MODEL_ORDER = [
    "gr4j_structured",
    "hbv_structured",
    "established_ensemble",
    "single_reservoir",
    "two_reservoir",
    "soil_bucket_dual_route",
    "conceptual_ensemble",
    "seasonal_climatology",
    "seasonal_ar1_null",
    "lagged_q_lower_bound",
]

INDEPENDENT_MODELS = [
    "gr4j_structured",
    "hbv_structured",
    "established_ensemble",
    "single_reservoir",
    "two_reservoir",
    "soil_bucket_dual_route",
    "conceptual_ensemble",
    "seasonal_climatology",
    "seasonal_ar1_null",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Maximum CAMELS-GB gauges; 0 means all.")
    parser.add_argument("--min-eval-years", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def gr4j_structured_unit(
    precip: np.ndarray,
    pet: np.ndarray,
    x1: float,
    x3: float,
    x4: float,
    pet_scale: float,
) -> np.ndarray:
    """Compact GR4J-like production store plus two routing components."""
    store = 0.5 * x1
    routed = np.zeros_like(precip, dtype=float)
    eps = 1e-9
    for i, (p, e) in enumerate(zip(precip, pet)):
        net = float(p - pet_scale * e)
        ratio = np.clip(store / max(x1, eps), 0.0, 1.0)
        if net >= 0.0:
            tanh_term = np.tanh(net / max(x1, eps))
            infil = x1 * (1.0 - ratio**2) * tanh_term / max(1.0 + ratio * tanh_term, eps)
            infil = min(max(infil, 0.0), net)
            store = min(x1, store + infil)
            excess = net - infil
        else:
            demand = -net
            tanh_term = np.tanh(demand / max(x1, eps))
            evap = store * (2.0 - ratio) * tanh_term / max(1.0 + (1.0 - ratio) * tanh_term, eps)
            evap = min(max(evap, 0.0), store)
            store -= evap
            excess = 0.0
        perc = store * (1.0 - (1.0 + (store / max(2.25 * x1, eps)) ** 4) ** -0.25)
        perc = min(max(perc, 0.0), store)
        store -= perc
        routed[i] = max(excess + perc, 0.0)
    quick = r23.linear_filter(0.10 * routed, max(1.0, 0.5 * x4))
    slow = r23.linear_filter(0.90 * routed, max(1.0, x3))
    return quick + slow


def hbv_structured_unit(
    precip: np.ndarray,
    pet: np.ndarray,
    fc: float,
    beta: float,
    k_fast: float,
    k_slow: float,
    pet_scale: float,
) -> np.ndarray:
    """Compact HBV-like soil store, upper box and lower groundwater box."""
    soil = 0.5 * fc
    upper = 0.0
    lower = 0.0
    out = np.zeros_like(precip, dtype=float)
    for i, (p, e) in enumerate(zip(precip, pet)):
        rel = np.clip(soil / max(fc, 1e-9), 0.0, 1.0)
        recharge = max(p, 0.0) * rel**beta
        soil = min(fc, max(0.0, soil + max(p, 0.0) - recharge))
        aet = min(soil, pet_scale * max(e, 0.0) * np.clip(soil / max(fc, 1e-9), 0.0, 1.0))
        soil -= aet
        upper += 0.55 * recharge
        lower += 0.45 * recharge
        quick = k_fast * upper
        upper -= quick
        percolation = min(upper, 0.06 * upper)
        upper -= percolation
        lower += percolation
        slow = k_slow * lower
        lower -= slow
        out[i] = max(quick + slow, 0.0)
    return out


def fit_gr4j_structured(p: np.ndarray, pet: np.ndarray, q: np.ndarray, train_idx: np.ndarray) -> dict:
    best: dict | None = None
    grid = itertools.product([120.0, 400.0, 900.0], [18.0, 70.0, 180.0], [1.5, 3.5], [0.7, 1.0])
    for x1, x3, x4, pet_scale in grid:
        unit = gr4j_structured_unit(p, pet, x1, x3, x4, pet_scale)
        pred, score, intercept, slope = r23.scaled_prediction(unit, q, train_idx)
        if best is None or (np.isfinite(score) and score > best["train_nse"]):
            best = {
                "model_type": "gr4j_structured",
                "pred": pred,
                "train_nse": score,
                "x1_production_store": x1,
                "x3_routing_store": x3,
                "x4_route_lag": x4,
                "pet_scale": pet_scale,
                "intercept": intercept,
                "slope": slope,
            }
    if best is None:
        raise RuntimeError("GR4J-inspired grid failed")
    return best


def fit_hbv_structured(p: np.ndarray, pet: np.ndarray, q: np.ndarray, train_idx: np.ndarray) -> dict:
    best: dict | None = None
    grid = itertools.product([120.0, 350.0, 800.0], [1.0, 2.0, 4.0], [0.08, 0.22], [0.01, 0.04], [0.7, 1.0])
    for fc, beta, k_fast, k_slow, pet_scale in grid:
        unit = hbv_structured_unit(p, pet, fc, beta, k_fast, k_slow, pet_scale)
        pred, score, intercept, slope = r23.scaled_prediction(unit, q, train_idx)
        if best is None or (np.isfinite(score) and score > best["train_nse"]):
            best = {
                "model_type": "hbv_structured",
                "pred": pred,
                "train_nse": score,
                "fc_soil_capacity": fc,
                "beta_runoff_shape": beta,
                "k_fast": k_fast,
                "k_slow": k_slow,
                "pet_scale": pet_scale,
                "intercept": intercept,
                "slope": slope,
            }
    if best is None:
        raise RuntimeError("HBV-inspired grid failed")
    return best


def metric_row(
    gauge_id: str,
    model_type: str,
    eval_dates: pd.DatetimeIndex,
    obs: np.ndarray,
    pred: np.ndarray,
    train_nse: float,
    params: dict,
    tau_obs: float,
    obs_curve: pd.DataFrame,
) -> tuple[dict | None, pd.DataFrame | None]:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 365 * 4:
        return None, None
    try:
        tau_pred, pred_curve = r21.beta_curve(eval_dates[mask], pred[mask], model_type)
    except Exception:
        return None, None
    row = {
        "gauge_id": gauge_id,
        "model_type": model_type,
        "model_label": MODEL_LABELS[model_type],
        "n_eval_years": float(mask.sum() / 365.25),
        "train_nse": float(train_nse),
        "eval_nse": r21.nse(pred[mask], obs[mask]),
        "eval_kge": r21.kge(pred[mask], obs[mask]),
        "obs_tau_eval_days": float(tau_obs),
        "model_tau_eval_days": float(tau_pred),
        "abs_log10_tau_error": float(abs(np.log10(tau_pred / tau_obs))) if tau_obs > 0 and tau_pred > 0 else np.nan,
        "beta_de_median_abs_distance": r21.curve_distance(obs_curve, pred_curve, "de"),
        "beta_raw_median_abs_distance": r21.curve_distance(obs_curve, pred_curve, "frequency_cpd"),
    }
    for key, value in params.items():
        if key != "pred":
            row[f"param_{key}"] = value
    curves = pred_curve.copy()
    curves["gauge_id"] = gauge_id
    curves["model_type"] = model_type
    curves["tau_days"] = tau_pred
    return row, curves


def evaluate_gauge(path: Path, rng: np.random.Generator, min_eval_years: float) -> tuple[list[dict], list[pd.DataFrame]]:
    gauge_id = gauge_id_from_path(path)
    data = r21.load_one(path)
    q_series = data["discharge_spec"].astype(float)
    train_idx, eval_idx = r21.train_eval_split(q_series)
    if eval_idx.size < 365 * min_eval_years:
        return [], []

    p = data["precipitation_cehgear"].to_numpy(dtype=float)
    pet = data["pet_chess"].to_numpy(dtype=float)
    q = q_series.to_numpy(dtype=float)
    eval_dates = data.index[eval_idx]
    obs = q[eval_idx]
    try:
        tau_obs, obs_curve = r21.beta_curve(eval_dates, obs, "observed")
    except Exception:
        return [], []

    fitted = [
        fit_gr4j_structured(p, pet, q, train_idx),
        fit_hbv_structured(p, pet, q, train_idx),
        r23.fit_single_reservoir(p, pet, q, train_idx),
        r23.fit_two_reservoir(p, pet, q, train_idx),
        r23.fit_soil_bucket(p, pet, q, train_idx),
    ]

    established_members = [m["pred"] for m in fitted if m["model_type"] in {"gr4j_structured", "hbv_structured"}]
    fitted.append(
        {
            "model_type": "established_ensemble",
            "pred": np.nanmean(np.vstack(established_members), axis=0),
            "train_nse": r21.nse(np.nanmean(np.vstack(established_members), axis=0)[train_idx], q[train_idx]),
            "members": "gr4j_structured+hbv_structured",
        }
    )

    conceptual_members = [m["pred"] for m in fitted if m["model_type"] in {"single_reservoir", "two_reservoir", "soil_bucket_dual_route"}]
    fitted.append(
        {
            "model_type": "conceptual_ensemble",
            "pred": np.nanmean(np.vstack(conceptual_members), axis=0),
            "train_nse": r21.nse(np.nanmean(np.vstack(conceptual_members), axis=0)[train_idx], q[train_idx]),
            "members": "single+two+bucket",
        }
    )

    climatology_pred = r23.seasonal_climatology(q_series, train_idx, eval_idx)
    fitted.append(
        {
            "model_type": "seasonal_climatology",
            "pred": np.full_like(q, np.nan, dtype=float),
            "train_nse": np.nan,
            "note": "train-period daily climatology",
        }
    )
    fitted[-1]["pred"][eval_idx] = climatology_pred

    ar1_pred = r22.seasonal_ar1_null(q_series, train_idx, eval_idx, rng)
    fitted.append(
        {
            "model_type": "seasonal_ar1_null",
            "pred": np.full_like(q, np.nan, dtype=float),
            "train_nse": np.nan,
            "note": "seasonal AR(1) generated from train residuals",
        }
    )
    fitted[-1]["pred"][eval_idx] = ar1_pred

    lag_pred = r22.one_day_persistence(q, train_idx, eval_idx)
    fitted.append(
        {
            "model_type": "lagged_q_lower_bound",
            "pred": np.full_like(q, np.nan, dtype=float),
            "train_nse": np.nan,
            "note": "uses observed previous-day held-out discharge",
        }
    )
    fitted[-1]["pred"][eval_idx] = lag_pred

    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    for fit in fitted:
        model_type = fit["model_type"]
        pred_eval = np.asarray(fit["pred"], dtype=float)[eval_idx]
        params = {k: v for k, v in fit.items() if k not in {"model_type", "train_nse"}}
        row, curve = metric_row(gauge_id, model_type, eval_dates, obs, pred_eval, fit["train_nse"], params, tau_obs, obs_curve)
        if row is not None and curve is not None:
            rows.append(row)
            curves.append(curve)
    return rows, curves


def evaluate_job(job: tuple[str, int, float]) -> tuple[str, list[dict], list[pd.DataFrame], str | None]:
    path_text, seed, min_eval_years = job
    path = Path(path_text)
    rng = np.random.default_rng(seed)
    try:
        rows, curves = evaluate_gauge(path, rng, min_eval_years)
        return gauge_id_from_path(path), rows, curves, None
    except Exception as exc:
        return gauge_id_from_path(path), [], [], str(exc)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_type in MODEL_ORDER:
        group = metrics[metrics["model_type"] == model_type].replace([np.inf, -np.inf], np.nan)
        valid = group.dropna(subset=["eval_nse", "eval_kge", "beta_de_median_abs_distance"])
        rho, pval = stats.spearmanr(valid["eval_nse"], valid["beta_de_median_abs_distance"]) if len(valid) >= 5 else (np.nan, np.nan)
        rows.append(
            {
                "model_type": model_type,
                "model_label": MODEL_LABELS[model_type],
                "n_gauges": int(len(valid)),
                "median_eval_nse": float(valid["eval_nse"].median()) if not valid.empty else np.nan,
                "median_eval_kge": float(valid["eval_kge"].median()) if not valid.empty else np.nan,
                "median_beta_de_distance": float(valid["beta_de_median_abs_distance"].median()) if not valid.empty else np.nan,
                "median_abs_log10_tau_error": float(valid["abs_log10_tau_error"].median()) if not valid.empty else np.nan,
                "spearman_nse_vs_beta_de_distance": float(rho),
                "spearman_p_value": float(pval),
            }
        )
    return pd.DataFrame(rows)


def pairwise_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(index="gauge_id", columns="model_type", values="beta_de_median_abs_distance", aggfunc="first")
    rows = []
    baselines = ["seasonal_ar1_null", "established_ensemble", "lagged_q_lower_bound"]
    for baseline in baselines:
        if baseline not in pivot.columns:
            continue
        for model in [m for m in MODEL_ORDER if m in pivot.columns and m != baseline]:
            paired = pivot[[model, baseline]].dropna()
            if len(paired) < 10:
                continue
            delta = paired[model] - paired[baseline]
            try:
                stat, pval = stats.wilcoxon(delta)
            except Exception:
                stat, pval = np.nan, np.nan
            rows.append(
                {
                    "model_type": model,
                    "baseline_type": baseline,
                    "n": int(len(paired)),
                    "median_delta_beta_de_distance": float(delta.median()),
                    "mean_delta_beta_de_distance": float(delta.mean()),
                    "fraction_model_lower_distance": float((delta < 0).mean()),
                    "wilcoxon_statistic": float(stat),
                    "p_value": float(pval),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh_by_baseline"] = out.groupby("baseline_type")["p_value"].transform(r23.bh_qvalues)
    return out


def rank_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gauge_id, group in metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)].groupby("gauge_id"):
        group = group.dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        if len(group) < 4:
            continue
        nse_rank = group["eval_nse"].rank(ascending=False, method="average")
        beta_rank = group["beta_de_median_abs_distance"].rank(ascending=True, method="average")
        tau, pval = stats.kendalltau(nse_rank, beta_rank)
        rows.append(
            {
                "gauge_id": gauge_id,
                "n_models": int(len(group)),
                "rank_kendall_tau_nse_vs_beta": float(tau),
                "rank_kendall_p": float(pval),
                "best_nse_model": group.loc[group["eval_nse"].idxmax(), "model_type"],
                "best_beta_model": group.loc[group["beta_de_median_abs_distance"].idxmin(), "model_type"],
                "best_nse_is_best_beta": bool(
                    group.loc[group["eval_nse"].idxmax(), "model_type"]
                    == group.loc[group["beta_de_median_abs_distance"].idxmin(), "model_type"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_results(metrics: pd.DataFrame, pairwise: pd.DataFrame) -> None:
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
    colors = {
        "gr4j_structured": "#255C99",
        "hbv_structured": "#2A9D8F",
        "established_ensemble": "#264653",
        "single_reservoir": "#8AB17D",
        "two_reservoir": "#E9C46A",
        "soil_bucket_dual_route": "#F4A261",
        "conceptual_ensemble": "#E76F51",
        "seasonal_climatology": "#A78BFA",
        "seasonal_ar1_null": "#8D2F2F",
        "lagged_q_lower_bound": "#6B7280",
    }
    fig = plt.figure(figsize=(7.6, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    vals = [
        metrics.loc[metrics["model_type"] == model, "beta_de_median_abs_distance"].dropna().to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]
    bp = ax0.boxplot(vals, tick_labels=[MODEL_LABELS[m] for m in MODEL_ORDER], vert=False, widths=0.56, patch_artist=True, showfliers=False)
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.78 if "structured" in model or "ensemble" in model else 0.62)
    ax0.set_xlabel("held-out beta(De) distance")
    ax0.set_title("a  GR4J-/HBV-inspired structures in the diagnostic", loc="left", fontsize=8.8, fontweight="bold")
    ax0.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    all_scatter = metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)].dropna(
        subset=["eval_nse", "beta_de_median_abs_distance"]
    )
    ax1.scatter(
        all_scatter["eval_nse"],
        all_scatter["beta_de_median_abs_distance"],
        s=5,
        color="#9AA5B1",
        alpha=0.055,
        linewidths=0,
        rasterized=True,
    )
    short_labels = {
        "gr4j_structured": "GR4J",
        "hbv_structured": "HBV",
        "established_ensemble": "G/H insp. ens.",
        "single_reservoir": "1R",
        "two_reservoir": "2R",
        "soil_bucket_dual_route": "soil",
        "conceptual_ensemble": "C ens.",
        "seasonal_climatology": "clim.",
        "seasonal_ar1_null": "AR1",
    }
    for model in INDEPENDENT_MODELS:
        data = metrics[metrics["model_type"] == model].dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        if data.empty:
            continue
        xq = data["eval_nse"].quantile([0.25, 0.50, 0.75])
        yq = data["beta_de_median_abs_distance"].quantile([0.25, 0.50, 0.75])
        ax1.errorbar(
            xq.loc[0.50],
            yq.loc[0.50],
            xerr=[[xq.loc[0.50] - xq.loc[0.25]], [xq.loc[0.75] - xq.loc[0.50]]],
            yerr=[[yq.loc[0.50] - yq.loc[0.25]], [yq.loc[0.75] - yq.loc[0.50]]],
            fmt="o",
            ms=4.2 if "structured" in model or "ensemble" in model else 3.8,
            lw=1.0,
            capsize=2.0,
            color=colors[model],
            mec="white",
            mew=0.45,
        )
        ax1.text(xq.loc[0.50] + 0.012, yq.loc[0.50] + 0.012, short_labels[model], color=colors[model], fontsize=5.9)
    ax1.set_xlabel("held-out NSE")
    ax1.set_ylabel("held-out beta(De) distance")
    ax1.set_title("b  Hydrograph and spectral-memory skill diverge", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)

    if "baseline_type" in pairwise.columns:
        pwork = pairwise[pairwise["baseline_type"] == "seasonal_ar1_null"].copy()
        pwork = pwork[pwork["model_type"].isin([m for m in INDEPENDENT_MODELS if m != "seasonal_ar1_null"])]
    else:
        pwork = pd.DataFrame()
    if not pwork.empty:
        pwork = pwork.sort_values("median_delta_beta_de_distance")
        y = np.arange(len(pwork))
        ax2.axvline(0, color="#475467", lw=0.9)
        ax2.barh(y, pwork["median_delta_beta_de_distance"], color=[colors[m] for m in pwork["model_type"]], alpha=0.86)
        ax2.set_yticks(y, [MODEL_LABELS[m] for m in pwork["model_type"]])
        xmax = max(0.08, float(np.nanmax(np.abs(pwork["median_delta_beta_de_distance"]))) * 1.18)
        ax2.set_xlim(-0.04, xmax)
        for yi, row in zip(y, pwork.itertuples(index=False)):
            ax2.text(xmax * 0.98, yi, f"q={row.q_value_bh_by_baseline:.2g}", ha="right", va="center", fontsize=5.8)
    ax2.set_xlabel("median beta(De) distance delta vs seasonal AR(1)")
    ax2.set_title("c  A stochastic memory null remains competitive", loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    rank_rows = []
    for _, group in metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)].groupby("gauge_id"):
        group = group.dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        if len(group) < 4:
            continue
        rank_rows.append(
            {
                "best_nse_model": group.loc[group["eval_nse"].idxmax(), "model_type"],
                "best_beta_model": group.loc[group["beta_de_median_abs_distance"].idxmin(), "model_type"],
            }
        )
    if rank_rows:
        counts = pd.crosstab(pd.DataFrame(rank_rows)["best_nse_model"], pd.DataFrame(rank_rows)["best_beta_model"])
        counts = counts.reindex(index=INDEPENDENT_MODELS, columns=INDEPENDENT_MODELS, fill_value=0)
        im = ax3.imshow(counts.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax3.set_xticks(range(len(INDEPENDENT_MODELS)), [MODEL_LABELS[m] for m in INDEPENDENT_MODELS], rotation=45, ha="right")
        ax3.set_yticks(range(len(INDEPENDENT_MODELS)), [MODEL_LABELS[m] for m in INDEPENDENT_MODELS])
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                val = int(counts.iloc[i, j])
                if val:
                    ax3.text(j, i, str(val), ha="center", va="center", fontsize=5.8, color="white" if val > counts.to_numpy().max() * 0.35 else "#111827")
        fig.colorbar(im, ax=ax3, shrink=0.74, pad=0.02, label="gauges")
    ax3.set_xlabel("best beta(De)-distance output")
    ax3.set_ylabel("best NSE output")
    ax3.set_title("d  Best output depends on the evaluation axis", loc="left", fontsize=8.8, fontweight="bold")

    for suffix, kwargs in [(".png", {"dpi": 320}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(summary: pd.DataFrame, pairwise: pd.DataFrame, ranks: pd.DataFrame) -> None:
    agreement = float(ranks["best_nse_is_best_beta"].mean()) if not ranks.empty else np.nan
    lines = [
        "# R38 GR4J-/HBV-Inspired Structural Benchmark",
        "",
        "R38 adds compact GR4J-/HBV-inspired rainfall-runoff candidates to the",
        "same CAMELS-GB held-out diagnostic used in R23.",
        "",
        "## Model summaries",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Pairwise beta(De)-distance tests",
        "",
        pairwise.to_markdown(index=False, floatfmt=".4g") if not pairwise.empty else "No pairwise tests available.",
        "",
        "## Rank disagreement",
        "",
        f"Fraction of gauges where the best NSE model is also the best beta(De)-distance model: {agreement:.3f}",
        "",
        ranks.describe(include='all').to_markdown() if not ranks.empty else "No rank table available.",
        "",
        "## Safe interpretation",
        "",
        "This round materially improves the model-use case by adding compact",
        "GR4J-/HBV-inspired structural surrogates. It still does not claim an",
        "operational GR4J/HBV software intercomparison, neural hydrological benchmark,",
        "or universal model ranking. The defensible claim is that beta(De)-curve",
        "distance remains a separate held-out diagnostic axis in this candidate set.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_submission_outputs() -> None:
    source = LATEX / "source_data"
    figures = LATEX / "figures"
    source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png", ".svg"]:
        src = FIGURES / f"{OUT}{suffix}"
        if src.exists():
            shutil.copy2(src, figures / f"fig6_multimodel_benchmark{suffix}")
    for name in [
        f"{OUT}_metrics.csv",
        f"{OUT}_curves.csv",
        f"{OUT}_summary.csv",
        f"{OUT}_pairwise_tests.csv",
        f"{OUT}_rank_disagreement.csv",
    ]:
        src = TABLES / name
        if src.exists():
            shutil.copy2(src, source / name)


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(DAILY_DIR).glob("*.csv"))
    if args.limit:
        files = files[: args.limit]
    rng_master = np.random.default_rng(args.seed)

    jobs = [
        (str(path), int(rng_master.integers(0, 2**31 - 1)), args.min_eval_years)
        for path in files
    ]
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    if args.workers <= 1:
        iterator = map(evaluate_job, jobs)
        for idx, (gauge_id, gauge_rows, gauge_curves, error) in enumerate(iterator, start=1):
            if error:
                print(f"  [{gauge_id}] skipped: {error}", flush=True)
            rows.extend(gauge_rows)
            curves.extend(gauge_curves)
            if idx % 10 == 0:
                print(f"  processed {idx}/{len(files)} files; rows={len(rows)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(evaluate_job, job): job[0] for job in jobs}
            for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                gauge_id, gauge_rows, gauge_curves, error = future.result()
                if error:
                    print(f"  [{gauge_id}] skipped: {error}", flush=True)
                rows.extend(gauge_rows)
                curves.extend(gauge_curves)
                if idx % 10 == 0:
                    print(f"  processed {idx}/{len(files)} files; rows={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No usable R38 model-benchmark rows.")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / f"{OUT}_metrics.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(TABLES / f"{OUT}_curves.csv", index=False)

    summary = summarize(metrics)
    pairwise = pairwise_tests(metrics)
    ranks = rank_table(metrics)
    summary.to_csv(TABLES / f"{OUT}_summary.csv", index=False)
    pairwise.to_csv(TABLES / f"{OUT}_pairwise_tests.csv", index=False)
    ranks.to_csv(TABLES / f"{OUT}_rank_disagreement.csv", index=False)
    plot_results(metrics, pairwise)
    write_note(summary, pairwise, ranks)
    copy_submission_outputs()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
