"""R23 formal conceptual-model benchmark for beta(De) evaluation.

This script extends the R21/R22 CAMELS-GB proof-of-concept from a single
two-reservoir model into a small but formal intercomparison. Every model is
trained on the same early-period data, evaluated on the same held-out period,
and scored with both hydrograph metrics (NSE/KGE) and beta(De)-curve distance.

The model set is deliberately transparent:

* single linear reservoir driven by P - PET;
* two-reservoir model from R21;
* soil-bucket dual-routing conceptual model;
* equal-weight ensemble of the three conceptual rainfall-runoff models;
* seasonal discharge climatology;
* seasonal AR(1) memory-null from R22;
* one-day lagged-Q diagnostic lower bound from R22.

This is a formal conceptual-model benchmark, not a comparison of operational
hydrological modelling systems.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

import run_r21_hydrological_model_benchmark as r21
import run_r22_model_null_context as r22
from run_camels_gb_replication import DAILY_DIR, gauge_id_from_path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r23_multimodel_hydrology_benchmark"

MODEL_LABELS = {
    "single_reservoir": "single reservoir",
    "two_reservoir": "two reservoirs (compact)",
    "soil_bucket_dual_route": "soil bucket + dual route",
    "conceptual_ensemble": "conceptual ensemble",
    "seasonal_climatology": "seasonal climatology",
    "seasonal_ar1_null": "seasonal AR(1) null",
    "lagged_q_lower_bound": "lagged-Q lower bound",
}

MODEL_ORDER = [
    "single_reservoir",
    "two_reservoir",
    "soil_bucket_dual_route",
    "conceptual_ensemble",
    "seasonal_climatology",
    "seasonal_ar1_null",
    "lagged_q_lower_bound",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=160, help="Maximum CAMELS-GB gauges to process; 0 means all.")
    parser.add_argument("--min-eval-years", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def linear_filter(x: np.ndarray, tau_days: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    a = float(np.exp(-1.0 / tau_days))
    return signal.lfilter([1.0 - a], [1.0, -a], x)


def scaled_prediction(unit: np.ndarray, q_values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    intercept, slope = r21.fit_nonnegative_linear(q_values[train_idx], unit[train_idx])
    pred = np.clip(intercept + slope * unit, 0.0, None)
    return pred, float(r21.nse(pred[train_idx], q_values[train_idx])), float(intercept), float(slope)


def effective_input(precip: np.ndarray, pet: np.ndarray, pet_scale: float) -> np.ndarray:
    return np.maximum(precip - pet_scale * pet, 0.0)


def fit_single_reservoir(p: np.ndarray, pet: np.ndarray, q: np.ndarray, train_idx: np.ndarray) -> dict:
    best: dict | None = None
    for pet_scale in [0.0, 0.5, 1.0]:
        x = effective_input(p, pet, pet_scale)
        for tau in [2.0, 8.0, 32.0, 128.0]:
            unit = linear_filter(x, tau)
            pred, score, intercept, slope = scaled_prediction(unit, q, train_idx)
            if best is None or (np.isfinite(score) and score > best["train_nse"]):
                best = {
                    "model_type": "single_reservoir",
                    "pred": pred,
                    "train_nse": score,
                    "pet_scale": pet_scale,
                    "tau": tau,
                    "intercept": intercept,
                    "slope": slope,
                }
    if best is None:
        raise RuntimeError("single-reservoir grid failed")
    return best


def fit_two_reservoir(p: np.ndarray, pet: np.ndarray, q: np.ndarray, train_idx: np.ndarray) -> dict:
    best: dict | None = None
    for pet_scale in [0.0, 0.5, 1.0]:
        x = effective_input(p, pet, pet_scale)
        for fast_tau in [2.0, 8.0]:
            fast = linear_filter(x, fast_tau)
            for slow_tau in [32.0, 128.0]:
                if slow_tau <= fast_tau:
                    continue
                slow = linear_filter(x, slow_tau)
                for slow_frac in [0.25, 0.5, 0.75]:
                    unit = (1.0 - slow_frac) * fast + slow_frac * slow
                    pred, score, intercept, slope = scaled_prediction(unit, q, train_idx)
                    if best is None or (np.isfinite(score) and score > best["train_nse"]):
                        best = {
                            "model_type": "two_reservoir",
                            "pred": pred,
                            "train_nse": score,
                            "pet_scale": pet_scale,
                            "fast_tau": fast_tau,
                            "slow_tau": slow_tau,
                            "slow_frac": slow_frac,
                            "intercept": intercept,
                            "slope": slope,
                        }
    if best is None:
        raise RuntimeError("two-reservoir grid failed")
    return best


def soil_bucket_unit(
    precip: np.ndarray,
    pet: np.ndarray,
    capacity: float,
    pet_scale: float,
    runoff_exp: float,
    fast_tau: float,
    slow_tau: float,
    slow_frac: float,
) -> np.ndarray:
    storage = capacity * 0.5
    runoff = np.zeros_like(precip, dtype=float)
    for i, (p, e) in enumerate(zip(precip, pet)):
        wetness = np.clip(storage / max(capacity, 1e-6), 0.0, 2.0)
        evap = min(storage, pet_scale * e * min(wetness, 1.0))
        storage = max(storage + p - evap, 0.0)
        quick = max(storage - capacity, 0.0)
        if quick > 0:
            storage -= quick
        leak = min(storage, (wetness ** runoff_exp) * storage * 0.035)
        storage -= leak
        runoff[i] = quick + leak
    fast = linear_filter(runoff, fast_tau)
    slow = linear_filter(runoff, slow_tau)
    return (1.0 - slow_frac) * fast + slow_frac * slow


def fit_soil_bucket(p: np.ndarray, pet: np.ndarray, q: np.ndarray, train_idx: np.ndarray) -> dict:
    best: dict | None = None
    for capacity in [100.0, 300.0]:
        for pet_scale in [0.5, 1.0]:
            for runoff_exp in [1.0, 2.0]:
                fast_tau = 4.0
                slow_tau = 64.0
                slow_frac = 0.5
                unit = soil_bucket_unit(p, pet, capacity, pet_scale, runoff_exp, fast_tau, slow_tau, slow_frac)
                pred, score, intercept, slope = scaled_prediction(unit, q, train_idx)
                if best is None or (np.isfinite(score) and score > best["train_nse"]):
                    best = {
                        "model_type": "soil_bucket_dual_route",
                        "pred": pred,
                        "train_nse": score,
                        "capacity": capacity,
                        "pet_scale": pet_scale,
                        "runoff_exp": runoff_exp,
                        "fast_tau": fast_tau,
                        "slow_tau": slow_tau,
                        "slow_frac": slow_frac,
                        "intercept": intercept,
                        "slope": slope,
                    }
    if best is None:
        raise RuntimeError("soil-bucket grid failed")
    return best


def seasonal_climatology(q: pd.Series, train_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    train = pd.DataFrame({"q": q.iloc[train_idx].to_numpy(dtype=float)}, index=q.index[train_idx])
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    if train.empty:
        return np.full(eval_idx.size, np.nan)
    clim = train.groupby(train.index.dayofyear)["q"].median()
    fallback = float(train["q"].median())
    return np.array([clim.get(day, fallback) for day in q.index[eval_idx].dayofyear], dtype=float)


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
        fit_single_reservoir(p, pet, q, train_idx),
        fit_two_reservoir(p, pet, q, train_idx),
        fit_soil_bucket(p, pet, q, train_idx),
    ]
    ensemble_pred = np.nanmean(np.vstack([m["pred"] for m in fitted]), axis=0)
    fitted.append(
        {
            "model_type": "conceptual_ensemble",
            "pred": ensemble_pred,
            "train_nse": r21.nse(ensemble_pred[train_idx], q[train_idx]),
            "members": "single+two+bucket",
        }
    )

    climatology_pred = seasonal_climatology(q_series, train_idx, eval_idx)
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
    for baseline in ["seasonal_ar1_null", "two_reservoir", "lagged_q_lower_bound"]:
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
        out["q_value_bh_by_baseline"] = out.groupby("baseline_type")["p_value"].transform(bh_qvalues)
    return out


def rank_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gauge_id, group in metrics.groupby("gauge_id"):
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
            }
        )
    return pd.DataFrame(rows)


def plot_results(metrics: pd.DataFrame, summary: pd.DataFrame, pairwise: pd.DataFrame, ranks: pd.DataFrame) -> None:
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
        "single_reservoir": "#4E79A7",
        "two_reservoir": "#2F6FA8",
        "soil_bucket_dual_route": "#59A14F",
        "conceptual_ensemble": "#8CD17D",
        "seasonal_climatology": "#B07AA1",
        "seasonal_ar1_null": "#C4513F",
        "lagged_q_lower_bound": "#6B7A34",
    }
    fig = plt.figure(figsize=(7.6, 5.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    vals = [
        metrics.loc[metrics["model_type"] == model, "beta_de_median_abs_distance"].dropna().to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]
    bp = ax0.boxplot(vals, tick_labels=[MODEL_LABELS[m] for m in MODEL_ORDER], widths=0.55, patch_artist=True, showfliers=False, vert=False)
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.72)
    ax0.set_xlabel("beta(De) median absolute distance")
    ax0.set_title("a  Spectral-memory error across model families", loc="left", fontsize=8.8, fontweight="bold")
    ax0.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    scatter_order = [m for m in MODEL_ORDER if m != "lagged_q_lower_bound"]
    for model in scatter_order:
        data = metrics[(metrics["model_type"] == model)].dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        ax1.scatter(
            data["eval_nse"],
            data["beta_de_median_abs_distance"],
            s=14,
            color=colors[model],
            alpha=0.58,
            linewidths=0,
            label=MODEL_LABELS[model],
        )
    ax1.set_xlabel("evaluation NSE")
    ax1.set_ylabel("beta(De) distance")
    ax1.set_title("b  Hydrograph skill and beta(De) skill are separate axes", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)
    ax1.legend(loc="upper right", fontsize=5.3, ncols=1)

    pwork = pairwise[pairwise["baseline_type"] == "seasonal_ar1_null"].copy()
    pwork = pwork[pwork["model_type"].isin(["single_reservoir", "two_reservoir", "soil_bucket_dual_route", "conceptual_ensemble", "seasonal_climatology"])]
    if not pwork.empty:
        pwork["model_label"] = pwork["model_type"].map(MODEL_LABELS)
        pwork = pwork.sort_values("median_delta_beta_de_distance")
        y = np.arange(len(pwork))
        ax2.axvline(0, color="#667085", lw=0.9)
        ax2.barh(
            y,
            pwork["median_delta_beta_de_distance"],
            color=[colors[m] for m in pwork["model_type"]],
            alpha=0.85,
        )
        ax2.set_yticks(y, pwork["model_label"])
        for yi, row in zip(y, pwork.itertuples(index=False)):
            ax2.text(row.median_delta_beta_de_distance, yi, f"  q={row.q_value_bh_by_baseline:.2g}", va="center", fontsize=6)
    ax2.set_xlabel("median distance delta vs seasonal AR(1)")
    ax2.set_title("c  Memory-null remains a hard comparison", loc="left", fontsize=8.8, fontweight="bold")
    ax2.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    rank_models = [m for m in MODEL_ORDER if m != "lagged_q_lower_bound"]
    rank_rows = []
    for gauge_id, group in metrics[metrics["model_type"].isin(rank_models)].groupby("gauge_id"):
        group = group.dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
        if len(group) < 3:
            continue
        rank_rows.append(
            {
                "best_nse_model": group.loc[group["eval_nse"].idxmax(), "model_type"],
                "best_beta_model": group.loc[group["beta_de_median_abs_distance"].idxmin(), "model_type"],
            }
        )
    if rank_rows:
        counts = pd.crosstab(pd.DataFrame(rank_rows)["best_nse_model"], pd.DataFrame(rank_rows)["best_beta_model"])
        counts = counts.reindex(index=rank_models, columns=rank_models, fill_value=0)
        im = ax3.imshow(counts.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
        ax3.set_xticks(range(len(rank_models)), [MODEL_LABELS[m] for m in rank_models], rotation=45, ha="right")
        ax3.set_yticks(range(len(rank_models)), [MODEL_LABELS[m] for m in rank_models])
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                val = int(counts.iloc[i, j])
                if val:
                    ax3.text(j, i, str(val), ha="center", va="center", fontsize=6, color="#111827")
        fig.colorbar(im, ax=ax3, shrink=0.72, pad=0.02, label="gauges")
    ax3.set_xlabel("best beta(De)-distance model")
    ax3.set_ylabel("best NSE model")
    ax3.set_title("d  Best non-lower-bound model depends on the axis", loc="left", fontsize=8.8, fontweight="bold")

    for suffix, kwargs in [
        (".png", {"dpi": 300}),
        (".svg", {}),
        (".pdf", {}),
    ]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(summary: pd.DataFrame, pairwise: pd.DataFrame, ranks: pd.DataFrame) -> None:
    lines = [
        "# R23 Multi-Model Hydrology Benchmark",
        "",
        "R23 expands the CAMELS-GB model-use result from a single conceptual model",
        "into a formal same-split conceptual-model intercomparison.",
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
        ranks.describe(include='all').to_markdown() if not ranks.empty else "No rank table available.",
        "",
        "## Safe interpretation",
        "",
        "This is a formal conceptual-model benchmark. It supports using beta(De)",
        "distance as a model-evaluation axis alongside NSE/KGE. It does not claim",
        "a comprehensive operational model ranking.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(DAILY_DIR).glob("*.csv"))
    if args.limit:
        files = files[: args.limit]
    rng_master = np.random.default_rng(args.seed)

    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    for idx, path in enumerate(files, start=1):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        try:
            gauge_rows, gauge_curves = evaluate_gauge(path, rng, args.min_eval_years)
        except Exception as exc:
            print(f"  [{gauge_id_from_path(path)}] skipped: {exc}", flush=True)
            continue
        rows.extend(gauge_rows)
        curves.extend(gauge_curves)
        if idx % 20 == 0:
            print(f"  processed {idx}/{len(files)} files; rows={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No usable model-benchmark rows.")
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
    plot_results(metrics, summary, pairwise, ranks)
    write_note(summary, pairwise, ranks)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
