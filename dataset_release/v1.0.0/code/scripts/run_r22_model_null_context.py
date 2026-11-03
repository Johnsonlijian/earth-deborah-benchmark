"""R22 null-contextualized hydrological model benchmark.

R21 showed that a simple precipitation/PET-driven two-reservoir model can have
held-out hydrograph skill that is weakly associated with beta(De)-curve error.
This R22 increment adds explicit null context for that claim. It compares the
R21 model against (i) a one-day persistence baseline and (ii) a seasonal AR(1)
memory-null generated from training-period discharge residuals.

The analysis is a diagnostic benchmark, not a model-ranking study. Persistence
uses observed held-out lagged discharge and is therefore an optimistic baseline,
whereas the seasonal AR(1) null is a stochastic memory-only comparator.
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

import run_r21_hydrological_model_benchmark as r21
from run_camels_gb_replication import ATTR_DIR, DAILY_DIR, gauge_id_from_path, load_attributes

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r22_model_null_context"

MODEL_LABELS = {
    "two_reservoir": "two-reservoir",
    "persistence_1d": "lagged-Q baseline",
    "seasonal_ar1_null": "seasonal AR(1) null",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=160, help="Match the R21 gauge subset by default.")
    parser.add_argument("--min-eval-years", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def file_map(limit: int) -> dict[str, Path]:
    files = sorted(Path(DAILY_DIR).glob("*.csv"))
    if limit:
        files = files[:limit]
    return {r21.gauge_key(gauge_id_from_path(path)): path for path in files}


def doy_climatology(values: pd.Series, indices: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"value": values.iloc[indices].to_numpy(dtype=float)}, index=values.index[indices])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby(frame.index.dayofyear)["value"].mean()


def one_day_persistence(q_values: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    train_mean = float(np.nanmean(q_values[train_idx]))
    pred = np.full(eval_idx.size, train_mean, dtype=float)
    for j, idx in enumerate(eval_idx):
        prev = idx - 1
        if prev >= 0 and np.isfinite(q_values[prev]):
            pred[j] = q_values[prev]
    return np.clip(pred, 0.0, None)


def fit_ar1(residual: np.ndarray) -> tuple[float, float, float]:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size < 30 or np.nanstd(residual) <= 0:
        return 0.0, float(np.nanmean(residual)) if residual.size else 0.0, 0.0
    x = residual[:-1]
    y = residual[1:]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 30 or np.var(x) <= 0:
        return 0.0, float(np.nanmean(residual)), float(np.nanstd(residual))
    phi = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    phi = float(np.clip(phi, -0.98, 0.98))
    mu = float(np.nanmean(residual))
    innovation = y - (mu + phi * (x - mu))
    sigma = float(np.nanstd(innovation))
    return phi, mu, max(sigma, 1e-6)


def seasonal_ar1_null(
    q: pd.Series,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    q_values = q.to_numpy(dtype=float)
    train_dates = q.index[train_idx]
    train_log = np.log1p(np.clip(q_values[train_idx], 0.0, None))
    train_frame = pd.DataFrame({"logq": train_log}, index=train_dates).replace([np.inf, -np.inf], np.nan).dropna()
    if train_frame.empty:
        return np.full(eval_idx.size, np.nan)

    clim = train_frame.groupby(train_frame.index.dayofyear)["logq"].mean()
    fallback = float(train_frame["logq"].mean())
    train_clim = np.array([clim.get(day, fallback) for day in train_frame.index.dayofyear], dtype=float)
    residual = train_frame["logq"].to_numpy(dtype=float) - train_clim
    phi, mu, sigma = fit_ar1(residual)

    start = float(residual[-1]) if residual.size else mu
    sim_resid = np.empty(eval_idx.size, dtype=float)
    state = start
    for i in range(eval_idx.size):
        state = mu + phi * (state - mu) + rng.normal(0.0, sigma)
        sim_resid[i] = state

    eval_dates = q.index[eval_idx]
    eval_clim = np.array([clim.get(day, fallback) for day in eval_dates.dayofyear], dtype=float)
    pred = np.expm1(eval_clim + sim_resid)
    return np.clip(pred, 0.0, None)


def metric_row(
    gauge_id: str,
    model_type: str,
    eval_dates: pd.DatetimeIndex,
    obs: np.ndarray,
    pred: np.ndarray,
    train_nse: float = np.nan,
) -> tuple[dict[str, float | str] | None, pd.DataFrame | None]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 365 * 4:
        return None, None
    try:
        tau_obs, obs_curve = r21.beta_curve(eval_dates[mask], obs[mask], "observed")
        tau_pred, pred_curve = r21.beta_curve(eval_dates[mask], pred[mask], model_type)
    except Exception:
        return None, None

    row: dict[str, float | str] = {
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
    curves = pd.concat([obs_curve, pred_curve], ignore_index=True)
    curves["gauge_id"] = gauge_id
    curves["model_type"] = model_type
    curves["tau_days"] = curves["series"].map({"observed": tau_obs, model_type: tau_pred})
    return row, curves


def load_r21_rows() -> pd.DataFrame:
    path = TABLES / "r21_hydrological_model_benchmark_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path)
    metrics["gauge_key"] = metrics["gauge_id"].map(r21.gauge_key)
    rows = metrics.copy()
    rows["gauge_id"] = rows["gauge_key"]
    rows["model_type"] = "two_reservoir"
    rows["model_label"] = MODEL_LABELS["two_reservoir"]
    keep = [
        "gauge_id",
        "gauge_key",
        "model_type",
        "model_label",
        "n_eval_years",
        "train_nse",
        "eval_nse",
        "eval_kge",
        "obs_tau_eval_days",
        "model_tau_eval_days",
        "abs_log10_tau_error",
        "beta_de_median_abs_distance",
        "beta_raw_median_abs_distance",
    ]
    return rows[keep].copy()


def add_null_rows(args: argparse.Namespace, r21_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = file_map(args.limit)
    metrics: list[dict[str, float | str]] = []
    curves: list[pd.DataFrame] = []
    rng_master = np.random.default_rng(args.seed)
    gauge_seeds = {key: int(rng_master.integers(0, 2**31 - 1)) for key in r21_rows["gauge_key"].unique()}

    for idx, key in enumerate(r21_rows["gauge_key"].unique(), start=1):
        path = paths.get(key)
        if path is None:
            continue
        data = r21.load_one(path)
        q = data["discharge_spec"].astype(float)
        train_idx, eval_idx = r21.train_eval_split(q)
        if eval_idx.size < 365 * args.min_eval_years:
            continue
        q_values = q.to_numpy(dtype=float)
        eval_dates = data.index[eval_idx]
        eval_obs = q_values[eval_idx]

        pred_map = {
            "persistence_1d": one_day_persistence(q_values, train_idx, eval_idx),
            "seasonal_ar1_null": seasonal_ar1_null(q, train_idx, eval_idx, np.random.default_rng(gauge_seeds[key])),
        }
        for model_type, pred in pred_map.items():
            row, curve = metric_row(key, model_type, eval_dates, eval_obs, pred)
            if row is not None and curve is not None:
                metrics.append(row)
                curves.append(curve)
        if idx % 50 == 0:
            print(f"  processed null context for {idx} R21 gauges")

    return pd.DataFrame(metrics), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(index="gauge_id", columns="model_type", values="beta_de_median_abs_distance", aggfunc="first")
    rows = []
    if "two_reservoir" not in pivot.columns:
        return pd.DataFrame()
    for other in [col for col in pivot.columns if col != "two_reservoir"]:
        paired = pivot[["two_reservoir", other]].dropna()
        if paired.empty:
            continue
        delta = paired["two_reservoir"] - paired[other]
        try:
            stat, pval = stats.wilcoxon(delta)
        except Exception:
            stat, pval = np.nan, np.nan
        rows.append(
            {
                "comparison": f"two_reservoir_minus_{other}",
                "n": int(len(paired)),
                "median_delta_beta_de_distance": float(delta.median()),
                "mean_delta_beta_de_distance": float(delta.mean()),
                "fraction_two_reservoir_lower_distance": float((delta < 0).mean()),
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p_value": float(pval),
            }
        )
    return pd.DataFrame(rows)


def rank_disagreement(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gauge_id, group in metrics.dropna(subset=["eval_nse", "beta_de_median_abs_distance"]).groupby("gauge_id"):
        if group["model_type"].nunique() < 3:
            continue
        # Higher NSE is better; lower beta distance is better.
        nse_rank = group["eval_nse"].rank(ascending=False, method="average")
        beta_rank = group["beta_de_median_abs_distance"].rank(ascending=True, method="average")
        tau, pval = stats.kendalltau(nse_rank, beta_rank)
        rows.append({"gauge_id": gauge_id, "model_rank_kendall_tau": tau, "model_rank_kendall_p": pval})
    return pd.DataFrame(rows)


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


def attribute_associations(metrics: pd.DataFrame) -> pd.DataFrame:
    attrs = load_attributes(Path(ATTR_DIR)).reset_index()
    attrs["gauge_key"] = attrs["gauge_id"].map(r21.gauge_key)
    two = metrics[metrics["model_type"] == "two_reservoir"].copy()
    ar1 = metrics[metrics["model_type"] == "seasonal_ar1_null"][["gauge_id", "beta_de_median_abs_distance"]].rename(
        columns={"beta_de_median_abs_distance": "seasonal_ar1_beta_distance"}
    )
    two = two.merge(ar1, on="gauge_id", how="left")
    two["two_reservoir_minus_ar1_beta_distance"] = two["beta_de_median_abs_distance"] - two["seasonal_ar1_beta_distance"]
    work = two.merge(attrs, on="gauge_key", how="left", suffixes=("", "_attr"))
    candidates = [
        "aridity",
        "p_seasonality",
        "frac_snow",
        "high_prec_freq",
        "low_prec_freq",
        "baseflow_index",
        "runoff_ratio",
        "slope_fdc",
        "zero_q_freq",
        "area",
        "dpsbar",
        "elev_mean",
    ]
    responses = [
        "beta_de_median_abs_distance",
        "abs_log10_tau_error",
        "two_reservoir_minus_ar1_beta_distance",
    ]
    rows = []
    for response in responses:
        for attr in candidates:
            if attr not in work.columns:
                continue
            data = work[[response, attr]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 30:
                continue
            rho, pval = stats.spearmanr(data[attr], data[response])
            rows.append(
                {
                    "response": response,
                    "attribute": attr,
                    "n": int(len(data)),
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh_by_response"] = out.groupby("response")["p_value"].transform(bh_qvalues)
    return out


def summarize(metrics: pd.DataFrame, pairwise: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_type, group in metrics.groupby("model_type"):
        valid = group.dropna(subset=["eval_nse", "eval_kge", "beta_de_median_abs_distance"])
        rho, pval = stats.spearmanr(valid["eval_nse"], valid["beta_de_median_abs_distance"]) if len(valid) >= 5 else (np.nan, np.nan)
        rows.append(
            {
                "model_type": model_type,
                "model_label": MODEL_LABELS.get(model_type, model_type),
                "n_gauges": int(len(valid)),
                "median_eval_nse": float(valid["eval_nse"].median()) if not valid.empty else np.nan,
                "median_eval_kge": float(valid["eval_kge"].median()) if not valid.empty else np.nan,
                "median_beta_de_distance": float(valid["beta_de_median_abs_distance"].median()) if not valid.empty else np.nan,
                "spearman_nse_vs_beta_de_distance": float(rho),
                "spearman_p_value": float(pval),
            }
        )
    summary = pd.DataFrame(rows)
    if not ranks.empty:
        summary["median_within_gauge_rank_kendall_tau"] = float(ranks["model_rank_kendall_tau"].median())
    return summary


def plot_results(metrics: pd.DataFrame, pairwise: pd.DataFrame, assoc: pd.DataFrame, ranks: pd.DataFrame) -> None:
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
    order = ["two_reservoir", "persistence_1d", "seasonal_ar1_null"]
    colors = {
        "two_reservoir": "#2F6FA8",
        "persistence_1d": "#6B7A34",
        "seasonal_ar1_null": "#C4513F",
    }
    fig = plt.figure(figsize=(7.2, 5.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    vals = [metrics.loc[metrics["model_type"] == model, "beta_de_median_abs_distance"].dropna().to_numpy(dtype=float) for model in order]
    bp = ax0.boxplot(vals, tick_labels=[MODEL_LABELS[m] for m in order], widths=0.55, patch_artist=True, showfliers=False)
    for patch, model in zip(bp["boxes"], order):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.72)
    for model, x, arr in zip(order, range(1, len(order) + 1), vals):
        jitter = np.linspace(-0.14, 0.14, max(len(arr), 1))[: len(arr)]
        ax0.scatter(np.full(len(arr), x) + jitter, arr, s=6, color=colors[model], alpha=0.28, linewidths=0)
    ax0.set_ylabel("beta(De) median absolute distance")
    ax0.set_title("a  Null context for beta(De) error", loc="left", fontsize=8.6, fontweight="bold")
    ax0.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)
    ax0.tick_params(axis="x", rotation=15)

    scatter = metrics.dropna(subset=["eval_nse", "beta_de_median_abs_distance"])
    for model in order:
        data = scatter[scatter["model_type"] == model]
        ax1.scatter(
            data["eval_nse"],
            data["beta_de_median_abs_distance"],
            s=16,
            alpha=0.68,
            linewidths=0,
            color=colors[model],
            label=MODEL_LABELS[model],
        )
    ax1.set_xlabel("evaluation NSE")
    ax1.set_ylabel("beta(De) distance")
    ax1.set_title("b  Fit score and spectral error separate", loc="left", fontsize=8.6, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)
    ax1.legend(loc="upper right", fontsize=6)

    if not pairwise.empty:
        pwork = pairwise.copy()
        pwork["comparison_label"] = pwork["comparison"].str.replace("two_reservoir_minus_", "", regex=False).map(MODEL_LABELS)
        ax2.axvline(0, color="#667085", lw=0.9)
        y = np.arange(len(pwork))
        ax2.barh(y, pwork["median_delta_beta_de_distance"], color=["#C4513F" if v > 0 else "#2F6FA8" for v in pwork["median_delta_beta_de_distance"]], alpha=0.82)
        for yi, row in zip(y, pwork.itertuples(index=False)):
            ax2.text(row.median_delta_beta_de_distance, yi, f"  p={row.wilcoxon_p_value:.2g}", va="center", fontsize=6)
        ax2.set_yticks(y, pwork["comparison_label"])
    ax2.set_xlabel("median distance delta: two-reservoir minus baseline")
    ax2.set_title("c  Baseline comparison is a gate", loc="left", fontsize=8.6, fontweight="bold")
    ax2.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    awork = assoc[assoc["response"] == "two_reservoir_minus_ar1_beta_distance"].copy() if not assoc.empty else pd.DataFrame()
    if awork.empty:
        awork = assoc[assoc["response"] == "beta_de_median_abs_distance"].copy() if not assoc.empty else pd.DataFrame()
    if not awork.empty:
        awork = awork.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).head(8).sort_values("spearman_rho")
        y = np.arange(len(awork))
        ax3.axvline(0, color="#667085", lw=0.9)
        bar_colors = ["#C4513F" if v > 0 else "#2F6FA8" for v in awork["spearman_rho"]]
        ax3.barh(y, awork["spearman_rho"], color=bar_colors, alpha=0.86)
        labels = [name.replace("_", " ") for name in awork["attribute"]]
        ax3.set_yticks(y, labels)
        for yi, row in zip(y, awork.itertuples(index=False)):
            ax3.text(row.spearman_rho, yi, f"  q={row.q_value_bh_by_response:.2g}", va="center", fontsize=6)
    ax3.set_xlabel("Spearman rho")
    ax3.set_title("d  Boundary attributes", loc="left", fontsize=8.6, fontweight="bold")
    ax3.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    for suffix, kwargs in [
        (".png", {"dpi": 300}),
        (".svg", {}),
        (".pdf", {}),
    ]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", **kwargs)
    plt.close(fig)


def write_note(summary: pd.DataFrame, pairwise: pd.DataFrame, ranks: pd.DataFrame, assoc: pd.DataFrame) -> None:
    lines = [
        "# R22 Model Null Context",
        "",
        "This round contextualizes the R21 two-reservoir model benchmark against",
        "a one-day persistence baseline and a seasonal AR(1) memory-null.",
        "",
        "## Model summaries",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Pairwise beta(De) distance tests",
        "",
        pairwise.to_markdown(index=False, floatfmt=".4g") if not pairwise.empty else "No paired tests available.",
        "",
        "## Within-gauge model-rank disagreement",
        "",
        ranks.describe().to_markdown(floatfmt=".4g") if not ranks.empty else "No rank summary available.",
        "",
        "## Attribute associations",
        "",
        assoc.sort_values(["response", "q_value_bh_by_response"]).head(20).to_markdown(index=False, floatfmt=".4g")
        if not assoc.empty
        else "No attribute associations available.",
        "",
        "## Safe interpretation",
        "",
        "R22 upgrades the R21 model-use case from an isolated proof-of-concept to",
        "a null-contextualized diagnostic. It still does not constitute a full",
        "multi-model hydrological intercomparison.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    two = load_r21_rows()
    null_metrics, null_curves = add_null_rows(args, two)
    metrics = pd.concat([two, null_metrics], ignore_index=True, sort=False)
    metrics.to_csv(TABLES / f"{OUT}_metrics.csv", index=False)
    if not null_curves.empty:
        null_curves.to_csv(TABLES / f"{OUT}_curves.csv", index=False)

    pairwise = paired_tests(metrics)
    pairwise.to_csv(TABLES / f"{OUT}_pairwise_tests.csv", index=False)
    ranks = rank_disagreement(metrics)
    ranks.to_csv(TABLES / f"{OUT}_rank_disagreement.csv", index=False)
    assoc = attribute_associations(metrics)
    assoc.to_csv(TABLES / f"{OUT}_attribute_associations.csv", index=False)
    summary = summarize(metrics, pairwise, ranks)
    summary.to_csv(TABLES / f"{OUT}_summary.csv", index=False)

    plot_results(metrics, pairwise, assoc, ranks)
    write_note(summary, pairwise, ranks, assoc)
    print(summary.to_string(index=False))
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
