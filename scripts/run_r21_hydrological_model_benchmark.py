"""R21 hydrological model benchmark demonstration.

This proof-of-concept asks whether a simple precipitation/PET-driven
rainfall-runoff model can fit daily discharge while failing the beta(De)
spectral-memory benchmark. The analysis uses CAMELS-GB v2 hydrometeorological
time series because the local project contains daily precipitation, PET and
specific discharge in one archive.

The goal is not to present a state-of-the-art hydrological model. It is to show
that beta(De) can operate as an additional model-evaluation axis beyond NSE/KGE.
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

from edb.signal.de import beta_vs_de
from edb.signal.psd import welch_psd
from edb.signal.slopes import local_loglog_slope
from edb.signal.timescales import integral_autocorrelation_time
from run_camels_gb_replication import DAILY_DIR, gauge_id_from_path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
OUT = "r21_hydrological_model_benchmark"
SECONDS_PER_DAY = 86_400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=160, help="Maximum gauges to process; 0 means all.")
    parser.add_argument("--min-eval-years", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]
    obs = obs[mask]
    if sim.size < 10 or np.nanstd(obs) <= 0 or np.nanstd(sim) <= 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]
    obs = obs[mask]
    if sim.size < 10:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def linear_reservoir_filter(x: np.ndarray, tau_days: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    a = float(np.exp(-1.0 / tau_days))
    for i in range(1, x.size):
        out[i] = a * out[i - 1] + (1.0 - a) * x[i]
    return out


def fit_nonnegative_linear(y_train: np.ndarray, x_train: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_train) & np.isfinite(x_train)
    if mask.sum() < 30:
        return 0.0, 0.0
    x = x_train[mask]
    y = y_train[mask]
    var = float(np.var(x))
    if var <= 0:
        return max(float(np.mean(y)), 0.0), 0.0
    slope = float(np.cov(x, y, ddof=0)[0, 1] / var)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return max(intercept, 0.0), max(slope, 0.0)


def simulate_candidate(precip: np.ndarray, pet: np.ndarray, pet_scale: float, fast_tau: float, slow_tau: float, slow_frac: float) -> np.ndarray:
    effective = np.maximum(precip - pet_scale * pet, 0.0)
    fast = linear_reservoir_filter(effective, fast_tau)
    slow = linear_reservoir_filter(effective, slow_tau)
    return (1.0 - slow_frac) * fast + slow_frac * slow


def load_one(path: Path) -> pd.DataFrame:
    usecols = ["date", "precipitation_cehgear", "pet_chess", "discharge_spec"]
    data = pd.read_csv(path, usecols=usecols, na_values=["NaN", "nan", ""])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in usecols[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").set_index("date").asfreq("D")
    data["precipitation_cehgear"] = data["precipitation_cehgear"].interpolate(limit=7).fillna(0.0).clip(lower=0.0)
    data["pet_chess"] = data["pet_chess"].interpolate(limit=7).fillna(0.0).clip(lower=0.0)
    data["discharge_spec"] = data["discharge_spec"].where(data["discharge_spec"] >= 0)
    return data


def train_eval_split(q: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    finite_idx = np.flatnonzero(np.isfinite(q.to_numpy(dtype=float)))
    if finite_idx.size < 365 * 12:
        return np.array([], dtype=int), np.array([], dtype=int)
    split = finite_idx[int(0.7 * finite_idx.size)]
    all_idx = np.arange(len(q))
    train = all_idx[(all_idx <= split) & np.isfinite(q.to_numpy(dtype=float))]
    eval_idx = all_idx[(all_idx > split) & np.isfinite(q.to_numpy(dtype=float))]
    return train, eval_idx


def prepare_anomaly(dates: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.DataFrame({"value": values}, index=dates).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    if len(frame) < 365 * 4:
        raise ValueError("too few data for spectral analysis")
    frame = frame.asfreq("D").interpolate(limit=30).dropna()
    frame["doy"] = frame.index.dayofyear
    frame["anom"] = frame["value"] - frame.groupby("doy")["value"].transform("mean")
    med = float(frame["anom"].median())
    mad = float(np.median(np.abs(frame["anom"].to_numpy(dtype=float) - med)))
    if mad > 0:
        frame["anom"] = (frame["anom"] - med) / mad
    return frame.index, frame["anom"].to_numpy(dtype=float)


def beta_curve(dates: pd.DatetimeIndex, values: np.ndarray, label: str) -> tuple[float, pd.DataFrame]:
    d, x = prepare_anomaly(dates, values)
    tau_seconds = integral_autocorrelation_time(x, dt_seconds=SECONDS_PER_DAY, max_lag=730, stop_at_zero=True)
    tau_days = tau_seconds / SECONDS_PER_DAY
    psd = welch_psd(d, x, fs=1.0 / SECONDS_PER_DAY)
    beta = local_loglog_slope(psd["frequency"], psd["psd"], window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_seconds)
    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency", "de", "beta"])
    curve = curve[(curve["frequency"] > 0) & (curve["de"] > 0)].copy()
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["series"] = label
    return tau_days, curve[["series", "frequency", "frequency_cpd", "de", "beta"]]


def gauge_key(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def curve_distance(obs_curve: pd.DataFrame, sim_curve: pd.DataFrame, value_col: str = "de", n_bins: int = 18) -> float:
    obs = obs_curve[["series", value_col, "beta"]].copy()
    sim = sim_curve[["series", value_col, "beta"]].copy()
    all_log = np.log10(pd.concat([obs[value_col], sim[value_col]], ignore_index=True).to_numpy(dtype=float))
    lo, hi = np.nanpercentile(all_log, [10, 90])
    edges = np.linspace(lo, hi, n_bins + 1)

    rows = []
    for label, data in [("obs", obs), ("sim", sim)]:
        work = data.copy()
        work["bin"] = pd.cut(np.log10(work[value_col]), bins=edges, labels=False, include_lowest=True)
        med = work.dropna(subset=["bin"]).groupby("bin")["beta"].median()
        rows.append(med.rename(label))
    merged = pd.concat(rows, axis=1).dropna()
    if merged.empty:
        return np.nan
    return float(np.nanmedian(np.abs(merged["obs"] - merged["sim"])))


def evaluate_gauge(path: Path) -> tuple[dict[str, float | str] | None, pd.DataFrame | None, pd.DataFrame | None]:
    gauge_id = gauge_id_from_path(path)
    data = load_one(path)
    q = data["discharge_spec"].astype(float)
    train_idx, eval_idx = train_eval_split(q)
    if eval_idx.size < 365 * 6:
        return None, None, None

    p = data["precipitation_cehgear"].to_numpy(dtype=float)
    pet = data["pet_chess"].to_numpy(dtype=float)
    q_values = q.to_numpy(dtype=float)
    pet_scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    fast_taus = [1.0, 2.0, 4.0, 8.0]
    slow_taus = [16.0, 32.0, 64.0, 128.0]
    slow_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

    best = None
    for pet_scale in pet_scales:
        for fast_tau in fast_taus:
            for slow_tau in slow_taus:
                if slow_tau <= fast_tau:
                    continue
                for slow_frac in slow_fracs:
                    unit = simulate_candidate(p, pet, pet_scale, fast_tau, slow_tau, slow_frac)
                    intercept, slope = fit_nonnegative_linear(q_values[train_idx], unit[train_idx])
                    sim = intercept + slope * unit
                    score = nse(sim[train_idx], q_values[train_idx])
                    if best is None or (np.isfinite(score) and score > best["train_nse"]):
                        best = {
                            "pet_scale": pet_scale,
                            "fast_tau": fast_tau,
                            "slow_tau": slow_tau,
                            "slow_frac": slow_frac,
                            "intercept": intercept,
                            "slope": slope,
                            "train_nse": score,
                            "sim": sim,
                        }
    if best is None:
        return None, None, None
    sim = np.asarray(best["sim"], dtype=float)
    eval_dates = data.index[eval_idx]
    eval_obs = q_values[eval_idx]
    eval_sim = sim[eval_idx]
    try:
        tau_obs, obs_curve = beta_curve(eval_dates, eval_obs, "observed")
        tau_sim, sim_curve = beta_curve(eval_dates, eval_sim, "model")
    except Exception:
        return None, None, None

    de_distance = curve_distance(obs_curve, sim_curve, "de")
    raw_distance = curve_distance(obs_curve, sim_curve, "frequency_cpd")
    metric = {
        "gauge_id": gauge_id,
        "n_eval_years": float(eval_idx.size / 365.25),
        "train_nse": float(best["train_nse"]),
        "eval_nse": nse(eval_sim, eval_obs),
        "eval_kge": kge(eval_sim, eval_obs),
        "obs_tau_eval_days": float(tau_obs),
        "model_tau_eval_days": float(tau_sim),
        "abs_log10_tau_error": float(abs(np.log10(tau_sim / tau_obs))) if tau_obs > 0 and tau_sim > 0 else np.nan,
        "beta_de_median_abs_distance": de_distance,
        "beta_raw_median_abs_distance": raw_distance,
        "pet_scale": float(best["pet_scale"]),
        "fast_tau": float(best["fast_tau"]),
        "slow_tau": float(best["slow_tau"]),
        "slow_frac": float(best["slow_frac"]),
        "intercept": float(best["intercept"]),
        "slope": float(best["slope"]),
    }
    curves = pd.concat([obs_curve, sim_curve], ignore_index=True)
    curves["gauge_id"] = gauge_id
    curves["tau_days"] = curves["series"].map({"observed": tau_obs, "model": tau_sim})

    # Store a compact 2-year example window from the evaluation period.
    window = min(730, len(eval_idx))
    ts = pd.DataFrame(
        {
            "gauge_id": gauge_id,
            "date": eval_dates[:window],
            "observed_q": eval_obs[:window],
            "model_q": eval_sim[:window],
        }
    )
    return metric, curves, ts


def plot_results(metrics: pd.DataFrame, curves: pd.DataFrame, examples: pd.DataFrame) -> None:
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
    work = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["eval_nse", "beta_de_median_abs_distance"]).copy()
    if work.empty:
        raise RuntimeError("No model metrics available for plotting.")
    work["_gauge_key"] = work["gauge_id"].map(gauge_key)
    curves = curves.copy()
    examples = examples.copy()
    curves["_gauge_key"] = curves["gauge_id"].map(gauge_key)
    examples["_gauge_key"] = examples["gauge_id"].map(gauge_key)
    nse_q75 = float(work["eval_nse"].quantile(0.75))
    dist_q75 = float(work["beta_de_median_abs_distance"].quantile(0.75))
    high_fit_fail = work[(work["eval_nse"] >= nse_q75) & (work["beta_de_median_abs_distance"] >= dist_q75)]
    if high_fit_fail.empty:
        example_gid = work.sort_values(["eval_nse", "beta_de_median_abs_distance"], ascending=[False, False]).iloc[0]["_gauge_key"]
    else:
        example_gid = high_fit_fail.sort_values("eval_nse", ascending=False).iloc[0]["_gauge_key"]

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    sc = ax0.scatter(
        work["eval_nse"],
        work["beta_de_median_abs_distance"],
        c=np.log10(work["obs_tau_eval_days"].clip(lower=0.1)),
        cmap="viridis",
        s=22,
        alpha=0.82,
        linewidths=0,
    )
    ax0.axvline(nse_q75, color="#7B8794", ls="--", lw=0.8)
    ax0.axhline(dist_q75, color="#7B8794", ls="--", lw=0.8)
    ax0.set_xlabel("evaluation NSE")
    ax0.set_ylabel(r"median $|\beta_{\mathrm{obs}}(D_e)-\beta_{\mathrm{model}}(D_e)|$")
    ax0.set_title("a  Hydrograph fit does not guarantee spectral-memory fit", loc="left", fontsize=9, fontweight="bold")
    ax0.grid(True, color="#E4E9EF", linewidth=0.55)
    cbar = fig.colorbar(sc, ax=ax0, shrink=0.78, pad=0.02)
    cbar.set_label(r"$\log_{10}\tau_{\mathrm{obs}}$ [d]")

    groups = [
        ("all gauges", work["beta_de_median_abs_distance"]),
        ("top-NSE quartile", work.loc[work["eval_nse"] >= nse_q75, "beta_de_median_abs_distance"]),
        ("high fit + high\nspectral error", high_fit_fail["beta_de_median_abs_distance"] if not high_fit_fail.empty else pd.Series(dtype=float)),
    ]
    vals = [g[1].dropna().to_numpy(dtype=float) for g in groups]
    ax1.boxplot(vals, tick_labels=[g[0] for g in groups], patch_artist=True, widths=0.55)
    ax1.set_ylabel("beta(De) median absolute distance")
    ax1.set_title("b  Spectral error remains visible among better hydrograph fits", loc="left", fontsize=9, fontweight="bold")
    ax1.grid(True, axis="y", color="#E4E9EF", linewidth=0.55)

    ts = examples[examples["_gauge_key"] == example_gid].copy()
    if not ts.empty:
        ax2.plot(pd.to_datetime(ts["date"]), ts["observed_q"], color="#2F6FA8", lw=1.0, label="observed")
        ax2.plot(pd.to_datetime(ts["date"]), ts["model_q"], color="#C4513F", lw=0.9, alpha=0.85, label="model")
    em = work[work["_gauge_key"] == example_gid].iloc[0]
    ax2.set_title(
        f"c  Example gauge {example_gid}: NSE={em['eval_nse']:.2f}, beta(De) distance={em['beta_de_median_abs_distance']:.2f}",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax2.set_ylabel("specific discharge [mm d$^{-1}$]")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, color="#E4E9EF", linewidth=0.55)
    ax2.legend(loc="upper right", fontsize=6)

    cc = curves[curves["_gauge_key"] == example_gid].copy()
    for series, color in [("observed", "#2F6FA8"), ("model", "#C4513F")]:
        data = cc[cc["series"] == series].dropna(subset=["de", "beta"])
        if data.empty:
            continue
        log_de = np.log10(data["de"].to_numpy(dtype=float))
        lo, hi = np.nanpercentile(log_de, [5, 95])
        bins = np.linspace(lo, hi, 22)
        data["bin"] = pd.cut(log_de, bins=bins, labels=False, include_lowest=True)
        prof = data.dropna(subset=["bin"]).groupby("bin").agg(de=("de", "median"), beta=("beta", "median"))
        ax3.semilogx(prof["de"], prof["beta"], color=color, lw=1.7, label=series)
    ax3.set_xlabel("De")
    ax3.set_ylabel("local beta")
    ax3.set_title("d  The benchmark exposes timescale-structure mismatch", loc="left", fontsize=9, fontweight="bold")
    ax3.grid(True, color="#E4E9EF", linewidth=0.55)
    ax3.legend(loc="upper left", fontsize=6)

    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{OUT}{suffix}", **kwargs)
    plt.close(fig)


def write_note(metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    work = metrics.replace([np.inf, -np.inf], np.nan)
    valid = work.dropna(subset=["eval_nse", "eval_kge", "beta_de_median_abs_distance"])
    rho, pval = stats.spearmanr(valid["eval_nse"], valid["beta_de_median_abs_distance"]) if len(valid) >= 5 else (np.nan, np.nan)
    nse_q75 = float(valid["eval_nse"].quantile(0.75)) if not valid.empty else np.nan
    dist_q75 = float(valid["beta_de_median_abs_distance"].quantile(0.75)) if not valid.empty else np.nan
    high_fit_fail = valid[(valid["eval_nse"] >= nse_q75) & (valid["beta_de_median_abs_distance"] >= dist_q75)]
    top_nse = valid["eval_nse"] >= nse_q75 if not valid.empty else pd.Series(dtype=bool)
    top_error = valid["beta_de_median_abs_distance"] >= dist_q75 if not valid.empty else pd.Series(dtype=bool)
    overlap_a = int((top_nse & top_error).sum()) if not valid.empty else 0
    overlap_b = int((top_nse & ~top_error).sum()) if not valid.empty else 0
    overlap_c = int((~top_nse & top_error).sum()) if not valid.empty else 0
    overlap_d = int((~top_nse & ~top_error).sum()) if not valid.empty else 0
    if min(overlap_a, overlap_b, overlap_c, overlap_d) >= 0 and len(valid) >= 5:
        fisher_odds, fisher_p = stats.fisher_exact([[overlap_a, overlap_b], [overlap_c, overlap_d]])
    else:
        fisher_odds, fisher_p = np.nan, np.nan
    summary = pd.DataFrame(
        [
            {
                "n_gauges": int(len(valid)),
                "median_eval_nse": float(valid["eval_nse"].median()) if not valid.empty else np.nan,
                "median_eval_kge": float(valid["eval_kge"].median()) if not valid.empty else np.nan,
                "median_beta_de_distance": float(valid["beta_de_median_abs_distance"].median()) if not valid.empty else np.nan,
                "spearman_nse_vs_beta_de_distance": float(rho),
                "spearman_p_value": float(pval),
                "top_nse_quartile_threshold": nse_q75,
                "high_fit_high_spectral_error_count": int(len(high_fit_fail)),
                "top_nse_top_error_fisher_odds": float(fisher_odds),
                "top_nse_top_error_fisher_p": float(fisher_p),
            }
        ]
    )
    summary.to_csv(TABLES / f"{OUT}_summary.csv", index=False)
    lines = [
        "# R21 Hydrological Model Benchmark Demonstration",
        "",
        "This proof-of-concept uses CAMELS-GB v2 daily precipitation, PET and",
        "specific discharge to calibrate a simple conceptual reservoir model by",
        "daily-discharge NSE, then evaluates whether the fitted model reproduces",
        "the observed beta(De) spectral-memory curve.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Interpretation",
        "",
        "Safe claim: beta(De) can expose timescale-structure mismatch that is not",
        "reducible to a single hydrograph-fit score in this proof-of-concept.",
        "The top-NSE/top-spectral-error overlap is treated as an existence",
        "example rather than enrichment evidence.",
        "",
        "Boundary: the model is intentionally simple and CAMELS-GB-only in R21.",
        "This is a benchmark demonstration, not a comprehensive hydrological-model",
        "intercomparison.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{OUT}_metrics.csv`",
        f"- `reports/tables/{OUT}_curves.csv`",
        f"- `reports/tables/{OUT}_example_timeseries.csv`",
        f"- `reports/tables/{OUT}_summary.csv`",
        f"- `reports/figures/{OUT}.png/svg/pdf`",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(DAILY_DIR).glob("*.csv"))
    if args.limit:
        files = files[: args.limit]

    metrics = []
    curves = []
    examples = []
    for idx, path in enumerate(files, start=1):
        result, curve, ts = evaluate_gauge(path)
        if result is not None and curve is not None and ts is not None:
            metrics.append(result)
            curves.append(curve)
            examples.append(ts)
        if idx % 25 == 0:
            print(f"  processed {idx}/{len(files)} files; usable={len(metrics)}")

    if not metrics:
        raise RuntimeError("No gauges produced usable model benchmark results.")
    metrics_df = pd.DataFrame(metrics)
    curves_df = pd.concat(curves, ignore_index=True)
    examples_df = pd.concat(examples, ignore_index=True)
    metrics_df.to_csv(TABLES / f"{OUT}_metrics.csv", index=False)
    curves_df.to_csv(TABLES / f"{OUT}_curves.csv", index=False)
    examples_df.to_csv(TABLES / f"{OUT}_example_timeseries.csv", index=False)
    plot_results(metrics_df, curves_df, examples_df)
    write_note(metrics_df, args)


if __name__ == "__main__":
    main()
