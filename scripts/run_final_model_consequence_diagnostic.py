from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = "final_model_consequence"


def spearman_pair(df: pd.DataFrame, x: str, y: str) -> dict:
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return {"x": x, "y": y, "n": len(d), "spearman_rho": np.nan, "p_value": np.nan}
    rho, p = stats.spearmanr(d[x], d[y])
    return {"x": x, "y": y, "n": len(d), "spearman_rho": float(rho), "p_value": float(p)}


def bootstrap_rho(df: pd.DataFrame, x: str, y: str, n_draws: int = 2000, seed: int = 44) -> tuple[float, float, float]:
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(d) < 8:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values = []
    idx = np.arange(len(d))
    for _ in range(n_draws):
        sample = d.iloc[rng.choice(idx, size=len(idx), replace=True)]
        rho, _ = stats.spearmanr(sample[x], sample[y])
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray(values)
    return (float(np.median(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TABLE_DIR / "r39_open_model_intercomparison_metrics.csv")
    df = df.replace([np.inf, -np.inf], np.nan)
    df["hydrograph_skill_loss"] = 1.0 - df["eval_nse"]
    df["spectral_memory_advantage"] = df["beta_raw_median_abs_distance"] - df["beta_de_median_abs_distance"]
    df["log10_tau_ratio_model_obs"] = np.log10(df["model_tau_eval_days"] / df["obs_tau_eval_days"])

    variables = [
        ("beta_de_median_abs_distance", "abs_log10_tau_error"),
        ("beta_de_median_abs_distance", "eval_nse"),
        ("beta_de_median_abs_distance", "eval_kge"),
        ("beta_raw_median_abs_distance", "abs_log10_tau_error"),
        ("eval_nse", "abs_log10_tau_error"),
        ("eval_kge", "abs_log10_tau_error"),
    ]
    rows = []
    for model, g in df.groupby("model_label"):
        for x, y in variables:
            row = spearman_pair(g, x, y)
            med, lo, hi = bootstrap_rho(g, x, y, seed=44 + len(rows))
            row.update({"model_label": model, "rho_bootstrap_median": med, "rho_ci_low": lo, "rho_ci_high": hi})
            rows.append(row)
    correlations = pd.DataFrame(rows)

    summary = (
        df.groupby("model_label")
        .agg(
            gauges=("gauge_id", "nunique"),
            median_eval_nse=("eval_nse", "median"),
            median_eval_kge=("eval_kge", "median"),
            median_beta_de_distance=("beta_de_median_abs_distance", "median"),
            median_beta_raw_distance=("beta_raw_median_abs_distance", "median"),
            median_abs_log10_tau_error=("abs_log10_tau_error", "median"),
            median_obs_tau_days=("obs_tau_eval_days", "median"),
            median_model_tau_days=("model_tau_eval_days", "median"),
        )
        .reset_index()
    )
    summary["median_de_minus_raw_distance"] = summary["median_beta_de_distance"] - summary["median_beta_raw_distance"]

    df.to_csv(TABLE_DIR / f"{OUT}_points.csv", index=False)
    correlations.to_csv(TABLE_DIR / f"{OUT}_correlations.csv", index=False)
    summary.to_csv(TABLE_DIR / f"{OUT}_summary.csv", index=False)
    return df, summary, correlations


def plot(df: pd.DataFrame, summary: pd.DataFrame, correlations: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
        }
    )
    plot_models = ["RRMPG GR4J", "RRMPG HBV-Edu", "global LSTM", "seasonal AR(1) null"]
    df = df[df["model_label"].isin(plot_models)].copy()
    summary = summary[summary["model_label"].isin(plot_models)].copy()
    correlations = correlations[correlations["model_label"].isin(plot_models)].copy()
    order = [m for m in plot_models if m in set(summary["model_label"])]
    palette = {
        "global LSTM": "#1f77b4",
        "RRMPG GR4J": "#ff7f0e",
        "RRMPG HBV-Edu": "#2ca02c",
        "seasonal AR(1) null": "#7f7f7f",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    ax = axes[0, 0]
    for label in order:
        g = df[df["model_label"] == label]
        ax.scatter(
            g["eval_nse"],
            g["beta_de_median_abs_distance"],
            s=12,
            alpha=0.35,
            color=palette.get(label, "#666666"),
            label=label,
        )
    ax.set_xlabel("Evaluation NSE")
    ax.set_ylabel("Median |beta(De) error|")
    ax.set_xlim(-1.0, 1.0)
    ax.set_title("a  Spectral error is not NSE (NSE clipped)")
    ax.legend(frameon=False, markerscale=1.5, fontsize=8)

    ax = axes[0, 1]
    for label in order:
        g = df[df["model_label"] == label]
        ax.scatter(
            g["abs_log10_tau_error"],
            g["beta_de_median_abs_distance"],
            s=12,
            alpha=0.35,
            color=palette.get(label, "#666666"),
        )
    ax.set_xlabel("Absolute log10 memory-time error")
    ax.set_ylabel("Median |beta(De) error|")
    ax.set_title("b  Diagnostic relation to memory error")

    ax = axes[1, 0]
    x = np.arange(len(order))
    w = 0.34
    s = summary.set_index("model_label").loc[order]
    ax.bar(x - w / 2, s["median_beta_de_distance"], width=w, color="#4c78a8", label="De axis")
    ax.bar(x + w / 2, s["median_beta_raw_distance"], width=w, color="#f58518", label="raw frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Median spectral distance")
    ax.set_title("c  Distance contract by model")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    heat = correlations[
        correlations["x"].eq("beta_de_median_abs_distance")
    ].pivot_table(index="model_label", columns="y", values="spearman_rho", aggfunc="first")
    heat = heat.reindex(order)
    cols = ["abs_log10_tau_error", "eval_nse", "eval_kge"]
    data = heat.reindex(columns=cols).to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["tau error", "NSE", "KGE"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("d  Spearman rho with |beta(De) error|")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{OUT}_diagnostic.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df, summary, correlations = build_tables()
    plot(df, summary, correlations)
    print(summary.to_string(index=False))
    print(correlations[correlations["x"].eq("beta_de_median_abs_distance")].to_string(index=False))


if __name__ == "__main__":
    main()
