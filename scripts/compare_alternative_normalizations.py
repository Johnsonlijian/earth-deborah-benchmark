"""Compare De normalization against alternative memory scalings.

The goal is reviewer-facing: test whether tau_acf is doing specific
storage-memory work, or whether arbitrary/plausible alternative x-axis scalings
produce the same beta-curve dispersion reduction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
CURVES_CSV = TABLES / "camels_673_beta_curve_points.csv"
ATTR_CSV = TABLES / "camels_673_attributed_diagnostics.csv"
OUT_CSV = TABLES / "camels_673_alternative_normalization_metrics.csv"
SHUFFLE_CSV = TABLES / "camels_673_random_tau_normalization_null.csv"
OUT_MD = NOTES / "alternative_normalizations_20260531.md"


def station_bin_medians(curves: pd.DataFrame, axis_col: str, n_bins: int = 24) -> pd.DataFrame:
    values = curves[axis_col].to_numpy(dtype=float)
    keep = np.isfinite(values) & (values > 0) & np.isfinite(curves["beta"].to_numpy(dtype=float))
    work = curves.loc[keep, ["gauge_id", axis_col, "beta"]].copy()
    x = np.log10(work[axis_col].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work["bin_id"] = pd.cut(np.log10(work[axis_col]), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    grouped = (
        work.groupby(["gauge_id", "bin_id"], as_index=False)
        .agg(beta_median=("beta", "median"), n_points=("beta", "size"))
    )
    grouped["axis_value"] = grouped["bin_id"].map({i: 10 ** labels[i] for i in range(len(labels))})
    return grouped


def weighted_variance(binned: pd.DataFrame, min_stations: int = 50) -> tuple[float, int, int]:
    rows: list[tuple[float, int]] = []
    for _, group in binned.groupby("bin_id"):
        n_stations = int(group["gauge_id"].nunique())
        if n_stations < min_stations:
            continue
        var = float(group["beta_median"].var())
        if np.isfinite(var):
            rows.append((var, n_stations))
    if not rows:
        return np.nan, 0, 0
    variances = np.array([item[0] for item in rows], dtype=float)
    weights = np.array([item[1] for item in rows], dtype=float)
    return float(np.average(variances, weights=weights)), len(rows), int(weights.sum())


def metric_for_axis(curves: pd.DataFrame, method: str, axis_col: str, description: str) -> dict[str, float | int | str]:
    cols = ["gauge_id", "frequency_cpd", "beta", axis_col]
    work = curves[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[(work["frequency_cpd"] > 0) & (work[axis_col] > 0)]
    raw = station_bin_medians(work, "frequency_cpd")
    alt = station_bin_medians(work, axis_col)
    raw_var, raw_bins, raw_values = weighted_variance(raw)
    alt_var, alt_bins, alt_values = weighted_variance(alt)
    return {
        "method": method,
        "description": description,
        "n_gauges": int(work["gauge_id"].nunique()),
        "raw_weighted_variance_matched": raw_var,
        "method_weighted_variance": alt_var,
        "variance_reduction_vs_matched_raw": 1 - alt_var / raw_var if np.isfinite(raw_var) and raw_var > 0 else np.nan,
        "raw_bins": raw_bins,
        "method_bins": alt_bins,
        "raw_station_bin_values": raw_values,
        "method_station_bin_values": alt_values,
    }


def rank_proxy_tau(attrs: pd.DataFrame, proxy_col: str, descending: bool = False) -> pd.Series:
    work = attrs[["gauge_id", "tau_acf_days", proxy_col]].dropna().copy()
    work = work[(work["tau_acf_days"] > 0) & np.isfinite(work[proxy_col])]
    sorted_tau = np.sort(work["tau_acf_days"].to_numpy(dtype=float))
    order = work.sort_values(proxy_col, ascending=not descending).index
    assigned = pd.Series(index=work.index, dtype=float)
    assigned.loc[order] = sorted_tau
    return pd.Series(assigned.to_numpy(), index=work["gauge_id"].astype(str).str.zfill(8))


def attach_axis(curves: pd.DataFrame, tau_map: pd.Series, axis_name: str) -> pd.DataFrame:
    out = curves.copy()
    out[axis_name] = out["frequency_cpd"] * out["gauge_id"].map(tau_map)
    return out


def run_random_tau_null(curves: pd.DataFrame, attrs: pd.DataFrame, n_draws: int = 300, seed: int = 20260531) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tau = attrs.dropna(subset=["tau_acf_days"]).copy()
    tau = tau[tau["tau_acf_days"] > 0]
    gauges = tau["gauge_id"].astype(str).str.zfill(8).to_numpy()
    tau_values = tau["tau_acf_days"].to_numpy(dtype=float)
    base = curves[curves["gauge_id"].isin(gauges)].copy()
    rows = []
    for draw in range(n_draws):
        shuffled = rng.permutation(tau_values)
        tau_map = pd.Series(shuffled, index=gauges)
        work = attach_axis(base, tau_map, "de_random_tau")
        row = metric_for_axis(work, "random_tau", "de_random_tau", "randomly shuffled tau_acf assignments")
        row["draw"] = draw
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, random_null: pd.DataFrame) -> None:
    plot_order = [
        "tau_acf",
        "tau_recession_matched",
        "bfi_rank_tau",
        "area_rank_tau",
        "constant_tau",
    ]
    plot = metrics[metrics["method"].isin(plot_order)].copy()
    plot["reduction_pct"] = 100 * plot["variance_reduction_vs_matched_raw"]
    random_vals = 100 * random_null["variance_reduction_vs_matched_raw"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    labels = {
        "tau_acf": "tau_acf",
        "tau_recession_matched": "tau_recession",
        "bfi_rank_tau": "BFI rank",
        "area_rank_tau": "area rank",
        "constant_tau": "constant tau",
    }
    colors = {
        "tau_acf": "#1f6f78",
        "tau_recession_matched": "#4a8f5a",
        "bfi_rank_tau": "#8f6bb8",
        "area_rank_tau": "#c7892d",
        "constant_tau": "#7a7a7a",
    }
    x = np.arange(len(plot_order))
    values = [float(plot.loc[plot["method"] == m, "reduction_pct"].iloc[0]) for m in plot_order]
    axes[0].bar(x, values, color=[colors[m] for m in plot_order], alpha=0.9)
    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[m] for m in plot_order], rotation=30, ha="right")
    axes[0].set_ylabel("dispersion reduction [%]")
    axes[0].set_title("Alternative normalizations", loc="left", fontsize=9)
    axes[0].grid(True, axis="y", color="#e5e2dc", lw=0.6)

    axes[1].hist(random_vals, bins=28, color="#c4513f", alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[1].axvline(values[0], color="#1f6f78", lw=1.8, label="tau_acf")
    axes[1].axvline(np.nanpercentile(random_vals, 95), color="#333333", ls="--", lw=1.0, label="random 95%")
    axes[1].set_xlabel("random tau reduction [%]")
    axes[1].set_ylabel("draws")
    axes[1].set_title("Random tau null", loc="left", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    out = FIGURES / "nature_extended_alternative_normalizations"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write_report(metrics: pd.DataFrame, random_null: pd.DataFrame) -> None:
    m = metrics.set_index("method")
    random_reductions = random_null["variance_reduction_vs_matched_raw"].dropna().to_numpy(dtype=float)
    random_p = float((np.sum(random_reductions >= float(m.loc["tau_acf", "variance_reduction_vs_matched_raw"])) + 1) / (len(random_reductions) + 1))
    lines = [
        "# Alternative Normalization Comparison",
        "",
        "Date: 2026-05-31",
        "",
        "## Result",
        "",
        f"- tau_acf normalization reduction: {float(m.loc['tau_acf', 'variance_reduction_vs_matched_raw']):.1%} (n={int(m.loc['tau_acf', 'n_gauges'])}).",
        f"- tau_recession normalization on its matched subset: {float(m.loc['tau_recession_matched', 'variance_reduction_vs_matched_raw']):.1%} (n={int(m.loc['tau_recession_matched', 'n_gauges'])}).",
        f"- tau_acf on the same recession subset: {float(m.loc['tau_acf_recession_subset', 'variance_reduction_vs_matched_raw']):.1%}.",
        f"- BFI-rank proxy reduction: {float(m.loc['bfi_rank_tau', 'variance_reduction_vs_matched_raw']):.1%}.",
        f"- drainage-area-rank proxy reduction: {float(m.loc['area_rank_tau', 'variance_reduction_vs_matched_raw']):.1%}.",
        f"- constant-tau reduction: {float(m.loc['constant_tau', 'variance_reduction_vs_matched_raw']):.1%}.",
        f"- random tau null median reduction: {np.nanmedian(random_reductions):.1%}; 95th percentile: {np.nanpercentile(random_reductions, 95):.1%}.",
        f"- Empirical probability that random tau >= tau_acf reduction: {random_p:.4f}.",
        "",
        "## Interpretation",
        "",
        "This is an independent-reviewer control. If tau_acf beats random tau and drainage-area rank, the central claim is not merely an arbitrary x-axis rescaling or the old area-memory intuition. Similar performance by tau_recession is supportive cross-validation. Partial performance by BFI-rank is mechanism-consistent but remains observational because BFI is streamflow-derived.",
        "",
        "## Outputs",
        "",
        "- `reports/tables/camels_673_alternative_normalization_metrics.csv`",
        "- `reports/tables/camels_673_random_tau_normalization_null.csv`",
        "- `reports/figures/nature_extended_alternative_normalizations.png`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


def main() -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(CURVES_CSV)
    curves["gauge_id"] = curves["gauge_id"].astype(str).str.zfill(8)
    attrs = pd.read_csv(ATTR_CSV)
    attrs["gauge_id"] = attrs["gauge_id"].astype(str).str.zfill(8)

    rows = []
    rows.append(metric_for_axis(curves, "tau_acf", "de", "measured positive-integral autocorrelation memory"))
    constant = curves.copy()
    constant["de_constant_tau"] = constant["frequency_cpd"] * float(attrs["tau_acf_days"].median())
    rows.append(metric_for_axis(constant, "constant_tau", "de_constant_tau", "constant median tau; should match raw frequency up to scaling"))

    rec_tau = attrs.dropna(subset=["tau_recession_days"]).copy()
    rec_tau = rec_tau[rec_tau["tau_recession_days"] > 0]
    rec_map = rec_tau.set_index("gauge_id")["tau_recession_days"]
    rec_curves = attach_axis(curves[curves["gauge_id"].isin(rec_map.index)].copy(), rec_map, "de_tau_recession")
    rows.append(metric_for_axis(rec_curves, "tau_recession_matched", "de_tau_recession", "independent recession-time memory proxy"))
    rows.append(metric_for_axis(rec_curves, "tau_acf_recession_subset", "de", "tau_acf on the tau_recession-valid subset"))

    area_map = rank_proxy_tau(attrs, "drainage_area_km2")
    area_curves = attach_axis(curves[curves["gauge_id"].isin(area_map.index)].copy(), area_map, "de_area_rank_tau")
    rows.append(metric_for_axis(area_curves, "area_rank_tau", "de_area_rank_tau", "rank-preserving area proxy with tau_acf distribution"))

    bfi_map = rank_proxy_tau(attrs, "baseflow_index")
    bfi_curves = attach_axis(curves[curves["gauge_id"].isin(bfi_map.index)].copy(), bfi_map, "de_bfi_rank_tau")
    rows.append(metric_for_axis(bfi_curves, "bfi_rank_tau", "de_bfi_rank_tau", "rank-preserving BFI proxy with tau_acf distribution"))

    metrics = pd.DataFrame(rows)
    random_null = run_random_tau_null(curves, attrs, n_draws=300)
    metrics.to_csv(OUT_CSV, index=False)
    random_null.to_csv(SHUFFLE_CSV, index=False)
    plot_metrics(metrics, random_null)
    write_report(metrics, random_null)


if __name__ == "__main__":
    main()
