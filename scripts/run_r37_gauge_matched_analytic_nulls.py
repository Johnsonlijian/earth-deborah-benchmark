"""R37 full-archive gauge-matched analytic stochastic-memory nulls.

R36 added process-family simulations but left a heavy stochastic
gauge-matched multiscale ensemble as a compute-expensive follow-up. This script
closes the reviewer-facing evidence gap with deterministic analytic nulls on
the full CAMELS-US and CAMELS-GB beta-curve supports.

For every observed gauge, the script uses that gauge's measured tau_acf and
frequency grid, computes analytic PSD families, re-estimates local beta with
the manuscript's sliding log-frequency window, maps beta to De=tau_acf*f, and
applies the same raw-frequency versus De dispersion metric. The output is a
full-archive artifact-boundary table, not a fitted catchment model.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edb.signal.de import beta_vs_de
from edb.signal.slopes import local_loglog_slope
from run_r20_gauge_matched_surrogate_ladder import archive_dispersion


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX_SOURCE = ROOT / "source_data"
SUPP_FIGURES = ROOT / "figures"
SECONDS_PER_DAY = 86_400.0
DEFAULT_PREFIX = "r37_gauge_matched_analytic_nulls"

ARCHIVE_CURVES = {
    "CAMELS-US": "camels_673_beta_curve_points.csv",
    "CAMELS-GB": "camels_gb_v2_replication_beta_curve_points.csv",
}
MIN_UNITS = {"CAMELS-US": 50, "CAMELS-GB": 45}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--arfima-d", nargs="+", type=float, default=[0.15, 0.28, 0.40])
    return parser.parse_args()


def ar1_psd_cpd(frequency_cpd: np.ndarray, tau_days: float) -> np.ndarray:
    phi = float(np.exp(-1.0 / max(float(tau_days), 0.2)))
    phi = float(np.clip(phi, -0.995, 0.995))
    omega = 2.0 * np.pi * np.clip(frequency_cpd, 1e-8, 0.499999)
    return 1.0 / np.maximum(1.0 + phi * phi - 2.0 * phi * np.cos(omega), 1e-12)


def two_timescale_psd_cpd(frequency_cpd: np.ndarray, tau_days: float) -> np.ndarray:
    fast_tau = max(1.0, float(tau_days) / 6.0)
    slow_tau = min(730.0, max(float(tau_days) * 6.0, fast_tau + 1.0))
    fast_weight = (float(tau_days) - slow_tau) / (fast_tau - slow_tau)
    fast_weight = float(np.clip(fast_weight, 0.25, 0.90))
    return fast_weight * ar1_psd_cpd(frequency_cpd, fast_tau) + (1.0 - fast_weight) * ar1_psd_cpd(
        frequency_cpd, slow_tau
    )


def arfima_psd_cpd(frequency_cpd: np.ndarray, tau_days: float, d_value: float) -> np.ndarray:
    d_value = float(np.clip(d_value, 0.02, 0.45))
    omega_half = np.sin(np.pi * np.clip(frequency_cpd, 1e-8, 0.499999))
    long_memory = np.maximum(2.0 * omega_half, 1e-8) ** (-2.0 * d_value)
    return ar1_psd_cpd(frequency_cpd, tau_days) * long_memory


def safe_archive_dispersion(curves: pd.DataFrame, min_units: int) -> dict[str, float | int]:
    metrics = archive_dispersion(curves, min_units=min_units)
    raw_var = float(metrics["raw_axis_weighted_variance"])
    de_var = float(metrics["de_axis_weighted_variance"])
    if not np.isfinite(raw_var) or raw_var <= 1e-12:
        metrics["reduction_vs_raw"] = np.nan
        metrics["reduction_note"] = "undefined_raw_axis_variance"
    else:
        metrics["reduction_vs_raw"] = 1.0 - de_var / raw_var
        metrics["reduction_note"] = "defined"
    return metrics


def load_observed_curves(archive: str) -> pd.DataFrame:
    path = TABLES / ARCHIVE_CURVES[archive]
    data = pd.read_csv(path)
    data["gauge_id"] = data["gauge_id"].astype(str).str.zfill(8) if archive == "CAMELS-US" else data["gauge_id"].astype(str)
    required = ["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]
    data = data[required].replace([np.inf, -np.inf], np.nan).dropna()
    return data[(data["tau_acf_days"] > 0) & (data["frequency"] > 0) & (data["frequency_cpd"] > 0)].copy()


def analytic_curve_for_gauge(gauge: pd.DataFrame, family: str, d_value: float | None = None) -> pd.DataFrame:
    tau_days = float(gauge["tau_acf_days"].iloc[0])
    frequency_cpd = gauge["frequency_cpd"].to_numpy(dtype=float)
    frequency = gauge["frequency"].to_numpy(dtype=float)
    if family == "analytic_ar1":
        psd = ar1_psd_cpd(frequency_cpd, tau_days)
    elif family == "analytic_two_timescale_ar":
        psd = two_timescale_psd_cpd(frequency_cpd, tau_days)
    elif family == "analytic_arfima":
        if d_value is None:
            raise ValueError("d_value required for analytic_arfima")
        psd = arfima_psd_cpd(frequency_cpd, tau_days, d_value)
    else:
        raise ValueError(f"unknown family: {family}")
    beta = local_loglog_slope(frequency, psd, window_decades=0.9, min_points=8)
    curve = beta_vs_de(beta["f_center"], beta["beta"], tau=tau_days * SECONDS_PER_DAY)
    curve["frequency_cpd"] = curve["frequency"] * SECONDS_PER_DAY
    curve["tau_acf_days"] = tau_days
    curve["gauge_id"] = str(gauge["gauge_id"].iloc[0])
    return curve[["gauge_id", "tau_acf_days", "frequency", "frequency_cpd", "de", "beta"]]


def build_family_curves(observed: pd.DataFrame, family: str, d_value: float | None = None) -> pd.DataFrame:
    rows = []
    failures = []
    for _, gauge in observed.groupby("gauge_id", sort=False):
        try:
            rows.append(analytic_curve_for_gauge(gauge, family, d_value=d_value))
        except Exception as exc:
            failures.append((str(gauge["gauge_id"].iloc[0]), repr(exc)))
    if failures:
        preview = "; ".join(f"{gid}: {err}" for gid, err in failures[:5])
        raise RuntimeError(f"{family} failed for {len(failures)} gauges; first failures: {preview}")
    if not rows:
        raise RuntimeError(f"no analytic curves for {family}")
    out = pd.concat(rows, ignore_index=True)
    expected_units = int(observed["gauge_id"].nunique())
    observed_units = int(out["gauge_id"].nunique())
    if observed_units != expected_units:
        raise RuntimeError(f"{family} retained {observed_units}/{expected_units} gauges")
    if family == "analytic_arfima":
        out["family"] = f"analytic_arfima_d{d_value:.2f}".replace(".", "p")
        out["d_value"] = d_value
    else:
        out["family"] = family
        out["d_value"] = np.nan
    return out


def summarize_archive(archive: str, observed: pd.DataFrame, analytic_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_metrics = safe_archive_dispersion(observed, min_units=MIN_UNITS[archive])
    rows = [
        {
            "archive": archive,
            "family": "observed",
            "d_value": np.nan,
            "observed_reduction": obs_metrics["reduction_vs_raw"],
            "null_reduction": np.nan,
            "observed_minus_null": np.nan,
            **{f"observed_{k}": v for k, v in obs_metrics.items()},
        }
    ]
    all_curves = []
    for curves in analytic_frames:
        family = str(curves["family"].iloc[0])
        d_value = float(curves["d_value"].dropna().iloc[0]) if curves["d_value"].notna().any() else np.nan
        metrics = safe_archive_dispersion(curves, min_units=MIN_UNITS[archive])
        rows.append(
            {
                "archive": archive,
                "family": family,
                "d_value": d_value,
                "observed_reduction": obs_metrics["reduction_vs_raw"],
                "null_reduction": metrics["reduction_vs_raw"],
                "observed_minus_null": obs_metrics["reduction_vs_raw"] - metrics["reduction_vs_raw"],
                **{f"null_{k}": v for k, v in metrics.items()},
            }
        )
        curves = curves.copy()
        curves["archive"] = archive
        all_curves.append(curves)
    return pd.DataFrame(rows), pd.concat(all_curves, ignore_index=True)


def binned_median_curve(curves: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
    work = curves.replace([np.inf, -np.inf], np.nan).dropna(subset=["de", "beta"])
    work = work[(work["de"] > 0) & np.isfinite(work["de"]) & np.isfinite(work["beta"])].copy()
    x = np.log10(work["de"].to_numpy(dtype=float))
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    work["bin_id"] = pd.cut(np.log10(work["de"]), edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bin_id"])
    work["bin_id"] = work["bin_id"].astype(int)
    out = (
        work.groupby("bin_id", as_index=False)
        .agg(de=("de", "median"), beta=("beta", "median"), n=("beta", "size"))
        .sort_values("de")
    )
    out["axis_value"] = out["bin_id"].map({idx: 10 ** labels[idx] for idx in range(len(labels))})
    return out


def plot_summary(summary: pd.DataFrame, curves: pd.DataFrame, prefix: str) -> None:
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
    labels = {
        "analytic_ar1": "analytic AR(1)",
        "analytic_two_timescale_ar": "analytic two-timescale AR",
        "analytic_arfima_d0p15": "ARFIMA d=0.15",
        "analytic_arfima_d0p28": "ARFIMA d=0.28",
        "analytic_arfima_d0p40": "ARFIMA d=0.40",
    }
    colors = {
        "observed": "#2E6FAD",
        "analytic_ar1": "#C4513F",
        "analytic_two_timescale_ar": "#2F855A",
        "analytic_arfima_d0p15": "#8E75D8",
        "analytic_arfima_d0p28": "#6C5CE7",
        "analytic_arfima_d0p40": "#4C3FB5",
    }
    order = ["analytic_ar1", "analytic_two_timescale_ar", "analytic_arfima_d0p15", "analytic_arfima_d0p28", "analytic_arfima_d0p40"]
    archives = [a for a in ["CAMELS-US", "CAMELS-GB"] if a in set(summary["archive"])]
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, len(archives), height_ratios=[0.95, 1.15])
    top_axes = [fig.add_subplot(gs[0, idx]) for idx in range(len(archives))]
    bottom_axes = [fig.add_subplot(gs[1, idx]) for idx in range(len(archives))]
    for ax, archive in zip(top_axes, archives):
        data = summary[(summary["archive"] == archive) & (summary["family"] != "observed")].copy()
        data["family"] = pd.Categorical(data["family"], categories=order, ordered=True)
        data = data.sort_values("family")
        obs = float(data["observed_reduction"].iloc[0])
        vals = 100.0 * data["null_reduction"].to_numpy(dtype=float)
        x = np.arange(len(data))
        ax.bar(x, vals, color=[colors[str(f)] for f in data["family"]], alpha=0.78, width=0.66)
        ax.axhline(100.0 * obs, color=colors["observed"], lw=1.1, ls="--")
        ax.text(len(data) - 0.15, 100.0 * obs, "observed", ha="right", va="bottom", fontsize=6, color=colors["observed"])
        for xx, yy in zip(x, vals):
            ax.text(xx, yy + (1.2 if yy >= 0 else -2.2), f"{yy:.1f}%", ha="center", va="center", fontsize=5.8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[str(f)] for f in data["family"]], rotation=28, ha="right", fontsize=5.9)
        ax.set_title(archive, loc="left", fontweight="bold")
        ax.set_ylabel("De-axis dispersion reduction [%]")
        ax.grid(True, axis="y", color="#E4E9EF", lw=0.55)
    for ax, archive in zip(bottom_axes, archives):
        for family in order:
            data = curves[(curves["archive"] == archive) & (curves["family"] == family)]
            if data.empty:
                continue
            med = binned_median_curve(data)
            ax.semilogx(med["de"], med["beta"], lw=1.2, color=colors[family], label=labels[family])
        ax.axvspan(0.5, 2.0, color="#C4513F", alpha=0.06, lw=0)
        ax.set_xlabel(r"$De=\tau_{acf}f$")
        ax.set_ylabel("median analytic local beta")
        ax.set_title("Gauge-matched analytic beta(De) curves", loc="left")
        ax.grid(True, color="#E4E9EF", lw=0.55)
    bottom_axes[0].legend(ncol=1, fontsize=5.8, loc="best")
    for suffix in [".png", ".svg", ".pdf"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{prefix}{suffix}", **kwargs)
    plt.close(fig)


def write_note(prefix: str, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# R37 Gauge-Matched Analytic Nulls",
        "",
        "Date: 2026-06-07",
        "",
        "## Purpose",
        "",
        "This analysis closes the R36 deferred full-archive multiscale null gap",
        "with deterministic analytic nulls evaluated on every CAMELS-US and",
        "CAMELS-GB gauge's measured tau_acf and frequency support.",
        "",
        "## Aggregate summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Claim boundary",
        "",
        "The analytic nulls are artifact-boundary calculations, not fitted",
        "hydrological models. If they exceed observed De reductions, the safe",
        "claim is bounded diagnostic utility rather than excess-over-null storage",
        "mechanism evidence.",
        "",
        "## Outputs",
        "",
        f"- `reports/tables/{prefix}_summary.csv`",
        f"- `reports/tables/{prefix}_curves.csv`",
        f"- `reports/figures/{prefix}.png/svg/pdf`",
    ]
    (NOTES / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_outputs(prefix: str) -> None:
    LATEX_SOURCE.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ["summary.csv", "curves.csv"]:
        src = TABLES / f"{prefix}_{suffix}"
        shutil.copy2(src, LATEX_SOURCE / src.name)
    for ext in [".pdf", ".svg", ".png"]:
        src = FIGURES / f"{prefix}{ext}"
        shutil.copy2(src, SUPP_FIGURES / f"supp_fig9_gauge_matched_analytic_nulls{ext}")


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    curve_frames = []
    for archive in ARCHIVE_CURVES:
        observed = load_observed_curves(archive)
        analytic_frames = [
            build_family_curves(observed, "analytic_ar1"),
            build_family_curves(observed, "analytic_two_timescale_ar"),
        ]
        for d_value in args.arfima_d:
            analytic_frames.append(build_family_curves(observed, "analytic_arfima", d_value=d_value))
        summary, curves = summarize_archive(archive, observed, analytic_frames)
        summary_frames.append(summary)
        curve_frames.append(curves)

    summary = pd.concat(summary_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    summary.to_csv(TABLES / f"{args.prefix}_summary.csv", index=False)
    curves.to_csv(TABLES / f"{args.prefix}_curves.csv", index=False)
    plot_summary(summary, curves, args.prefix)
    write_note(args.prefix, summary, args)
    copy_outputs(args.prefix)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
