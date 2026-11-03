"""Parallel, resumable CAMELS storage-memory null analysis.

This script mirrors `run_camels_full.py` but writes partial rows as stations
finish. It is intended for higher surrogate counts where a single serial run can
take too long and should not lose all progress if interrupted.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from run_camels_full import (
    SECONDS_PER_DAY,
    beta_quick,
    estimate_recession_tau_raw,
    load_camels_attributes,
    load_one_camels_flow,
    prepare_daily_anomaly,
    wy_shuffle,
)
from edb.signal.nulls import ar1_surrogate, phase_randomized_surrogate, seasonal_noise_surrogate
from edb.signal.timescales import integral_autocorrelation_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-dir", default="data/external/camels/usgs_streamflow")
    parser.add_argument("--attr-dir", default="data/external/camels/camels_attributes_v2.0")
    parser.add_argument("--output-prefix", default="camels_673_n30_parallel")
    parser.add_argument("--min-years", type=float, default=20.0)
    parser.add_argument("--max-missing", type=float, default=0.30)
    parser.add_argument("--n-sample", type=int, default=0)
    parser.add_argument("--n-null", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, gauge_id: str, null_name: str, index: int) -> int:
    text = f"{seed}:{gauge_id}:{null_name}:{index}".encode("utf-8")
    return zlib.crc32(text) & 0x7FFFFFFF


def process_one_station(
    filepath: str,
    n_null: int,
    seed: int,
    min_years: float,
    max_missing: float,
) -> dict[str, Any] | None:
    gid, flow_df = load_one_camels_flow(Path(filepath))
    if flow_df.empty or "q" not in flow_df.columns:
        return None
    series = flow_df["q"]
    dates, values = prepare_daily_anomaly(series)
    n_years = len(values) / 365.25
    missing = float(np.isnan(values).mean())
    if n_years < min_years or missing > max_missing:
        return None

    tau_sec = integral_autocorrelation_time(
        values,
        dt_seconds=SECONDS_PER_DAY,
        max_lag=730,
        stop_at_zero=True,
    )
    tau_acf = tau_sec / SECONDS_PER_DAY
    tau_rec = estimate_recession_tau_raw(series)
    try:
        _, obs = beta_quick(dates, values, tau_sec)
    except Exception:
        return None

    nulls: dict[str, Any] = {}
    generators = {
        "seasonal": lambda s: seasonal_noise_surrogate(
            pd.Series(dates),
            values,
            seed=s,
            seasonal_key="dayofyear",
        ),
        "ar1": lambda s: ar1_surrogate(values, seed=s),
        "wy_shuffle": lambda s: wy_shuffle(values, dates, seed=s),
        "phase_rand": lambda s: phase_randomized_surrogate(values, seed=s),
    }
    obs_mean = obs.get("mean_beta_de_window", np.nan)
    for null_name, gen in generators.items():
        samples = []
        for ni in range(n_null):
            try:
                _, ns = beta_quick(dates, gen(stable_seed(seed, gid, null_name, ni)), tau_sec)
                samples.append(ns["mean_beta_de_window"])
            except Exception:
                continue
        arr = np.asarray(samples, dtype=float)
        valid = arr[np.isfinite(arr)]
        nulls[f"{null_name}_n"] = int(valid.size)
        nulls[f"{null_name}_median"] = float(np.median(valid)) if valid.size else np.nan
        nulls[f"{null_name}_p_ge"] = (
            float(np.mean(valid >= obs_mean)) if valid.size and np.isfinite(obs_mean) else np.nan
        )

    return {
        "gauge_id": gid,
        "n_years": n_years,
        "missing_frac": missing,
        "tau_acf_days": tau_acf,
        "tau_recession_days": tau_rec,
        **obs,
        **nulls,
    }


def add_attributes(summary: pd.DataFrame, attr_dir: str) -> pd.DataFrame:
    attrs = load_camels_attributes(attr_dir)
    if attrs.empty or summary.empty:
        return summary
    rows = []
    for _, row in summary.iterrows():
        out = row.to_dict()
        gid = str(row["gauge_id"]).zfill(8)
        if gid in attrs.index:
            for col in [
                "gauge_name",
                "drainage_area_km2",
                "p_mean",
                "pet_mean",
                "aridity",
                "frac_snow",
                "elev_mean",
                "slope_mean",
                "q_mean",
                "runoff_ratio",
                "huc_02",
                "baseflow_index",
                "slope_fdc",
                "stream_elas",
                "hfd_mean",
                "high_q_freq",
                "low_q_freq",
                "zero_q_freq",
                "carbonate_rocks_frac",
                "geol_porostiy",
                "geol_permeability",
            ]:
                if col in attrs.columns:
                    out[col] = attrs.loc[gid, col]
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    flow_dir = Path(args.flow_dir)
    files = sorted(flow_dir.glob("*.txt"))
    if args.n_sample > 0:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, min(args.n_sample, len(files)), replace=False))

    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{args.output_prefix}_partial.csv"
    final_path = out_dir / f"{args.output_prefix}_summary.csv"

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if args.resume and partial_path.exists():
        partial = pd.read_csv(partial_path, dtype={"gauge_id": str})
        rows = partial.to_dict("records")
        done = set(partial["gauge_id"].astype(str).str.zfill(8))
        print(f"Resuming from {len(done)} completed stations")

    todo = [fp for fp in files if fp.stem.zfill(8) not in done]
    print(
        f"Processing {len(todo)} CAMELS files with n_null={args.n_null}, "
        f"workers={args.workers}"
    )

    with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_station,
                str(fp),
                args.n_null,
                args.seed,
                args.min_years,
                args.max_missing,
            ): fp
            for fp in todo
        }
        for idx, future in enumerate(cf.as_completed(futures), start=1):
            fp = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"  failed {fp.stem}: {exc}")
                continue
            if row is not None:
                rows.append(row)
            if idx % 10 == 0 or idx == len(futures):
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                print(f"  completed {idx}/{len(futures)} futures; rows={len(rows)}")

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["gauge_id"] = summary["gauge_id"].astype(str).str.zfill(8)
        summary = summary.sort_values("tau_acf_days")
        summary = add_attributes(summary, args.attr_dir)
    summary.to_csv(final_path, index=False)

    print(f"Saved {final_path}")
    print(f"Rows: {len(summary)}")
    for prefix in ["seasonal", "ar1", "wy_shuffle", "phase_rand"]:
        n_col = f"{prefix}_n"
        p_col = f"{prefix}_p_ge"
        if n_col in summary.columns and p_col in summary.columns:
            valid = summary[n_col].fillna(0) > 0
            n = summary.loc[valid, n_col]
            p = summary.loc[valid, p_col]
            print(
                f"{prefix}: median N={n.median():.0f}; "
                f"p<=1/N count={(p <= 1.0 / n).sum()}; "
                f"strict count={(p == 0).sum()}"
            )
    if "drainage_area_km2" in summary.columns:
        sub = summary.dropna(subset=["tau_acf_days", "drainage_area_km2"])
        if len(sub) >= 10:
            rho, p_value = stats.spearmanr(sub["tau_acf_days"], np.log10(sub["drainage_area_km2"]))
            print(f"tau_acf vs area: rho={rho:.3f}, p={p_value:.4g}")


if __name__ == "__main__":
    main()
