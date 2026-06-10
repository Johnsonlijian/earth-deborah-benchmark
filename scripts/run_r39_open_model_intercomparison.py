"""R39 open-implementation GR4J/HBV-Edu/LSTM intercomparison for beta(De).

This round replaces the R38 "inspired candidate" model-use case with an
open-implementation model intercomparison:

* RRMPG GR4J, a Python implementation of the daily GR4J model;
* RRMPG HBVEdu, an educational HBV-family model with snow, soil and two runoff
  reservoirs;
* a lightweight global PyTorch LSTM trained only on meteorological forcings and
  the same 70/30 within-gauge split.

The script still avoids overclaiming: this is not an airGR/TUWmodel or
NeuralHydrology operational benchmark, and the calibration budgets are recorded
in the outputs. The intended manuscript claim is narrower: beta(De) exposes a
separate spectral-memory diagnostic axis under named open model classes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import platform
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

import run_r21_hydrological_model_benchmark as r21
import run_r22_model_null_context as r22
import run_r23_multimodel_hydrology_benchmark as r23
from run_camels_gb_replication import ATTR_DIR, DAILY_DIR, gauge_id_from_path

try:
    from rrmpg.models import GR4J, HBVEdu

    HAS_RRMPG = True
except Exception:
    GR4J = None
    HBVEdu = None
    HAS_RRMPG = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    HAS_TORCH = False


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
NOTES = ROOT / "reports" / "manuscript_notes"
LATEX = ROOT
OUT = "r39_open_model_intercomparison"

MODEL_LABELS = {
    "rrmpg_gr4j": "RRMPG GR4J",
    "rrmpg_hbvedu": "RRMPG HBV-Edu",
    "global_lstm": "global LSTM",
    "seasonal_climatology": "seasonal climatology",
    "seasonal_ar1_null": "seasonal AR(1) null",
    "lagged_q_lower_bound": "lagged-Q lower bound",
}

MODEL_ORDER = [
    "rrmpg_gr4j",
    "rrmpg_hbvedu",
    "global_lstm",
    "seasonal_climatology",
    "seasonal_ar1_null",
    "lagged_q_lower_bound",
]

INDEPENDENT_MODELS = [
    "rrmpg_gr4j",
    "rrmpg_hbvedu",
    "global_lstm",
    "seasonal_climatology",
    "seasonal_ar1_null",
]


@dataclass
class GaugeData:
    gauge_id: str
    dates: pd.DatetimeIndex
    precip: np.ndarray
    pet: np.ndarray
    temp: np.ndarray
    month: np.ndarray
    q: np.ndarray
    train_idx: np.ndarray
    eval_idx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Maximum CAMELS-GB gauges; 0 means all.")
    parser.add_argument("--min-eval-years", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--rrmpg-samples", type=int, default=128, help="Latin-hypercube parameter samples per RRMPG model.")
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--lstm-batches", type=int, default=260)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument("--lstm-seq-len", type=int, default=180)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument(
        "--allow-missing-model-deps",
        action="store_true",
        help="Allow downgraded diagnostic runs if RRMPG or PyTorch is missing. Do not use for the formal R39 submission benchmark.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the R39 runtime manifest from existing outputs without recomputing models.",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def package_direct_url(name: str) -> str:
    try:
        dist = metadata.distribution(name)
        return dist.read_text("direct_url.json") or ""
    except Exception:
        return ""


def ensure_required_model_deps(args: argparse.Namespace) -> None:
    missing: list[str] = []
    if not HAS_RRMPG:
        missing.append("RRMPG")
    if not HAS_TORCH and not args.skip_lstm:
        missing.append("PyTorch")
    if missing and not args.allow_missing_model_deps:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"R39 formal GR4J/HBV/LSTM intercomparison requires {joined}. "
            "Install model dependencies with `python -m pip install -e \".[dev,model]\"` "
            "or use `--allow-missing-model-deps` only for explicitly downgraded diagnostics."
        )


def load_gauge(path: Path, min_eval_years: float) -> GaugeData | None:
    usecols = [
        "date",
        "precipitation_cehgear",
        "pet_chess",
        "temperature_chess",
        "discharge_spec",
    ]
    data = pd.read_csv(path, usecols=usecols, na_values=["NaN", "nan", ""])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in usecols[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").set_index("date").asfreq("D")
    data["precipitation_cehgear"] = data["precipitation_cehgear"].interpolate(limit=7).fillna(0.0).clip(lower=0.0)
    data["pet_chess"] = data["pet_chess"].interpolate(limit=7).fillna(0.0).clip(lower=0.0)
    data["temperature_chess"] = data["temperature_chess"].interpolate(limit=30).bfill().ffill()
    data["discharge_spec"] = data["discharge_spec"].where(data["discharge_spec"] >= 0)

    q_series = data["discharge_spec"].astype(float)
    train_idx, eval_idx = r21.train_eval_split(q_series)
    if eval_idx.size < 365 * min_eval_years:
        return None

    return GaugeData(
        gauge_id=gauge_id_from_path(path),
        dates=data.index,
        precip=data["precipitation_cehgear"].to_numpy(dtype=float),
        pet=data["pet_chess"].to_numpy(dtype=float),
        temp=data["temperature_chess"].to_numpy(dtype=float),
        month=data.index.month.to_numpy(dtype=np.int8),
        q=q_series.to_numpy(dtype=float),
        train_idx=train_idx,
        eval_idx=eval_idx,
    )


def lhs_params(model: Any, n: int, seed: int) -> np.ndarray:
    names = model.get_parameter_names()
    bounds = model.get_default_bounds()
    dtype = model.get_dtype()
    samples = np.zeros(n, dtype=dtype)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n)
    for j, name in enumerate(names):
        lo, hi = bounds[name]
        samples[name] = lo + unit[:, j] * (hi - lo)
    return samples


def scale_many(unit: np.ndarray, q_values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float, float, int]:
    unit = np.asarray(unit, dtype=float)
    if unit.ndim == 1:
        unit = unit[:, None]
    train_unit = unit[train_idx, :]
    y = q_values[train_idx]
    mask = np.isfinite(y)
    train_unit = train_unit[mask, :]
    y = y[mask]
    if y.size < 30:
        raise RuntimeError("too few training observations for scaling")
    y_mean = float(np.mean(y))
    denom = float(np.sum((y - y_mean) ** 2))
    if denom <= 0:
        raise RuntimeError("zero training variance")
    x_mean = np.nanmean(train_unit, axis=0)
    x_var = np.nanmean((train_unit - x_mean) ** 2, axis=0)
    cov = np.nanmean((train_unit - x_mean) * (y[:, None] - y_mean), axis=0)
    slopes = np.divide(cov, x_var, out=np.zeros_like(cov), where=x_var > 0)
    slopes = np.clip(slopes, 0.0, None)
    intercepts = np.clip(y_mean - slopes * x_mean, 0.0, None)
    pred_train = np.clip(intercepts[None, :] + slopes[None, :] * train_unit, 0.0, None)
    scores = 1.0 - np.nansum((pred_train - y[:, None]) ** 2, axis=0) / denom
    if not np.isfinite(scores).any():
        raise RuntimeError("all scaled candidates failed")
    best = int(np.nanargmax(scores))
    pred = np.clip(intercepts[best] + slopes[best] * unit[:, best], 0.0, None)
    return pred, float(scores[best]), float(intercepts[best]), float(slopes[best]), best


def monthly_train_means(values: np.ndarray, months: np.ndarray, train_idx: np.ndarray, fallback: float) -> np.ndarray:
    out = np.zeros(12, dtype=float)
    train_months = months[train_idx]
    train_values = values[train_idx]
    for month in range(1, 13):
        vals = train_values[(train_months == month) & np.isfinite(train_values)]
        out[month - 1] = float(np.mean(vals)) if vals.size else fallback
    return out


def fit_rrmpg_models(gauge: GaugeData, rrmpg_samples: int, seed: int) -> list[dict]:
    if not HAS_RRMPG:
        return []
    rows: list[dict] = []

    gr4j = GR4J()
    gr4j_params = lhs_params(gr4j, rrmpg_samples, seed)
    gr4j_unit = gr4j.simulate(gauge.precip, gauge.pet, s_init=0.5, r_init=0.5, params=gr4j_params)
    gr4j_pred, gr4j_score, gr4j_intercept, gr4j_slope, gr4j_best = scale_many(gr4j_unit, gauge.q, gauge.train_idx)
    gr4j_param = gr4j_params[gr4j_best]
    rows.append(
        {
            "model_type": "rrmpg_gr4j",
            "pred": gr4j_pred,
            "train_nse": gr4j_score,
            "calibration": f"latin_hypercube_n={rrmpg_samples};train_nse_scaling",
            "intercept": gr4j_intercept,
            "slope": gr4j_slope,
            "x1": float(gr4j_param["x1"]),
            "x2": float(gr4j_param["x2"]),
            "x3": float(gr4j_param["x3"]),
            "x4": float(gr4j_param["x4"]),
        }
    )

    hbv = HBVEdu()
    hbv_params = lhs_params(hbv, rrmpg_samples, seed + 7919)
    pe_m = monthly_train_means(gauge.pet, gauge.month, gauge.train_idx, float(np.nanmean(gauge.pet)))
    t_m = monthly_train_means(gauge.temp, gauge.month, gauge.train_idx, float(np.nanmean(gauge.temp)))
    hbv_unit = hbv.simulate(
        gauge.temp,
        gauge.precip,
        gauge.month.copy(),
        pe_m,
        t_m,
        snow_init=0.0,
        soil_init=100.0,
        s1_init=0.0,
        s2_init=0.0,
        params=hbv_params,
    )
    hbv_pred, hbv_score, hbv_intercept, hbv_slope, hbv_best = scale_many(hbv_unit, gauge.q, gauge.train_idx)
    hbv_param = hbv_params[hbv_best]
    row = {
        "model_type": "rrmpg_hbvedu",
        "pred": hbv_pred,
        "train_nse": hbv_score,
        "calibration": f"latin_hypercube_n={rrmpg_samples};train_nse_scaling",
        "intercept": hbv_intercept,
        "slope": hbv_slope,
    }
    for name in hbv.get_parameter_names():
        row[name] = float(hbv_param[name])
    rows.append(row)
    return rows


def metric_row(
    gauge: GaugeData,
    model_type: str,
    pred: np.ndarray,
    train_nse: float,
    params: dict[str, Any],
    obs_curve: pd.DataFrame,
    tau_obs: float,
) -> tuple[dict | None, pd.DataFrame | None]:
    eval_dates = gauge.dates[gauge.eval_idx]
    obs = gauge.q[gauge.eval_idx]
    sim = np.asarray(pred, dtype=float)[gauge.eval_idx]
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 365 * 4:
        return None, None
    try:
        tau_pred, pred_curve = r21.beta_curve(eval_dates[mask], sim[mask], model_type)
    except Exception:
        return None, None
    row = {
        "gauge_id": gauge.gauge_id,
        "model_type": model_type,
        "model_label": MODEL_LABELS[model_type],
        "n_eval_years": float(mask.sum() / 365.25),
        "train_nse": float(train_nse) if np.isfinite(train_nse) else np.nan,
        "eval_nse": r21.nse(sim[mask], obs[mask]),
        "eval_kge": r21.kge(sim[mask], obs[mask]),
        "obs_tau_eval_days": float(tau_obs),
        "model_tau_eval_days": float(tau_pred),
        "abs_log10_tau_error": float(abs(np.log10(tau_pred / tau_obs))) if tau_obs > 0 and tau_pred > 0 else np.nan,
        "beta_de_median_abs_distance": r21.curve_distance(obs_curve, pred_curve, "de"),
        "beta_raw_median_abs_distance": r21.curve_distance(obs_curve, pred_curve, "frequency_cpd"),
    }
    for key, value in params.items():
        if key == "pred":
            continue
        if isinstance(value, (int, float, str, np.integer, np.floating)):
            row[f"param_{key}"] = value
    curves = pred_curve.copy()
    curves["gauge_id"] = gauge.gauge_id
    curves["model_type"] = model_type
    curves["tau_days"] = tau_pred
    return row, curves


def evaluate_rrmpg_job(job: tuple[str, int, int, float]) -> tuple[str, list[dict], list[pd.DataFrame], str | None]:
    path_text, seed, rrmpg_samples, min_eval_years = job
    path = Path(path_text)
    gauge = load_gauge(path, min_eval_years)
    if gauge is None:
        return gauge_id_from_path(path), [], [], None
    try:
        tau_obs, obs_curve = r21.beta_curve(gauge.dates[gauge.eval_idx], gauge.q[gauge.eval_idx], "observed")
    except Exception as exc:
        return gauge.gauge_id, [], [], str(exc)
    try:
        fits = fit_rrmpg_models(gauge, rrmpg_samples, seed)
        rows: list[dict] = []
        curves: list[pd.DataFrame] = []
        for fit in fits:
            model_type = fit["model_type"]
            row, curve = metric_row(gauge, model_type, fit["pred"], fit["train_nse"], fit, obs_curve, tau_obs)
            if row is not None and curve is not None:
                rows.append(row)
                curves.append(curve)
        return gauge.gauge_id, rows, curves, None
    except Exception as exc:
        return gauge.gauge_id, [], [], str(exc)


if HAS_TORCH:

    class GlobalLSTM(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y, _ = self.lstm(x)
            return self.head(y).squeeze(-1)


def lstm_normalizers(gauges: list[GaugeData]) -> dict[str, np.ndarray | float]:
    features = []
    targets = []
    for gauge in gauges:
        idx = gauge.train_idx
        features.append(np.column_stack([gauge.precip[idx], gauge.pet[idx], gauge.temp[idx]]))
        q = gauge.q[idx]
        targets.append(np.log1p(q[np.isfinite(q)]))
    x = np.vstack(features)
    y = np.concatenate(targets)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std <= 0] = 1.0
    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y))
    if y_std <= 0:
        y_std = 1.0
    return {"x_mean": mean, "x_std": std, "y_mean": y_mean, "y_std": y_std}


def standardized_features(gauge: GaugeData, norms: dict[str, np.ndarray | float]) -> np.ndarray:
    x = np.column_stack([gauge.precip, gauge.pet, gauge.temp]).astype(np.float32)
    return ((x - norms["x_mean"]) / norms["x_std"]).astype(np.float32)


def sample_lstm_batch(
    gauges: list[GaugeData],
    features: dict[str, np.ndarray],
    norms: dict[str, np.ndarray | float],
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_batch = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
    m_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
    for b in range(batch_size):
        for _ in range(80):
            gauge = gauges[int(rng.integers(0, len(gauges)))]
            if gauge.train_idx.size <= seq_len + 10:
                continue
            max_end = int(gauge.train_idx[-1])
            if max_end <= seq_len:
                continue
            start = int(rng.integers(0, max_end - seq_len + 1))
            stop = start + seq_len
            idx = np.arange(start, stop)
            target = gauge.q[idx]
            train_mask = np.isin(idx, gauge.train_idx, assume_unique=False)
            mask = np.isfinite(target) & train_mask
            if mask.sum() < max(12, seq_len // 12):
                continue
            x_batch[b] = features[gauge.gauge_id][start:stop]
            y = np.zeros(seq_len, dtype=np.float32)
            y[mask] = ((np.log1p(target[mask]) - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
            y_batch[b] = y
            m_batch[b] = mask.astype(np.float32)
            break
    return x_batch, y_batch, m_batch


def train_global_lstm(gauges: list[GaugeData], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if not HAS_TORCH or args.skip_lstm:
        return {}, pd.DataFrame()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    rng = np.random.default_rng(args.seed + 913)
    norms = lstm_normalizers(gauges)
    features = {g.gauge_id: standardized_features(g, norms) for g in gauges}
    model = GlobalLSTM(args.lstm_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    log_rows = []
    model.train()
    for epoch in range(1, args.lstm_epochs + 1):
        losses = []
        for _ in range(args.lstm_batches):
            xb, yb, mb = sample_lstm_batch(gauges, features, norms, args.lstm_batch_size, args.lstm_seq_len, rng)
            x_t = torch.from_numpy(xb)
            y_t = torch.from_numpy(yb)
            m_t = torch.from_numpy(mb)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_t)
            denom = torch.clamp(m_t.sum(), min=1.0)
            loss = (((pred - y_t) ** 2) * m_t).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else np.nan
        log_rows.append({"epoch": epoch, "mean_masked_mse": mean_loss})
        print(f"  LSTM epoch {epoch}/{args.lstm_epochs}: masked MSE={mean_loss:.4f}", flush=True)

    model.eval()
    predictions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for idx, gauge in enumerate(gauges, start=1):
            x = torch.from_numpy(features[gauge.gauge_id][None, :, :])
            out = model(x).squeeze(0).cpu().numpy()
            pred = np.expm1(out * norms["y_std"] + norms["y_mean"])
            predictions[gauge.gauge_id] = np.clip(pred.astype(float), 0.0, None)
            if idx % 100 == 0:
                print(f"  LSTM predicted {idx}/{len(gauges)} gauges", flush=True)
    return predictions, pd.DataFrame(log_rows)


def evaluate_lstm_and_baselines(
    gauges: list[GaugeData],
    lstm_predictions: dict[str, np.ndarray],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    for idx, gauge in enumerate(gauges, start=1):
        try:
            tau_obs, obs_curve = r21.beta_curve(gauge.dates[gauge.eval_idx], gauge.q[gauge.eval_idx], "observed")
        except Exception:
            continue
        fits: list[dict[str, Any]] = []
        if gauge.gauge_id in lstm_predictions:
            pred = lstm_predictions[gauge.gauge_id]
            fits.append(
                {
                    "model_type": "global_lstm",
                    "pred": pred,
                    "train_nse": r21.nse(pred[gauge.train_idx], gauge.q[gauge.train_idx]),
                    "calibration": "global_lstm_same_split_meteorological_forcings_only",
                }
            )
        climatology_pred = r23.seasonal_climatology(pd.Series(gauge.q, index=gauge.dates), gauge.train_idx, gauge.eval_idx)
        climatology_full = np.full_like(gauge.q, np.nan, dtype=float)
        climatology_full[gauge.eval_idx] = climatology_pred
        fits.append(
            {
                "model_type": "seasonal_climatology",
                "pred": climatology_full,
                "train_nse": np.nan,
                "calibration": "train-period daily climatology",
            }
        )
        ar1_pred = r22.seasonal_ar1_null(pd.Series(gauge.q, index=gauge.dates), gauge.train_idx, gauge.eval_idx, rng)
        ar1_full = np.full_like(gauge.q, np.nan, dtype=float)
        ar1_full[gauge.eval_idx] = ar1_pred
        fits.append(
            {
                "model_type": "seasonal_ar1_null",
                "pred": ar1_full,
                "train_nse": np.nan,
                "calibration": "single seasonal AR(1) null draw",
            }
        )
        lag_pred = r22.one_day_persistence(gauge.q, gauge.train_idx, gauge.eval_idx)
        lag_full = np.full_like(gauge.q, np.nan, dtype=float)
        lag_full[gauge.eval_idx] = lag_pred
        fits.append(
            {
                "model_type": "lagged_q_lower_bound",
                "pred": lag_full,
                "train_nse": np.nan,
                "calibration": "observed previous-day held-out discharge",
            }
        )
        for fit in fits:
            row, curve = metric_row(gauge, fit["model_type"], fit["pred"], fit["train_nse"], fit, obs_curve, tau_obs)
            if row is not None and curve is not None:
                rows.append(row)
                curves.append(curve)
        if idx % 100 == 0:
            print(f"  evaluated LSTM/baselines {idx}/{len(gauges)} gauges", flush=True)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


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
    baselines = ["seasonal_ar1_null", "rrmpg_gr4j", "rrmpg_hbvedu", "global_lstm", "lagged_q_lower_bound"]
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
    work = metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)]
    for gauge_id, group in work.groupby("gauge_id"):
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
        "rrmpg_gr4j": "#2F5D8C",
        "rrmpg_hbvedu": "#2A9D8F",
        "global_lstm": "#B83280",
        "seasonal_climatology": "#A78BFA",
        "seasonal_ar1_null": "#8D2F2F",
        "lagged_q_lower_bound": "#6B7280",
    }
    fig = plt.figure(figsize=(7.6, 5.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0])
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
        patch.set_alpha(0.78)
    ax0.set_xlabel("held-out beta(De) distance")
    ax0.set_title("a  Diagnostic stress test: not a model-family ranking", loc="left", fontsize=8.4, fontweight="bold")
    ax0.text(
        0.98,
        0.03,
        "implementation-level\nbenchmark only",
        transform=ax0.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color="#475467",
    )
    ax0.grid(True, axis="x", color="#E4E9EF", linewidth=0.55)

    scatter = metrics[metrics["model_type"].isin(INDEPENDENT_MODELS)].dropna(
        subset=["eval_nse", "beta_de_median_abs_distance"]
    )
    x_lo, x_hi = -1.25, 0.85
    y_lo, y_hi = -0.05, 1.35
    visible = scatter[
        (scatter["eval_nse"].between(x_lo, x_hi))
        & (scatter["beta_de_median_abs_distance"].between(y_lo, y_hi))
    ]
    ax1.scatter(
        visible["eval_nse"],
        visible["beta_de_median_abs_distance"],
        s=5,
        color="#9AA5B1",
        alpha=0.06,
        linewidths=0,
        rasterized=True,
    )
    short = {
        "rrmpg_gr4j": "GR4J",
        "rrmpg_hbvedu": "HBV",
        "global_lstm": "LSTM",
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
            ms=4.4,
            lw=1.0,
            capsize=2.0,
            color=colors[model],
            mec="white",
            mew=0.45,
        )
        ax1.text(xq.loc[0.50] + 0.012, yq.loc[0.50] + 0.012, short[model], color=colors[model], fontsize=6.0)
    clipped = int(len(scatter) - len(visible))
    if clipped:
        ax1.text(
            0.02,
            0.98,
            f"{clipped} points clipped for readability; all gauge-level values in Source Data",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="#475467",
        )
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xlabel("held-out NSE")
    ax1.set_ylabel("held-out beta(De) distance")
    ax1.set_title("b  Hydrograph and memory diagnostics separate", loc="left", fontsize=8.8, fontweight="bold")
    ax1.grid(True, color="#E4E9EF", linewidth=0.55)

    pwork = pairwise[pairwise.get("baseline_type", pd.Series(dtype=str)) == "seasonal_ar1_null"].copy() if not pairwise.empty else pd.DataFrame()
    pwork = pwork[pwork["model_type"].isin([m for m in INDEPENDENT_MODELS if m != "seasonal_ar1_null"])] if not pwork.empty else pwork
    if not pwork.empty:
        pwork = pwork.sort_values("median_delta_beta_de_distance")
        y = np.arange(len(pwork))
        ax2.axvline(0, color="#475467", lw=0.9)
        ax2.barh(y, pwork["median_delta_beta_de_distance"], color=[colors[m] for m in pwork["model_type"]], alpha=0.88)
        ax2.set_yticks(y, [MODEL_LABELS[m] for m in pwork["model_type"]])
        xmax = max(0.08, float(np.nanmax(np.abs(pwork["median_delta_beta_de_distance"]))) * 1.2)
        ax2.set_xlim(-xmax, xmax)
        for yi, row in zip(y, pwork.itertuples(index=False)):
            delta = float(row.median_delta_beta_de_distance)
            xpos = delta + (0.006 if delta >= 0 else -0.006)
            xpos = min(max(xpos, -xmax * 0.96), xmax * 0.96)
            ax2.text(
                xpos,
                yi,
                f"q={row.q_value_bh_by_baseline:.2g}",
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=5.8,
            )
    ax2.set_xlabel("median beta(De) distance delta vs seasonal AR(1)")
    ax2.set_title("c  Memory-null comparison remains a hard test", loc="left", fontsize=8.8, fontweight="bold")
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
        im = ax3.imshow(counts.to_numpy(dtype=float), cmap="magma", aspect="auto")
        ax3.set_xticks(range(len(INDEPENDENT_MODELS)), [MODEL_LABELS[m] for m in INDEPENDENT_MODELS], rotation=45, ha="right")
        ax3.set_yticks(range(len(INDEPENDENT_MODELS)), [MODEL_LABELS[m] for m in INDEPENDENT_MODELS])
        max_count = max(1, int(counts.to_numpy().max()))
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                val = int(counts.iloc[i, j])
                if val:
                    ax3.text(
                        j,
                        i,
                        str(val),
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="white" if val < max_count * 0.45 else "#111827",
                    )
        fig.colorbar(im, ax=ax3, shrink=0.74, pad=0.02, label="gauges")
    ax3.set_xlabel("best beta(De)-distance output")
    ax3.set_ylabel("best NSE output")
    ax3.set_title("d  Best model depends on evaluation axis", loc="left", fontsize=8.8, fontweight="bold")

    for suffix, kwargs in [(".png", {"dpi": 320}), (".svg", {}), (".pdf", {})]:
        fig.savefig(FIGURES / f"{OUT}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_note(summary: pd.DataFrame, pairwise: pd.DataFrame, ranks: pd.DataFrame, args: argparse.Namespace) -> None:
    agreement = float(ranks["best_nse_is_best_beta"].mean()) if not ranks.empty else np.nan
    lines = [
        "# R39 Open-Implementation GR4J/HBV-Edu/LSTM Intercomparison",
        "",
        "R39 replaces the R38 inspired-candidate wording with a named open-implementation benchmark.",
        "",
        "## Provenance and calibration",
        "",
        f"- RRMPG available: {HAS_RRMPG}.",
        f"- PyTorch available: {HAS_TORCH}.",
        f"- RRMPG parameter budget: Latin-hypercube n = {args.rrmpg_samples} per gauge/model, selected by training NSE with non-negative scaling.",
        f"- LSTM: hidden = {args.lstm_hidden}, sequence length = {args.lstm_seq_len}, epochs = {args.lstm_epochs}, batches/epoch = {args.lstm_batches}, batch size = {args.lstm_batch_size}.",
        "- Same within-gauge 70/30 chronological split as R21/R23/R38; LSTM inputs are precipitation, PET and temperature only.",
        "- Boundary: this is not an airGR/TUWmodel/NeuralHydrology operational benchmark and does not claim model-family superiority.",
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
        "## Safe manuscript claim",
        "",
        "The defensible claim is that a named open GR4J/HBV-Edu/LSTM model set makes",
        "the De diagnostic materially more reviewable: hydrograph skill and spectral-memory",
        "skill remain non-identical axes, and the stochastic memory-null remains a",
        "necessary guard against overinterpreting storage-like model behaviour.",
    ]
    (NOTES / f"{OUT}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_manifest(
    args: argparse.Namespace,
    files: list[Path],
    gauges: list[GaugeData],
    summary: pd.DataFrame,
) -> None:
    attr_files = sorted(Path(ATTR_DIR).glob("*.csv")) if Path(ATTR_DIR).exists() else []
    row: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "rrmpg_available": HAS_RRMPG,
        "rrmpg_version": package_version("rrmpg"),
        "rrmpg_direct_url": package_direct_url("rrmpg"),
        "rrmpg_module": getattr(GR4J, "__module__", "") if HAS_RRMPG else "",
        "torch_available": HAS_TORCH,
        "torch_version": getattr(torch, "__version__", "") if HAS_TORCH else "",
        "seed": args.seed,
        "workers": args.workers,
        "min_eval_years": args.min_eval_years,
        "rrmpg_samples": args.rrmpg_samples,
        "lstm_hidden": args.lstm_hidden,
        "lstm_seq_len": args.lstm_seq_len,
        "lstm_epochs": args.lstm_epochs,
        "lstm_batches": args.lstm_batches,
        "lstm_batch_size": args.lstm_batch_size,
        "skip_lstm": args.skip_lstm,
        "allow_missing_model_deps": args.allow_missing_model_deps,
        "hydromet_daily_dir": str(Path(DAILY_DIR)),
        "hydromet_daily_csv_count": len(files),
        "attribute_dir": str(Path(ATTR_DIR)),
        "attribute_csv_count": len(attr_files),
        "loaded_gauge_count": len(gauges),
        "formal_r39_boundary": (
            "Implementation-level RRMPG GR4J/RRMPG HBV-Edu/in-project PyTorch LSTM diagnostic; "
            "not an airGR/TUWmodel/NeuralHydrology operational model ranking."
        ),
    }
    if not summary.empty:
        for _, item in summary.iterrows():
            key = str(item["model_type"])
            row[f"summary_n_gauges_{key}"] = int(item["n_gauges"])
            row[f"summary_median_beta_de_distance_{key}"] = float(item["median_beta_de_distance"])
            row[f"summary_median_eval_nse_{key}"] = float(item["median_eval_nse"])
    pd.DataFrame([row]).to_csv(TABLES / f"{OUT}_run_manifest.csv", index=False)


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
        f"{OUT}_lstm_training_log.csv",
        f"{OUT}_run_manifest.csv",
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

    print(f"R39 files={len(files)} rrmpg={HAS_RRMPG} torch={HAS_TORCH}", flush=True)
    if args.manifest_only:
        gauges = [g for path in files if (g := load_gauge(path, args.min_eval_years)) is not None]
        summary_path = TABLES / f"{OUT}_summary.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        write_run_manifest(args, files, gauges, summary)
        copy_submission_outputs()
        print(f"Wrote {TABLES / f'{OUT}_run_manifest.csv'}", flush=True)
        return

    ensure_required_model_deps(args)
    jobs = [
        (str(path), int(rng_master.integers(0, 2**31 - 1)), args.rrmpg_samples, args.min_eval_years)
        for path in files
    ]

    rrmpg_rows: list[dict] = []
    rrmpg_curves: list[pd.DataFrame] = []
    if HAS_RRMPG:
        if args.workers <= 1:
            iterator = map(evaluate_rrmpg_job, jobs)
            for idx, (gauge_id, rows, curves, error) in enumerate(iterator, start=1):
                if error:
                    print(f"  [{gauge_id}] RRMPG skipped: {error}", flush=True)
                rrmpg_rows.extend(rows)
                rrmpg_curves.extend(curves)
                if idx % 20 == 0:
                    print(f"  RRMPG processed {idx}/{len(files)} files; rows={len(rrmpg_rows)}", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map = {executor.submit(evaluate_rrmpg_job, job): job[0] for job in jobs}
                for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                    gauge_id, rows, curves, error = future.result()
                    if error:
                        print(f"  [{gauge_id}] RRMPG skipped: {error}", flush=True)
                    rrmpg_rows.extend(rows)
                    rrmpg_curves.extend(curves)
                    if idx % 20 == 0:
                        print(f"  RRMPG processed {idx}/{len(files)} files; rows={len(rrmpg_rows)}", flush=True)

    gauges = [g for path in files if (g := load_gauge(path, args.min_eval_years)) is not None]
    print(f"  loaded {len(gauges)} gauges for LSTM/baselines", flush=True)
    lstm_predictions, lstm_log = train_global_lstm(gauges, args)
    if not lstm_log.empty:
        lstm_log.to_csv(TABLES / f"{OUT}_lstm_training_log.csv", index=False)
    extra_metrics, extra_curves = evaluate_lstm_and_baselines(gauges, lstm_predictions, args.seed + 33)

    metrics_parts = []
    curve_parts = []
    if rrmpg_rows:
        metrics_parts.append(pd.DataFrame(rrmpg_rows))
    if not extra_metrics.empty:
        metrics_parts.append(extra_metrics)
    if rrmpg_curves:
        curve_parts.append(pd.concat(rrmpg_curves, ignore_index=True))
    if not extra_curves.empty:
        curve_parts.append(extra_curves)
    if not metrics_parts:
        raise RuntimeError("No usable R39 model-intercomparison rows.")

    metrics = pd.concat(metrics_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    metrics.to_csv(TABLES / f"{OUT}_metrics.csv", index=False)
    if not curves.empty:
        curves.to_csv(TABLES / f"{OUT}_curves.csv", index=False)

    summary = summarize(metrics)
    pairwise = pairwise_tests(metrics)
    ranks = rank_table(metrics)
    summary.to_csv(TABLES / f"{OUT}_summary.csv", index=False)
    pairwise.to_csv(TABLES / f"{OUT}_pairwise_tests.csv", index=False)
    ranks.to_csv(TABLES / f"{OUT}_rank_disagreement.csv", index=False)
    plot_results(metrics, pairwise)
    write_note(summary, pairwise, ranks, args)
    write_run_manifest(args, files, gauges, summary)
    copy_submission_outputs()
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
