"""Preprocessing utilities for geophysical time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _to_numeric_time(t: ArrayLike) -> np.ndarray:
    values = pd.Series(t)
    if pd.api.types.is_datetime64_any_dtype(values):
        seconds = (pd.to_datetime(values) - pd.to_datetime(values).min()).dt.total_seconds()
        return seconds.to_numpy(dtype=float)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def robust_zscore(x: ArrayLike) -> np.ndarray:
    """Return a median/MAD-based z-score."""

    values = np.asarray(x, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out

    center = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - center))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(values[finite])
    if not np.isfinite(scale) or scale == 0:
        out[finite] = 0.0
        return out

    out[finite] = (values[finite] - center) / scale
    return out


def remove_linear_trend(t: ArrayLike, x: ArrayLike) -> np.ndarray:
    """Remove an ordinary least-squares linear trend from finite observations."""

    time = _to_numeric_time(t)
    values = np.asarray(x, dtype=float)
    if time.shape != values.shape:
        raise ValueError("t and x must have the same shape")

    out = values.astype(float, copy=True)
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() < 2:
        return out

    t_centered = time[finite] - np.nanmean(time[finite])
    slope, intercept = np.polyfit(t_centered, values[finite], deg=1)
    trend = slope * (time - np.nanmean(time[finite])) + intercept
    out[finite] = values[finite] - trend[finite]
    return out


def remove_seasonal_cycle(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    freq: str = "D",
) -> pd.DataFrame:
    """Remove a climatological seasonal cycle from a time series dataframe."""

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values(time_col)

    freq_upper = freq.upper()
    if freq_upper.startswith("M"):
        group_key = out[time_col].dt.month
    elif freq_upper.startswith("W"):
        group_key = out[time_col].dt.isocalendar().week.astype(int)
    else:
        group_key = out[time_col].dt.dayofyear

    seasonal = out.groupby(group_key)[value_col].transform("median")
    out["seasonal_cycle"] = seasonal
    out[value_col] = out[value_col] - seasonal
    return out


def resample_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    rule: str,
) -> pd.DataFrame:
    """Resample a dataframe to a regular time grid using the mean value."""

    if not rule:
        raise ValueError("rule must be a non-empty pandas offset alias")

    series = df.loc[:, [time_col, value_col]].copy()
    series[time_col] = pd.to_datetime(series[time_col])
    resampled = (
        series.sort_values(time_col)
        .set_index(time_col)[value_col]
        .resample(rule)
        .mean()
        .rename("value")
        .reset_index()
        .rename(columns={time_col: "time"})
    )
    return resampled
