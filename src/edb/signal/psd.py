"""Power spectral density estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import lombscargle, welch


def _numeric_time(t: ArrayLike) -> np.ndarray:
    values = pd.Series(t)
    if pd.api.types.is_datetime64_any_dtype(values):
        return (pd.to_datetime(values) - pd.to_datetime(values).min()).dt.total_seconds().to_numpy(float)
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def _clean_pair(t: ArrayLike, x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    time = _numeric_time(t)
    values = np.asarray(x, dtype=float)
    if time.shape != values.shape:
        raise ValueError("t and x must have the same shape")
    finite = np.isfinite(time) & np.isfinite(values)
    return time[finite], values[finite]


def _infer_fs(time: np.ndarray) -> float:
    if time.size < 2:
        raise ValueError("at least two time points are required to infer fs")
    dt = np.diff(np.sort(time))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("time values must contain at least two distinct finite points")
    return 1.0 / float(np.median(dt))


def welch_psd(
    t: ArrayLike,
    x: ArrayLike,
    fs: float | None = None,
    nperseg: int | None = None,
) -> pd.DataFrame:
    """Estimate PSD for a regular time series with Welch's method."""

    time, values = _clean_pair(t, x)
    if values.size < 4:
        raise ValueError("at least four finite observations are required")

    sample_rate = fs if fs is not None else _infer_fs(time)
    if sample_rate <= 0:
        raise ValueError("fs must be positive")

    segment = nperseg if nperseg is not None else min(256, values.size)
    frequencies, power = welch(
        values - np.nanmean(values),
        fs=sample_rate,
        nperseg=segment,
        detrend="constant",
        scaling="density",
    )
    keep = (frequencies > 0) & np.isfinite(power) & (power > 0)
    return pd.DataFrame({"frequency": frequencies[keep], "psd": power[keep]})


def lomb_scargle_psd(
    t: ArrayLike,
    x: ArrayLike,
    min_freq: float,
    max_freq: float,
    n_freqs: int,
) -> pd.DataFrame:
    """Estimate PSD for an irregular time series with Lomb-Scargle."""

    if min_freq <= 0 or max_freq <= min_freq:
        raise ValueError("require 0 < min_freq < max_freq")
    if n_freqs < 2:
        raise ValueError("n_freqs must be at least 2")

    time, values = _clean_pair(t, x)
    if values.size < 4:
        raise ValueError("at least four finite observations are required")

    time = time - np.nanmin(time)
    frequencies = np.geomspace(min_freq, max_freq, n_freqs)
    angular = 2.0 * np.pi * frequencies
    centered = values - np.nanmean(values)
    power = lombscargle(time, centered, angular, normalize=True)
    keep = np.isfinite(power) & (power > 0)
    return pd.DataFrame({"frequency": frequencies[keep], "psd": power[keep]})


def multitaper_psd(*args: object, **kwargs: object) -> pd.DataFrame:
    """Reserved API for a validated multitaper estimator."""

    raise NotImplementedError("multitaper PSD is not implemented in this release")
