"""Characteristic timescale estimators for activity time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def autocorrelation(x: ArrayLike, max_lag: int | None = None) -> np.ndarray:
    """Estimate normalized autocorrelation for lags ``0..max_lag``."""

    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("at least three finite observations are required")
    values = values - np.mean(values)
    variance = float(np.dot(values, values))
    if variance <= 0:
        return np.ones(1 if max_lag is None else max_lag + 1)

    lag_max = values.size - 1 if max_lag is None else min(int(max_lag), values.size - 1)
    if lag_max < 0:
        raise ValueError("max_lag must be non-negative")
    acf = np.empty(lag_max + 1, dtype=float)
    for lag in range(lag_max + 1):
        acf[lag] = float(np.dot(values[: values.size - lag], values[lag:]) / variance)
    return acf


def integral_autocorrelation_time(
    x: ArrayLike,
    dt_seconds: float,
    max_lag: int | None = None,
    stop_at_zero: bool = True,
) -> float:
    """Estimate integral autocorrelation time in seconds.

    The estimate integrates the positive autocorrelation sequence. It is a
    pragmatic first-pass memory scale for Deborah normalization, not a unique
    physical relaxation constant.
    """

    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    acf = autocorrelation(x, max_lag=max_lag)
    if stop_at_zero:
        non_positive = np.flatnonzero(acf[1:] <= 0)
        if non_positive.size:
            acf = acf[: non_positive[0] + 1]
    positive = acf[np.isfinite(acf) & (acf > 0)]
    if positive.size == 0:
        return float(dt_seconds)
    return float(np.trapezoid(positive, dx=dt_seconds))
