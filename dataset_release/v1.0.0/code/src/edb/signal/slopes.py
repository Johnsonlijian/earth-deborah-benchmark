"""Local log-log PSD slope estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import linregress


def local_loglog_slope(
    freq: ArrayLike,
    psd: ArrayLike,
    window_decades: float = 0.75,
    min_points: int = 8,
) -> pd.DataFrame:
    """Estimate local PSD slope beta in sliding log-frequency windows."""

    if window_decades <= 0:
        raise ValueError("window_decades must be positive")
    if min_points < 2:
        raise ValueError("min_points must be at least 2")

    frequency = np.asarray(freq, dtype=float)
    power = np.asarray(psd, dtype=float)
    if frequency.shape != power.shape:
        raise ValueError("freq and psd must have the same shape")

    valid = np.isfinite(frequency) & np.isfinite(power) & (frequency > 0) & (power > 0)
    log_f = np.log10(frequency[valid])
    log_p = np.log10(power[valid])
    freq_valid = frequency[valid]

    rows: list[dict[str, float | int]] = []
    half_window = window_decades / 2.0
    for center in log_f:
        in_window = (log_f >= center - half_window) & (log_f <= center + half_window)
        if int(in_window.sum()) < min_points:
            continue

        fit = linregress(log_f[in_window], log_p[in_window])
        r2 = fit.rvalue**2 if np.isfinite(fit.rvalue) else np.nan
        rows.append(
            {
                "f_center": float(10.0**center),
                "beta": float(-fit.slope),
                "intercept": float(fit.intercept),
                "n_points": int(in_window.sum()),
                "r2": float(r2),
                "log10_f_min": float(log_f[in_window].min()),
                "log10_f_max": float(log_f[in_window].max()),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "f_center",
                "beta",
                "intercept",
                "n_points",
                "r2",
                "log10_f_min",
                "log10_f_max",
            ]
        )
    return result.sort_values("f_center").reset_index(drop=True)
