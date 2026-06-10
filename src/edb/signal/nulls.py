"""Null and surrogate time-series generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _finite_values(x: ArrayLike) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("x must contain at least one finite value")
    return values


def _match_mean_std(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    finite_target = target[np.isfinite(target)]
    target_mean = float(np.nanmean(finite_target))
    target_std = float(np.nanstd(finite_target))
    if target_std == 0 or not np.isfinite(target_std):
        return np.full(values.shape, target_mean)
    values_std = float(np.nanstd(values))
    if values_std == 0 or not np.isfinite(values_std):
        return np.full(values.shape, target_mean)
    return (values - np.nanmean(values)) / values_std * target_std + target_mean


def white_noise_surrogate(x: ArrayLike, seed: int | None = None) -> np.ndarray:
    """Generate Gaussian white noise with the input mean and standard deviation."""

    target = _finite_values(x)
    rng = np.random.default_rng(seed)
    finite = target[np.isfinite(target)]
    return rng.normal(float(np.nanmean(finite)), float(np.nanstd(finite)), size=target.shape)


def ar1_surrogate(x: ArrayLike, seed: int | None = None) -> np.ndarray:
    """Generate an AR(1) surrogate estimated from the input series."""

    target = _finite_values(x)
    finite = target[np.isfinite(target)]
    if finite.size < 3:
        raise ValueError("at least three finite observations are required")

    centered = finite - np.nanmean(finite)
    denom = float(np.dot(centered[:-1], centered[:-1]))
    phi = float(np.dot(centered[:-1], centered[1:]) / denom) if denom > 0 else 0.0
    phi = float(np.clip(phi, -0.99, 0.99))
    residual = centered[1:] - phi * centered[:-1]
    sigma = float(np.nanstd(residual))
    if sigma == 0 or not np.isfinite(sigma):
        sigma = float(np.nanstd(centered))

    rng = np.random.default_rng(seed)
    out = np.empty(target.size, dtype=float)
    out[0] = rng.normal(0.0, sigma)
    for idx in range(1, target.size):
        out[idx] = phi * out[idx - 1] + rng.normal(0.0, sigma)
    return _match_mean_std(out, target)


def phase_randomized_surrogate(x: ArrayLike, seed: int | None = None) -> np.ndarray:
    """Generate a phase-randomized surrogate preserving the amplitude spectrum."""

    target = _finite_values(x)
    if target.size < 4:
        raise ValueError("at least four observations are required")

    filled = pd.Series(target).interpolate(limit_direction="both").to_numpy(float)
    centered = filled - np.nanmean(filled)
    spectrum = np.fft.rfft(centered)

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.size)
    phases[0] = 0.0
    if target.size % 2 == 0:
        phases[-1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    out = np.fft.irfft(randomized, n=target.size)
    return _match_mean_std(out, target)


def seasonal_noise_surrogate(
    dates: ArrayLike,
    x: ArrayLike,
    seed: int | None = None,
    seasonal_key: str = "dayofyear",
) -> np.ndarray:
    """Generate a surrogate from seasonal medians plus resampled residuals."""

    target = _finite_values(x)
    date_index = pd.to_datetime(pd.Series(dates))
    if date_index.shape[0] != target.shape[0]:
        raise ValueError("dates and x must have the same length")

    frame = pd.DataFrame({"date": date_index, "value": target})
    if seasonal_key == "dayofyear":
        key = frame["date"].dt.dayofyear
    elif seasonal_key == "month":
        key = frame["date"].dt.month
    else:
        raise ValueError("seasonal_key must be 'dayofyear' or 'month'")
    seasonal = frame.groupby(key)["value"].transform("median").to_numpy(float)
    residual = target - seasonal
    residual_pool = residual[np.isfinite(residual)]
    if residual_pool.size == 0:
        residual_pool = np.array([0.0])

    rng = np.random.default_rng(seed)
    sampled = rng.choice(residual_pool, size=target.size, replace=True)
    return seasonal + sampled
