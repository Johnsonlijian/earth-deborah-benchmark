"""Deborah-number transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def deborah_number(freq: ArrayLike, tau: float) -> np.ndarray:
    """Return Deborah number for frequency values and relaxation time."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    frequency = np.asarray(freq, dtype=float)
    return frequency * tau


def beta_vs_de(freq: ArrayLike, beta: ArrayLike, tau: float) -> pd.DataFrame:
    """Combine local slope estimates with Deborah-number coordinates."""

    frequency = np.asarray(freq, dtype=float)
    beta_values = np.asarray(beta, dtype=float)
    if frequency.shape != beta_values.shape:
        raise ValueError("freq and beta must have the same shape")

    de_values = deborah_number(frequency, tau)
    out = pd.DataFrame(
        {
            "frequency": frequency,
            "beta": beta_values,
            "tau": tau,
            "de": de_values,
        }
    )
    out["log10_de"] = np.where(out["de"] > 0, np.log10(out["de"]), np.nan)
    return out
