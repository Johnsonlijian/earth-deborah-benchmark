"""Small summary helpers for benchmark diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_beta_window(
    beta_de: pd.DataFrame,
    de_min: float = 0.5,
    de_max: float = 2.0,
) -> dict[str, float | int]:
    """Summarize beta values overall and inside a target De window."""

    if de_min <= 0 or de_max <= de_min:
        raise ValueError("require 0 < de_min < de_max")
    required = {"beta", "de"}
    missing = required.difference(beta_de.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"beta_de is missing required columns: {missing_cols}")

    beta = pd.to_numeric(beta_de["beta"], errors="coerce")
    de = pd.to_numeric(beta_de["de"], errors="coerce")
    valid = beta[np.isfinite(beta)]
    in_window = beta[(de >= de_min) & (de <= de_max) & np.isfinite(beta)]
    return {
        "n_beta_rows": int(valid.shape[0]),
        "mean_beta_all": float(valid.mean()) if not valid.empty else np.nan,
        "median_beta_all": float(valid.median()) if not valid.empty else np.nan,
        "n_beta_de_window": int(in_window.shape[0]),
        "mean_beta_de_window": float(in_window.mean()) if not in_window.empty else np.nan,
        "median_beta_de_window": float(in_window.median()) if not in_window.empty else np.nan,
        "de_window_min": float(de_min),
        "de_window_max": float(de_max),
    }
