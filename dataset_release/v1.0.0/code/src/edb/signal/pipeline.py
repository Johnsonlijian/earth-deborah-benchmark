"""Composable analysis pipeline for standardized activity series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edb.signal.de import beta_vs_de
from edb.signal.preprocess import remove_linear_trend, remove_seasonal_cycle, resample_time_series, robust_zscore
from edb.signal.psd import lomb_scargle_psd, welch_psd
from edb.signal.slopes import local_loglog_slope


@dataclass(slots=True)
class AnalysisResult:
    """Container for one benchmark pipeline result."""

    processed_series: pd.DataFrame
    psd_df: pd.DataFrame
    beta_df: pd.DataFrame
    beta_de_df: pd.DataFrame


def analyze_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    tau_seconds: float,
    resample_rule: str | None = None,
    remove_seasonality: bool = True,
    psd_method: str = "welch",
) -> AnalysisResult:
    """Run preprocessing, PSD, local slope, and Deborah-number mapping."""

    if tau_seconds <= 0:
        raise ValueError("tau_seconds must be positive")

    work = df.loc[:, [time_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.sort_values(time_col)

    if resample_rule is not None:
        work = resample_time_series(work, time_col, value_col, resample_rule)
        time_col = "time"
        value_col = "value"

    if remove_seasonality:
        work = remove_seasonal_cycle(work, time_col, value_col)

    work[value_col] = remove_linear_trend(work[time_col], work[value_col])
    work[value_col] = robust_zscore(work[value_col])
    processed = work.rename(columns={time_col: "time", value_col: "value"})

    method = psd_method.lower()
    if method == "welch":
        psd_df = welch_psd(processed["time"], processed["value"])
    elif method == "lomb_scargle":
        time_seconds = (processed["time"] - processed["time"].min()).dt.total_seconds()
        max_freq = 0.5 / float(time_seconds.diff().median())
        min_freq = 1.0 / float(time_seconds.max() - time_seconds.min())
        psd_df = lomb_scargle_psd(processed["time"], processed["value"], min_freq, max_freq, 512)
    else:
        raise ValueError(f"unsupported psd_method: {psd_method}")

    beta_df = local_loglog_slope(psd_df["frequency"], psd_df["psd"])
    beta_de_df = beta_vs_de(beta_df["f_center"], beta_df["beta"], tau_seconds)
    return AnalysisResult(processed, psd_df, beta_df, beta_de_df)
