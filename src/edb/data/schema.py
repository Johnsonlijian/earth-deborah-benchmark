"""Common data schema for benchmark activity series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class GeoActivitySeries:
    """Standard metadata wrapper for a single geophysical activity series."""

    process: str
    region: str
    variable: str
    time_unit: str
    value_unit: str
    tau_seconds: float | None
    tau_source: str | None
    dataframe: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"time", "value"}
        missing = required.difference(self.dataframe.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"dataframe is missing required columns: {missing_cols}")
        if self.tau_seconds is not None and self.tau_seconds <= 0:
            raise ValueError("tau_seconds must be positive when provided")


def to_standard_dataframe(series: GeoActivitySeries) -> pd.DataFrame:
    """Return a tidy dataframe with metadata columns attached."""

    out = series.dataframe.loc[:, ["time", "value"]].copy()
    out["time"] = pd.to_datetime(out["time"])
    out["process"] = series.process
    out["region"] = series.region
    out["variable"] = series.variable
    return out
