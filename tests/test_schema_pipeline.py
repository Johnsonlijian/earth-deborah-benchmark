import numpy as np
import pandas as pd

from edb.data.schema import GeoActivitySeries, to_standard_dataframe
from edb.signal.pipeline import analyze_series


def test_geo_activity_series_to_standard_dataframe() -> None:
    series = GeoActivitySeries(
        process="test_process",
        region="test_region",
        variable="test_variable",
        time_unit="day",
        value_unit="unit",
        tau_seconds=86400.0,
        tau_source="synthetic",
        dataframe=pd.DataFrame({"time": pd.date_range("2020-01-01", periods=3), "value": [1, 2, 3]}),
    )

    out = to_standard_dataframe(series)

    assert list(out.columns) == ["time", "value", "process", "region", "variable"]
    assert out["process"].eq("test_process").all()


def test_analyze_series_returns_expected_frames() -> None:
    n = 512
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = np.sin(2 * np.pi * np.arange(n) / 32.0) + 0.1 * np.random.default_rng(3).normal(size=n)
    df = pd.DataFrame({"time": dates, "value": x})

    result = analyze_series(
        df,
        time_col="time",
        value_col="value",
        tau_seconds=86400.0 * 30,
        remove_seasonality=False,
        psd_method="welch",
    )

    assert not result.processed_series.empty
    assert not result.psd_df.empty
    assert not result.beta_df.empty
    assert not result.beta_de_df.empty
