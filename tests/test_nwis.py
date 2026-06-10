import pandas as pd

from edb.data.nwis import CFS_TO_M3S, build_nwis_daily_values_url, parse_nwis_daily_values_json


MOCK_NWIS_JSON = {
    "value": {
        "timeSeries": [
            {
                "sourceInfo": {
                    "siteName": "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
                    "siteCode": [{"value": "01646500"}],
                },
                "variable": {"variableName": "Streamflow, ft3/s", "unit": {"unitCode": "ft3/s"}},
                "values": [
                    {
                        "value": [
                            {"value": "1000", "dateTime": "1980-01-01T00:00:00.000", "qualifiers": ["A"]},
                            {"value": "2000", "dateTime": "1980-01-02T00:00:00.000", "qualifiers": ["A"]},
                        ]
                    }
                ],
            }
        ]
    }
}


def test_build_nwis_daily_values_url() -> None:
    url = build_nwis_daily_values_url(["01646500", "09380000"], "1980-01-01", "1980-01-31")

    assert "sites=01646500%2C09380000" in url
    assert "parameterCd=00060" in url
    assert "statCd=00003" in url


def test_parse_nwis_daily_values_json() -> None:
    out = parse_nwis_daily_values_json(MOCK_NWIS_JSON)

    assert out.shape[0] == 2
    assert out.loc[0, "station_id"] == "01646500"
    assert out.loc[0, "country_code"] == "US"
    pd.testing.assert_series_equal(
        out["discharge_m3s"].reset_index(drop=True),
        pd.Series([1000 * CFS_TO_M3S, 2000 * CFS_TO_M3S], name="discharge_m3s"),
    )

