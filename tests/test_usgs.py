import math

import numpy as np
import pandas as pd

from edb.data.usgs import (
    build_comcat_url,
    build_energy_rate_series,
    build_event_rate_series,
    earthquake_energy_joule,
    fetch_comcat_events,
    fetch_comcat_events_paginated,
    merge_comcat_geojson_pages,
    parse_comcat_geojson,
)


MOCK_GEOJSON = {
    "type": "FeatureCollection",
    "metadata": {"count": 2, "status": 200},
    "features": [
        {
            "type": "Feature",
            "id": "us_test_1",
            "properties": {
                "mag": 5.5,
                "place": "Test region 1",
                "time": 946684800000,
                "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us_test_1",
                "magType": "mw",
            },
            "geometry": {"type": "Point", "coordinates": [100.0, 20.0, 10.0]},
        },
        {
            "type": "Feature",
            "id": "us_test_2",
            "properties": {
                "mag": 6.0,
                "place": "Test region 2",
                "time": 946771200000,
                "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us_test_2",
                "magType": "mww",
            },
            "geometry": {"type": "Point", "coordinates": [101.0, 21.0, 11.0]},
        },
    ],
}


class MockResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class MockSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        return MockResponse(MOCK_GEOJSON)


class PagedMockSession:
    def __init__(self):
        self.urls = []
        self.pages = [
            {
                "type": "FeatureCollection",
                "metadata": {"count": 1},
                "features": [MOCK_GEOJSON["features"][0]],
            },
            {
                "type": "FeatureCollection",
                "metadata": {"count": 1},
                "features": [MOCK_GEOJSON["features"][1]],
            },
            {
                "type": "FeatureCollection",
                "metadata": {"count": 0},
                "features": [],
            },
        ]

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        return MockResponse(self.pages.pop(0))


def test_build_comcat_url_contains_expected_parameters() -> None:
    url = build_comcat_url(
        starttime="2000-01-01",
        endtime="2000-01-31",
        minmagnitude=5.5,
        minlatitude=-10,
        maxlatitude=10,
        offset=1,
    )

    assert url.startswith("https://earthquake.usgs.gov/fdsnws/event/1/query?")
    assert "format=geojson" in url
    assert "starttime=2000-01-01" in url
    assert "endtime=2000-01-31" in url
    assert "minmagnitude=5.5" in url
    assert "eventtype=earthquake" in url
    assert "orderby=time-asc" in url
    assert "offset=1" in url


def test_fetch_comcat_events_uses_session_and_returns_json() -> None:
    session = MockSession()

    payload = fetch_comcat_events("2000-01-01", "2000-01-02", 5.5, session=session, timeout=3.0)

    assert payload["metadata"]["count"] == 2
    assert len(session.urls) == 1
    assert session.urls[0][1] == 3.0


def test_merge_comcat_geojson_pages_deduplicates_event_ids() -> None:
    page_one = {"metadata": {}, "features": [MOCK_GEOJSON["features"][0]]}
    page_two = {"metadata": {}, "features": [MOCK_GEOJSON["features"][0], MOCK_GEOJSON["features"][1]]}

    merged = merge_comcat_geojson_pages([page_one, page_two])

    assert merged["metadata"]["count"] == 2
    assert merged["metadata"]["page_count"] == 2
    assert [feature["id"] for feature in merged["features"]] == ["us_test_1", "us_test_2"]


def test_fetch_comcat_events_paginated_uses_1_based_offsets() -> None:
    session = PagedMockSession()

    payload = fetch_comcat_events_paginated(
        "2000-01-01",
        "2000-01-02",
        5.5,
        page_limit=1,
        max_pages=3,
        session=session,
    )

    assert payload["metadata"]["count"] == 2
    assert "offset=1" in session.urls[0][0]
    assert "offset=2" in session.urls[1][0]
    assert "offset=3" in session.urls[2][0]


def test_parse_comcat_geojson_returns_expected_columns() -> None:
    out = parse_comcat_geojson(MOCK_GEOJSON)

    assert list(out.columns) == [
        "event_id",
        "time",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "mag_type",
        "place",
        "url",
    ]
    assert out.shape[0] == 2
    assert out.loc[0, "event_id"] == "us_test_1"
    assert out.loc[0, "time"] == pd.Timestamp("2000-01-01T00:00:00Z")
    assert out.loc[1, "magnitude"] == 6.0


def test_earthquake_energy_joule_usgs_formula() -> None:
    energy = earthquake_energy_joule(5.5)
    expected = 10 ** (5.24 + 1.44 * 5.5)

    assert math.isclose(energy, expected)

    vector = earthquake_energy_joule(np.array([5.5, 6.0]))
    np.testing.assert_allclose(vector, 10 ** (5.24 + 1.44 * np.array([5.5, 6.0])))


def test_build_event_rate_series_daily() -> None:
    df = parse_comcat_geojson(MOCK_GEOJSON)

    out = build_event_rate_series(df, rule="D")

    assert list(out.columns) == ["time", "event_count", "value"]
    np.testing.assert_allclose(out["event_count"], [1, 1])
    np.testing.assert_allclose(out["value"], [1.0, 1.0])


def test_build_energy_rate_series_daily() -> None:
    df = parse_comcat_geojson(MOCK_GEOJSON)

    out = build_energy_rate_series(df, rule="D")

    assert list(out.columns) == ["time", "energy_joule", "value"]
    assert out.shape[0] == 2
    assert (out["energy_joule"] > 0).all()
