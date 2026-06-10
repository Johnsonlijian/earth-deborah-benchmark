from edb.data.itslive import summarize_stac_items


def test_summarize_stac_items() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "item-1",
                "collection": "itslive-cubes",
                "bbox": [-1, -1, 1, 1],
                "properties": {"datetime": "2020-01-01T00:00:00Z"},
                "assets": {"zarr": {"href": "s3://example"}},
                "links": [{"rel": "self", "href": "https://example.test/item-1"}],
            }
        ],
    }

    out = summarize_stac_items(payload)

    assert out.shape[0] == 1
    assert out.loc[0, "item_id"] == "item-1"
    assert out.loc[0, "asset_keys"] == "zarr"
