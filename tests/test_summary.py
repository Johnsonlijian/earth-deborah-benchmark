import numpy as np
import pandas as pd

from edb.signal.summary import summarize_beta_window


def test_summarize_beta_window() -> None:
    df = pd.DataFrame({"de": [0.1, 0.6, 1.0, 3.0], "beta": [0.2, 1.5, 1.7, 0.4]})

    out = summarize_beta_window(df, de_min=0.5, de_max=2.0)

    assert out["n_beta_rows"] == 4
    assert out["n_beta_de_window"] == 2
    assert np.isclose(out["mean_beta_de_window"], 1.6)
