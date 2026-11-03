"""Cross-series summary utilities."""

from __future__ import annotations

import pandas as pd


def summarize_beta_by_process(beta_de_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize beta values by process when process metadata are available."""

    required = {"process", "beta"}
    missing = required.difference(beta_de_df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"beta_de_df is missing required columns: {missing_cols}")
    return beta_de_df.groupby("process", as_index=False)["beta"].agg(["count", "median", "mean", "std"]).reset_index()
