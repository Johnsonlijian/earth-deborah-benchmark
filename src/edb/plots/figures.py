"""Figure output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_figure(fig: Any, filename: str | Path, figures_dir: str | Path = "reports/figures", **kwargs: Any) -> Path:
    """Save a matplotlib figure under the reports/figures directory."""

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output_path = figures_path / Path(filename).name
    fig.savefig(output_path, bbox_inches="tight", **kwargs)
    return output_path
