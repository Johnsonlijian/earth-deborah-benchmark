"""Small file-system helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
