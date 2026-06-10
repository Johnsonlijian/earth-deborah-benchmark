"""Bootstrap utilities."""

from __future__ import annotations

import numpy as np


def bootstrap_indices(n: int, n_boot: int, seed: int | None = None) -> np.ndarray:
    """Return simple bootstrap sample indices."""

    if n <= 0:
        raise ValueError("n must be positive")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))
