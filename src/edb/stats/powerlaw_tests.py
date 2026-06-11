"""Power-law testing extension points."""

from __future__ import annotations


def fit_powerlaw_distribution(*args: object, **kwargs: object) -> None:
    """Future distributional power-law tests are not part of this release."""

    raise NotImplementedError("distributional power-law tests are planned for event-size analyses")
