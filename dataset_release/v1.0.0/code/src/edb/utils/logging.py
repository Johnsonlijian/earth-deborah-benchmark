"""Logging configuration."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a package logger with a basic handler when needed."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
