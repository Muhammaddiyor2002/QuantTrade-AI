"""Structured logging utility with Rich console output."""

from __future__ import annotations

import logging
import os
from logging import Logger

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("QT_LOG_LEVEL") or "INFO").upper()
    handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    """Return a configured logger."""
    configure_logging()
    return logging.getLogger(name)
