"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import cast

import structlog


def configure_logging(stage: str, log_dir: Path | None = None) -> structlog.BoundLogger:
    """Configure structlog with stage-bound logger and pretty output to stderr.

    If ``log_dir`` is provided, ensures the directory exists so future tasks
    can attach a file sink there. The directory is currently unused beyond
    that — file logging will be added in a later task.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    structlog.configure(
        processors=[*processors, structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    log = cast(structlog.BoundLogger, structlog.get_logger().bind(stage=stage))
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    return log
