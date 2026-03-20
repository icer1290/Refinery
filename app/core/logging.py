"""Structured logging configuration."""

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging() -> None:
    """Configure structured logging."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Determine if colors should be enabled:
    # 1. FORCE_COLOR=1/true forces colors (for Docker/lazydocker)
    # 2. NO_COLOR=1 disables colors
    # 3. Otherwise, use TTY detection
    force_color = os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes")
    no_color = os.environ.get("NO_COLOR", "").lower() in ("1", "true", "yes")
    use_colors = force_color or (sys.stdout.isatty() and not no_color)

    if use_colors:
        processors.append(structlog.dev.ConsoleRenderer(colors=True, force_colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


# Initialize logging on module import
setup_logging()