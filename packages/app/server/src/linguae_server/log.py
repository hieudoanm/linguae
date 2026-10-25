"""Logging configuration for the Linguae server."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors and timestamps to log messages."""

    COLORS: ClassVar[dict[str, str]] = {
        "TRACE": "\033[90m",  # Grey
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "FATAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with a timestamp, level, and color."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        levelname = record.levelname

        if levelname == "CRITICAL":
            levelname = "FATAL"

        color = self.COLORS.get(levelname, "")
        message = record.getMessage()

        return f"{color}[{timestamp}] [{levelname}]{self.RESET} {message}"


class TraceLogger(logging.Logger):
    """Logger with TRACE level support."""

    TRACE_LEVEL: ClassVar[int] = 5

    def trace(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log a message at TRACE level."""
        if self.isEnabledFor(self.TRACE_LEVEL):
            self._log(self.TRACE_LEVEL, msg, args, **kwargs)


def get_logger(name: str | None = None) -> TraceLogger:
    """Get or create a logger instance."""
    logging.setLoggerClass(TraceLogger)
    logging.addLevelName(TraceLogger.TRACE_LEVEL, "TRACE")
    logging.addLevelName(logging.CRITICAL, "FATAL")

    logger_instance = logging.getLogger(name)

    if not logger_instance.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger_instance.addHandler(handler)
        logger_instance.setLevel(TraceLogger.TRACE_LEVEL)
        logger_instance.propagate = False

    return logger_instance


logger = get_logger("linguae_server")
