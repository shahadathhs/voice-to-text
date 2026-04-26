"""Logging configuration for Voice-to-Text application."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _logger

from app.core.config import settings


def setup_logging() -> None:
    """Setup application logging."""
    # Remove default handler
    _logger.remove()

    # Console handler with formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    _logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level,
        colorize=True,
    )

    # File handler if specified
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        _logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=settings.log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )

    def handle_exception(exc: Exception) -> None:
        """Handle uncaught exceptions."""
        _logger.exception(f"Uncaught exception: {exc}")

    sys.excepthook = handle_exception

    _logger.info(f"Logging initialized - Level: {settings.log_level}")


# Initialize logging on import
setup_logging()

# Export logger
logger = _logger
