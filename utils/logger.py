"""
Centralized logging configuration for NeuroPilots.

All NeuroPilots components should use this logging module instead of
configuring their own logging handlers.

The logger is designed to provide:

    - Consistent log formatting
    - Configurable log level
    - Console logging during development
    - File logging for persistent application diagnostics
    - Reusable component-specific loggers
    - Protection against duplicate handlers

Typical usage:

    from utils.logger import get_logger

    logger = get_logger(__name__)

    logger.info("Dataset analysis started")
    logger.warning("Missing validation metrics")
    logger.error("Experiment execution failed")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config.settings import LOG_LEVEL


# ============================================================================
# CONSTANTS
# ============================================================================

# Root logger name for the entire NeuroPilots application.
LOGGER_NAME = "neuropilots"

# Directory used for persistent application logs.
LOG_DIRECTORY = Path("artifacts") / "logs"

# Default log file.
LOG_FILE = LOG_DIRECTORY / "neuropilots.log"

# Standard log format.
#
# Example:
# 2026-09-07 11:30:15 | INFO | neuropilots.ingestion.dataset_analyzer |
# Dataset analysis started
LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

# Date/time format used in log messages.
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# LOG LEVEL VALIDATION
# ============================================================================

def _get_log_level(level_name: Optional[str]) -> int:
    """
    Convert a textual log level into the corresponding logging constant.

    Parameters
    ----------
    level_name:
        Log level such as ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``.

    Returns
    -------
    int
        Python logging level.

    Notes
    -----
    If an invalid level is supplied, INFO is used as a safe default.
    """

    if not level_name:
        return logging.INFO

    normalized_level = level_name.strip().upper()

    level = getattr(logging, normalized_level, None)

    if not isinstance(level, int):
        return logging.INFO

    return level


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

def configure_logging(
    log_file: Path = LOG_FILE,
    level_name: Optional[str] = LOG_LEVEL,
) -> logging.Logger:
    """
    Configure the central NeuroPilots logger.

    The configuration is intentionally centralized so that all components
    share the same logging behavior.

    Parameters
    ----------
    log_file:
        File where persistent logs are written.

    level_name:
        Logging level, normally obtained from application configuration.

    Returns
    -------
    logging.Logger
        Configured NeuroPilots root logger.

    Notes
    -----
    The function is idempotent. Calling it multiple times will not create
    duplicate handlers.
    """

    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(_get_log_level(level_name))

    # Allow child loggers such as:
    #
    #     neuropilots.ingestion.dataset_analyzer
    #
    # to propagate messages to this central logger.
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    # ------------------------------------------------------------------------
    # Console Handler
    # ------------------------------------------------------------------------

    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_get_log_level(level_name))
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    # ------------------------------------------------------------------------
    # File Handler
    # ------------------------------------------------------------------------

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path.resolve()
        for handler in logger.handlers
    )

    if not has_file_handler:
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        file_handler.setLevel(_get_log_level(level_name))
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


# ============================================================================
# COMPONENT LOGGER
# ============================================================================

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger for a specific NeuroPilots component.

    Parameters
    ----------
    name:
        Component/module name.

        When ``__name__`` is supplied, the logger becomes:

            neuropilots.<module_name>

    Returns
    -------
    logging.Logger
        Configured component logger.

    Examples
    --------
    In ``ingestion/dataset_analyzer.py``:

        from utils.logger import get_logger

        logger = get_logger(__name__)

    This produces a logger similar to:

        neuropilots.ingestion.dataset_analyzer
    """

    root_logger = configure_logging()

    if not name:
        return root_logger

    # Avoid duplicating the root logger name when callers explicitly pass
    # "neuropilots.*".
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)

    return logging.getLogger(f"{LOGGER_NAME}.{name}")


# ============================================================================
# INITIALIZE CENTRAL LOGGER
# ============================================================================

# Configure the logger when this module is imported.
#
# This means any NeuroPilots component can immediately call:
#
#     logger = get_logger(__name__)
#
# without having to configure logging independently.
configure_logging()