"""Logging utilities for ASR training."""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file.

    Args:
        name:     Logger name (usually ``__name__``).
        log_file: Optional path to write log output to.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
