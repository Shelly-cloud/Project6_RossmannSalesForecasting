"""Project-wide logging (Task 1.2).

Every module calls :func:`get_logger` instead of ``print`` so that a full
audit trail of each pipeline run lands in ``logs/rossmann.log`` as well as the
console. Handlers are attached once per logger name to avoid the duplicate-line
problem you get when a notebook re-imports a module.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import LOGS_DIR

LOG_FILE = LOGS_DIR / "rossmann.log"

_FMT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to both the console and the rotating log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Re-importing a module in a notebook would otherwise stack handlers and
    # print every message two or three times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_dataframe(logger: logging.Logger, df, label: str) -> None:
    """Log the shape, memory footprint and null count of a dataframe."""
    nulls = int(df.isna().sum().sum())
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(
        "%s | shape=%s | nulls=%d | memory=%.1f MB",
        label,
        df.shape,
        nulls,
        mem_mb,
    )
