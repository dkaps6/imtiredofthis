# scripts/utils/logging_utils.py
"""
Structured logging helper used across the pipeline.
Writes both to console and optional logfile.
"""

import logging
from pathlib import Path

def setup_logger(name="engine", log_file="logs/daily/run.log", level=logging.INFO):
    """Create a named logger that outputs to both console and file."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate handlers if reimported
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_section(logger, section):
    """Pretty section header in logs."""
    sep = "=" * 30
    logger.info(f"\n{sep}\n>>> {section}\n{sep}\n")
