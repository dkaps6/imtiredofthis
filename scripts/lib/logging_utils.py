import logging
import os
import sys


def get_logger(name: str = "fetch_props", level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))
    return logger
