"""Backward-compatible nflverse PBP accessors.

New code should import ``get_pbp`` from ``scripts.utils.pbp``.  The historical
``get_pbp_2025`` name remains only so old callers do not crash; it now resolves
the active runtime season instead of silently forcing 2025.
"""
from __future__ import annotations

import os

import pandas as pd

from scripts.utils.pbp import get_pbp as _get_pbp


def get_pbp(season: int, min_rows: int = 0) -> pd.DataFrame:
    return _get_pbp(int(season), min_rows=min_rows)


def get_pbp_2025(min_rows: int = 80000) -> pd.DataFrame:
    """Deprecated compatibility alias; loads the active ``SEASON``.

    Keeping the old function name avoids breaking legacy imports while removing
    the dangerous behavior that always fetched 2025 data during a 2026 run.
    """
    season = int(os.getenv("SEASON", "2026"))
    return _get_pbp(season, min_rows=min_rows)
