"""Utilities for fetching nflverse play-by-play data with robust fallbacks."""

from __future__ import annotations

from typing import List

import pandas as pd


def _load_pbp_with_nfl_data_py(seasons: List[int]) -> pd.DataFrame:
    # Try both import names across versions
    try:
        from nfl_data_py import import_pbp
        df = import_pbp(seasons)
    except Exception:
        try:
            from nfl_data_py import import_pbp_data
            df = import_pbp_data(seasons)
        except Exception as e:
            raise RuntimeError(f"nfl_data_py unavailable: {e}")
    if df is None or len(df) == 0:
        raise RuntimeError("nfl_data_py returned empty PBP")
    return df


def _load_pbp_with_nflreadpy(seasons: List[int]) -> pd.DataFrame:
    # nflreadpy API differences accounted for
    try:
        from nflreadpy import load_pbp
        df = load_pbp(seasons)
    except Exception:
        try:
            from nflreadpy import import_pbp
            df = import_pbp(season=seasons[0] if len(seasons) == 1 else seasons)
        except Exception as e:
            raise RuntimeError(f"nflreadpy unavailable: {e}")
    # Convert Polars → pandas if needed
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
    except Exception:
        pass
    if df is None or len(df) == 0:
        raise RuntimeError("nflreadpy returned empty PBP")
    return df


def get_pbp_2025(min_rows: int = 80000) -> pd.DataFrame:
    """
    Load 2025 PBP using maintained libraries (multiple fallbacks).
    Enforce a minimum row threshold; raise RuntimeError on failure.
    """

    seasons = [2025]
    errors = []

    try:
        df = _load_pbp_with_nfl_data_py(seasons)
        if len(df) < min_rows:
            raise RuntimeError(
                f"nfl_data_py PBP rows too small: {len(df)} < {min_rows}"
            )
        return df.reset_index(drop=True)
    except Exception as e:
        errors.append(f"nfl_data_py: {e}")

    try:
        df = _load_pbp_with_nflreadpy(seasons)
        if len(df) < min_rows:
            raise RuntimeError(
                f"nflreadpy PBP rows too small: {len(df)} < {min_rows}"
            )
        return df.reset_index(drop=True)
    except Exception as e:
        errors.append(f"nflreadpy: {e}")

    raise RuntimeError(
        "Unable to fetch 2025 PBP via libraries. Errors: " + " | ".join(errors)
    )
