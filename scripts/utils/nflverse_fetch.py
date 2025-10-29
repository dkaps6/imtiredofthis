"""Utilities for fetching nflverse play-by-play data with robust fallbacks."""

from __future__ import annotations

from typing import List

import pandas as pd


def _load_pbp_with_nfl_data_py(seasons: List[int]) -> pd.DataFrame:
    try:
        from nfl_data_py import import_pbp
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(f"nfl_data_py not available: {e}")
    df = import_pbp(seasons)
    if df is None or df.empty:
        raise RuntimeError("nfl_data_py.import_pbp returned empty DataFrame")
    return df


def _load_pbp_with_nflreadpy(seasons: List[int]) -> pd.DataFrame:
    try:
        from nflreadpy import load_pbp
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(f"nflreadpy not available: {e}")
    df = load_pbp(seasons)
    try:
        import polars as pl  # type: ignore

        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
    except Exception:
        pass
    if df is None or len(df) == 0:
        raise RuntimeError("nflreadpy.load_pbp returned empty DataFrame")
    return df


def get_pbp_2025(min_rows: int = 100000) -> pd.DataFrame:
    """Load the 2025 play-by-play data via maintained libraries.

    The loader prioritizes :mod:`nfl_data_py` with :mod:`nflreadpy` as a
    fallback. A minimum row threshold is enforced to avoid propagating partial
    or empty results through downstream builders.
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
