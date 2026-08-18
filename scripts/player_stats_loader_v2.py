#!/usr/bin/env python3
"""Compatibility loader for nflverse weekly player statistics.

The nflreadpy API maps the old nfl_data_py.import_weekly_data() behavior to
load_player_stats(). Newer nflreadpy releases no longer accept stat_type;
weekly data is requested with summary_level="week" when that keyword exists.
"""
from __future__ import annotations

import inspect
from typing import Any

import pandas as pd


def _to_pandas(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _validate_weekly(df: pd.DataFrame, season: int, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError(f"{source} returned 0 rows for season={season}")
    out = _to_pandas(df).copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {"season", "week"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"{source} weekly stats missing required columns {sorted(missing)}")
    season_num = pd.to_numeric(out["season"], errors="coerce")
    week_num = pd.to_numeric(out["week"], errors="coerce")
    out = out.loc[season_num.eq(int(season)) & week_num.notna()].copy()
    if out.empty:
        raise RuntimeError(f"{source} returned no usable weekly rows after filtering season={season}")
    out["season"] = int(season)
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)


def _load_nflreadpy(season: int) -> pd.DataFrame:
    import nflreadpy as nflv

    loader = nflv.load_player_stats
    params = inspect.signature(loader).parameters
    kwargs: dict[str, Any] = {"seasons": [int(season)]}
    if "summary_level" in params:
        kwargs["summary_level"] = "week"
    # Deliberately do not pass stat_type. It was removed/deprecated by nflverse.
    raw = loader(**kwargs)
    return _validate_weekly(_to_pandas(raw), season, "nflreadpy")


def _load_nfl_data_py(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    if not hasattr(nfl, "import_weekly_data"):
        raise RuntimeError("nfl_data_py has no import_weekly_data")
    raw = nfl.import_weekly_data([int(season)], downcast=True)
    return _validate_weekly(_to_pandas(raw), season, "nfl_data_py")


def load_weekly_player_stats(season: int) -> pd.DataFrame:
    """Load one season of weekly player stats, preferring nflreadpy."""
    errors: list[str] = []
    for label, loader in (("nflreadpy", _load_nflreadpy), ("nfl_data_py", _load_nfl_data_py)):
        try:
            df = loader(int(season))
            print(
                f"[player_stats_loader_v2] source={label} season={season} "
                f"rows={len(df)} weeks={df['week'].nunique()} cols={len(df.columns)}"
            )
            return df
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError(
        f"Unable to load weekly player stats for {season}: {' | '.join(errors)}"
    )
