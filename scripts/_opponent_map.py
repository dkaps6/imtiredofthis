from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.config import DATA_DIR
from scripts.utils.canonical_names import (
    canon_team,
    canon_team_series as _canon_team_series,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public canonical helpers
# ---------------------------------------------------------------------------

def canon_team_series(s: pd.Series) -> pd.Series:
    """
    Canonicalize team abbreviations using the shared canonical_names helper.

    This is the function other modules should import:

        from scripts._opponent_map import canon_team_series

    We also re-export `canon_team` below for backwards compatibility with
    callers such as make_team_week_map.py.
    """

    return _canon_team_series(s)


# ---------------------------------------------------------------------------
# Opponent-map I/O
# ---------------------------------------------------------------------------

def _default_opponent_map_path() -> Path:
    """
    Default location where the opponent map is written by the build step.

    This is produced by scripts/build/build_opponent_map_from_props.py.
    """

    return DATA_DIR / "opponent_map_from_props.csv"


def load_opponent_map(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the opponent map from disk.

    If the file does not exist or is empty, return an empty DataFrame and log
    a warning instead of raising. This keeps downstream steps from crashing.
    """

    if path is None:
        path = _default_opponent_map_path()

    path = Path(path)

    if not path.exists():
        LOG.warning(
            "[_opponent_map] opponent map file not found at %s; "
            "returning empty DataFrame",
            path,
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error(
            "[_opponent_map] failed to read opponent map at %s: %s", path, exc
        )
        return pd.DataFrame()

    if df.empty:
        LOG.warning(
            "[_opponent_map] opponent map at %s is empty; "
            "downstream merges will be no-ops",
            path,
        )

    return df


def build_opponent_map(
    season: int,
    schedule_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Backwards-compatible stub used by older callers.

    The actual *building* of the opponent map now lives in
    scripts/build/build_opponent_map_from_props.py and is executed as its own
    GitHub Actions step.

    For pipeline consumers, we just load whatever the build step already wrote.
    """

    if out_path is None:
        out_path = _default_opponent_map_path()

    df = load_opponent_map(out_path)
    LOG.info(
        "[_opponent_map] build_opponent_map shim returning %d rows for season %s",
        len(df),
        season,
    )
    return df


# ---------------------------------------------------------------------------
# Attach opponent info to per-team / per-player frames
# ---------------------------------------------------------------------------

def attach_opponent(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    week_col: str = "week",
    team_col: str = "team",
    out_col: str = "opponent",
    schedule_path: Optional[str] = None,  # deprecated, kept for compat
    opponent_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Backwards-compatible helper to attach opponent info to a per-team/per-player
    DataFrame.

    Expected opponent_map schema (from build_opponent_map_from_props step):
        season, week, team_abbr, opponent

    We:
      1. Ensure df has a team column (or fall back to 'team_abbr')
      2. Rename df columns to ('season', 'week', 'team_abbr') for the join
      3. Merge opponent_map
      4. Rename columns back and ensure the caller sees `out_col` for opponent
    """

    if df is None or df.empty:
        return df

    # Load the pre-built opponent map if the caller didn't supply one.
    if opponent_map is None:
        opponent_map = load_opponent_map()

    if opponent_map is None or opponent_map.empty:
        LOG.warning(
            "[_opponent_map] attach_opponent called with empty opponent map; "
            "returning dataframe unchanged"
        )
        return df

    df_work = df.copy()

    # Ensure we have a team column to join on.
    if team_col not in df_work.columns:
        if "team_abbr" in df_work.columns:
            df_work[team_col] = df_work["team_abbr"]
        else:
            LOG.warning(
                "[_opponent_map] attach_opponent: cannot find team_col %r "
                "or 'team_abbr' in df; returning unchanged",
                team_col,
            )
            return df

    # Sanity check required columns in df.
    for col in (season_col, week_col, team_col):
        if col not in df_work.columns:
            LOG.warning(
                "[_opponent_map] attach_opponent: missing column %r in df; "
                "returning unchanged",
                col,
            )
            return df

    # Rename df columns to match the opponent_map join keys.
    rename_map: dict[str, str] = {}
    if season_col != "season":
        rename_map[season_col] = "season"
    if week_col != "week":
        rename_map[week_col] = "week"
    if team_col != "team_abbr":
        rename_map[team_col] = "team_abbr"

    df_for_join = df_work.rename(columns=rename_map)

    LOG.info(
        "[_opponent_map] attaching opponent using keys ['season', 'week', 'team_abbr'] "
        "(df rows=%d, opp_map rows=%d)",
        len(df_for_join),
        len(opponent_map),
    )

    merged = df_for_join.merge(
        opponent_map,
        how="left",
        on=["season", "week", "team_abbr"],
        suffixes=("", "_opp"),
    )

    # Restore original column names if we renamed them for the join.
    inverse_rename = {v: k for k, v in rename_map.items()}
    merged = merged.rename(columns=inverse_rename)

    # Ensure callers see the opponent under out_col.
    if "opponent" in merged.columns:
        if out_col in merged.columns:
            merged[out_col] = merged[out_col].where(
                merged[out_col].notna(), merged["opponent"]
            )
        else:
            merged[out_col] = merged["opponent"]

    return merged


__all__ = [
    "canon_team",
    "canon_team_series",
    "load_opponent_map",
    "build_opponent_map",
    "attach_opponent",
]
