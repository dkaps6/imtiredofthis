from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from scripts.utils.canonical_names import (
    canon_team,
    canon_team_series as _canon_team_series,
)

canon_team_series = _canon_team_series


# ---------------------------------------------------------------------------
# Canonical team name / abbreviation helpers
# ---------------------------------------------------------------------------

# Canonical NFL team abbreviations used across the project.
CANON_TEAM_CODES = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LAR",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
}

# Weird codes observed in APIs/books + identity mapping for determinism.
TEAM_FIX: Dict[str, str] = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "ARZ": "ARI",
    "JAC": "JAX",
    "WSH": "WAS",
    "LA": "LAR",
    "LVG": "LV",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "SDG": "LAC",
}
TEAM_FIX.update({code: code for code in CANON_TEAM_CODES})

# ESPN schedule "city" names.
ESPN_CITY_TO_ABBR: Dict[str, str] = {
    "Washington": "WAS",
    "San Francisco": "SF",
    "Tampa Bay": "TB",
    "New England": "NE",
    "New York Jets": "NYJ",
    "New York Giants": "NYG",
    "Las Vegas": "LV",
    "Los Angeles Rams": "LAR",
    "Los Angeles Chargers": "LAC",
    "Jacksonville": "JAX",
    "Kansas City": "KC",
    "Green Bay": "GB",
    "New Orleans": "NO",
    "Minnesota": "MIN",
    "Cleveland": "CLE",
    "Chicago": "CHI",
    "Detroit": "DET",
    "Atlanta": "ATL",
    "Carolina": "CAR",
    "Buffalo": "BUF",
    "Houston": "HOU",
    "Tennessee": "TEN",
    "Miami": "MIA",
    "Philadelphia": "PHI",
    "Dallas": "DAL",
    "Baltimore": "BAL",
    "Cincinnati": "CIN",
    "Pittsburgh": "PIT",
    "Seattle": "SEA",
    "Arizona": "ARI",
    "Indianapolis": "IND",
    "Denver": "DEN",
    "Los Angeles": "LAR",  # ambiguous city, default to Rams
}

# Full franchise names → canonical abbreviations.
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "Washington Commanders": "WAS",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "New England Patriots": "NE",
    "New York Jets": "NYJ",
    "New York Giants": "NYG",
    "Las Vegas Raiders": "LV",
    "Los Angeles Rams": "LAR",
    "Los Angeles Chargers": "LAC",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Green Bay Packers": "GB",
    "New Orleans Saints": "NO",
    "Minnesota Vikings": "MIN",
    "Cleveland Browns": "CLE",
    "Chicago Bears": "CHI",
    "Detroit Lions": "DET",
    "Atlanta Falcons": "ATL",
    "Carolina Panthers": "CAR",
    "Buffalo Bills": "BUF",
    "Houston Texans": "HOU",
    "Tennessee Titans": "TEN",
    "Miami Dolphins": "MIA",
    "Philadelphia Eagles": "PHI",
    "Dallas Cowboys": "DAL",
    "Baltimore Ravens": "BAL",
    "Cincinnati Bengals": "CIN",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "Arizona Cardinals": "ARI",
    "Indianapolis Colts": "IND",
    "Denver Broncos": "DEN",
    # Legacy / alternate full names seen in data feeds.
    "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LAR",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}

# Common nicknames / shorthand names.
TEAM_NICKNAME_TO_ABBR: Dict[str, str] = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Jags": "JAX",
    "Chiefs": "KC",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Raiders": "LV",
    "Dolphins": "MIA",
    "Fins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Pats": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "Seahawks": "SEA",
    "Hawks": "SEA",
    "49ers": "SF",
    "Niners": "SF",
    "Buccaneers": "TB",
    "Bucs": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
    "Football Team": "WAS",
}

# Additional aliases that appear in feeds/HTML (city abbreviations, two-word combos).
ADDITIONAL_TEAM_ALIASES: Dict[str, str] = {
    "NY JETS": "NYJ",
    "NY GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "NEW YORK GIANTS": "NYG",
    "LA RAMS": "LAR",
    "LA CHARGERS": "LAC",
    "LOS ANGELES RAIDERS": "LV",
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LOS ANGELES RAMS": "LAR",
    "LOS ANGELES CHARGERS": "LAC",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
}


def _build_canon_team_abbr() -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    def _ingest(source: Dict[str, str], *, normalize: bool = True) -> None:
        for raw_key, value in source.items():
            if not raw_key or not value:
                continue
            key = raw_key.upper() if normalize else raw_key
            mapping[key] = value

    for code in CANON_TEAM_CODES:
        mapping[code] = code

    _ingest(TEAM_FIX)
    _ingest({k.upper(): v for k, v in ESPN_CITY_TO_ABBR.items()})
    _ingest({k.upper(): v for k, v in TEAM_NAME_TO_ABBR.items()})
    _ingest({k.upper(): v for k, v in TEAM_NICKNAME_TO_ABBR.items()})
    _ingest(ADDITIONAL_TEAM_ALIASES)

    return mapping


CANON_TEAM_ABBR = _build_canon_team_abbr()
TEAM_REMAP: Dict[str, str] = {key.upper(): val for key, val in CANON_TEAM_ABBR.items()}


__all__ = [
    "TEAM_FIX",
    "TEAM_NAME_TO_ABBR",
    "ESPN_CITY_TO_ABBR",
    "CANON_TEAM_ABBR",
    "canon_team",
    "canon_team_series",
    "CANON_TEAM_CODES",
    "TEAM_REMAP",
    "map_normalize_team",
    "normalize_team_series",
    "attach_opponent",
]


# ---------------------------------------------------------------------------
# Backwards-compatible helper for metrics / player-form pipeline
# ---------------------------------------------------------------------------


def attach_opponent(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    week_col: str = "week",
    team_col: str = "team",
    out_col: str = "opponent",
    schedule_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach an opponent column to `df` using data/team_week_map.csv.

    This is a lightweight, backwards-compatible implementation used by
    make_metrics.py and some legacy scripts. It expects that `df` has
    at least (season, week, team) columns (names configurable via args).

    Parameters
    ----------
    df : DataFrame
        Input frame containing per-team or per-player rows.
    season_col, week_col, team_col : str
        Column names in `df` that identify the game.
    out_col : str
        Name of the opponent column to add/overwrite in `df`.
    schedule_path : str | None
        Optional override path to the schedule CSV. Defaults to
        "data/team_week_map.csv".

    Returns
    -------
    DataFrame
        A copy of `df` with an opponent column merged in when possible.
    """
    if df is None or df.empty:
        return df

    schedule_path = schedule_path or "data/team_week_map.csv"
    path = Path(schedule_path)
    if not path.exists():
        # Keep behaviour non-fatal; callers can still operate without opponent.
        print(
            f"[attach_opponent] WARNING: schedule file not found at {path}; "
            "returning original dataframe."
        )
        return df

    try:
        sched = pd.read_csv(path)
    except Exception as err:
        print(f"[attach_opponent] WARNING: failed to read {path}: {err}")
        return df

    needed = {"season", "week", "team", "opponent"}
    if not needed.issubset(set(sched.columns)):
        print(
            "[attach_opponent] WARNING: team_week_map.csv is missing "
            f"required columns {needed}; returning original dataframe."
        )
        return df

    # Canonicalize team + opponent using the same helper as the props pipeline.
    sched = sched.copy()
    try:
        sched["team_canon"] = sched["team"].map(canon_team)
        sched["opp_canon"] = sched["opponent"].map(canon_team)
    except NameError:
        # Fallback: upper-case if canon_team is unavailable for some reason.
        sched["team_canon"] = sched["team"].astype(str).str.upper()
        sched["opp_canon"] = sched["opponent"].astype(str).str.upper()

    # Work on a copy of df to avoid mutating callers unexpectedly.
    out = df.copy()

    # If the caller uses a different team column name, normalize it to a temp key.
    if team_col not in out.columns:
        # If there is a 'team_abbr' column, fall back to that.
        if "team_abbr" in out.columns:
            out[team_col] = out["team_abbr"]
        else:
            # Nothing to join on; bail out quietly.
            print(
                f"[attach_opponent] WARNING: {team_col!r} column not found "
                "in input; returning original dataframe."
            )
            return df

    # Basic sanity: required columns in df.
    for col in (season_col, week_col, team_col):
        if col not in out.columns:
            print(
                f"[attach_opponent] WARNING: column {col!r} missing from "
                "input; returning original dataframe."
            )
            return df

    # Canonicalize the team column in df for the join.
    try:
        out["_team_canon"] = out[team_col].map(canon_team)
    except NameError:
        out["_team_canon"] = out[team_col].astype(str).str.upper()

    # Build a minimal schedule frame to join on.
    sched_key = sched[["season", "week", "team_canon", "opp_canon"]].drop_duplicates()

    # Perform the left join.
    merged = out.merge(
        sched_key,
        left_on=[season_col, week_col, "_team_canon"],
        right_on=["season", "week", "team_canon"],
        how="left",
    )

    # Fill / rename the opponent column.
    if out_col in merged.columns:
        merged[out_col] = merged[out_col].where(
            merged[out_col].notna(), merged["opp_canon"]
        )
    else:
        merged[out_col] = merged["opp_canon"]

    # Clean up helper columns.
    merged = merged.drop(
        columns=[c for c in ("_team_canon", "team_canon", "opp_canon") if c in merged.columns]
    )

    return merged


# ---------------------------------------------------------------------------
# Compatibility shims for older callers
# ---------------------------------------------------------------------------


def map_normalize_team(x: str | None) -> str | None:
    """
    Backwards-compatible wrapper used by older code that expected
    map_normalize_team in this module.

    It just delegates to scripts.utils.canonical_names.canon_team and
    normalizes empty / unknown values to None.
    """
    if x is None:
        return None
    val = canon_team(str(x))
    return val or None


def normalize_team_series(s: pd.Series) -> pd.Series:
    """
    Vectorised wrapper around map_normalize_team.

    This mirrors the old behaviour that the metrics and player-form
    pipelines expect when they call normalize_team_series from here.
    """
    return s.apply(map_normalize_team)
