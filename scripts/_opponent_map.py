from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

from scripts.utils.canonical_names import canon_team_series as _canon_team_series

# Re-export canon_team_series so downstream callers can rely on the shared
# canonical mapping logic from scripts.utils.canonical_names
canon_team_series = _canon_team_series
LOG = logging.getLogger(__name__)


# ------------------------------------------------------------------
# LEGACY NORMALIZATION SHIM (REQUIRED BY build_team_week_map)
# ------------------------------------------------------------------
_PUNCT = re.compile(r"[.\u2019\u2018'`]")
_WS = re.compile(r"\s+")


def _clean_token(name: str | None) -> str:
    text = "" if name is None else str(name)
    text = text.strip()
    if not text:
        return ""
    text = _PUNCT.sub("", text)
    text = _WS.sub(" ", text)
    return text


def canon_team(name: str | None) -> str:
    """
    Backwards compatible team normalizer.

    Delegate to scripts.utils.canonical_names.canon_team so there is a single
    source of truth for team alias handling.
    """

    from scripts.utils.canonical_names import canon_team as _canon_team  # local import to avoid import cycles

    return _canon_team(name)


def map_normalize_team(x):
    # Kept for backwards compatibility, but prefer normalize_team_series / canon_team_series
    if pd.isna(x):
        return x
    return canon_team(str(x))


def normalize_team_series(s: pd.Series) -> pd.Series:
    """
    Backwards-compatible team-series normalizer.

    Delegate to scripts.utils.canonical_names.canon_team_series so all codepaths
    share the same alias + cleanup rules.
    """

    return _canon_team_series(s)


# ---------------------------------------------------------------------------
# Default path helpers
# ---------------------------------------------------------------------------


def _default_opponent_map_path() -> Path:
    """
    Default location where the opponent map is written by the new build step.

    We intentionally centralize this in one place so both the build and
    pipeline layers stay in sync.
    """

    # This is produced by scripts/build/build_opponent_map_from_props.py.
    # Resolve relative to the repo root (parent of the scripts directory) so
    # this works both locally and in CI without relying on scripts.config.
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "opponent_map_from_props.csv"


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
    "normalize_team",
    "normalize_team_series",
    "attach_opponent",
]


# ---------------------------------------------------------------------------
# Backwards-compatible helper for metrics / player-form pipeline
# ---------------------------------------------------------------------------


def attach_opponent(
    df_players: pd.DataFrame,
    opponent_map: Optional[pd.DataFrame] = None,
    *,
    season_col: str = "season",
    week_col: str = "week",
    team_col: str = "team",
    out_col: str = "opponent",
    schedule_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach an opponent column using a provided map or the default schedule CSV.

    This remains backwards-compatible with earlier callers by falling back to
    ``data/team_week_map.csv`` when an explicit opponent map is not provided.
    """
    if df_players is None or df_players.empty:
        return df_players

    # Load the opponent map from disk when one was not supplied explicitly.
    if opponent_map is None:
        path = Path(schedule_path or Path(DATA_DIR) / "team_week_map.csv")
        if not path.exists():
            LOG.warning(
                "[_opponent_map] attach_opponent schedule file not found at %s; returning original frame",
                path,
            )
            return df_players

        try:
            opponent_map = pd.read_csv(path)
        except Exception as err:  # pragma: no cover - defensive
            LOG.warning(
                "[_opponent_map] attach_opponent failed to read %s: %s", path, err
            )
            return df_players

        needed_base = {"season", "week", "team"}
        opp_col_in_map: Optional[str] = None
        if out_col in opponent_map.columns:
            opp_col_in_map = out_col
        elif "opponent" in opponent_map.columns:
            opp_col_in_map = "opponent"

        if not needed_base.issubset(set(opponent_map.columns)) or opp_col_in_map is None:
            LOG.warning(
                "[_opponent_map] attach_opponent map %s missing required columns %s; returning original frame",
                path,
                needed_base.union({out_col}),
            )
            return df_players

        if opp_col_in_map != out_col:
            opponent_map = opponent_map.rename(columns={opp_col_in_map: out_col})

        # Canonicalize team values and expose a consistent join column.
        opponent_map = opponent_map.copy()
        try:
            opponent_map["team_abbr"] = opponent_map["team"].map(canon_team)
        except NameError:
            opponent_map["team_abbr"] = opponent_map["team"].astype(str).str.upper()

    # Work out a safe join key set.
    candidate_keys = [season_col, week_col, team_col, "team_abbr"]
    join_keys = [
        k for k in candidate_keys if k in df_players.columns and k in opponent_map.columns
    ]

    if not join_keys:
        LOG.warning(
            "[_opponent_map] attach_opponent found no common join keys between players (%s) and opponent map (%s); returning unchanged",
            list(df_players.columns),
            list(opponent_map.columns),
        )
        return df_players

    # Work on a copy so we don’t mutate the caller’s frame.
    players = df_players.copy()

    # If the player frame already has an opponent column, park it under a
    # different name so the merge doesn’t raise a ValueError for overlapping
    # columns. We’ll reconcile after the merge.
    existing_opp_col: Optional[str] = None
    if out_col in players.columns:
        existing_opp_col = f"{out_col}_existing"
        players[existing_opp_col] = players[out_col]
        players = players.drop(columns=[out_col])

    LOG.info(
        "[_opponent_map] attaching opponent info using keys %s (players=%d, opp_map=%d, had_existing_opp=%s)",
        join_keys,
        len(players),
        len(opponent_map),
        bool(existing_opp_col),
    )

    merged = players.merge(
        opponent_map,
        how="left",
        on=join_keys,
        suffixes=("", "_opp"),
    )

    # If the opponent map provided an opponent column, prefer that.
    opp_col_from_map = f"{out_col}_opp"
    if opp_col_from_map in merged.columns:
        merged[out_col] = merged[opp_col_from_map]
        merged = merged.drop(columns=[opp_col_from_map])
    elif "opponent_opp" in merged.columns:
        merged[out_col] = merged["opponent_opp"]
        merged = merged.drop(columns=["opponent_opp"])
    # Otherwise, fall back to any existing opponent we preserved.
    elif existing_opp_col:
        merged[out_col] = merged[existing_opp_col]

    # Clean up the parked column if we created one.
    if existing_opp_col and existing_opp_col in merged.columns:
        merged = merged.drop(columns=[existing_opp_col])

    return merged


# ---------------------------------------------------------------------------
# Compatibility shims for older callers
# ---------------------------------------------------------------------------


def normalize_team(team: str) -> str:
    """
    Normalize a team name or abbreviation to its canonical 3-letter team code.

    This wrapper delegates to `map_normalize_team` so callers can import
    `normalize_team` from this module (as expected by make_player_form.py).

    Examples:
        >>> normalize_team("Niners")
        'SF'
        >>> normalize_team("San Francisco 49ers")
        'SF'
        >>> normalize_team("sf")
        'SF'

    If the input cannot be normalized, the original value is returned.
    """
    try:
        return map_normalize_team(team)
    except Exception:
        return team

