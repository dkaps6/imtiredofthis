# scripts/_opponent_map.py
from __future__ import annotations

from typing import Dict, Union

import pandas as pd

# Merge these into your existing TEAM_FIX; do not delete any keys you already have.
TEAM_FIX: Dict[str, str] = {
    # Normalize odd vendor/team tokens:
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "ARZ": "ARI", "JAC": "JAX", "LA": "LAR", "WSH": "WAS",
    # Identity for standard codes:
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC", "LAC": "LAC", "LAR": "LAR",
    "LV": "LV", "MIA": "MIA", "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG", "NYJ": "NYJ",
    "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "SF": "SF", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
}


def canon_team(x: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """Canonicalize a single team code or a pandas Series of codes."""
    if isinstance(x, pd.Series):
        s = x.astype(str).str.upper()
        return s.map(TEAM_FIX).fillna(s)
    if x is None:
        return x
    s = str(x).upper()
    return TEAM_FIX.get(s, s)


# Back-compat alias expected by some modules:
def _canon_team_series(s: pd.Series) -> pd.Series:  # keep this name for legacy import sites
    return canon_team(s)


# Canonical 2–3 letter team abbreviations used by our model
CANON_TEAM_ABBR: Dict[str, str] = TEAM_FIX

# broader remap for free-form names we see in APIs / books
TEAM_REMAP: Dict[str, str] = {
    # full names → abbr
    "ARIZONA CARDINALS": "ARI", "CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL", "FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL", "RAVENS": "BAL", "BLT": "BAL",
    "BUFFALO BILLS": "BUF", "BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR", "PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI", "BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE", "BROWNS": "CLE", "CLV": "CLE",
    "DALLAS COWBOYS": "DAL", "COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN", "BRONCOS": "DEN",
    "DETROIT LIONS": "DET", "LIONS": "DET",
    "GREEN BAY PACKERS": "GB", "PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "TEXANS": "HOU", "HST": "HOU",
    "INDIANAPOLIS COLTS": "IND", "COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX", "JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC", "CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV", "RAIDERS": "LV", "OAKLAND RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC", "CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR", "RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA", "DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN", "VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE", "PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO", "SAINTS": "NO",
    "NEW YORK GIANTS": "NYG", "GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI", "EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT", "STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF", "49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA", "SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB", "BUCCANEERS": "TB", "TAMPA BAY": "TB",
    "TENNESSEE TITANS": "TEN", "TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS", "COMMANDERS": "WAS",
}


def map_normalize_team(x: str | None) -> str | None:
    if x is None:
        return None
    key = str(x).strip()
    if not key:
        return None
    key_upper = key.upper()
    if key_upper in TEAM_REMAP:
        key_upper = TEAM_REMAP[key_upper]
    canon = canon_team(key_upper)
    if not isinstance(canon, str):
        return None
    return TEAM_FIX.get(canon.upper())


def normalize_team_series(s: pd.Series) -> pd.Series:
    return s.apply(map_normalize_team)


__all__ = [
    "TEAM_FIX",
    "canon_team",
    "_canon_team_series",
    "CANON_TEAM_ABBR",
    "TEAM_REMAP",
    "map_normalize_team",
    "normalize_team_series",
]
