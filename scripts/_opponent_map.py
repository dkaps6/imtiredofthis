# scripts/_opponent_map.py
from __future__ import annotations
import pandas as pd
from typing import Dict

# Canonical 2–3 letter team abbreviations used by our model
CANON_TEAM_ABBR: Dict[str, str] = {
    # odd site aliases → canonical model keys
    "BLT": "BAL", "BAL": "BAL",
    "CLV": "CLE", "CLE": "CLE",
    "HST": "HOU", "HOU": "HOU",
    # regular teams pass-through
    "ARI": "ARI", "ATL": "ATL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "IND": "IND", "JAX": "JAX", "KC": "KC", "LAC": "LAC", "LAR": "LAR",
    "LV": "LV", "MIA": "MIA", "MIN": "MIN", "NE": "NE", "NO": "NO",
    "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "SF": "SF", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
}

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
    if not x:
        return None
    key = str(x).strip().upper()
    # First, normalize any long-form to an abbr
    if key in TEAM_REMAP:
        key = TEAM_REMAP[key]
    # Then coerce to canonical abbr set
    return CANON_TEAM_ABBR.get(key, CANON_TEAM_ABBR.get(TEAM_REMAP.get(key, key), None))

def normalize_team_series(s: pd.Series) -> pd.Series:
    return s.map(map_normalize_team)

__all__ = [
    "CANON_TEAM_ABBR",
    "TEAM_REMAP",
    "map_normalize_team",
    "normalize_team_series",
]
