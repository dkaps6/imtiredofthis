from __future__ import annotations

import pandas as pd
import re

# Existing house fixes? Keep them, but ensure keys are UPPERCASE
TEAM_FIX = {
    # weird book / legacy codes you already normalize:
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "ARZ": "ARI",
    "JAC": "JAX",
    "WSH": "WAS",
    # Identity for standard codes so map_normalize_team stays deterministic
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "KC",
    "LAC": "LAC",
    "LAR": "LAR",
    "LV": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
}

# ESPN city/long-name forms → your house abbreviations
ESPN_NAME_FIX = {
    "WASHINGTON": "WAS",
    "MIAMI": "MIA",
    "CAROLINA": "CAR",
    "ATLANTA": "ATL",
    "TAMPA BAY": "TB",
    "BUFFALO": "BUF",
    "HOUSTON": "HOU",
    "TENNESSEE": "TEN",
    "CHICAGO": "CHI",
    "MINNESOTA": "MIN",

    # helpful extras (common long forms)
    "SAN FRANCISCO": "SF",
    "ARIZONA": "ARI",
    "NEW ENGLAND": "NE",
    "DALLAS": "DAL",
    "PHILADELPHIA": "PHI",
    "NEW YORK JETS": "NYJ",
    "NY JETS": "NYJ",
    "NEW YORK GIANTS": "NYG",
    "NY GIANTS": "NYG",
    "LOS ANGELES RAMS": "LAR",
    "LA RAMS": "LAR",
    "LOS ANGELES CHARGERS": "LAC",
    "LA CHARGERS": "LAC",
    "LAS VEGAS": "LV",
    "JACKSONVILLE": "JAX",
    "KANSAS CITY": "KC",
    "GREEN BAY": "GB",
    "NEW ORLEANS": "NO",
    "CLEVELAND": "CLE",
    "CINCINNATI": "CIN",
    "BALTIMORE": "BAL",
    "PITTSBURGH": "PIT",
    "DETROIT": "DET",
    "INDIANAPOLIS": "IND",
    "DENVER": "DEN",
    "SEATTLE": "SEA",
    "LA": None,  # guard; specific LA teams handled above
}

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


def _canon_team_series(s: pd.Series) -> pd.Series:
    """
    Canonicalize team names coming from books, HTML pages, etc.
    Returns your house abbreviations (e.g., WAS, MIA, SF, NE).
    """
    x = s.fillna("").astype(str).str.strip()
    x = x.str.replace(r"\s+", " ", regex=True).str.upper()

    # apply house fixes (weird book codes first)
    x = x.replace(TEAM_FIX)

    # map ESPN-style names/long forms
    x = x.replace({k: v for k, v in ESPN_NAME_FIX.items() if v is not None})

    # if something is still the literal city "LA" etc., leave as-is for later logic
    return x


# Re-export alias some modules already import
canon_team = _canon_team_series

# broader remap for free-form names we see in APIs / books
TEAM_REMAP = {
    # full names → abbr
    "ARIZONA CARDINALS": "ARI",
    "CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "RAVENS": "BAL",
    "BLT": "BAL",
    "BUFFALO BILLS": "BUF",
    "BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "BROWNS": "CLE",
    "CLV": "CLE",
    "DALLAS COWBOYS": "DAL",
    "COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "TEXANS": "HOU",
    "HST": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "RAIDERS": "LV",
    "OAKLAND RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "BUCCANEERS": "TB",
    "TAMPA BAY": "TB",
    "TENNESSEE TITANS": "TEN",
    "TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
    "COMMANDERS": "WAS",
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
    canon = _canon_team_series(pd.Series([key_upper])).iloc[0]
    if not isinstance(canon, str):
        return None
    canon = canon.strip().upper()
    if not canon:
        return None
    if canon in CANON_TEAM_CODES:
        return canon
    if canon in TEAM_FIX:
        fixed = TEAM_FIX[canon]
        if fixed in CANON_TEAM_CODES:
            return fixed
    return None


def normalize_team_series(s: pd.Series) -> pd.Series:
    return s.apply(map_normalize_team)


__all__ = [
    "TEAM_FIX",
    "canon_team",
    "_canon_team_series",
    "CANON_TEAM_CODES",
    "TEAM_REMAP",
    "map_normalize_team",
    "normalize_team_series",
]
