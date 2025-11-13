from __future__ import annotations

import re
from typing import Dict

import pandas as pd

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

_PUNCT_PATTERN = re.compile(r"[\.\u2019\u2018'`]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _clean_token(name: str | None) -> str:
    text = "" if name is None else str(name)
    text = text.strip()
    if not text:
        return ""
    text = _PUNCT_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text


def canon_team(name: str) -> str:
    """Return the canonical 2–3 letter team abbreviation for *name*."""

    cleaned = _clean_token(name)
    if not cleaned:
        return ""

    upper = cleaned.upper()
    if upper in CANON_TEAM_ABBR:
        return CANON_TEAM_ABBR[upper]

    title = cleaned.title()
    if title in ESPN_CITY_TO_ABBR:
        return ESPN_CITY_TO_ABBR[title]
    if title in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[title]

    return upper


def _canon_team_series(s: pd.Series) -> pd.Series:
    """Vectorized wrapper returning canonical team abbreviations for a Series."""

    x = s.fillna("").astype(str)
    return x.apply(canon_team)


def canon_team_series(series: pd.Series) -> pd.Series:
    """Public alias so callers can import canon_team_series directly."""

    return _canon_team_series(series)


def map_normalize_team(x: str | None) -> str | None:
    if x is None:
        return None

    candidate = canon_team(str(x))
    if not candidate:
        return None

    candidate = candidate.upper()
    if candidate in CANON_TEAM_CODES:
        return candidate

    fixed = TEAM_FIX.get(candidate)
    if fixed and fixed in CANON_TEAM_CODES:
        return fixed

    mapped = CANON_TEAM_ABBR.get(candidate)
    if mapped and mapped in CANON_TEAM_CODES:
        return mapped

    return None


def normalize_team_series(s: pd.Series) -> pd.Series:
    return s.apply(map_normalize_team)


__all__ = [
    "TEAM_FIX",
    "TEAM_NAME_TO_ABBR",
    "ESPN_CITY_TO_ABBR",
    "CANON_TEAM_ABBR",
    "canon_team",
    "_canon_team_series",
    "canon_team_series",
    "CANON_TEAM_CODES",
    "TEAM_REMAP",
    "map_normalize_team",
    "normalize_team_series",
]
