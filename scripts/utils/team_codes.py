"""Helpers for canonical NFL team codes."""

TEAM_CANON = {
    # Official three-letter codes we want everywhere
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

ALIASES = {
    # If any upstreams ever send alternates, normalize them here
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WFT": "WAS",
    "WASFT": "WAS",
    "N.O.": "NO",
    "L.V.": "LV",
    "LA": "LAR",
    "LOS ANGELES RAMS": "LAR",
    "LOS ANGELES CHARGERS": "LAC",
}


def canon_team(x: str) -> str:
    """Return canonical three-letter team code when possible."""

    if x is None:
        return None
    s = str(x).strip().upper()
    s = ALIASES.get(s, s)
    return TEAM_CANON.get(s, s)
