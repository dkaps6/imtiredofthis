# scripts/_opponent_map.py
import logging
import pandas as pd

logger = logging.getLogger("opponent_map")

TEAM_ALIASES = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "ARZ": "ARI",
    "JAC": "JAX",
    "WSH": "WAS",
    "SD": "LAC",
    "LA": "LAR",
    "STL": "LAR",
    "OAK": "LV",
}

ALL_TEAMS = {
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


def _normalize_code(value: object) -> str:
    """Normalize a single team abbreviation and preserve BYE placeholders."""

    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    if not text:
        return ""
    if text == "BYE":
        return "BYE"
    canonical = TEAM_ALIASES.get(text, text)
    canonical = canonical.upper()
    if canonical in ALL_TEAMS or canonical == "BYE":
        return canonical
    return canonical

# --- light, safe utilities used by multiple scripts ---

def normalize_team(s: pd.Series) -> pd.Series:
    """Uppercase, trim, and apply known alias fixes for team abbreviations."""

    return s.astype(str).map(_normalize_code)

def _read_csv_silent(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def build_opponent_map(coverage_path: str = "data/coverage_cb.csv") -> dict:
    """
    Build offense_team -> defense_team map from coverage_cb.csv.
    Returns {} if file/columns not present.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.setLevel(logging.INFO)

    cov = _read_csv_silent(coverage_path)
    need = {"offense_team", "defense_team"}
    if not need.issubset(set(cov.columns)):
        logger.warning(
            "[OPPONENT-MAP] coverage file missing required columns: %s",
            ", ".join(sorted(need - set(cov.columns))),
        )
        return {}
    cov = cov.copy()
    cov["offense_team"] = normalize_team(cov["offense_team"])
    cov["defense_team"] = normalize_team(cov["defense_team"])

    mapping: dict[str, str] = {}
    teams_seen: set[str] = set()

    for off, deff in zip(cov["offense_team"], cov["defense_team"]):
        if not off:
            continue
        if off == "BYE":
            continue
        teams_seen.add(off)
        if deff and deff != "BYE":
            teams_seen.add(deff)
        if not deff or deff == "BYE":
            mapping[off] = "BYE"
            continue
        mapping[off] = deff
        mapping.setdefault(deff, off)

    normalized_count = len(teams_seen)
    teams_with_gaps = {
        team
        for team in teams_seen
        if team not in mapping or mapping.get(team, "") in {"", "BYE"}
    }
    logger.info(
        "[OPPONENT-MAP] Normalized %d teams, %d provisional BYE assignments",
        normalized_count,
        len(teams_with_gaps),
    )

    for team in ALL_TEAMS:
        if team not in mapping:
            mapping[team] = "BYE"

    print(f"[OPPONENT-MAP] Normalized {len(mapping)} teams.")
    missing = ALL_TEAMS - set(mapping.keys())
    if missing:
        print(f"[OPPONENT-MAP] WARNING: Missing mappings for {missing}")
    else:
        print("[OPPONENT-MAP] ✅ All 32 team mappings complete.")
    return mapping

def attach_opponent(df: pd.DataFrame,
                    team_col: str = "team",
                    coverage_path: str = "data/coverage_cb.csv",
                    opponent_col: str = "opponent",
                    inplace: bool = True) -> pd.DataFrame:
    """
    Fill/overwrite df[opponent_col] using offense->defense schedule map.
    Leaves df untouched if mapping can't be built.
    """
    opp_map = build_opponent_map(coverage_path)
    if not opp_map or team_col not in df.columns:
        return df
    out = df if inplace else df.copy()
    out[team_col] = normalize_team(out[team_col])
    if opponent_col not in out.columns:
        out[opponent_col] = pd.NA
    # only fill where missing or 'ALL'
    mask = out[opponent_col].isna() | (
        out[opponent_col].astype(str).str.upper().isin({"ALL", ""})
    )
    filled = out.loc[mask, team_col].map(opp_map)
    out.loc[mask, opponent_col] = filled
    if opponent_col in out.columns:
        bye_mask = out[team_col].astype(str).str.upper().eq("BYE")
        out.loc[bye_mask, opponent_col] = "BYE"
    return out
