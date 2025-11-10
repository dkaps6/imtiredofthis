# scripts/_opponent_map.py
import logging
from typing import Iterable, Optional, Union

import pandas as pd

logger = logging.getLogger("opponent_map")

TEAM_ALIASES = {
    # site oddities
    "BLT": "BAL",
    "BAL RAVENS": "BAL",
    "CLV": "CLE",
    "CLEVELAND BROWNS": "CLE",
    "HST": "HOU",
    "HOUSTON TEXANS": "HOU",
    # common book strings & spaces
    "JAX": "JAX",
    "JAC": "JAX",
    "WSH": "WAS",
    "WFT": "WAS",
    "COMMANDERS": "WAS",
    "NO": "NO",
    "NOR": "NO",
    "NOS": "NO",
    "N.O.": "NO",
    "TB": "TB",
    "T.B.": "TB",
    "TAM": "TB",
    "SD": "LAC",
    "S.D.": "LAC",
    "LA CHARGERS": "LAC",
    "STL": "LA",
    "LA RAMS": "LA",
    "LAR": "LA",
    "NEP": "NE",
    "N.E.": "NE",
    "GNB": "GB",
    "G.B.": "GB",
    "SFO": "SF",
    "S.F.": "SF",
    "ARI": "ARI",
    "ARZ": "ARI",
    "KCC": "KC",
    "K.C.": "KC",
    "N.Y. JETS": "NYJ",
    "NY JETS": "NYJ",
    "N.Y. GIANTS": "NYG",
    "NY GIANTS": "NYG",
    # extras retained from legacy map
    "BAL": "BAL",
    "CLE": "CLE",
    "HOU": "HOU",
    "WAS": "WAS",
    "OAK": "LV",
}

CANON_SET = set(
    [
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
        "LV",
        "LAC",
        "LA",
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
    ]
)


def _normalize_one(team: Union[str, object]) -> object:
    if isinstance(team, pd.Series):
        # legacy callers might still hand us a Series; normalize element-wise
        return team.apply(_normalize_one)
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return ""
    if isinstance(team, str) and not team.strip():
        return ""
    try:
        if pd.isna(team):  # handles pd.NA, numpy.nan, etc.
            return ""
    except TypeError:
        # objects that do not support isna checks fall through
        pass

    t = str(team).strip().upper()
    if t in {"", "NAN", "NONE", "NULL", "NA", "<NA>"}:
        return ""
    t = TEAM_ALIASES.get(t, t)
    # Drop stray punctuation/words
    t = t.replace(".", "").replace("  ", " ").strip()
    if t in CANON_SET:
        return t
    # last resort: collapse to 2–3 letters (keeps TB, NO, LA)
    return TEAM_ALIASES.get(t, t)


def normalize_team(val: Union[str, pd.Series, object]) -> object:
    """Backward-compatible entry point for scalar normalization."""

    if isinstance(val, pd.Series):
        return val.apply(_normalize_one)
    return _normalize_one(val)


def normalize_team_series(vals: Union[pd.Series, Iterable]) -> pd.Series:
    """Vectorized normalization for Series/arrays."""

    s = pd.Series(vals, copy=False)
    return s.apply(_normalize_one)


def map_normalize_team(x):
    """Element-wise scalar wrapper (safe for Series.apply)."""

    return _normalize_one(x)


def build_opponent_map(week: Optional[int] = 10, team_map_path: str = "data/team_week_map.csv") -> pd.DataFrame:
    try:
        tm = pd.read_csv(team_map_path)
    except FileNotFoundError:
        logger.warning("[OpponentMap] team_week_map.csv not found at %s", team_map_path)
        return pd.DataFrame(columns=["event_id", "week", "team", "opponent"])
    except Exception as exc:
        logger.error("[OpponentMap] Failed to read %s: %s", team_map_path, exc)
        return pd.DataFrame(columns=["event_id", "week", "team", "opponent"])

    required = {"event_id", "week", "team", "opponent"}
    if not required.issubset(tm.columns):
        missing = ", ".join(sorted(required - set(tm.columns)))
        logger.error("[OpponentMap] team_week_map missing columns: %s", missing)
        return pd.DataFrame(columns=["event_id", "week", "team", "opponent"])

    working = tm.copy()
    if "team" in working.columns:
        working["team"] = normalize_team_series(working["team"])
    if "opponent" in working.columns:
        working["opponent"] = normalize_team_series(working["opponent"])

    if week is not None and "week" in working.columns:
        working = working[working["week"] == week]

    out = []
    for _, r in working.iterrows():
        team_val = r.get("team", "")
        opponent_val = r.get("opponent", "")
        if not team_val:
            continue
        event_id = r.get("event_id")
        week_val = r.get("week")
        if opponent_val == "BYE":
            out.append({"event_id": event_id, "week": week_val, "team": team_val, "opponent": "BYE"})
        else:
            out.append({"event_id": event_id, "week": week_val, "team": team_val, "opponent": opponent_val})
            if opponent_val:
                out.append({"event_id": event_id, "week": week_val, "team": opponent_val, "opponent": team_val})

    df = pd.DataFrame(out).drop_duplicates(subset=["event_id", "team", "opponent"])
    print(f"[OpponentMap] {len(df)} rows written for week {week}")
    df.to_csv("data/opponent_map.csv", index=False)
    return df


def attach_opponent(
    df: pd.DataFrame,
    team_col: str = "team",
    coverage_path: str = "data/team_week_map.csv",
    opponent_col: str = "opponent",
    inplace: bool = True,
    week: Optional[int] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    target = df if inplace else df.copy()
    if team_col not in target.columns:
        return target

    opponent_map_df = build_opponent_map(week=week, team_map_path=coverage_path)
    if opponent_map_df.empty:
        return target

    target[team_col] = normalize_team_series(target[team_col])
    if opponent_col not in target.columns:
        target[opponent_col] = pd.NA

    mapping_event = {}
    mapping_team = {}
    for row in opponent_map_df.itertuples(index=False):
        team = getattr(row, "team", "")
        opponent = getattr(row, "opponent", "")
        event_id = getattr(row, "event_id", None)
        if team:
            mapping_team.setdefault(team, opponent)
            if event_id is not None:
                mapping_event[(event_id, team)] = opponent

    mask = target[opponent_col].isna() | (target[opponent_col].astype(str).str.strip() == "")

    if "event_id" in target.columns:
        event_keys = list(zip(target.loc[mask, "event_id"], target.loc[mask, team_col]))
        mapped_values = [mapping_event.get(key, mapping_team.get(key[1], "")) for key in event_keys]
        target.loc[mask, opponent_col] = mapped_values
    else:
        target.loc[mask, opponent_col] = target.loc[mask, team_col].map(mapping_team)

    return target
