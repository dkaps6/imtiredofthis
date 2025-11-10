# scripts/_opponent_map.py
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("opponent_map")

TEAM_ALIASES = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "WSH": "WAS",
    "JAC": "JAX",
    "SD": "LAC",
    "LA": "LAR",
    "STL": "LAR",
    "ARZ": "ARI",
    "OAK": "LV",
}


def normalize_team(team: Optional[object]) -> str:
    if pd.isna(team):
        return ""
    return TEAM_ALIASES.get(str(team).upper(), str(team).upper())


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
    working["team"] = working["team"].apply(normalize_team)
    working["opponent"] = working["opponent"].apply(normalize_team)

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

    target[team_col] = target[team_col].apply(normalize_team)
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
