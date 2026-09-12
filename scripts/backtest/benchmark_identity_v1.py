"""Fail-closed identity helpers for historical Vegas benchmark reconstruction.

The historical market benchmark must never grade a sportsbook row unless the
football projection and sportsbook event resolve to the same season/week/game.
This module is deliberately independent of betting logic: it only validates
identity and canonical team aliases.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from scripts._opponent_map import canon_team

_GAME_ID_RE = re.compile(r"^(?P<season>\d{4})_(?P<week>\d{1,2})_(?P<away>[A-Za-z0-9]+)_(?P<home>[A-Za-z0-9]+)$")


def parse_game_id(value: Any) -> dict[str, Any] | None:
    text = str(value or "").strip()
    m = _GAME_ID_RE.match(text)
    if not m:
        return None
    away = canon_team(m.group("away"))
    home = canon_team(m.group("home"))
    if not away or not home or away == home:
        return None
    return {
        "season": int(m.group("season")),
        "week": int(m.group("week")),
        "away": away,
        "home": home,
    }


def home_away_from_game_id(team: Any, game_id: Any) -> str:
    parsed = parse_game_id(game_id)
    t = canon_team(team)
    if not parsed or not t:
        return "UNKNOWN"
    if t == parsed["home"]:
        return "HOME"
    if t == parsed["away"]:
        return "AWAY"
    return "UNKNOWN"


def _identity_failures(
    frame: pd.DataFrame,
    *,
    require_team: bool,
    require_opponent: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "game_id"}
    if require_team:
        required.add("team")
    if require_opponent:
        required.add("opponent")
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"benchmark identity frame missing columns: {missing}")

    season = pd.to_numeric(x["season"], errors="coerce")
    week = pd.to_numeric(x["week"], errors="coerce")
    parsed = x["game_id"].map(parse_game_id)

    invalid_game_id = parsed.isna()
    parsed_season = parsed.map(lambda v: v.get("season") if isinstance(v, dict) else pd.NA)
    parsed_week = parsed.map(lambda v: v.get("week") if isinstance(v, dict) else pd.NA)
    parsed_away = parsed.map(lambda v: v.get("away") if isinstance(v, dict) else "")
    parsed_home = parsed.map(lambda v: v.get("home") if isinstance(v, dict) else "")

    season_mismatch = ~invalid_game_id & (pd.to_numeric(parsed_season, errors="coerce") != season)
    week_mismatch = ~invalid_game_id & (pd.to_numeric(parsed_week, errors="coerce") != week)

    team_mismatch = pd.Series(False, index=x.index)
    opponent_mismatch = pd.Series(False, index=x.index)
    if require_team:
        team = x["team"].map(canon_team)
        team_mismatch = ~invalid_game_id & ~((team == parsed_away) | (team == parsed_home))
        if require_opponent:
            opponent = x["opponent"].map(canon_team)
            expected_opponent = pd.Series("", index=x.index, dtype="object")
            expected_opponent.loc[team == parsed_away] = parsed_home.loc[team == parsed_away]
            expected_opponent.loc[team == parsed_home] = parsed_away.loc[team == parsed_home]
            opponent_mismatch = ~invalid_game_id & (~team_mismatch) & (opponent != expected_opponent)

    bad = invalid_game_id | season_mismatch | week_mismatch | team_mismatch | opponent_mismatch
    counts = {
        "rows": int(len(x)),
        "invalid_game_id": int(invalid_game_id.sum()),
        "season_mismatch": int(season_mismatch.sum()),
        "week_mismatch": int(week_mismatch.sum()),
        "team_mismatch": int(team_mismatch.sum()),
        "opponent_mismatch": int(opponent_mismatch.sum()),
        "bad_rows": int(bad.sum()),
    }
    sample_cols = [c for c in ["season", "week", "team", "opponent", "game_id", "player_clean_key", "market"] if c in x.columns]
    return x.loc[bad, sample_cols].head(20).copy(), counts


def assert_benchmark_identity(
    frame: pd.DataFrame,
    *,
    label: str,
    require_team: bool = False,
    require_opponent: bool = False,
) -> dict[str, int]:
    if frame is None or frame.empty:
        raise RuntimeError(f"{label}: benchmark identity validation received zero rows")
    sample, counts = _identity_failures(
        frame,
        require_team=require_team,
        require_opponent=require_opponent,
    )
    if counts["bad_rows"]:
        raise RuntimeError(
            f"{label}: benchmark identity failure counts={counts}; "
            f"sample={sample.to_dict(orient='records')}"
        )
    return counts
