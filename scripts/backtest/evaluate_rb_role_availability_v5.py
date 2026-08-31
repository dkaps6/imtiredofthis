"""M95G source-completeness correction v5.

V4 correctly identified the 2025 nflverse depth-chart schema switch but the
source timestamps are UTC-aware while schedule gamedays are date/naive. This
wrapper changes only date normalization so both are compared as timezone-naive
calendar dates before selecting the latest snapshot strictly before game day.
All M95G features, candidate models, regularization, selection rules,
thresholds, and validation gates remain frozen.
"""
from __future__ import annotations

import pandas as pd

import scripts.backtest.evaluate_rb_role_availability as g
import scripts.backtest.evaluate_rb_role_availability_v4 as v4


def _date_depth_2025_tzsafe(raw: pd.DataFrame, schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    x = g.lower(raw)
    need = {"dt", "team", "player_name"}
    if x.empty or not need.issubset(x.columns):
        return pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])

    pos_col = g.first_col(x, ["pos_abb", "pos_name", "pos_grp", "position"])
    rank_col = g.first_col(x, ["pos_rank", "depth_rank", "rank"])
    if not pos_col or not rank_col:
        return pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])

    x["dt"] = pd.to_datetime(x["dt"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    x["team"] = x["team"].map(g.canon)
    x["player_clean_key"] = x["player_name"].map(g.norm_name)
    x["position"] = x[pos_col].astype(str).str.upper().str.strip()
    x["depth_rank"] = g.num(x[rank_col])
    x = x.loc[
        x["dt"].notna() & x["player_clean_key"].ne("")
        & x["position"].str.contains("RB|HB|FB|RUNNING", regex=True, na=False)
    ].copy()
    if x.empty:
        return pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])

    s = g.lower(schedules)
    season_col = g.first_col(s, ["season"])
    week_col = g.first_col(s, ["week"])
    date_col = g.first_col(s, ["gameday", "game_date", "date"])
    home_col = g.first_col(s, ["home_team"])
    away_col = g.first_col(s, ["away_team"])
    type_col = g.first_col(s, ["game_type", "season_type"])
    if not all([season_col, week_col, date_col, home_col, away_col]):
        raise RuntimeError(f"M95G schedule schema incomplete for date-depth mapping: {list(s.columns)}")
    s = s.loc[g.num(s[season_col]).eq(season)].copy()
    if type_col:
        reg = s[type_col].astype(str).str.upper().isin(["REG", "REGULAR", "R"])
        if reg.any():
            s = s.loc[reg].copy()
    s["game_date"] = pd.to_datetime(s[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    games = []
    for _, r in s.iterrows():
        if pd.isna(r["game_date"]):
            continue
        for tc in (home_col, away_col):
            games.append({
                "season": season, "week": int(r[week_col]),
                "team": g.canon(r[tc]), "game_date": r["game_date"],
            })
    games = pd.DataFrame(games).drop_duplicates(g.TEAM_KEYS)

    pieces = []
    for _, game in games.iterrows():
        q = x.loc[(x["team"].eq(game["team"])) & (x["dt"].lt(game["game_date"]))].copy()
        if q.empty:
            continue
        latest = q["dt"].max()
        q = q.loc[q["dt"].eq(latest)].copy()
        q["season"] = season
        q["week"] = int(game["week"])
        q["depth_is_rb1"] = q["depth_rank"].eq(1).astype(int)
        pieces.append(q[g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"]])
    if not pieces:
        return pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates(g.PLAYER_KEYS, keep="last")


v4._date_depth_2025 = _date_depth_2025_tzsafe
g.load_provider_sources = v4._load_provider_sources_v4

if __name__ == "__main__":
    raise SystemExit(g.main())
