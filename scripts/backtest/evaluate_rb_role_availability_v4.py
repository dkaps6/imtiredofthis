"""M95G source-completeness corrections v4.

This wrapper preserves the exact M95G model/feature candidate grid, regularization
values, 2024 selection rule, operating-threshold grid, and 2025 validation gate.
It corrects source contracts only:

1. TEAM_KEYS are [season, week, team] when reconstructing prior-game RB leaders.
2. Frozen M95F holdout/validation target-specific raw_prob_20/raw_prob_25 are
   mapped to the OOF artifact's raw_score contract before fit/apply.
3. nflverse depth charts changed schema after 2024. 2024 is week-tagged; 2025
   is date-based (ESPN source). For 2025, each team-game receives the latest
   depth-chart snapshot STRICTLY BEFORE the scheduled game date. Same-day
   snapshots are deliberately excluded because the source has date rather than
   trustworthy pregame timestamps. This restores leakage-safe 2025 depth input
   instead of silently treating the new schema as missing.

No outcome-driven tuning or model changes are made.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

import scripts.backtest.evaluate_rb_role_availability as g


def _previous_team_leaders_fixed(trace: pd.DataFrame) -> pd.DataFrame:
    z = trace.copy()
    if "actual_carries" not in z.columns:
        if "actual_rush_att" in z.columns:
            z["actual_carries"] = g.num(z["actual_rush_att"])
        else:
            raise RuntimeError("M95B trace missing actual carry truth for prior-game leader construction")
    z = z.loc[z["season"].isin([2024, 2025])].copy()
    rows = []
    for (season, week, team), frame in z.groupby(g.TEAM_KEYS):
        q = frame.loc[g.num(frame["actual_carries"]).notna()].copy()
        if q.empty:
            continue
        q["actual_carries"] = g.num(q["actual_carries"])
        q = q.sort_values(["actual_carries", "player_clean_key"], ascending=[False, True])
        rows.append({
            "season": int(season), "team": g.canon(team), "week": int(week),
            "game_top1_key": str(q.iloc[0]["player_clean_key"]),
            "game_top1_carries": float(q.iloc[0]["actual_carries"]),
            "game_top2_key": str(q.iloc[1]["player_clean_key"]) if len(q) > 1 else "",
            "game_top2_carries": float(q.iloc[1]["actual_carries"]) if len(q) > 1 else 0.0,
        })
    game = pd.DataFrame(rows).sort_values(["season", "team", "week"])
    grp = game.groupby(["season", "team"], sort=False)
    game["prior_top1_key"] = grp["game_top1_key"].shift(1)
    game["prior_top1_carries"] = grp["game_top1_carries"].shift(1)
    game["prior_top2_key"] = grp["game_top2_key"].shift(1)
    game["prior_top2_carries"] = grp["game_top2_carries"].shift(1)
    return game[g.TEAM_KEYS + ["prior_top1_key", "prior_top1_carries", "prior_top2_key", "prior_top2_carries"]]


_original_fit_apply = g.fit_apply


def _fit_apply_score_contract(train, test, spec, c, target):
    tr = train.copy(); te = test.copy()
    suffix = "20" if target == "actual_20plus" else "25"
    if "raw_score" not in tr.columns and f"raw_prob_{suffix}" in tr.columns:
        tr["raw_score"] = tr[f"raw_prob_{suffix}"]
    if "raw_score" not in te.columns and f"raw_prob_{suffix}" in te.columns:
        te["raw_score"] = te[f"raw_prob_{suffix}"]
    return _original_fit_apply(tr, te, spec, c, target)


def _date_depth_2025(raw: pd.DataFrame, schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    x = g.lower(raw)
    need = {"dt", "team", "player_name"}
    if x.empty or not need.issubset(x.columns):
        return pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])

    pos_col = g.first_col(x, ["pos_abb", "pos_name", "pos_grp", "position"])
    rank_col = g.first_col(x, ["pos_rank", "depth_rank", "rank"])
    if not pos_col or not rank_col:
        return pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])

    x["dt"] = pd.to_datetime(x["dt"], errors="coerce").dt.normalize()
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
    s["game_date"] = pd.to_datetime(s[date_col], errors="coerce").dt.normalize()
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
    # Snapshot is team-level: choose the latest source date strictly before
    # the game, then take all RB rows from that snapshot.
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


def _load_provider_sources_v4(seasons: list[int]):
    import nflreadpy as nfl

    roster_parts = []
    depth_norm_parts = []
    audit = []
    for season in seasons:
        try:
            r = g.to_pandas(nfl.load_rosters_weekly(int(season)))
            roster_parts.append(r)
            audit.append({"source": "nflreadpy_weekly_rosters", "season": season, "rows": len(r), "status": "ok"})
        except Exception as exc:
            audit.append({"source": "nflreadpy_weekly_rosters", "season": season, "rows": 0, "status": f"error:{type(exc).__name__}:{exc}"})

        try:
            d = g.to_pandas(nfl.load_depth_charts(int(season)))
            audit.append({"source": "nflreadpy_depth_charts_raw", "season": season, "rows": len(d), "status": "loaded"})
            if season <= 2024:
                nd, weekly = g.normalize_depth(d)
                audit.append({"source": "depth_normalized_week_tagged", "season": season, "rows": len(nd), "status": "yes" if weekly else "no"})
            else:
                sched = g.to_pandas(nfl.load_schedules(int(season)))
                nd = _date_depth_2025(d, sched, int(season))
                audit.append({"source": "depth_normalized_date_to_pregame_week", "season": season, "rows": len(nd), "status": "strictly_before_game_date"})
            if not nd.empty:
                depth_norm_parts.append(nd)
        except Exception as exc:
            audit.append({"source": "nflreadpy_depth_charts", "season": season, "rows": 0, "status": f"unavailable:{type(exc).__name__}:{exc}"})

    try:
        inj = g.to_pandas(nfl.load_injuries(seasons=seasons))
        audit.append({"source": "nflreadpy_injuries", "season": 0, "rows": len(inj), "status": "ok"})
    except Exception as exc:
        inj = pd.DataFrame()
        audit.append({"source": "nflreadpy_injuries", "season": 0, "rows": 0, "status": f"error:{type(exc).__name__}:{exc}"})

    if not roster_parts:
        raise RuntimeError("M95G requires leakage-safe weekly roster data")
    rosters = g.normalize_rosters(pd.concat(roster_parts, ignore_index=True, sort=False))
    injuries = g.normalize_injuries(inj)
    depth = pd.concat(depth_norm_parts, ignore_index=True, sort=False) if depth_norm_parts else pd.DataFrame(columns=g.PLAYER_KEYS + ["depth_rank", "depth_is_rb1"])
    for season in seasons:
        audit.append({"source": "usable_depth_rows", "season": season, "rows": int((g.num(depth["season"]) == season).sum()) if not depth.empty else 0, "status": "leakage_safe"})
    return rosters, injuries, depth, pd.DataFrame(audit)


g.previous_team_leaders = _previous_team_leaders_fixed
g.fit_apply = _fit_apply_score_contract
g.load_provider_sources = _load_provider_sources_v4

if __name__ == "__main__":
    raise SystemExit(g.main())
