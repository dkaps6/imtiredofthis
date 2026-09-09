#!/usr/bin/env python3
"""R26 Part B: build/audit leakage-safe RB roster-transition state.

Uses only:
- target-week weekly roster snapshot (already accepted by historical universe contract),
- strictly earlier roster snapshot,
- strictly earlier depth snapshot.

No player outcome/stat table, participation data, sportsbook data, or model prediction
is loaded.  Same-week historical depth is intentionally NOT used.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.historical_inputs import _load_nflreadpy_weekly_sources, build_schedule_history

RB_POSITIONS = {"RB", "HB", "FB"}


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def first(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def normalize_roster(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["season", "week", "team", "player_key", "player", "position", "status"])
    d = lower(df)
    wk = first(d, ["week", "report_week"])
    team = first(d, ["team", "club_code", "team_abbr", "team_abbreviation", "club"])
    name = first(d, ["full_name", "football_name", "player_name", "player", "name"])
    pos = first(d, ["position", "pos"])
    status = first(d, ["status", "roster_status", "status_description_abbr"])
    if not all([wk, team, name, pos]):
        raise RuntimeError(f"roster {season} missing required week/team/name/position semantics")
    x = pd.DataFrame({
        "season": int(season),
        "week": pd.to_numeric(d[wk], errors="coerce"),
        "team": d[team].map(canon_team),
        "player_key": d[name].map(key),
        "player": d[name].astype(str).str.strip(),
        "position": d[pos].astype(str).str.upper().str.strip(),
        "status": d[status].astype(str).str.upper().str.strip() if status else "",
    })
    x = x.loc[x["position"].isin(RB_POSITIONS) & x["week"].notna() & x["player_key"].ne("")].copy()
    x["week"] = x["week"].astype(int)
    return x.drop_duplicates(["season", "week", "team", "player_key"], keep="last")


def normalize_depth(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["season", "week", "team", "player_key", "prior_depth_position", "prior_depth_team"])
    d = lower(df)
    if not {"season", "week"}.issubset(d.columns):
        # Date-based (2025+) is intentionally not made historical-current here.
        return pd.DataFrame(columns=["season", "week", "team", "player_key", "prior_depth_position", "prior_depth_team"])
    team = first(d, ["club_code", "team", "team_abbr", "team_abbreviation"])
    name = first(d, ["full_name", "football_name", "player_name", "player", "name"])
    pos = first(d, ["position", "pos", "depth_position"])
    dpos = first(d, ["depth_position", "position", "pos"])
    dteam = first(d, ["depth_team"])
    if not all([team, name, pos]):
        return pd.DataFrame(columns=["season", "week", "team", "player_key", "prior_depth_position", "prior_depth_team"])
    x = pd.DataFrame({
        "season": pd.to_numeric(d["season"], errors="coerce"),
        "week": pd.to_numeric(d["week"], errors="coerce"),
        "team": d[team].map(canon_team),
        "player_key": d[name].map(key),
        "position": d[pos].astype(str).str.upper().str.strip(),
        "prior_depth_position": d[dpos].astype(str).str.upper().str.strip() if dpos else "",
        "prior_depth_team": d[dteam].astype(str).str.strip() if dteam else "",
    })
    x = x.loc[
        x["season"].eq(int(season)) & x["week"].notna() & x["position"].isin(RB_POSITIONS) & x["player_key"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    return x.drop_duplicates(["season", "week", "team", "player_key"], keep="last")


def prior_snapshot(target_season: int, target_week: int, schedules: dict[int, list[int]]) -> tuple[int, int]:
    if int(target_week) > min(schedules[int(target_season)]):
        earlier = [w for w in schedules[int(target_season)] if w < int(target_week)]
        return int(target_season), int(max(earlier))
    ps = int(target_season) - 1
    if ps not in schedules or not schedules[ps]:
        raise RuntimeError(f"no prior-season schedule for {target_season} W{target_week}")
    return ps, int(max(schedules[ps]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-seasons", default="2015-2025")
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r26_safe_transition_state"))
    a = ap.parse_args()
    if "-" in a.target_seasons and "," not in a.target_seasons:
        lo, hi = a.target_seasons.split("-", 1)
        target_seasons = list(range(int(lo), int(hi) + 1))
    else:
        target_seasons = sorted({int(x.strip()) for x in a.target_seasons.split(",") if x.strip()})
    source_seasons = list(range(min(target_seasons) - 1, max(target_seasons) + 1))

    schedule = build_schedule_history(source_seasons)
    schedules = {
        int(s): sorted(pd.to_numeric(g["week"], errors="coerce").dropna().astype(int).unique().tolist())
        for s, g in schedule.groupby("season")
    }

    rosters: dict[int, pd.DataFrame] = {}
    depths: dict[int, pd.DataFrame] = {}
    source_rows = []
    for season in source_seasons:
        r, d = _load_nflreadpy_weekly_sources(int(season))
        rosters[int(season)] = normalize_roster(r, int(season))
        depths[int(season)] = normalize_depth(d, int(season))
        source_rows.append({
            "season": int(season),
            "rb_roster_rows": int(len(rosters[int(season)])),
            "laggable_week_depth_rows": int(len(depths[int(season)])),
            "week_tagged_depth": bool(len(depths[int(season)]) > 0),
        })

    player_rows = []
    room_rows = []
    for season in target_seasons:
        for week in schedules[int(season)]:
            cur = rosters[int(season)].loc[rosters[int(season)]["week"].eq(int(week))].copy()
            if cur.empty:
                continue
            ps, pw = prior_snapshot(int(season), int(week), schedules)
            prev = rosters[int(ps)].loc[rosters[int(ps)]["week"].eq(int(pw))].copy()
            pdepth = depths[int(ps)].loc[depths[int(ps)]["week"].eq(int(pw))].copy()

            prev_any = prev[["player_key", "team"]].drop_duplicates("player_key", keep="last").rename(columns={"team": "prior_any_team"})
            prev_same = prev[["team", "player_key"]].copy()
            prev_same["prior_same_team_roster"] = 1
            z = cur.merge(prev_same, on=["team", "player_key"], how="left", validate="one_to_one")
            z = z.merge(prev_any, on="player_key", how="left", validate="many_to_one")
            z["prior_same_team_roster"] = z["prior_same_team_roster"].fillna(0).astype(int)
            z["prior_any_nfl_roster"] = z["prior_any_team"].notna().astype(int)
            z["new_to_team_veteran"] = ((z["prior_same_team_roster"].eq(0)) & (z["prior_any_nfl_roster"].eq(1))).astype(int)
            z["no_prior_nfl_roster"] = z["prior_any_nfl_roster"].eq(0).astype(int)
            z["continuing_same_team"] = z["prior_same_team_roster"].astype(int)
            z["prior_snapshot_season"] = int(ps)
            z["prior_snapshot_week"] = int(pw)

            if not pdepth.empty:
                any_depth = pdepth[["player_key", "team", "prior_depth_position", "prior_depth_team"]].drop_duplicates("player_key", keep="last").rename(columns={"team": "prior_depth_team_club"})
                z = z.merge(any_depth, on="player_key", how="left", validate="many_to_one")
            else:
                z["prior_depth_position"] = ""
                z["prior_depth_team"] = ""
                z["prior_depth_team_club"] = ""
            z["prior_depth_available"] = z["prior_depth_position"].fillna("").astype(str).str.strip().ne("").astype(int)

            for team, cg in cur.groupby("team"):
                pg = prev.loc[prev["team"].eq(team)]
                cur_set = set(cg["player_key"])
                prev_set = set(pg["player_key"])
                entrants = cur_set - prev_set
                exits = prev_set - cur_set
                room_rows.append({
                    "season": int(season),
                    "week": int(week),
                    "team": team,
                    "prior_snapshot_season": int(ps),
                    "prior_snapshot_week": int(pw),
                    "current_rb_room_n": int(len(cur_set)),
                    "prior_rb_room_n": int(len(prev_set)),
                    "room_entrants_n": int(len(entrants)),
                    "room_exits_n": int(len(exits)),
                    "room_turnover_n": int(len(entrants) + len(exits)),
                    "room_turnover_flag": int(bool(entrants or exits)),
                })
            room = pd.DataFrame(room_rows[-cur["team"].nunique():])
            z = z.merge(room[["season", "week", "team", "current_rb_room_n", "prior_rb_room_n", "room_entrants_n", "room_exits_n", "room_turnover_n", "room_turnover_flag"]], on=["season", "week", "team"], how="left", validate="many_to_one")
            player_rows.append(z)

    players = pd.concat(player_rows, ignore_index=True) if player_rows else pd.DataFrame()
    rooms = pd.DataFrame(room_rows)
    if players.empty:
        raise RuntimeError("R26 safe transition-state audit produced zero player rows")

    summary_rows = []
    for season, g in players.groupby("season"):
        w1 = g.loc[g["week"].eq(1)]
        summary_rows.append({
            "season": int(season),
            "player_rows": int(len(g)),
            "continuing_same_team_rate": float(g["continuing_same_team"].mean()),
            "new_to_team_veteran_rate": float(g["new_to_team_veteran"].mean()),
            "no_prior_nfl_roster_rate": float(g["no_prior_nfl_roster"].mean()),
            "room_turnover_rate": float(g["room_turnover_flag"].mean()),
            "prior_depth_coverage": float(g["prior_depth_available"].mean()),
            "week1_rows": int(len(w1)),
            "week1_continuing_same_team_rate": float(w1["continuing_same_team"].mean()) if len(w1) else np.nan,
            "week1_new_to_team_veteran_rate": float(w1["new_to_team_veteran"].mean()) if len(w1) else np.nan,
            "week1_no_prior_nfl_roster_rate": float(w1["no_prior_nfl_roster"].mean()) if len(w1) else np.nan,
            "week1_room_turnover_rate": float(w1["room_turnover_flag"].mean()) if len(w1) else np.nan,
            "week1_prior_depth_coverage": float(w1["prior_depth_available"].mean()) if len(w1) else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values("season")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(a.out_dir / "r26_safe_transition_source_inventory.csv", index=False)
    players.to_csv(a.out_dir / "r26_safe_transition_player_state.csv", index=False)
    rooms.to_csv(a.out_dir / "r26_safe_transition_room_state.csv", index=False)
    summary.to_csv(a.out_dir / "r26_safe_transition_summary.csv", index=False)

    result = {
        "candidate": "RB_R26_ROLE_TRANSITION_ENTITLEMENT_V1",
        "stage": "SAFE_TRANSITION_STATE_SOURCE_AUDIT_ONLY",
        "scientific_candidate_scored": False,
        "player_outcomes_loaded": False,
        "same_week_depth_used": False,
        "target_game_participation_used": False,
        "sportsbook_inputs_used": False,
        "production_parameters_changed": False,
        "target_seasons": target_seasons,
        "rows": int(len(players)),
        "continuing_same_team_rate": float(players["continuing_same_team"].mean()),
        "new_to_team_veteran_rate": float(players["new_to_team_veteran"].mean()),
        "no_prior_nfl_roster_rate": float(players["no_prior_nfl_roster"].mean()),
        "room_turnover_rate": float(players["room_turnover_flag"].mean()),
        "prior_depth_coverage": float(players["prior_depth_available"].mean()),
        "week1_rows": int(players["week"].eq(1).sum()),
        "week1_prior_depth_coverage": float(players.loc[players["week"].eq(1), "prior_depth_available"].mean()),
        "contract": "current-week roster + strictly prior roster/depth only",
    }
    (a.out_dir / "r26_safe_transition_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_csv(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
