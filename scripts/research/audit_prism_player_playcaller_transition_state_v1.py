#!/usr/bin/env python3
"""PRISM named-player offensive playcaller transition source audit.

Source/parity only. Attaches frozen historical primary-playcaller identity to
stable named-player weekly roster rows. Scores no player outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.build_qb_playcaller_opening_leverage import (
    CALLER_OVERRIDES,
    SEASON_OPENING_CALLER,
    caller_for,
)

VERSION = "PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "16530e901489ab2853bbbae6131ff33013b40b3c"
SEASONS = (2024, 2025)
FAMILIES = {"QB", "RB", "WR", "TE"}
TEAMS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
    "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SF","SEA","TB","TEN","WAS",
}


def to_pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def norm_pos(v) -> str:
    s = "" if v is None or pd.isna(v) else str(v).upper().strip()
    if s in {"HB", "TB"} or s.startswith("RB"):
        return "RB"
    if s.startswith("QB"):
        return "QB"
    if s.startswith("WR") or s in {"LWR","RWR","SWR"}:
        return "WR"
    if s.startswith("TE"):
        return "TE"
    return s


def regular_schedule_pairs(season: int) -> set[tuple[int,str]]:
    import nflreadpy as nfl
    x = to_pd(nfl.load_schedules(int(season)))
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"schedule source empty {season}")
    type_col = "season_type" if "season_type" in x.columns else "game_type" if "game_type" in x.columns else None
    if type_col:
        t = x[type_col].astype("string").fillna("").str.upper().str.strip()
        reg = x.loc[t.isin(["REG","REGULAR","RS",""])].copy()
        if not reg.empty:
            x = reg
    if not {"week","home_team","away_team"} <= set(x.columns):
        raise RuntimeError(f"schedule missing required columns {season}")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    pairs:set[tuple[int,str]] = set()
    for r in x.itertuples(index=False):
        if pd.isna(r.week):
            continue
        week = int(r.week)
        for raw in (r.home_team, r.away_team):
            team = canon_team(raw)
            if team == "WSH":
                team = "WAS"
            if team in TEAMS:
                pairs.add((week,team))
    return pairs


def load_roster(season: int) -> tuple[pd.DataFrame,dict]:
    import nflreadpy as nfl
    x = to_pd(nfl.load_rosters_weekly(int(season)))
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"weekly roster source empty {season}")

    def first(cols, default=""):
        for c in cols:
            if c in x.columns:
                return x[c]
        return pd.Series(default, index=x.index)

    x["week"] = pd.to_numeric(first(["week"]), errors="coerce")
    x["team"] = first(["team","team_abbr","club_code"]).map(canon_team)
    x.loc[x["team"].eq("WSH"),"team"] = "WAS"
    x["family"] = first(["position","pos"]).map(norm_pos)
    x["stable_id"] = first(["gsis_id","player_id"]).astype("string").fillna("").str.strip()

    admitted = x.loc[
        x["week"].between(1,22,inclusive="both")
        & x["team"].isin(TEAMS)
        & x["family"].isin(FAMILIES)
    ].copy()
    if admitted.empty:
        raise RuntimeError(f"no admitted skill roster rows {season}")

    stable_cov = float(admitted["stable_id"].ne("").mean())
    stable = admitted.loc[admitted["stable_id"].ne("")].copy()
    stable["season"] = int(season)
    stable["week"] = stable["week"].astype(int)
    stable = stable[["season","week","team","stable_id","family"]].drop_duplicates()

    dup = int(stable.duplicated(["season","week","team","stable_id"]).sum())
    if dup:
        raise RuntimeError(f"duplicate stable roster identities {season}: {dup}")

    expected = regular_schedule_pairs(season)
    observed = set(zip(stable["week"].astype(int), stable["team"].astype(str)))
    team_week_cov = float(len(expected & observed) / len(expected)) if expected else 0.0

    return stable, {
        "raw_admitted_rows": int(len(admitted)),
        "stable_rows": int(len(stable)),
        "stable_id_coverage": stable_cov,
        "expected_team_weeks": int(len(expected)),
        "observed_expected_team_weeks": int(len(expected & observed)),
        "team_week_coverage": team_week_cov,
    }


def historical_mapping_audit(season: int) -> dict:
    opening = SEASON_OPENING_CALLER.get(season,{})
    opening_nonblank = {str(k):str(v).strip() for k,v in opening.items() if str(v).strip()}
    opening_teams = set(opening_nonblank)
    overrides = [
        (int(s),str(t),int(w),str(c).strip(),str(url).strip())
        for s,t,w,c,url in CALLER_OVERRIDES if int(s)==season
    ]
    bad_overrides = [
        {"season":s,"team":t,"week":w,"caller":c,"url":url}
        for s,t,w,c,url in overrides
        if t not in TEAMS or w < 1 or not c or not url.startswith("http")
    ]
    deterministic_blank = []
    for week in range(1,19):
        for team in sorted(TEAMS):
            if not caller_for(season,week,team):
                deterministic_blank.append((week,team))

    return {
        "season":season,
        "opening_team_count":int(len(opening_teams)),
        "opening_missing_teams":sorted(TEAMS-opening_teams),
        "opening_extra_teams":sorted(opening_teams-TEAMS),
        "override_count":int(len(overrides)),
        "bad_overrides":bad_overrides,
        "deterministic_blank_team_weeks":int(len(deterministic_blank)),
        "documented_inseason_transition_count":int(len(overrides)),
    }


def attach_players(roster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in roster.itertuples(index=False):
        season,week,team = int(r.season),int(r.week),str(r.team)
        cur = caller_for(season,week,team)
        prior = caller_for(season,week-1,team) if week > 1 else ""
        rows.append({
            "season":season,
            "week":week,
            "team":team,
            "stable_id":str(r.stable_id),
            "family":str(r.family),
            "current_playcaller":cur,
            "prior_week_playcaller":prior,
            "playcaller_transition":bool(week > 1 and prior and cur and prior != cur),
        })
    return pd.DataFrame(rows)


def prospective_2026_probe() -> dict:
    opening = SEASON_OPENING_CALLER.get(2026,{})
    opening_nonblank = {str(k):str(v).strip() for k,v in opening.items() if str(v).strip()}
    overrides = [
        (int(s),str(t),int(w),str(c).strip(),str(url).strip())
        for s,t,w,c,url in CALLER_OVERRIDES if int(s)==2026
    ]
    complete = set(opening_nonblank) == TEAMS and all(opening_nonblank.get(t,"") for t in TEAMS)
    return {
        "season":2026,
        "repository_season_opening_team_count":int(len(opening_nonblank)),
        "repository_season_opening_complete_32":bool(complete),
        "repository_2026_override_count":int(len(overrides)),
        "same_caller_for_semantics_available":bool(complete),
        "complete_through_current_season":False,
        "prospective_parity_ready":False,
        "blocker":(
            "repo has no complete immutable 2026 primary-playcaller mapping/effective-week handoff ledger"
            if not complete
            else "current-through-week handoff completeness is not certified"
        ),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    season_results=[]
    player_parts=[]
    for season in SEASONS:
        mapping=historical_mapping_audit(season)
        roster,ra=load_roster(season)
        players=attach_players(roster)
        player_parts.append(players)

        attach_cov=float(players["current_playcaller"].astype(str).str.len().gt(0).mean()) if len(players) else 0.0
        dup=int(players.duplicated(["season","week","team","stable_id"]).sum())
        transition_rows=int(players["playcaller_transition"].sum())
        transition_players=int(players.loc[players["playcaller_transition"],"stable_id"].nunique())

        gates={
            "opening_32_teams":mapping["opening_team_count"]==32 and not mapping["opening_missing_teams"] and not mapping["opening_extra_teams"],
            "override_metadata_valid":len(mapping["bad_overrides"])==0,
            "caller_deterministic_all_team_weeks":mapping["deterministic_blank_team_weeks"]==0,
            "duplicate_player_team_week_zero":dup==0,
            "weekly_roster_team_week_coverage_ge_095":ra["team_week_coverage"]>=0.95,
            "stable_player_identity_coverage_ge_099":ra["stable_id_coverage"]>=0.99,
            "player_playcaller_attachment_coverage_ge_099":attach_cov>=0.99,
            "documented_inseason_transition_nonzero":mapping["documented_inseason_transition_count"]>0,
            "target_game_outcomes_used_zero":True,
            "target_game_snaps_used_zero":True,
            "recent_share_fields_used_zero":True,
            "sportsbook_oddsapi_used_zero":True,
            "coefficients_fit_zero":True,
            "thresholds_searched_zero":True,
            "production_changed_false":True,
        }
        season_results.append({
            **mapping,
            **ra,
            "player_playcaller_attachment_coverage":attach_cov,
            "duplicate_player_team_week_identities":dup,
            "transition_player_rows":transition_rows,
            "transition_named_players":transition_players,
            "gates":gates,
            "historical_source_ready":bool(all(gates.values())),
        })

    players=pd.concat(player_parts,ignore_index=True)
    prospective=prospective_2026_probe()
    historical_ready=bool(all(s["historical_source_ready"] for s in season_results))
    source_ready=bool(historical_ready and prospective["prospective_parity_ready"])

    if source_ready:
        disposition="PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_READY"
    elif historical_ready:
        disposition="PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_BLOCKED_PROSPECTIVE_PARITY"
    else:
        disposition="PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_BLOCKED"

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "disposition":disposition,
        "historical_source_ready":historical_ready,
        "prospective_parity_ready":bool(prospective["prospective_parity_ready"]),
        "season_results":season_results,
        "prospective_2026":prospective,
        "player_outcomes_scored":0,
        "player_opportunity_relationships_scored":0,
        "target_game_snaps_used":0,
        "recent_share_fields_used":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    players.to_csv(
        a.out_dir/"player_playcaller_transition_state_rows.csv",
        index=False
    )
    (a.out_dir/"player_playcaller_transition_state_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
