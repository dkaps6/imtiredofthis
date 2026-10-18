#!/usr/bin/env python3
"""Source/support audit for PRISM RB short-yardage carry-type identity V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research import audit_player_high_value_scoring_opportunity_identity_v1 as hv
from scripts.utils.pbp import get_pbp

VERSION="PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_V1_SOURCE_AUDIT"
SEASONS=(2022,2023,2024,2025)
TARGET_SEASONS=(2024,2025)
MIN_PRIOR_SAME_TEAM_CARRIES=10
MIN_SUPPORTED_PLAYER_GAMES=500
MIN_SUPPORTED_PLAYERS=60


def num(s): return pd.to_numeric(s,errors="coerce")
def first(df,names): return next((c for c in names if c in df.columns),None)


def load_events(season:int, roster:pd.DataFrame):
    p=get_pbp(season,min_rows=1000).copy()
    p.columns=[str(c).lower() for c in p.columns]
    if "season_type" in p.columns:
        q=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if not q.empty: p=q
    wc=first(p,["week"]); tc=first(p,["posteam"]); gc=first(p,["game_id"])
    pc=first(p,["play_id"]); rc=first(p,["rusher_player_id","rusher_id","rusher_gsis_id"])
    dc=first(p,["down"]); yc=first(p,["ydstogo","yards_to_go"])
    if not all([wc,tc,gc,pc,rc,dc,yc]):
        raise RuntimeError(f"PBP short-yardage schema missing {season}")
    rush=num(p.get("rush_attempt",pd.Series(0,index=p.index))).eq(1)
    rpid=p[rc].astype("string").fillna("").str.strip()
    all_rush=pd.DataFrame({
        "season":season,
        "week":num(p[wc]),
        "team":p[tc].map(canon_team),
        "game_id":p[gc].astype(str),
        "play_id":p[pc].astype(str),
        "player_id":rpid,
        "down":num(p[dc]),
        "ydstogo":num(p[yc]),
    }).loc[rush].copy()
    rusher_id_coverage=float(all_rush.player_id.ne("").mean()) if len(all_rush) else 0.0
    q=all_rush.loc[all_rush.player_id.ne("")].copy()
    r=roster.loc[roster.season.eq(season),["season","week","team","player_id","family"]].drop_duplicates(
        ["season","week","team","player_id"]
    )
    q=q.merge(r,on=["season","week","team","player_id"],how="left",validate="many_to_one")
    rb=q.loc[q.family.eq("RB")].copy()
    rb["short_yardage"]=(
        rb["down"].isin([3,4])
        & rb["ydstogo"].gt(0)
        & rb["ydstogo"].le(2)
    ).astype(int)
    weeks=sorted(set(int(v) for v in rb.week.dropna().astype(int).unique()))
    short_by_week=rb.groupby("week").short_yardage.sum().to_dict()
    metrics={
        "season":season,
        "pbp_rows":int(len(p)),
        "all_rush_rows":int(len(all_rush)),
        "rb_rush_rows":int(len(rb)),
        "regular_weeks":len(weeks),
        "rusher_stable_identity_coverage":rusher_id_coverage,
        "rb_family_bridge_coverage":float(q.family.eq("RB").mean()) if len(q) else 0.0,
        "down_coverage_rb":float(rb["down"].notna().mean()) if len(rb) else 0.0,
        "ydstogo_coverage_rb":float(rb.ydstogo.notna().mean()) if len(rb) else 0.0,
        "short_yardage_events":int(rb.short_yardage.sum()),
        "weeks_with_short_yardage":sum(int(short_by_week.get(w,0)>0) for w in weeks),
        "duplicate_player_play_rows":int(rb.duplicated(["game_id","play_id","player_id"]).sum()),
        "short_flag_binary":bool(rb.short_yardage.isin([0,1]).all()),
    }
    # Family bridge is evaluated among stable-identity rushes against a valid roster family.
    metrics["skill_family_bridge_coverage"]=float(q.family.astype("string").fillna("").ne("").mean()) if len(q) else 0.0
    metrics.update({
        "gate_pbp_nonempty":bool(len(p)>1000),
        "gate_rusher_identity_ge_99":bool(rusher_id_coverage>=.99),
        "gate_rb_family_bridge_ge_98":bool(metrics["skill_family_bridge_coverage"]>=.98),
        "gate_down_coverage_ge_995":bool(metrics["down_coverage_rb"]>=.995),
        "gate_ydstogo_coverage_ge_995":bool(metrics["ydstogo_coverage_rb"]>=.995),
        "gate_short_every_week":bool(metrics["weeks_with_short_yardage"]==metrics["regular_weeks"] and metrics["regular_weeks"]>0),
        "gate_duplicate_player_play_zero":bool(metrics["duplicate_player_play_rows"]==0),
        "gate_short_flag_binary":bool(metrics["short_flag_binary"]),
    })
    return rb,metrics


def add_prior_support(events:pd.DataFrame)->pd.DataFrame:
    g=(events.groupby(["season","week","game_id","team","player_id"],as_index=False)
       .agg(carries=("play_id","size"),short_yardage_carries=("short_yardage","sum"))
       .sort_values(["player_id","team","season","week","game_id"],kind="stable")
       .reset_index(drop=True))
    g["prior_same_team_carries"]=(
        g.groupby(["player_id","team"],sort=False)["carries"].cumsum()-g["carries"]
    )
    g["prior_same_team_short_yardage_carries"]=(
        g.groupby(["player_id","team"],sort=False)["short_yardage_carries"].cumsum()-g["short_yardage_carries"]
    )
    return g


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    meta=[]; roster=hv.load_rosters(meta)
    ev=[]; rows=[]
    for s in SEASONS:
        q,m=load_events(s,roster); ev.append(q); rows.append(m)
    events=pd.concat(ev,ignore_index=True)
    games=add_prior_support(events)
    metrics=pd.DataFrame(rows)
    support=[]
    for s in TARGET_SEASONS:
        q=games.loc[games.season.eq(s)&games.prior_same_team_carries.ge(MIN_PRIOR_SAME_TEAM_CARRIES)].copy()
        n_pg=int(len(q)); n_pl=int(q.player_id.nunique())
        support.append({
            "season":s,
            "min_prior_same_team_carries":MIN_PRIOR_SAME_TEAM_CARRIES,
            "supported_player_games":n_pg,
            "supported_players":n_pl,
            "gate_supported_player_games_ge_500":bool(n_pg>=MIN_SUPPORTED_PLAYER_GAMES),
            "gate_supported_players_ge_60":bool(n_pl>=MIN_SUPPORTED_PLAYERS),
        })
    source_cols=[c for c in metrics.columns if c.startswith("gate_")]
    source_ok=bool(metrics[source_cols].all(axis=None))
    support_ok=all(x["gate_supported_player_games_ge_500"] and x["gate_supported_players_ge_60"] for x in support)
    ready=source_ok and support_ok
    result={
        "version":VERSION,
        "disposition":"PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_SOURCE_READY" if ready else "PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_SOURCE_BLOCKED",
        "frozen_definition":{"downs":[3,4],"ydstogo_min_exclusive":0,"ydstogo_max_inclusive":2},
        "frozen_min_prior_same_team_carries":MIN_PRIOR_SAME_TEAM_CARRIES,
        "metrics":rows,
        "support":support,
        "predictive_candidates_scored":0,
        "outcomes_2026_read":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }
    pd.DataFrame(meta).to_csv(a.out_dir/"source_manifest.csv",index=False)
    metrics.to_csv(a.out_dir/"source_metrics.csv",index=False)
    games.to_csv(a.out_dir/"player_game_prior_support.csv",index=False)
    pd.DataFrame(support).to_csv(a.out_dir/"support_metrics.csv",index=False)
    (a.out_dir/"source_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
