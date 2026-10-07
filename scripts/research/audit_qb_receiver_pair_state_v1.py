#!/usr/bin/env python3
"""QB-Receiver Pair State V1 source/support audit.

Public nflverse/nflreadpy only. 2026 target boundary is strictly through Week 4.
No sportsbook data, no Week-5 outcomes, no model fitting, no production mutation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION="QB_RECEIVER_PAIR_STATE_V1"
SEASONS=(2022,2023,2024,2025,2026)
TARGET_SEASON=2026
TARGET_WEEK=5
HISTORY_MAX_WEEK=4

def to_pd(x):
    if isinstance(x,pd.DataFrame): return x.copy()
    if hasattr(x,"to_pandas"): return x.to_pandas()
    return pd.DataFrame(x)

def lower(x):
    y=to_pd(x)
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y

def num(x): return pd.to_numeric(x,errors="coerce")

def clean(v):
    if v is None or pd.isna(v): return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s

def team(v):
    try:
        t=canon_team(v)
        return "WAS" if t=="WSH" else t
    except Exception:
        return clean(v).upper()

def first(df,names,default=""):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series(default,index=df.index)

def regular_only(df):
    x=df.copy()
    c="season_type" if "season_type" in x.columns else "game_type" if "game_type" in x.columns else None
    if c:
        s=x[c].astype(str).str.upper()
        keep=s.isin(["REG","REGULAR","RS",""])
        if keep.any(): x=x.loc[keep].copy()
    return x

def load_pbp(season):
    import nflreadpy as nfl
    x=regular_only(lower(nfl.load_pbp(seasons=[int(season)])))
    for c in [
        "season","week","game_id","posteam",
        "passer_player_id","passer_player_name",
        "receiver_player_id","receiver_player_name",
        "pass_attempt","sack","two_point_attempt","complete_pass",
        "passing_yards","air_yards","yards_after_catch"
    ]:
        if c not in x.columns: x[c]=np.nan
    x["season"]=num(x["season"]).fillna(season).astype(int)
    x["week"]=num(x["week"])
    x["team"]=x["posteam"].map(team)
    raw_pass=num(x["pass_attempt"]).fillna(0).eq(1)
    sack=num(x["sack"]).fillna(0).eq(1)
    two=num(x["two_point_attempt"]).fillna(0).eq(1)
    x["official_pass_attempt"]=(raw_pass & ~sack & ~two)
    x["passer_id"]=x["passer_player_id"].map(clean)
    x["receiver_id"]=x["receiver_player_id"].map(clean)
    x["passer_name"]=x["passer_player_name"].map(clean)
    x["receiver_name"]=x["receiver_player_name"].map(clean)
    x["target_like"]=x["official_pass_attempt"] & (x["receiver_id"].ne("") | x["receiver_name"].ne(""))
    if season==TARGET_SEASON:
        bad=sorted(x.loc[x["week"].ge(TARGET_WEEK),"week"].dropna().astype(int).unique().tolist())
        if bad:
            raise RuntimeError(f"2026 target/future PBP already present; refusing audit: weeks={bad}")
        x=x.loc[x["week"].le(HISTORY_MAX_WEEK)].copy()
    return x

def load_week5_schedule():
    import nflreadpy as nfl
    try: x=lower(nfl.load_schedules(seasons=[TARGET_SEASON]))
    except TypeError: x=lower(nfl.load_schedules(TARGET_SEASON))
    x=regular_only(x)
    x=x.loc[num(x["season"]).eq(TARGET_SEASON)&num(x["week"]).eq(TARGET_WEEK)].copy()
    hc="home_team" if "home_team" in x.columns else "home"
    ac="away_team" if "away_team" in x.columns else "away"
    teams=[]
    for r in x.itertuples(index=False):
        h=team(getattr(r,hc)); a=team(getattr(r,ac))
        if h: teams.append(h)
        if a: teams.append(a)
    return sorted(set(teams))

def load_position_map():
    import nflreadpy as nfl
    raw=lower(nfl.load_player_stats(seasons=[TARGET_SEASON],summary_level="week"))
    if "week" in raw.columns:
        bad=sorted(num(raw.loc[num(raw["week"]).ge(TARGET_WEEK),"week"]).dropna().astype(int).unique().tolist())
        if bad:
            raise RuntimeError(f"2026 target/future weekly stats already present; refusing audit: weeks={bad}")
        raw=raw.loc[num(raw["week"]).le(HISTORY_MAX_WEEK)].copy()
    raw["player_id"]=first(raw,["player_id","gsis_id","player_gsis_id"]).map(clean)
    raw["position"]=first(raw,["position","position_group","pos"]).astype(str).str.upper().str.strip()
    p=(raw.loc[raw["player_id"].ne("")&raw["position"].ne("")]
         [["player_id","position"]].drop_duplicates("player_id",keep="last"))
    return dict(zip(p["player_id"],p["position"]))

def source_summary(pbp):
    rows=[]
    for season,g in pbp.groupby("season",sort=True):
        off=g.loc[g["official_pass_attempt"]].copy()
        targ=g.loc[g["target_like"]].copy()
        joint=(targ["passer_id"].ne("") & targ["receiver_id"].ne(""))
        rows.append({
            "season":int(season),
            "pbp_rows":int(len(g)),
            "official_pass_attempts":int(len(off)),
            "target_events":int(len(targ)),
            "passer_id_coverage":float(targ["passer_id"].ne("").mean()) if len(targ) else 0.0,
            "receiver_id_coverage":float(targ["receiver_id"].ne("").mean()) if len(targ) else 0.0,
            "joint_pair_id_coverage":float(joint.mean()) if len(targ) else 0.0,
            "distinct_passers":int(targ.loc[targ["passer_id"].ne(""),"passer_id"].nunique()),
            "distinct_receivers":int(targ.loc[targ["receiver_id"].ne(""),"receiver_id"].nunique()),
            "distinct_pairs":int(targ.loc[joint,["passer_id","receiver_id"]].drop_duplicates().shape[0]),
            "max_week":int(num(g["week"]).max()) if num(g["week"]).notna().any() else None,
        })
    return pd.DataFrame(rows)

def live_pairs(pbp2026,scheduled,positions):
    x=pbp2026.loc[pbp2026["official_pass_attempt"]].copy()
    x=x.loc[x["team"].isin(scheduled)].copy()

    # Primary passer proxy: most official attempts through Week 4.
    att=(x.loc[x["passer_id"].ne("")]
           .groupby(["team","passer_id"],as_index=False)
           .size().rename(columns={"size":"attempts"}))
    if len(att):
        att=att.sort_values(["team","attempts","passer_id"],ascending=[True,False,True])
        primary=att.drop_duplicates("team")[["team","passer_id","attempts"]].rename(
            columns={"passer_id":"primary_passer_id","attempts":"primary_attempts"})
    else:
        primary=pd.DataFrame(columns=["team","primary_passer_id","primary_attempts"])

    targ=x.loc[x["target_like"] & x["passer_id"].ne("") & x["receiver_id"].ne("")].copy()
    targ["_complete"]=num(targ["complete_pass"]).fillna(0).eq(1)
    targ["_yards"]=num(targ["passing_yards"]).fillna(0.0)
    targ["_air"]=num(targ["air_yards"])
    targ["_yac"]=num(targ["yards_after_catch"])
    targ["_game"]=targ["game_id"].astype(str)

    pair=(targ.groupby(["team","passer_id","receiver_id"],as_index=False)
      .agg(
        pair_games=("_game","nunique"),
        pair_targets=("receiver_id","size"),
        pair_receptions=("_complete","sum"),
        pair_receiving_yards=("_yards","sum"),
        pair_air_yards=("_air","sum"),
        pair_yac=("_yac","sum"),
      ))
    pair["pair_catch_rate"]=np.where(pair["pair_targets"]>0,pair["pair_receptions"]/pair["pair_targets"],np.nan)
    pair["pair_ypt"]=np.where(pair["pair_targets"]>0,pair["pair_receiving_yards"]/pair["pair_targets"],np.nan)
    pair["pair_air_per_target"]=np.where(pair["pair_targets"]>0,pair["pair_air_yards"]/pair["pair_targets"],np.nan)
    pair["pair_yac_per_reception"]=np.where(pair["pair_receptions"]>0,pair["pair_yac"]/pair["pair_receptions"],np.nan)

    passer_tot=(targ.groupby(["team","passer_id"],as_index=False).size()
                  .rename(columns={"size":"passer_total_targets"}))
    recv_tot=(targ.groupby(["team","receiver_id"],as_index=False).size()
                .rename(columns={"size":"receiver_total_targets"}))
    pair=pair.merge(passer_tot,on=["team","passer_id"],how="left",validate="many_to_one")
    pair=pair.merge(recv_tot,on=["team","receiver_id"],how="left",validate="many_to_one")
    pair["receiver_share_of_passer_targets"]=pair["pair_targets"]/pair["passer_total_targets"]
    pair["receiver_targets_from_this_passer_share"]=pair["pair_targets"]/pair["receiver_total_targets"]

    live=primary.merge(pair,left_on=["team","primary_passer_id"],right_on=["team","passer_id"],how="left")
    live=live.loc[live["receiver_id"].notna()].copy()
    live["position"]=live["receiver_id"].map(positions).fillna("")
    live["scheduled_week5"]=True

    multi=(pair.groupby(["team","receiver_id"])["passer_id"].nunique().reset_index(name="distinct_passers"))
    live=live.merge(multi,on=["team","receiver_id"],how="left",validate="many_to_one")
    return primary,live

def audit():
    frames=[]
    for s in SEASONS:
        q=load_pbp(s)
        if q.empty: raise RuntimeError(f"zero PBP rows for {s}")
        frames.append(q)
    all_pbp=pd.concat(frames,ignore_index=True,sort=False)
    hist=source_summary(all_pbp)
    sched=load_week5_schedule()
    if len(sched)!=30:
        raise RuntimeError(f"expected 30 scheduled Week-5 teams, got {len(sched)}")
    pos=load_position_map()
    p2026=all_pbp.loc[all_pbp["season"].eq(TARGET_SEASON)].copy()
    primary,live=live_pairs(p2026,sched,pos)

    target_events=p2026.loc[p2026["target_like"]].copy()
    joint_cov=float((target_events["passer_id"].ne("")&target_events["receiver_id"].ne("")).mean()) if len(target_events) else 0.0
    primary_teams=int(primary["team"].nunique())
    live_pairs_n=int(len(live))
    pos_cov=float(live["position"].ne("").mean()) if len(live) else 0.0
    thresholds={str(t):int((num(live["pair_targets"])>=t).sum()) for t in (1,5,10,15)}
    game_thr={str(t):int((num(live["pair_games"])>=t).sum()) for t in (1,2,3)}
    multi_recv=int((num(live["distinct_passers"])>=2).sum()) if len(live) else 0

    source_all=all(int((hist.loc[hist["season"].eq(s),"pbp_rows"].iloc[0]))>0 for s in SEASONS)
    chrono_ok=bool(int(num(p2026["week"]).max())<=HISTORY_MAX_WEEK)
    ready=bool(source_all and chrono_ok and joint_cov>=.98 and primary_teams>=28 and live_pairs_n>=100 and pos_cov>=.75)
    safe=bool(source_all and chrono_ok and joint_cov>=.95)
    disposition=("QB_RECEIVER_PAIR_STATE_SOURCE_READY" if ready else
                 "QB_RECEIVER_PAIR_STATE_SOURCE_PARTIAL" if safe else
                 "QB_RECEIVER_PAIR_STATE_SOURCE_NOT_READY")
    result={
        "version":VERSION,
        "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "target_season":TARGET_SEASON,
        "target_week":TARGET_WEEK,
        "disposition":disposition,
        "scheduled_week5_teams":len(sched),
        "source_2022_2026_all_nonzero":source_all,
        "max_2026_source_week":int(num(p2026["week"]).max()),
        "joint_pair_id_coverage_2026_target_events":joint_cov,
        "scheduled_teams_with_primary_passer_proxy":primary_teams,
        "live_current_primary_passer_receiver_pairs":live_pairs_n,
        "live_pair_position_coverage":pos_cov,
        "pair_target_support_counts":thresholds,
        "pair_game_support_counts":game_thr,
        "live_receivers_with_multiple_2026_passers":multi_recv,
        "sportsbook_inputs_used":0,
        "week5_outcomes_read":0,
        "candidate_models_fit":0,
        "production_changed":False,
    }
    return hist,primary,live,result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    hist,primary,live,result=audit()
    hist.to_csv(a.out_dir/"qb_receiver_pair_state_source_summary.csv",index=False)
    primary.to_csv(a.out_dir/"qb_receiver_pair_state_week5_primary_passer_proxy.csv",index=False)
    live.to_csv(a.out_dir/"qb_receiver_pair_state_week5_live_pairs.csv",index=False)
    (a.out_dir/"qb_receiver_pair_state_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    print("\nSOURCE SUMMARY")
    print(hist.to_string(index=False))
    print("\nPRIMARY PASSERS")
    print(primary.to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
