#!/usr/bin/env python3
"""Source audit for named-player high-value scoring opportunity identity V1."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.utils.pbp import get_pbp

VERSION="PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1_SOURCE_AUDIT"
SEASONS=(2022,2023,2024,2025)
FAMS={"RB":{"RB","FB","HB"},"WR":{"WR","LWR","RWR","SWR"},"TE":{"TE"}}


def num(s): return pd.to_numeric(s,errors="coerce")


def family(pos):
    p=str(pos or "").upper().strip().replace("-","").replace(" ","")
    for fam,vals in FAMS.items():
        if p in vals: return fam
    if p.startswith("QB"): return "QB"
    return ""


def first(df,names):
    for c in names:
        if c in df.columns: return c
    return None


def normalize_roster(raw,season):
    if raw.empty: return pd.DataFrame()
    x=raw.copy(); x.columns=[str(c).lower() for c in x.columns]
    wc=first(x,["week"]); tc=first(x,["team","club_code"]); ic=first(x,["gsis_id","player_id"])
    pc=first(x,["position","depth_chart_position","ngs_position","pos"])
    if not all([wc,tc,ic,pc]): raise RuntimeError(f"roster schema missing {season}")
    out=pd.DataFrame({
        "season":season,"week":num(x[wc]),"team":x[tc].map(canon_team),
        "player_id":x[ic].astype("string").fillna("").str.strip(),
        "position":x[pc].astype("string").fillna("").str.upper().str.strip(),
    })
    out["family"]=out.position.map(family)
    out=out.loc[out.week.notna() & out.player_id.ne("") & out.team.astype(str).ne("")].copy()
    out["week"]=out.week.astype(int)
    return out.drop_duplicates(["season","week","team","player_id"])


def load_rosters(meta):
    parts=[]
    for s in SEASONS:
        q=m78.load_weekly_rosters(s,meta)
        parts.append(normalize_roster(q,s))
    return pd.concat(parts,ignore_index=True)


def id_col(df,cands):
    return next((c for c in cands if c in df.columns),None)


def audit_season(season,roster):
    p=get_pbp(season,min_rows=1000).copy()
    p.columns=[str(c).lower() for c in p.columns]
    if "season_type" in p.columns:
        q=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if not q.empty: p=q
    wc=first(p,["week"]); tc=first(p,["posteam"])
    yc=first(p,["yardline_100"]); gc=first(p,["game_id"]); pc=first(p,["play_id"])
    rc=id_col(p,["rusher_player_id","rusher_id","rusher_gsis_id"])
    xc=id_col(p,["receiver_player_id","receiver_id","receiver_gsis_id"])
    if not all([wc,tc,yc,rc,xc]): raise RuntimeError(f"PBP core schema missing {season}")

    week=num(p[wc]); team=p[tc].map(canon_team); yard=num(p[yc])
    rush=num(p.get("rush_attempt",pd.Series(0,index=p.index))).eq(1)
    pas=num(p.get("pass_attempt",pd.Series(0,index=p.index))).eq(1)
    rpid=p[rc].astype("string").fillna("").str.strip()
    xpid=p[xc].astype("string").fillna("").str.strip()

    rush_all=pd.DataFrame({
        "season":season,"week":week,"team":team,"yardline_100":yard,"player_id":rpid,
        "game_id":p[gc].astype(str) if gc else "",
        "play_id":p[pc].astype(str) if pc else p.index.astype(str),
    }).loc[rush].copy()
    receiver_name_col=first(p,["receiver_player_name","receiver_name"])
    has_receiver=(p[receiver_name_col].astype("string").fillna("").str.strip().ne("") if receiver_name_col else xpid.ne(""))
    tar_all=pd.DataFrame({
        "season":season,"week":week,"team":team,"yardline_100":yard,"player_id":xpid,
        "game_id":p[gc].astype(str) if gc else "",
        "play_id":p[pc].astype(str) if pc else p.index.astype(str),
        "air_yards":num(p["air_yards"]) if "air_yards" in p.columns else np.nan,
    }).loc[pas & has_receiver].copy()

    rusher_id_coverage=float(rush_all.player_id.ne("").mean()) if len(rush_all) else 0.0
    receiver_id_coverage=float(tar_all.player_id.ne("").mean()) if len(tar_all) else 0.0
    rushq=rush_all.loc[rush_all.player_id.ne("")].copy()
    tarq=tar_all.loc[tar_all.player_id.ne("")].copy()

    r=roster.loc[roster.season.eq(season),["season","week","team","player_id","position","family"]]
    rushq=rushq.merge(r,on=["season","week","team","player_id"],how="left",validate="many_to_one")
    tarq=tarq.merge(r,on=["season","week","team","player_id"],how="left",validate="many_to_one")

    rushq["inside20"]=rushq.yardline_100.le(20)
    rushq["inside10"]=rushq.yardline_100.le(10)
    rushq["inside5"]=rushq.yardline_100.le(5)
    tarq["inside20"]=tarq.yardline_100.le(20)
    tarq["inside10"]=tarq.yardline_100.le(10)

    weeks=sorted(set(int(v) for v in week.dropna().astype(int).unique()))
    rush_by_week=rushq.loc[rushq.inside20].groupby("week").size().to_dict()
    tar_by_week=tarq.loc[tarq.inside20].groupby("week").size().to_dict()

    duplicate_rush=int(rushq.duplicated(["game_id","play_id","player_id"]).sum())
    duplicate_tar=int(tarq.duplicated(["game_id","play_id","player_id"]).sum())

    rush_pos_bridge=float(rushq.family.astype("string").fillna("").ne("").mean()) if len(rushq) else 0.0
    tar_pos_bridge=float(tarq.family.astype("string").fillna("").isin(["RB","WR","TE"]).mean()) if len(tarq) else 0.0
    skill_bridge=min(rush_pos_bridge,tar_pos_bridge)

    playerweeks=[]
    for channel,q in [("rush",rushq),("target",tarq)]:
        if q.empty: continue
        z=q.groupby(["season","week","team","player_id","family"],dropna=False).agg(
            opportunities=("play_id","size"),
            inside20=("inside20","sum"),
            inside10=("inside10","sum"),
            inside5=("inside5","sum") if channel=="rush" else ("inside10","sum"),
        ).reset_index()
        z["channel"]=channel
        playerweeks.append(z)
    pw=pd.concat(playerweeks,ignore_index=True) if playerweeks else pd.DataFrame()

    metrics={
        "season":season,
        "pbp_rows":int(len(p)),
        "regular_weeks":len(weeks),
        "rush_rows":int(len(rushq)),
        "target_rows":int(len(tarq)),
        "week_team_coverage_rush":float((rushq.week.notna() & rushq.team.astype(str).ne("")).mean()) if len(rushq) else 0.0,
        "week_team_coverage_target":float((tarq.week.notna() & tarq.team.astype(str).ne("")).mean()) if len(tarq) else 0.0,
        "yardline_coverage_rush":float(rushq.yardline_100.notna().mean()) if len(rushq) else 0.0,
        "yardline_coverage_target":float(tarq.yardline_100.notna().mean()) if len(tarq) else 0.0,
        "rusher_stable_identity_coverage":rusher_id_coverage,
        "receiver_stable_identity_coverage":receiver_id_coverage,
        "rusher_position_bridge":rush_pos_bridge,
        "receiver_position_bridge":tar_pos_bridge,
        "skill_position_bridge":skill_bridge,
        "weeks_with_rz_rush":sum(int(rush_by_week.get(w,0)>0) for w in weeks),
        "weeks_with_rz_target":sum(int(tar_by_week.get(w,0)>0) for w in weeks),
        "inside10_rushes":int(rushq.inside10.sum()),
        "inside5_rushes":int(rushq.inside5.sum()),
        "air_yards_coverage_target":float(tarq.air_yards.notna().mean()) if len(tarq) else 0.0,
        "duplicate_rush_player_play_rows":duplicate_rush,
        "duplicate_target_player_play_rows":duplicate_tar,
        "playerweeks":int(len(pw)),
    }
    gates={
        "pbp_nonempty":len(p)>1000,
        "week_team_coverage_ge_99":min(metrics["week_team_coverage_rush"],metrics["week_team_coverage_target"])>=.99,
        "yardline_coverage_ge_99":min(metrics["yardline_coverage_rush"],metrics["yardline_coverage_target"])>=.99,
        "rusher_identity_ge_99":metrics["rusher_stable_identity_coverage"]>=.99,
        "receiver_identity_ge_99":metrics["receiver_stable_identity_coverage"]>=.99,
        "position_bridge_ge_98":metrics["skill_position_bridge"]>=.98,
        "rz_rush_every_week":metrics["weeks_with_rz_rush"]==metrics["regular_weeks"],
        "rz_target_every_week":metrics["weeks_with_rz_target"]==metrics["regular_weeks"],
        "inside10_and_inside5_rush_present":metrics["inside10_rushes"]>0 and metrics["inside5_rushes"]>0,
        "duplicate_player_play_zero":duplicate_rush==0 and duplicate_tar==0,
    }
    metrics.update({f"gate_{k}":bool(v) for k,v in gates.items()})
    metrics["optional_endzone_air_yards_gate"]=metrics["air_yards_coverage_target"]>=.95
    return metrics,pw


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    meta=[]; roster=load_rosters(meta)
    metrics=[]; pws=[]
    for s in SEASONS:
        m,pw=audit_season(s,roster); metrics.append(m); pw["audit_season"]=s; pws.append(pw)
    md=pd.DataFrame(metrics); playerweeks=pd.concat(pws,ignore_index=True)

    core_cols=[c for c in md.columns if c.startswith("gate_")]
    source_ready=bool(len(md) and md[core_cols].all(axis=None))
    optional_endzone=bool(md.optional_endzone_air_yards_gate.all())
    result={
        "version":VERSION,
        "disposition":"PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY" if source_ready else "PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_BLOCKED",
        "optional_endzone_target_source_ready":optional_endzone,
        "seasons":list(SEASONS),
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "outcomes_2026_read":0,
        "predictive_models_fit":0,
        "production_changed":False,
        "metrics":metrics,
    }
    pd.DataFrame(meta).to_csv(a.out_dir/"high_value_opportunity_source_manifest.csv",index=False)
    roster.to_csv(a.out_dir/"high_value_opportunity_roster.csv",index=False)
    md.to_csv(a.out_dir/"high_value_opportunity_source_metrics.csv",index=False)
    playerweeks.to_csv(a.out_dir/"high_value_opportunity_playerweeks.csv",index=False)
    (a.out_dir/"high_value_opportunity_source_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
