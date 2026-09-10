#!/usr/bin/env python3
"""Reconcile current depth, weekly injury status and official inactives.

Implementation branch only; not wired to production Full Slate until locked and
verified. Sportsbook data is never read.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

DEF_UNAVAILABLE={"OUT","IR","PUP","RESERVE/INJURED","INJURED RESERVE"}
UNCERTAIN={"DOUBTFUL","QUESTIONABLE"}

def text(v): return "" if v is None or pd.isna(v) else str(v).strip()
def key(v): return player_name_key(v,strip_suffix=True)
def norm_status(v): return " ".join(text(v).upper().replace("-"," ").split())

def resolve_state(*,official_inactive,official_complete,injury_status,injury_designation,ourlads_status):
    if bool(official_complete) and bool(official_inactive): return "UNAVAILABLE_OFFICIAL_INACTIVE","official_inactive","listed on complete official game-day inactive section"
    vals={norm_status(injury_status),norm_status(injury_designation)}-{""}
    if any(v in DEF_UNAVAILABLE or v.startswith("OUT ") for v in vals): return "UNAVAILABLE_REPORTED","weekly_injury_report","definitive reported non-participation"
    if norm_status(ourlads_status)=="INACTIVE": return "UNAVAILABLE_DEPTH_SOURCE","ourlads_depth","Ourlads depth source marked inactive"
    if any(v in UNCERTAIN for v in vals): return "UNCERTAIN","weekly_injury_report","non-definitive game-status uncertainty"
    if bool(official_complete) and official_inactive is False: return "AVAILABLE_OFFICIAL_ACTIVE","official_inactive","absent from complete official inactive section"
    if vals: return "AVAILABLE_REPORTED","weekly_injury_report","report present without definitive unavailable designation"
    if norm_status(ourlads_status)=="ACTIVE": return "AVAILABLE_DEPTH_SOURCE","ourlads_depth","present active on current depth source"
    return "UNKNOWN","none","no authoritative availability fact"

def rerank(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); out["role_after_availability"]=""; out["role_rank_after_availability"]=pd.NA
    eligible=out[~out.definitive_unavailable.eq(1)].copy()
    for (team,grp),g in eligible.groupby(["team","position_group"],dropna=False):
        grp=str(grp).upper(); idx=g.sort_values(["depth_index","raw_depth_role","player_clean_key"],na_position="last").index
        if grp in {"QB","RB","FB","TE"}:
            prefix="RB" if grp in {"RB","FB"} else grp
            for rank,i in enumerate(idx,1): out.at[i,"role_after_availability"]=f"{prefix}{rank}"; out.at[i,"role_rank_after_availability"]=rank
        else:
            for i in idx: out.at[i,"role_after_availability"]=text(out.at[i,"raw_depth_role"]); out.at[i,"role_rank_after_availability"]=out.at[i,"depth_index"]
    return out

def build(depth:pd.DataFrame,injuries:pd.DataFrame,official:pd.DataFrame|None=None)->tuple[pd.DataFrame,dict]:
    d=depth.copy(); d.columns=[str(c).lower() for c in d.columns]
    need={"team","player","status","role","position","position_group","depth_index"}; miss=need-set(d.columns)
    if miss: raise RuntimeError(f"depth/status missing {sorted(miss)}")
    d["team"]=d.team.map(canon_team); d["player_clean_key"]=d.player.map(key); d=d[d.player_clean_key.ne("")].copy()
    d=d.sort_values(["team","player_clean_key","depth_index"],na_position="last").drop_duplicates(["team","player_clean_key"],keep="first").rename(columns={"status":"ourlads_status","role":"raw_depth_role"})
    i=injuries.copy() if injuries is not None else pd.DataFrame()
    if not i.empty:
        i.columns=[str(c).lower() for c in i.columns]; i["team"]=i.team.map(canon_team); i["player_clean_key"]=i.player.map(key)
        keep=[c for c in ["team","player_clean_key","status","designation","practice_status","source","report_date"] if c in i]
        i=i[keep].drop_duplicates(["team","player_clean_key"],keep="last").rename(columns={"status":"injury_status","source":"injury_source"}); d=d.merge(i,on=["team","player_clean_key"],how="left",validate="one_to_one")
    else: d["injury_status"]=""; d["designation"]=""; d["injury_source"]=""
    o=official.copy() if official is not None else pd.DataFrame()
    if not o.empty:
        o.columns=[str(c).lower() for c in o.columns]
        for req in ["team","section_complete"]:
            if req not in o: raise RuntimeError(f"official inactive source missing {req}")
        o["team"]=o.team.map(canon_team); o["player_clean_key"]=o.get("player",pd.Series("",index=o.index)).map(key)
        teams_complete=set(o.loc[pd.to_numeric(o.section_complete,errors="coerce").fillna(0).eq(1),"team"]); inactive_keys=set(zip(o.loc[o.player_clean_key.ne(""),"team"],o.loc[o.player_clean_key.ne(""),"player_clean_key"]))
        d["official_inactive_section_complete"]=d.team.isin(teams_complete).astype(int); d["official_inactive"]=[bool((t,k) in inactive_keys) if t in teams_complete else pd.NA for t,k in zip(d.team,d.player_clean_key)]
    else: d["official_inactive_section_complete"]=0; d["official_inactive"]=pd.NA
    states=[]
    for r in d.itertuples(index=False):
        oi=False if pd.isna(r.official_inactive) else bool(r.official_inactive)
        states.append(resolve_state(official_inactive=oi,official_complete=bool(r.official_inactive_section_complete),injury_status=getattr(r,"injury_status",""),injury_designation=getattr(r,"designation",""),ourlads_status=r.ourlads_status))
    d[["final_availability_state","availability_authority","availability_reason"]]=pd.DataFrame(states,index=d.index)
    d["definitive_unavailable"]=d.final_availability_state.str.startswith("UNAVAILABLE_").astype(int); d=rerank(d); d["eligible_for_opportunity"]=(1-d.definitive_unavailable).astype(int); d["availability_generated_at_utc"]=datetime.now(timezone.utc).isoformat()
    unavailable=d[d.definitive_unavailable.eq(1)]
    if unavailable.role_after_availability.astype(str).str.strip().ne("").any(): raise RuntimeError("definitive unavailable player retained active reconciled role")
    meta={"rows":int(len(d)),"teams":int(d.team.nunique()),"definitive_unavailable":int(d.definitive_unavailable.sum()),"uncertain":int(d.final_availability_state.eq("UNCERTAIN").sum()),"unknown":int(d.final_availability_state.eq("UNKNOWN").sum()),"official_complete_teams":int(d.loc[d.official_inactive_section_complete.eq(1),"team"].nunique()),"sportsbook_inputs_used":0,"production_wired":False,"generated_at_utc":datetime.now(timezone.utc).isoformat()}
    return d,meta

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--depth",type=Path,default=Path("data/roles_ourlads_status_v1.csv")); ap.add_argument("--injuries",type=Path,default=Path("data/injuries.csv")); ap.add_argument("--official",type=Path,default=Path("data/official_inactives_v1.csv")); ap.add_argument("--out",type=Path,default=Path("data/current_player_availability.csv")); ap.add_argument("--status",type=Path,default=Path("data/current_player_availability_status.json")); a=ap.parse_args()
    depth=pd.read_csv(a.depth); injuries=pd.read_csv(a.injuries) if a.injuries.exists() and a.injuries.stat().st_size else pd.DataFrame(); official=pd.read_csv(a.official) if a.official.exists() and a.official.stat().st_size else pd.DataFrame()
    out,meta=build(depth,injuries,official); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); a.status.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(meta,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
