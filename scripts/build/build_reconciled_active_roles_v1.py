#!/usr/bin/env python3
"""Materialize production-compatible active roles from locked availability state.

Unavailable players remain in current_player_availability.csv for audit, but are
excluded from this active-role artifact. No model or sportsbook input is read.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from scripts.utils.player_identity_v3 import player_name_key

REQ={"team","player","player_clean_key","position","position_group","raw_depth_role","depth_index","definitive_unavailable","role_after_availability","availability_authority","final_availability_state","availability_generated_at_utc"}

def build(df:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    x=df.copy(); x.columns=[str(c).lower() for c in x.columns]; miss=REQ-set(x.columns)
    if miss: raise RuntimeError(f"availability artifact missing {sorted(miss)}")
    if x.duplicated(["team","player_clean_key"]).any(): raise RuntimeError("duplicate current availability identity")
    active=x[pd.to_numeric(x.definitive_unavailable,errors="coerce").fillna(1).eq(0)].copy(); active["role"]=active.role_after_availability.astype(str).str.strip()
    needs_role=active.position_group.astype(str).str.upper().isin(["QB","RB","FB","TE"])
    if active.loc[needs_role,"role"].eq("").any(): raise RuntimeError("eligible ordinal-role player missing reconciled role")
    active["player_key"]=active.get("player_key",active.player.map(player_name_key)).astype(str); active["source_asof_utc"]=active.get("source_asof_utc",active.availability_generated_at_utc)
    cols=[c for c in ["player","team","role","position","position_group","player_key","player_clean_key","depth_index","raw_depth_role","availability_authority","final_availability_state","source_asof_utc","availability_generated_at_utc"] if c in active.columns]
    out=active[cols].copy().sort_values(["team","position_group","depth_index","player_clean_key"],na_position="last").reset_index(drop=True)
    if out.duplicated(["team","player_clean_key"]).any(): raise RuntimeError("duplicate active role identity")
    bad=[]
    for (team,grp),g in out[out.position_group.astype(str).str.upper().isin(["QB","RB","FB","TE"])].groupby(["team","position_group"]):
        prefix="RB" if str(grp).upper() in {"RB","FB"} else str(grp).upper(); ranks=[]
        for r in g.role.astype(str):
            if r.startswith(prefix) and r[len(prefix):].isdigit(): ranks.append(int(r[len(prefix):]))
        if ranks and sorted(ranks)!=list(range(1,len(ranks)+1)): bad.append({"team":team,"group":grp,"ranks":sorted(ranks)})
    if bad: raise RuntimeError(f"non-gap-free active ordinal roles: {bad[:5]}")
    meta={"rows":int(len(out)),"teams":int(out.team.nunique()),"excluded_definitive_unavailable":int(len(x)-len(active)),"sportsbook_inputs_used":0,"source":"current_player_availability_v1","production_candidate_only":True}
    return out,meta

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--availability",type=Path,default=Path("data/current_player_availability.csv")); ap.add_argument("--out",type=Path,default=Path("data/roles_ourlads_active_v1.csv")); ap.add_argument("--status",type=Path,default=Path("data/roles_ourlads_active_v1_status.json")); a=ap.parse_args()
    x=pd.read_csv(a.availability); out,meta=build(x); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); a.status.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(meta,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
