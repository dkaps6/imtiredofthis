#!/usr/bin/env python3
"""Grade the frozen Week-4+ projection-authority move-direction forward shadow.

This implements docs/research/WEEK4_PLUS_PROJECTION_AUTHORITY_MOVE_DIRECTION_FORWARD_V1_PLAN.md
without changing the frozen definitions or gates.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TOL=1e-12
SEED=42040
BOOT=10000

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--graded",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    d=pd.read_csv(a.graded,low_memory=False)
    d=d.loc[d["bet_result"].isin(["WIN","LOSS"])].copy()
    for c in ("mc_proj","model_proj","vegas_line","model_closer_than_vegas","unit_result"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["mc_gap"]=d["mc_proj"]-d["vegas_line"]
    d["final_gap"]=d["model_proj"]-d["vegas_line"]
    same=((d.mc_gap>TOL)&(d.final_gap>TOL))|((d.mc_gap<-TOL)&(d.final_gap<-TOL))
    strengthened=same&(d.final_gap.abs()>d.mc_gap.abs()+TOL)
    weakened=same&(d.final_gap.abs()<d.mc_gap.abs()-TOL)
    d["authority_move_state"]="CROSSED_OR_ON_LINE"
    d.loc[strengthened,"authority_move_state"]="STRENGTHENED"
    d.loc[weakened,"authority_move_state"]="WEAKENED"
    d.loc[same&~strengthened&~weakened,"authority_move_state"]="UNCHANGED_DISTANCE"
    d["win"]=d["bet_result"].eq("WIN").astype(float)

    def cell(q):
        return {
            "rows":int(len(q)),
            "games":int(q["event_id"].nunique()),
            "win_rate":float(q["win"].mean()) if len(q) else None,
            "units":float(q["unit_result"].sum()) if len(q) else 0.0,
            "model_closer_rate":float(q["model_closer_than_vegas"].mean()) if len(q) else None,
        }
    cells={s:cell(d.loc[d.authority_move_state.eq(s)]) for s in
           ("STRENGTHENED","WEAKENED","UNCHANGED_DISTANCE","CROSSED_OR_ON_LINE")}
    s=d.loc[d.authority_move_state.eq("STRENGTHENED")]
    w=d.loc[d.authority_move_state.eq("WEAKENED")]
    point_win=float(s.win.mean()-w.win.mean())
    point_closer=float(s.model_closer_than_vegas.mean()-w.model_closer_than_vegas.mean())

    sw=d.loc[d.authority_move_state.isin(["STRENGTHENED","WEAKENED"])].copy()
    games=sorted(sw.event_id.dropna().unique().tolist())
    arr=np.zeros((len(games),2,4),dtype=float)
    states=("STRENGTHENED","WEAKENED")
    for i,eid in enumerate(games):
        q=sw.loc[sw.event_id.eq(eid)]
        for j,state in enumerate(states):
            z=q.loc[q.authority_move_state.eq(state)]
            arr[i,j,0]=len(z)
            arr[i,j,1]=z.win.sum()
            cc=z.model_closer_than_vegas.dropna()
            arr[i,j,2]=cc.sum()
            arr[i,j,3]=len(cc)
    rng=np.random.default_rng(SEED)
    counts=rng.multinomial(len(games),[1.0/len(games)]*len(games),size=BOOT)
    agg=np.einsum("bg,gsk->bsk",counts,arr)
    wr=agg[:,:,1]/np.where(agg[:,:,0]>0,agg[:,:,0],np.nan)
    cr=agg[:,:,2]/np.where(agg[:,:,3]>0,agg[:,:,3],np.nan)
    win_diff=wr[:,0]-wr[:,1]
    closer_diff=cr[:,0]-cr[:,1]

    weeks=int(d["week"].nunique())
    support_ok=(weeks>=8 and len(s)>=400 and len(w)>=400)
    if not support_ok:
        disposition="FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT"
    else:
        win_ci=np.nanpercentile(win_diff,[2.5,97.5])
        closer_ci=np.nanpercentile(closer_diff,[2.5,97.5])
        confirmed=(
            point_win<=-0.05 and point_closer<=-0.08
            and win_ci[1]<0 and closer_ci[1]<0
        )
        disposition=("AUTHORITY_MOVE_DIRECTION_FORWARD_CONFIRMED" if confirmed
                     else "AUTHORITY_MOVE_DIRECTION_FORWARD_NOT_CONFIRMED")

    payload={
        "disposition":disposition,
        "frozen_plan":"WEEK4_PLUS_PROJECTION_AUTHORITY_MOVE_DIRECTION_FORWARD_V1",
        "weeks_observed":weeks,
        "strengthened_rows":int(len(s)),
        "weakened_rows":int(len(w)),
        "support_required":{"weeks":8,"strengthened_rows":400,"weakened_rows":400},
        "cells":cells,
        "primary":{
            "strengthened_minus_weakened_win_rate":point_win,
            "strengthened_minus_weakened_model_closer_rate":point_closer,
            "win_rate_diff_cluster_bootstrap_95ci":[float(x) for x in np.nanpercentile(win_diff,[2.5,97.5])],
            "model_closer_diff_cluster_bootstrap_95ci":[float(x) for x in np.nanpercentile(closer_diff,[2.5,97.5])],
            "bootstrap_reps":BOOT,
            "bootstrap_seed":SEED,
            "game_clusters":len(games),
        },
        "production_change_authorized":False,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
