#!/usr/bin/env python3
"""Outcome-free diagnostics for strict-prior player/room regime transitions.

Summarizes how often pregame context indicates player usage movement, team change,
and room continuity loss. This is engineering/qualification evidence only: it never
reads target-game outcomes, sportsbook data, or fits a model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_transition_diagnostics(df: pd.DataFrame, usage_delta_threshold: float = 0.05,
                                 room_overlap_threshold: float = 0.70) -> tuple[pd.DataFrame,pd.DataFrame]:
    x=df.copy(); x.columns=[str(c).strip().lower() for c in x.columns]
    required={"season","week","position","player_identity_key","team_change_prior",
              "target_share_delta_vs_roll3","rush_share_delta_vs_roll3",
              "returning_target_opportunity_overlap","returning_rush_opportunity_overlap",
              "any_context_unknown_flag"}
    missing=required-set(x.columns)
    if missing: raise RuntimeError(f"missing columns: {sorted(missing)}")
    key=["season","week","player_identity_key"]
    if x.duplicated(key).any(): raise RuntimeError("duplicate player-period keys")

    td=pd.to_numeric(x.target_share_delta_vs_roll3,errors="coerce")
    rd=pd.to_numeric(x.rush_share_delta_vs_roll3,errors="coerce")
    to=pd.to_numeric(x.returning_target_opportunity_overlap,errors="coerce")
    ro=pd.to_numeric(x.returning_rush_opportunity_overlap,errors="coerce")
    team_change=pd.to_numeric(x.team_change_prior,errors="coerce").fillna(0).gt(0)
    known=pd.to_numeric(x.any_context_unknown_flag,errors="coerce").fillna(1).eq(0)

    x["target_usage_transition_flag"]=(td.abs()>=usage_delta_threshold) & td.notna()
    x["rush_usage_transition_flag"]=(rd.abs()>=usage_delta_threshold) & rd.notna()
    x["target_room_churn_flag"]=(to<room_overlap_threshold) & to.notna()
    x["rush_room_churn_flag"]=(ro<room_overlap_threshold) & ro.notna()
    x["team_change_flag"]=team_change
    x["known_context_flag"]=known
    flags=["target_usage_transition_flag","rush_usage_transition_flag","target_room_churn_flag",
           "rush_room_churn_flag","team_change_flag"]
    x["any_transition_flag"]=x[flags].any(axis=1)
    x["joint_player_room_transition_flag"]=(
        (x.target_usage_transition_flag & x.target_room_churn_flag) |
        (x.rush_usage_transition_flag & x.rush_room_churn_flag)
    )

    rows=[]
    for (season,position),g in x.groupby(["season","position"],dropna=False):
        k=g[g.known_context_flag]
        row={"season":season,"position":position,"rows":len(g),"known_rows":len(k),
             "known_context_rate":len(k)/len(g) if len(g) else 0.0}
        for f in [*flags,"any_transition_flag","joint_player_room_transition_flag"]:
            row[f.replace("_flag","_rate")]=float(k[f].mean()) if len(k) else np.nan
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values(["season","position"]).reset_index(drop=True)
    return x,summary


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--detail-out",type=Path,required=True)
    p.add_argument("--summary-out",type=Path,required=True)
    p.add_argument("--usage-delta-threshold",type=float,default=0.05)
    p.add_argument("--room-overlap-threshold",type=float,default=0.70)
    a=p.parse_args()
    detail,summary=build_transition_diagnostics(pd.read_csv(a.input),a.usage_delta_threshold,a.room_overlap_threshold)
    a.detail_out.parent.mkdir(parents=True,exist_ok=True); a.summary_out.parent.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.detail_out,index=False); summary.to_csv(a.summary_out,index=False)
    print(summary.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
