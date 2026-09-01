#!/usr/bin/env python3
"""Mechanical repair wrapper for frozen RB STACK7 protocol.

Only replaces the pandas-incompatible inactive teammate-share diagnostic in v1.
No scientific candidate, gate, threshold, feature, or weight changes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scripts.backtest.evaluate_rb_stack7_official_availability as v1


def add_candidate_fixed(x: pd.DataFrame) -> pd.DataFrame:
    z=x.copy();gkeys=["season","week","team"]
    pool=z.groupby(gkeys)["parent_att"].transform("sum");z["team_parent_pool"]=pool;z["parent_share"]=np.where(pool.gt(0),z.parent_att/pool,0.0)
    inactive_component=z.parent_share*z.official_inactive
    team_inactive_share=inactive_component.groupby([z.season,z.week,z.team]).transform("sum")
    z["inactive_comp_parent_share"]=team_inactive_share-inactive_component
    z["returned_share_component"]=z.parent_share*z.returned_from_official_inactive
    team_ret=z.groupby(gkeys)["returned_share_component"].transform("sum")
    z["returned_comp_parent_share"]=team_ret-z.returned_share_component
    score=z.parent_share.where(z.official_inactive.eq(0),0.0)
    final=pd.Series(np.nan,index=z.index,dtype=float)
    affected=pd.Series(False,index=z.index)
    for _,g in z.groupby(gkeys,sort=False):
        s=score.loc[g.index].clip(lower=0);tot=float(s.sum())
        if float(g.official_inactive.sum())>0:affected.loc[g.index]=True
        if float(g.team_parent_pool.iloc[0])>0 and tot<=0:raise RuntimeError(f"all projected RB/FB inactive for {tuple(g[gkeys].iloc[0])}")
        final.loc[g.index]=s/tot if tot>0 else g.parent_share
    z["affected_team_game"]=affected.astype(int);z["candidate_share"]=final;z["candidate_att"]=z.candidate_share*pool
    implied=np.where(z.parent_att.gt(.10),z.parent_yards/z.parent_att,np.nan);implied=pd.Series(implied,index=z.index).replace([np.inf,-np.inf],np.nan)
    fallback=v1.num(z.get("stack_eff",pd.Series(np.nan,index=z.index))).replace([np.inf,-np.inf],np.nan)
    implied=implied.fillna(fallback).fillna(4.2);z["parent_implied_ypc"]=implied
    z["candidate_yards"]=z.candidate_att*z.parent_implied_ypc
    return z


v1.add_candidate=add_candidate_fixed

if __name__=="__main__":
    v1.main()
