#!/usr/bin/env python3
"""RB STACK6K: no-fit within-state rushing-tendency oracle."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

START_WEEK = 6
ALPHA = 0.75
EXPECTED_N = 388
EXPECTED_BASE_MAE = 6.203454780519527
EXPECTED_OCC_MAE = 5.518381962346741
RATE_HEADROOM = 3.240694394219671
OCC_RECOVERY = 0.6850728181727863
REMAINING_HEADROOM = RATE_HEADROOM - OCC_RECOVERY
STATES = ("lead", "neutral", "trail")
TEAM_MAP = {"JAX":"JAC","LAR":"LA","STL":"LA","OAK":"LV","SD":"LAC"}


def num(v): return pd.to_numeric(v, errors="coerce")
def canon(v):
    s=str(v).strip().upper(); return TEAM_MAP.get(s,s)
def lower(df):
    z=df.copy(); z.columns=[str(c).strip().lower() for c in z.columns]; return z
def one(root:Path,name:str):
    hits=list(root.rglob(name))
    if len(hits)!=1: raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0],low_memory=False))

def metric(y,p):
    y,p=num(y),num(p); ok=y.notna()&p.notna(); y,p=y[ok],p[ok]; e=p-y
    return {"n":len(y),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(p.corr(y)) if len(y)>=3 and p.nunique()>1 and y.nunique()>1 else np.nan}

def score(df,label):
    rows=[]
    for arm in ["BASE_M94C_TOTAL_RUSH","ORACLE_STATE_OCCUPANCY","ORACLE_OCC_PLUS_TENDENCY"]:
        rows.append({"population":label,"arm":arm,**metric(df.actual_team_rush_att,df[arm])})
    out=pd.DataFrame(rows); b=float(out.loc[out.arm.eq("BASE_M94C_TOTAL_RUSH"),"mae"].iloc[0])
    out["mae_recovery_vs_base"]=b-out.mae
    occ=float(out.loc[out.arm.eq("ORACLE_STATE_OCCUPANCY"),"mae"].iloc[0])
    out["incremental_tendency_recovery_vs_occupancy"]=np.where(out.arm.eq("ORACLE_OCC_PLUS_TENDENCY"),occ-out.mae,np.nan)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--stack6h-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    m=one(a.m94c_root,"m94c_2025_team_trace.csv"); h=one(a.stack6h_root,"stack6h_team_trace.csv")
    for d in (m,h): d["season"]=num(d.season).astype(int); d["week"]=num(d.week).astype(int); d["team"]=d.team.map(canon)
    req=["season","week","team","actual_team_rush_att","actual_off_plays","actual_rush_att_pbp","baseline_team_rush_att","pred_off_plays","candidate_team_rush_att"]
    for s in STATES: req += [f"{s}_play_share",f"gs_team_{s}_rush_rate_shrunk"]
    missing=[c for c in req if c not in m.columns]
    if missing: raise RuntimeError(f"M94C missing {missing}")
    bins=["pool_over_5","pool_under_5","pool_abs_5","non_extreme_abs_lt3"]
    t=m[req].merge(h[["season","week","team",*bins]],on=["season","week","team"],how="inner",validate="one_to_one")
    for c in req:
        if c!="team": t[c]=num(t[c])
    if len(t)!=544 or t.actual_off_plays.isna().any() or t.actual_off_plays.le(0).any(): raise RuntimeError("STACK6K join/denominator integrity failure")
    occ_rate=pd.Series(0.0,index=t.index)
    for s in STATES: occ_rate += t[f"{s}_play_share"]*t[f"gs_team_{s}_rush_rate_shrunk"]
    t["realized_pbp_rush_rate"]=t.actual_rush_att_pbp/t.actual_off_plays
    t["BASE_M94C_TOTAL_RUSH"]=t.candidate_team_rush_att
    t["ORACLE_STATE_OCCUPANCY"]=(1-ALPHA)*t.baseline_team_rush_att+ALPHA*t.pred_off_plays*occ_rate
    t["ORACLE_OCC_PLUS_TENDENCY"]=(1-ALPHA)*t.baseline_team_rush_att+ALPHA*t.pred_off_plays*t.realized_pbp_rush_rate
    t["pbp_weekly_diff"]=t.actual_rush_att_pbp-t.actual_team_rush_att
    w=t.loc[t.season.eq(2025)&t.week.ge(START_WEEK)].copy()
    overall=score(w,"ALL_W6_18")
    masks={"POOL_OVER_5":w.pool_over_5.eq(1),"POOL_UNDER_5":w.pool_under_5.eq(1),"POOL_ABS_5":w.pool_abs_5.eq(1),"NON_EXTREME_ABS_LT3":w.non_extreme_abs_lt3.eq(1)}
    bin_scores=pd.concat([score(w.loc[v],k) for k,v in masks.items()],ignore_index=True)
    bridge_m=metric(w.actual_team_rush_att,w.actual_rush_att_pbp); ad=w.pbp_weekly_diff.abs()
    bridge=pd.DataFrame([{**bridge_m,"exact_match_rate":float(ad.le(1e-12).mean()),"abs_diff_gt1_rate":float(ad.gt(1).mean()),"abs_diff_gt2_rate":float(ad.gt(2).mean())}])
    def inc(table,pop):
        q=table.loc[table.population.eq(pop)&table.arm.eq("ORACLE_OCC_PLUS_TENDENCY"),"incremental_tendency_recovery_vs_occupancy"]; return float(q.iloc[0])
    overall_inc=inc(overall,"ALL_W6_18"); over_inc=inc(bin_scores,"POOL_OVER_5"); under_inc=inc(bin_scores,"POOL_UNDER_5"); abs5_inc=inc(bin_scores,"POOL_ABS_5"); nonext_inc=inc(bin_scores,"NON_EXTREME_ABS_LT3")
    frac=overall_inc/REMAINING_HEADROOM
    if overall_inc>=1.0 and frac>=0.50 and over_inc>0 and under_inc>0: disp="WITHIN_STATE_TENDENCY_DOMINANT"
    elif overall_inc>=0.50 and over_inc>0 and under_inc>0: disp="WITHIN_STATE_TENDENCY_MATERIAL"
    else: disp="WITHIN_STATE_TENDENCY_NOT_PRIMARY"
    base_mae=float(overall.loc[overall.arm.eq("BASE_M94C_TOTAL_RUSH"),"mae"].iloc[0]); occ_mae=float(overall.loc[overall.arm.eq("ORACLE_STATE_OCCUPANCY"),"mae"].iloc[0])
    bridge_pass=int(float(bridge.iloc[0]["mae"])<=0.50 and float(bridge.iloc[0]["corr"])>=0.98 and float(bridge.iloc[0]["abs_diff_gt2_rate"])<=0.05)
    integrity_pass=int(len(w)==EXPECTED_N and abs(base_mae-EXPECTED_BASE_MAE)<=1e-9 and abs(occ_mae-EXPECTED_OCC_MAE)<=1e-9 and bridge_pass)
    if not bridge_pass: disp="STACK6K_TRUTH_BRIDGE_FAILURE_DO_NOT_INTERPRET"
    elif not integrity_pass: disp="STACK6K_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    integrity=pd.DataFrame([{"m94c_rows":len(m),"stack6h_rows":len(h),"joined_rows":len(t),"w6_18_n":len(w),"expected_base_mae":EXPECTED_BASE_MAE,"observed_base_mae":base_mae,"expected_occupancy_mae":EXPECTED_OCC_MAE,"observed_occupancy_mae":occ_mae,"truth_bridge_pass":bridge_pass,"integrity_pass":integrity_pass,"fitted_models":0,"feature_search":0,"hyperparameter_search":0,"threshold_search":0,"sportsbook_inputs":0,"target_game_pbp_used_as_oracle_only":1}])
    disposition=pd.DataFrame([{"incremental_tendency_recovery":overall_inc,"remaining_stack6i_rate_headroom":REMAINING_HEADROOM,"tendency_fraction_of_remaining":frac,"pool_over5_incremental_recovery":over_inc,"pool_under5_incremental_recovery":under_inc,"pool_abs5_incremental_recovery":abs5_inc,"non_extreme_incremental_recovery":nonext_inc,"disposition":disp,"production_change":0,"predictive_model_authorized":0}])
    t.to_csv(a.out_dir/"stack6k_team_trace.csv",index=False); integrity.to_csv(a.out_dir/"stack6k_integrity.csv",index=False); bridge.to_csv(a.out_dir/"stack6k_truth_bridge.csv",index=False); overall.to_csv(a.out_dir/"stack6k_overall_scores.csv",index=False); bin_scores.to_csv(a.out_dir/"stack6k_bin_scores.csv",index=False); disposition.to_csv(a.out_dir/"stack6k_disposition.csv",index=False)
    print("=== STACK6K integrity ==="); print(integrity.to_string(index=False)); print("=== truth bridge ==="); print(bridge.to_string(index=False)); print("=== overall ==="); print(overall.to_string(index=False)); print("=== bins ==="); print(bin_scores.to_string(index=False)); print("=== disposition ==="); print(disposition.to_string(index=False)); print(f"STACK6K_DISPOSITION={disp}")
    return 0
if __name__=="__main__": raise SystemExit(main())
