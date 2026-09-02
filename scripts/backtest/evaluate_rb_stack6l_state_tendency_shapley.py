#!/usr/bin/env python3
"""RB STACK6L: exact no-fit Shapley attribution of lead/neutral/trail tendency error."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

STATES = ("lead", "neutral", "trail")
START_WEEK = 6
ALPHA = 0.75
EXPECTED_N = 388
EXPECTED_OCC_MAE = 5.518381962346741
EXPECTED_ALL_MAE = 3.4503279031625445
EXPECTED_TOTAL_RECOVERY = EXPECTED_OCC_MAE - EXPECTED_ALL_MAE
STATE_THRESHOLD = 3.0


def num(v): return pd.to_numeric(v, errors="coerce")
def lower(df):
    z=df.copy(); z.columns=[str(c).strip().lower() for c in z.columns]; return z
def one(root:Path,name:str):
    hits=list(root.rglob(name))
    if len(hits)!=1: raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0],low_memory=False))

def metric(y,p):
    y,p=num(y),num(p); ok=y.notna()&p.notna(); y,p=y[ok],p[ok]; e=p-y
    return {"n":len(y),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(p.corr(y)) if len(y)>=3 and p.nunique()>1 and y.nunique()>1 else np.nan}

def pbp_states():
    import nflreadpy as nfl
    p=lower(nfl.load_pbp(seasons=[2025]).to_pandas())
    if "season_type" in p.columns:
        reg=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if len(reg): p=reg
    p["team"]=p.posteam.map(canon_team)
    p["rush_attempt"]=num(p.rush_attempt).fillna(0); p["qb_dropback"]=num(p.qb_dropback).fillna(0)
    p["off_play"]=(p.rush_attempt.eq(1)|p.qb_dropback.eq(1)).astype(int)
    p=p.loc[p.off_play.eq(1)&p.team.ne("")].copy()
    if "score_differential" in p.columns: diff=num(p.score_differential)
    else: diff=num(p.posteam_score)-num(p.defteam_score)
    p["score_diff"]=diff.fillna(0.0)
    p["state"]=np.select([p.score_diff.gt(STATE_THRESHOLD),p.score_diff.lt(-STATE_THRESHOLD)],["lead","trail"],default="neutral")
    rows=[]
    for (season,week,team),g in p.groupby(["season","week","team"]):
        rec={"season":int(season),"week":int(week),"team":canon_team(team),"pbp_actual_off_plays":float(len(g)),"pbp_actual_rush_att":float(g.rush_attempt.sum())}
        for s in STATES:
            q=g.loc[g.state.eq(s)]; plays=float(len(q)); rush=float(q.rush_attempt.sum())
            rec[f"pbp_{s}_plays"]=plays; rec[f"pbp_{s}_rushes"]=rush; rec[f"pbp_{s}_play_share"]=plays/len(g) if len(g) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def subset_name(sub): return "NONE" if not sub else "+".join(s.upper() for s in STATES if s in sub)
def all_subsets():
    out=[]
    for k in range(4): out.extend(frozenset(x) for x in itertools.combinations(STATES,k))
    return out

def score_subsets(df,label):
    rows=[]
    base_occ=None
    for sub in all_subsets():
        col=f"ORACLE_{subset_name(sub)}"
        r=metric(df.actual_team_rush_att,df[col]); rows.append({"population":label,"subset":subset_name(sub),"corrected_states":";".join(sorted(sub)),**r})
        if not sub: base_occ=r["mae"]
    z=pd.DataFrame(rows); z["recovery_vs_occupancy"]=float(base_occ)-z.mae
    return z

def shapley(table,pop):
    q=table.loc[table.population.eq(pop)].copy(); val={}
    for _,r in q.iterrows():
        sub=frozenset(x for x in str(r.corrected_states).split(";") if x)
        val[sub]=float(r.recovery_vs_occupancy)
    rows=[]; n=3; total=val[frozenset(STATES)]
    for st in STATES:
        phi=0.0
        others=[x for x in STATES if x!=st]
        for k in range(3):
            for combo in itertools.combinations(others,k):
                S=frozenset(combo); w=math.factorial(len(S))*math.factorial(n-len(S)-1)/math.factorial(n)
                phi += w*(val[S|{st}]-val[S])
        rows.append({"population":pop,"state":st,"shapley_recovery":phi,"fraction_of_total":phi/total if abs(total)>1e-12 else np.nan,"total_tendency_recovery":total})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--stack6h-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    m=one(a.m94c_root,"m94c_2025_team_trace.csv"); h=one(a.stack6h_root,"stack6h_team_trace.csv"); p=pbp_states()
    for d in (m,h): d["season"]=num(d.season).astype(int); d["week"]=num(d.week).astype(int); d["team"]=d.team.map(canon_team)
    req=["season","week","team","actual_team_rush_att","actual_off_plays","actual_rush_att_pbp","baseline_team_rush_att","pred_off_plays"]
    for s in STATES: req += [f"{s}_play_share",f"gs_team_{s}_rush_rate_shrunk"]
    bins=["pool_over_5","pool_under_5","pool_abs_5","non_extreme_abs_lt3"]
    t=m[req].merge(h[["season","week","team",*bins]],on=["season","week","team"],how="inner",validate="one_to_one").merge(p,on=["season","week","team"],how="inner",validate="one_to_one")
    for c in req:
        if c!="team": t[c]=num(t[c])
    if len(t)!=544: raise RuntimeError(f"expected 544 joins, got {len(t)}")
    w=t.loc[t.week.ge(START_WEEK)].copy()
    bridge={"actual_off_plays_max_abs_diff":float((w.actual_off_plays-w.pbp_actual_off_plays).abs().max()),"actual_rush_att_pbp_max_abs_diff":float((w.actual_rush_att_pbp-w.pbp_actual_rush_att).abs().max())}
    for s in STATES: bridge[f"{s}_share_max_abs_diff"]=float((w[f"{s}_play_share"]-w[f"pbp_{s}_play_share"]).abs().max())
    pbp_repro_pass=int(max(bridge.values())<=1e-9)
    for sub in all_subsets():
        rate=pd.Series(0.0,index=t.index)
        for s in STATES:
            if s in sub: rate += t[f"pbp_{s}_rushes"]/t.pbp_actual_off_plays
            else: rate += t[f"{s}_play_share"]*t[f"gs_team_{s}_rush_rate_shrunk"]
        t[f"ORACLE_{subset_name(sub)}"]=(1-ALPHA)*t.baseline_team_rush_att+ALPHA*t.pred_off_plays*rate
    w=t.loc[t.week.ge(START_WEEK)].copy()
    subset_scores=score_subsets(w,"ALL_W6_18")
    masks={"POOL_OVER_5":w.pool_over_5.eq(1),"POOL_UNDER_5":w.pool_under_5.eq(1),"POOL_ABS_5":w.pool_abs_5.eq(1),"NON_EXTREME_ABS_LT3":w.non_extreme_abs_lt3.eq(1)}
    all_scores=[subset_scores]+[score_subsets(w.loc[v],k) for k,v in masks.items()]; subset_scores=pd.concat(all_scores,ignore_index=True)
    pops=["ALL_W6_18",*masks.keys()]; shp=pd.concat([shapley(subset_scores,pop) for pop in pops],ignore_index=True)
    ov=shp.loc[shp.population.eq("ALL_W6_18")].sort_values("shapley_recovery",ascending=False); top=ov.iloc[0]; top_state=str(top.state); top_phi=float(top.shapley_recovery); top_frac=float(top.fraction_of_total)
    over_phi=float(shp.loc[(shp.population.eq("POOL_OVER_5"))&(shp.state.eq(top_state)),"shapley_recovery"].iloc[0]); under_phi=float(shp.loc[(shp.population.eq("POOL_UNDER_5"))&(shp.state.eq(top_state)),"shapley_recovery"].iloc[0])
    if top_phi>=0.75 and top_frac>=0.45 and over_phi>0 and under_phi>0: disp=f"{top_state.upper()}_TENDENCY_DOMINANT"
    else: disp="MULTI_STATE_TENDENCY"
    occ_mae=float(subset_scores.loc[(subset_scores.population.eq("ALL_W6_18"))&(subset_scores.subset.eq("NONE")),"mae"].iloc[0]); all_mae=float(subset_scores.loc[(subset_scores.population.eq("ALL_W6_18"))&(subset_scores.subset.eq("LEAD+NEUTRAL+TRAIL")),"mae"].iloc[0]); total_rec=occ_mae-all_mae; shap_sum=float(ov.shapley_recovery.sum()); shap_err=abs(shap_sum-total_rec)
    integrity_pass=int(len(w)==EXPECTED_N and pbp_repro_pass and abs(occ_mae-EXPECTED_OCC_MAE)<=1e-9 and abs(all_mae-EXPECTED_ALL_MAE)<=1e-9 and shap_err<=1e-9)
    if not pbp_repro_pass: disp="STACK6L_PBP_REPRODUCTION_FAILURE_DO_NOT_INTERPRET"
    elif not integrity_pass: disp="STACK6L_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    integrity=pd.DataFrame([{**bridge,"m94c_rows":len(m),"pbp_team_games":len(p),"joined_rows":len(t),"w6_18_n":len(w),"pbp_reproduction_pass":pbp_repro_pass,"expected_occupancy_mae":EXPECTED_OCC_MAE,"observed_occupancy_mae":occ_mae,"expected_all_tendency_mae":EXPECTED_ALL_MAE,"observed_all_tendency_mae":all_mae,"expected_total_tendency_recovery":EXPECTED_TOTAL_RECOVERY,"observed_total_tendency_recovery":total_rec,"shapley_sum":shap_sum,"shapley_sum_abs_error":shap_err,"integrity_pass":integrity_pass,"fitted_models":0,"feature_search":0,"hyperparameter_search":0,"threshold_search":0,"sportsbook_inputs":0,"target_game_pbp_used_as_oracle_only":1}])
    disposition=pd.DataFrame([{"top_state":top_state,"top_state_shapley_recovery":top_phi,"top_state_fraction":top_frac,"top_state_pool_over5_shapley":over_phi,"top_state_pool_under5_shapley":under_phi,"disposition":disp,"production_change":0,"predictive_model_authorized":0}])
    t.to_csv(a.out_dir/"stack6l_team_trace.csv",index=False); integrity.to_csv(a.out_dir/"stack6l_integrity.csv",index=False); subset_scores.to_csv(a.out_dir/"stack6l_subset_scores.csv",index=False); shp.to_csv(a.out_dir/"stack6l_shapley.csv",index=False); disposition.to_csv(a.out_dir/"stack6l_disposition.csv",index=False)
    print("=== STACK6L integrity ==="); print(integrity.to_string(index=False)); print("=== STACK6L overall subsets ==="); print(subset_scores.loc[subset_scores.population.eq("ALL_W6_18")].to_string(index=False)); print("=== STACK6L shapley ==="); print(shp.to_string(index=False)); print("=== STACK6L disposition ==="); print(disposition.to_string(index=False)); print(f"STACK6L_DISPOSITION={disp}")
    return 0
if __name__=="__main__": raise SystemExit(main())
