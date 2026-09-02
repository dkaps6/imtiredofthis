#!/usr/bin/env python3
"""RB STACK6M: exact no-fit Shapley attribution within trailing-state contexts."""
from __future__ import annotations

import argparse, itertools, math
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

CONTEXTS=("close_early","deep_early","close_late","deep_late")
START_WEEK=6; ALPHA=0.75
EXPECTED_N=388
EXPECTED_EMPTY_MAE=5.518381962346741
EXPECTED_ALL_MAE=4.503012474954635
EXPECTED_RECOVERY=EXPECTED_EMPTY_MAE-EXPECTED_ALL_MAE

def num(v): return pd.to_numeric(v,errors="coerce")
def lower(df):
    z=df.copy(); z.columns=[str(c).strip().lower() for c in z.columns]; return z
def one(root:Path,name:str):
    hits=list(root.rglob(name));
    if len(hits)!=1: raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0],low_memory=False))
def metric(y,p):
    y,p=num(y),num(p); ok=y.notna()&p.notna(); y,p=y[ok],p[ok]; e=p-y
    return {"n":len(y),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(p.corr(y)) if len(y)>=3 and p.nunique()>1 and y.nunique()>1 else np.nan}
def sname(sub): return "NONE" if not sub else "+".join(c.upper() for c in CONTEXTS if c in sub)
def subsets():
    out=[]
    for k in range(5): out.extend(frozenset(x) for x in itertools.combinations(CONTEXTS,k))
    return out

def pbp_contexts():
    import nflreadpy as nfl
    p=lower(nfl.load_pbp(seasons=[2025]).to_pandas())
    if "season_type" in p.columns:
        r=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy(); p=r if len(r) else p
    p["team"]=p.posteam.map(canon_team); p["rush_attempt"]=num(p.rush_attempt).fillna(0); p["qb_dropback"]=num(p.qb_dropback).fillna(0)
    p=p.loc[(p.rush_attempt.eq(1)|p.qb_dropback.eq(1))&p.team.ne("")].copy()
    if "score_differential" in p.columns: diff=num(p.score_differential)
    else: diff=num(p.posteam_score)-num(p.defteam_score)
    p["score_diff"]=diff.fillna(0.0); p["qtr_num"]=num(p.qtr).fillna(0)
    p["trail"]=p.score_diff.lt(-3)
    close=p.score_diff.between(-8,-4,inclusive="both"); deep=p.score_diff.le(-9); late=p.qtr_num.ge(4)
    p["context"]=np.select([p.trail&close&~late,p.trail&deep&~late,p.trail&close&late,p.trail&deep&late],CONTEXTS,default="other")
    rows=[]
    for (season,week,team),g in p.groupby(["season","week","team"]):
        n=float(len(g)); rec={"season":int(season),"week":int(week),"team":canon_team(team),"pbp_actual_off_plays":n,"pbp_trail_plays":float(g.trail.sum()),"pbp_trail_share":float(g.trail.mean())}
        for c in CONTEXTS:
            q=g.loc[g.context.eq(c)]; rec[f"{c}_plays"]=float(len(q)); rec[f"{c}_rushes"]=float(q.rush_attempt.sum()); rec[f"{c}_share"]=float(len(q)/n) if n else np.nan; rec[f"{c}_rush_rate"]=float(q.rush_attempt.mean()) if len(q) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def score_subsets(df,label):
    rows=[]; empty=None
    for sub in subsets():
        col=f"ORACLE_{sname(sub)}"; r=metric(df.actual_team_rush_att,df[col]); rows.append({"population":label,"subset":sname(sub),"corrected_contexts":";".join(sorted(sub)),**r})
        if not sub: empty=r["mae"]
    z=pd.DataFrame(rows); z["recovery_vs_empty"]=float(empty)-z.mae; return z

def shapley(table,pop):
    q=table.loc[table.population.eq(pop)]; values={}
    for _,r in q.iterrows(): values[frozenset(x for x in str(r.corrected_contexts).split(";") if x)]=float(r.recovery_vs_empty)
    n=len(CONTEXTS); total=values[frozenset(CONTEXTS)]; out=[]
    for c in CONTEXTS:
        others=[x for x in CONTEXTS if x!=c]; phi=0.0
        for k in range(n):
            for co in itertools.combinations(others,k):
                S=frozenset(co); w=math.factorial(len(S))*math.factorial(n-len(S)-1)/math.factorial(n); phi+=w*(values[S|{c}]-values[S])
        out.append({"population":pop,"context":c,"shapley_recovery":phi,"fraction_of_trail_recovery":phi/total if abs(total)>1e-12 else np.nan,"total_trail_recovery":total})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--stack6h-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    m=one(a.m94c_root,"m94c_2025_team_trace.csv"); h=one(a.stack6h_root,"stack6h_team_trace.csv"); p=pbp_contexts()
    for d in (m,h): d["season"]=num(d.season).astype(int); d["week"]=num(d.week).astype(int); d["team"]=d.team.map(canon_team)
    req=["season","week","team","actual_team_rush_att","actual_off_plays","baseline_team_rush_att","pred_off_plays","lead_play_share","neutral_play_share","trail_play_share","gs_team_lead_rush_rate_shrunk","gs_team_neutral_rush_rate_shrunk","gs_team_trail_rush_rate_shrunk"]
    bins=["pool_over_5","pool_under_5","pool_abs_5","non_extreme_abs_lt3"]
    t=m[req].merge(h[["season","week","team",*bins]],on=["season","week","team"],how="inner",validate="one_to_one").merge(p,on=["season","week","team"],how="inner",validate="one_to_one")
    if len(t)!=544: raise RuntimeError(f"expected 544 joined rows; got {len(t)}")
    for c in req:
        if c!="team": t[c]=num(t[c])
    context_sum=sum(t[f"{c}_share"] for c in CONTEXTS)
    t["context_trail_share_sum"]=context_sum
    lead_con=t.lead_play_share*t.gs_team_lead_rush_rate_shrunk; neutral_con=t.neutral_play_share*t.gs_team_neutral_rush_rate_shrunk
    for sub in subsets():
        trail_con=pd.Series(0.0,index=t.index)
        for c in CONTEXTS:
            if c in sub: trail_con += t[f"{c}_rushes"]/t.pbp_actual_off_plays
            else: trail_con += t[f"{c}_share"]*t.gs_team_trail_rush_rate_shrunk
        t[f"ORACLE_{sname(sub)}"]=(1-ALPHA)*t.baseline_team_rush_att+ALPHA*t.pred_off_plays*(lead_con+neutral_con+trail_con)
    w=t.loc[t.week.ge(START_WEEK)].copy()
    scores=score_subsets(w,"ALL_W6_18"); masks={"POOL_OVER_5":w.pool_over_5.eq(1),"POOL_UNDER_5":w.pool_under_5.eq(1),"POOL_ABS_5":w.pool_abs_5.eq(1),"NON_EXTREME_ABS_LT3":w.non_extreme_abs_lt3.eq(1)}; scores=pd.concat([scores]+[score_subsets(w.loc[v],k) for k,v in masks.items()],ignore_index=True)
    shp=pd.concat([shapley(scores,p) for p in ["ALL_W6_18",*masks]],ignore_index=True)
    context_summary=[]
    for c in CONTEXTS:
        context_summary.append({"context":c,"team_games_with_plays":int(w[f"{c}_plays"].gt(0).sum()),"mean_plays_per_team_game":float(w[f"{c}_plays"].mean()),"total_plays":float(w[f"{c}_plays"].sum()),"total_rushes":float(w[f"{c}_rushes"].sum()),"aggregate_rush_rate":float(w[f"{c}_rushes"].sum()/w[f"{c}_plays"].sum()) if w[f"{c}_plays"].sum()>0 else np.nan})
    context_summary=pd.DataFrame(context_summary)
    ov=shp.loc[shp.population.eq("ALL_W6_18")].sort_values("shapley_recovery",ascending=False); top=ov.iloc[0]; tc=str(top.context); phi=float(top.shapley_recovery); frac=float(top.fraction_of_trail_recovery); over=float(shp.loc[(shp.population.eq("POOL_OVER_5"))&shp.context.eq(tc),"shapley_recovery"].iloc[0]); under=float(shp.loc[(shp.population.eq("POOL_UNDER_5"))&shp.context.eq(tc),"shapley_recovery"].iloc[0])
    disp=f"{tc.upper()}_DOMINANT" if phi>=0.30 and frac>=0.40 and over>0 and under>0 else "TRAIL_CONTEXT_DISTRIBUTED"
    empty=float(scores.loc[(scores.population.eq("ALL_W6_18"))&scores.subset.eq("NONE"),"mae"].iloc[0]); allname=sname(frozenset(CONTEXTS)); allmae=float(scores.loc[(scores.population.eq("ALL_W6_18"))&scores.subset.eq(allname),"mae"].iloc[0]); recovery=empty-allmae; shsum=float(ov.shapley_recovery.sum())
    bridge_trail=float((w.trail_play_share-w.pbp_trail_share).abs().max()); context_err=float((w.context_trail_share_sum-w.pbp_trail_share).abs().max())
    integrity_pass=int(len(w)==EXPECTED_N and bridge_trail<=1e-9 and context_err<=1e-9 and abs(empty-EXPECTED_EMPTY_MAE)<=1e-9 and abs(allmae-EXPECTED_ALL_MAE)<=1e-9 and abs(shsum-EXPECTED_RECOVERY)<=1e-9)
    if not integrity_pass: disp="STACK6M_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    integrity=pd.DataFrame([{"m94c_rows":len(m),"pbp_team_games":len(p),"joined_rows":len(t),"w6_18_n":len(w),"trail_share_max_abs_diff":bridge_trail,"context_sum_vs_trail_max_abs_diff":context_err,"expected_empty_mae":EXPECTED_EMPTY_MAE,"observed_empty_mae":empty,"expected_all_context_mae":EXPECTED_ALL_MAE,"observed_all_context_mae":allmae,"expected_direct_trail_recovery":EXPECTED_RECOVERY,"observed_direct_trail_recovery":recovery,"shapley_sum":shsum,"shapley_sum_abs_error":abs(shsum-recovery),"integrity_pass":integrity_pass,"fitted_models":0,"feature_search":0,"hyperparameter_search":0,"threshold_search":0,"sportsbook_inputs":0,"target_game_pbp_used_as_oracle_only":1}])
    disposition=pd.DataFrame([{"top_context":tc,"top_context_shapley_recovery":phi,"top_context_fraction":frac,"top_context_pool_over5_shapley":over,"top_context_pool_under5_shapley":under,"disposition":disp,"production_change":0,"predictive_model_authorized":0}])
    t.to_csv(a.out_dir/"stack6m_team_trace.csv",index=False); scores.to_csv(a.out_dir/"stack6m_subset_scores.csv",index=False); shp.to_csv(a.out_dir/"stack6m_shapley.csv",index=False); context_summary.to_csv(a.out_dir/"stack6m_context_summary.csv",index=False); integrity.to_csv(a.out_dir/"stack6m_integrity.csv",index=False); disposition.to_csv(a.out_dir/"stack6m_disposition.csv",index=False)
    print("=== integrity ==="); print(integrity.to_string(index=False)); print("=== context summary ==="); print(context_summary.to_string(index=False)); print("=== shapley ==="); print(shp.to_string(index=False)); print("=== disposition ==="); print(disposition.to_string(index=False)); print(f"STACK6M_DISPOSITION={disp}")
    return 0
if __name__=="__main__": raise SystemExit(main())
