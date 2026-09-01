#!/usr/bin/env python3
"""RB-STACK3 frozen pregame state composition.

No fitting. Uses frozen STACK2, M95F, and M95I row-level outputs. 2025 is
retrospective development evidence. Sportsbook joins only after football arms
are frozen.
"""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np,pandas as pd

TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
def tm(v):
    s=str(v).strip().upper() if not pd.isna(v) else ""; return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v): return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def lower(x): x=x.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x
def one(root,name):
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name} under {root}, found {len(h)}")
    return lower(pd.read_csv(h[0],low_memory=False))
def num(s): return pd.to_numeric(s,errors="coerce")
def prep(x):
    z=x.copy(); z["season"]=num(z.get("season",2025)).fillna(2025).astype(int); z["week"]=num(z.week).astype(int); z["team"]=z.team.map(tm)
    if "player_clean_key" in z.columns: z["join_key"]=z.player_clean_key.astype(str).map(nk)
    elif "name_key" in z.columns: z["join_key"]=z.name_key.astype(str).map(nk)
    else: z["join_key"]=z.player.map(nk)
    return z

def metric(y,p):
    y=num(y);p=num(p);ok=y.notna()&p.notna();y=y[ok].astype(float);p=p[ok].astype(float)
    if not len(y): return {"n":0}
    e=p-y
    return {"n":int(len(y)),"mae":float(np.abs(e).mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(np.corrcoef(p,y)[0,1]) if len(y)>1 and y.std()>0 and p.std()>0 else np.nan,"actual_mean":float(y.mean()),"pred_mean":float(p.mean())}

def arms(x):
    z=x.copy(); w=num(z.week)
    p=num(z.arch_enriched_opp_stack_eff_yards); stack=num(z.stack_yards); enr=num(z.enriched_yards); stackeff=num(z.stack_implied_ypc)
    risk=num(z.cal_prob_20).fillna(0).ge(.25)|num(z.m95f_p90).fillna(0).ge(20)
    vac=num(z.prior_top1_unavailable).fillna(0).eq(1)
    tail=num(z.m95i_tail_eligible).fillna(0).gt(0)
    m95i_y=num(z.m95i_rush_att)*stackeff
    z["state_m95f_risk"]=risk.astype(int);z["state_vacancy"]=vac.astype(int);z["state_m95i_tail"]=tail.astype(int);z["m95i_carry_stack_eff_yards"]=m95i_y
    z["arm_stack2_parent"]=p
    z["arm_week1_stack"]=np.where(w.eq(1),stack,p)
    z["arm_m95f_risk_enriched"]=np.where(risk,enr,p)
    z["arm_week1_m95f_risk"]=np.where(w.eq(1),stack,np.where(risk,enr,p))
    z["arm_m95i_tail_stackeff"]=np.where(tail,m95i_y,p)
    z["arm_vacancy_enriched"]=np.where(vac,enr,p)
    z["arm_week1_risk_vacancy"]=np.where(w.eq(1),stack,np.where(risk|vac,enr,p))
    z["arm_week1_risk_m95i_tail"]=np.where(w.eq(1),stack,np.where(tail,m95i_y,np.where(risk,enr,p)))
    return z
ARM={
"STACK2_PARENT":"arm_stack2_parent",
"WEEK1_STACK_OVERRIDE":"arm_week1_stack",
"M95F_RISK_ENRICHED_OVERRIDE":"arm_m95f_risk_enriched",
"WEEK1_PLUS_M95F_RISK":"arm_week1_m95f_risk",
"M95I_CARRY_STACK_EFF":"arm_m95i_tail_stackeff",
"VACANCY_ENRICHED_OVERRIDE":"arm_vacancy_enriched",
"WEEK1_RISK_VACANCY_COMPOSITE":"arm_week1_risk_vacancy",
"WEEK1_RISK_M95I_TAIL_COMPOSITE":"arm_week1_risk_m95i_tail",
}
def slice_masks(x):
    a=num(x.actual_rush_att);w=num(x.week);risk=num(x.state_m95f_risk).eq(1);vac=num(x.state_vacancy).eq(1);tail=num(x.state_m95i_tail).eq(1)
    return {"all_rb":pd.Series(True,index=x.index),"week1":w.eq(1),"weeks2_18":w.ge(2),"actual_0_5":a.le(5),"actual_6_10":a.between(6,10),"actual_11_14":a.between(11,14),"actual_15_19":a.between(15,19),"actual_20_plus":a.ge(20),"actual_25_plus":a.ge(25),"pregame_m95f_risk":risk,"pregame_m95f_nonrisk":~risk,"pregame_vacancy":vac,"pregame_incumbent":~vac,"pregame_m95i_tail":tail,"pregame_m95i_nontail":~tail}
def score(x):
    rows=[]
    for sl,mask in slice_masks(x).items():
        q=x.loc[mask]
        for a,c in ARM.items(): rows.append({"slice":sl,"arm":a,**metric(q.actual_rush_yards,q[c])})
    return pd.DataFrame(rows)
def market_score(x,cb):
    c=prep(cb); keys=["week","team","join_key"]
    q=x.merge(c[keys+["consensus_line"]].drop_duplicates(keys),on=keys,how="inner",validate="one_to_one")
    rows=[];edges=[];bins=[0,2.5,5,10,np.inf];labs=["0_2.5","2.5_5","5_10","10_plus"]
    for a,col in {**ARM,"VEGAS_CONSENSUS":"consensus_line"}.items(): rows.append({"arm":a,**metric(q.actual_rush_yards,q[col])})
    for a,col in ARM.items():
        z=q[["actual_rush_yards","consensus_line",col]].dropna().copy();z["edge"]=num(z[col])-num(z.consensus_line);z["abs_edge"]=z.edge.abs();z["bucket"]=pd.cut(z.abs_edge,bins=bins,labels=labs,right=False);z["closer"]=(np.abs(num(z[col])-num(z.actual_rush_yards))<np.abs(num(z.consensus_line)-num(z.actual_rush_yards))).astype(float)
        for b,g in z.groupby("bucket",observed=True): edges.append({"arm":a,"edge_bucket":str(b),"n":len(g),"model_closer_rate":float(g.closer.mean()),"model_mae":float(np.abs(num(g[col])-num(g.actual_rush_yards)).mean()),"vegas_mae":float(np.abs(num(g.consensus_line)-num(g.actual_rush_yards)).mean()),"mean_edge":float(g.edge.mean())})
    return q,pd.DataFrame(rows),pd.DataFrame(edges)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stack2-root",type=Path,required=True);ap.add_argument("--m95f-root",type=Path,required=True);ap.add_argument("--m95i-root",type=Path,required=True);ap.add_argument("--market-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    s=prep(one(a.stack2_root,"stack2_2025_casebook.csv"));f=prep(one(a.m95f_root,"m95f_2025_rb_trace.csv"));i=prep(one(a.m95i_root,"m95i_2025_trace.csv"));cb=one(a.market_root,"rb_market_casebook.csv")
    keys=["season","week","team","join_key"]
    fcols=keys+[c for c in ["cal_prob_20","cal_prob_25","m95f_p90","m95f_p95"] if c in f.columns]
    icols=keys+[c for c in ["prior_top1_unavailable","m95i_rush_att","m95i_tail_eligible","p20_joint","p25_joint"] if c in i.columns]
    x=s.merge(f[fcols].drop_duplicates(keys),on=keys,how="left",validate="one_to_one").merge(i[icols].drop_duplicates(keys),on=keys,how="left",validate="one_to_one")
    required=["cal_prob_20","m95f_p90","prior_top1_unavailable","m95i_rush_att","m95i_tail_eligible"]
    miss=[c for c in required if c not in x.columns];
    if miss: raise RuntimeError(f"missing frozen state columns {miss}")
    x=arms(x);sc=score(x);mq,mm,ed=market_score(x,cb)
    cov=pd.DataFrame([{"rows":len(x),"m95f_p20_coverage":float(num(x.cal_prob_20).notna().mean()),"m95f_p90_coverage":float(num(x.m95f_p90).notna().mean()),"m95i_vacancy_coverage":float(num(x.prior_top1_unavailable).notna().mean()),"m95i_carry_coverage":float(num(x.m95i_rush_att).notna().mean()),"m95f_risk_rows":int(num(x.state_m95f_risk).sum()),"vacancy_rows":int(num(x.state_vacancy).sum()),"m95i_tail_rows":int(num(x.state_m95i_tail).sum()),"market_rows":len(mq)}])
    disp=pd.DataFrame([{"disposition":"STACK3_FROZEN_STATE_COMPOSITION_DEVELOPMENT_ONLY","sportsbook_upstream":0,"model_fit":0,"threshold_search":0,"weight_search":0,"validation_status":"2025_EXPOSED_DEVELOPMENT_FREEZE_FOR_2026_IF_RETAINED","next":"CAPABILITY_REVIEW_THEN_EFFICIENCY_ENVIRONMENT_ABLATIONS"}])
    for n,d in [("stack3_coverage.csv",cov),("stack3_slice_metrics.csv",sc),("stack3_market_metrics.csv",mm),("stack3_market_edge_buckets.csv",ed),("stack3_2025_casebook.csv",x),("stack3_disposition.csv",disp)]: d.to_csv(a.out_dir/n,index=False)
    print("=== coverage ===");print(cov.to_string(index=False));print("=== all-RB ===");print(sc.loc[sc.slice.eq("all_rb")].to_string(index=False));print("=== Week1 / 20+ / 25+ ===");print(sc.loc[sc.slice.isin(["week1","actual_20_plus","actual_25_plus","pregame_m95f_risk","pregame_vacancy"])].to_string(index=False));print("=== market ===");print(mm.to_string(index=False));print("=== edges ===");print(ed.to_string(index=False))
if __name__=="__main__": main()
