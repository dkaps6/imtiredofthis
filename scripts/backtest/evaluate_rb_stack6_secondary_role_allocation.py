#!/usr/bin/env python3
"""RB STACK6 / ND3: secondary-back situational role allocation ablation.

Frozen protocol:
- role model fits 2024 only;
- all role features are strictly lagged from completed prior games on the same team;
- Week 1 remains the frozen P3 parent;
- primary arm is 75% current allocation / 25% role signal for depth-rank >=2,
  non-M95F-risk backs with valid prior role history;
- team RB opportunity pool is conserved;
- sportsbook is downstream benchmark only and cannot select/pass the module.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
ALIAS={"audricestime":"audricestim"}
KEYS=["season","week","team","join_key"]
ALPHA=10.0
ROLE_WEIGHT=0.25

# Frozen scientific gates, written before first result is exposed.
MIN_ALL_YARD_GAIN=0.05
MIN_SECONDARY_YARD_GAIN=0.20
MIN_SECONDARY_CARRY_GAIN=0.05
MAX_DEPTH1_YARD_REG=0.10
MAX_RISK_YARD_REG=0.10
MAX_ALL_RMSE_REG=0.15
MAX_ABS_BIAS_DETERIORATION=0.50

ROLE_BASE=[
    "snap_share","early_down_share","third_down_share","two_minute_share",
    "short_yardage_share","red_zone_share","inside_10_share","inside_5_share",
    "single_back_share","multi_back_share","drive_share","rush_per_snap",
]
ROLE_FEATURES=[]
for c in ROLE_BASE:
    ROLE_FEATURES.extend([f"prior1_{c}",f"prior3_{c}"])
ROLE_FEATURES += ["prior_role_games"]


def lower(x):
    z=x.copy(); z.columns=[str(c).strip().lower() for c in z.columns]; return z

def num(s): return pd.to_numeric(s,errors="coerce")
def tm(v):
    s=str(v).strip().upper() if not pd.isna(v) else ""
    return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v):
    s=re.sub(r"[^a-z0-9]","",str(v or "").lower())
    return ALIAS.get(s,s)
def one(root:Path,name:str):
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected exactly one {name} under {root}; found {len(h)}")
    return lower(pd.read_csv(h[0],low_memory=False))
def metric(y,p):
    q=pd.DataFrame({"a":num(y),"p":num(p)}).dropna()
    if q.empty: return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan,"actual_mean":np.nan,"pred_mean":np.nan}
    e=q.p-q.a
    corr=q.a.corr(q.p) if len(q)>=3 and q.a.nunique()>1 and q.p.nunique()>1 else np.nan
    return {"n":int(len(q)),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(corr) if pd.notna(corr) else np.nan,"actual_mean":float(q.a.mean()),"pred_mean":float(q.p.mean())}


def prep_role(raw:pd.DataFrame)->pd.DataFrame:
    r=raw.copy()
    r["season"]=num(r.season).astype(int); r["week"]=num(r.week).astype(int); r["team"]=r.team.map(tm)
    r["join_key"]=r.player_name.astype(str).map(nk)
    count_cols=["snap","early_down","third_down","two_minute","short_yardage","red_zone","inside_10","inside_5","single_back_on_field","multi_back_on_field","is_rusher","drives_seen"]
    for c in count_cols: r[c]=num(r.get(c,pd.Series(np.nan,index=r.index))).fillna(0.0)
    # Denominators are sums of RB/FB participant counts in each team-game, which
    # intentionally preserve multi-back overlap rather than pretending each play
    # contains only one back.
    gkeys=["season","week","team"]
    mapping={
        "snap":"snap_share","early_down":"early_down_share","third_down":"third_down_share",
        "two_minute":"two_minute_share","short_yardage":"short_yardage_share","red_zone":"red_zone_share",
        "inside_10":"inside_10_share","inside_5":"inside_5_share","single_back_on_field":"single_back_share",
        "multi_back_on_field":"multi_back_share","drives_seen":"drive_share",
    }
    for src,out in mapping.items():
        den=r.groupby(gkeys)[src].transform("sum")
        r[out]=np.where(den.gt(0),r[src]/den,0.0)
    r["rush_per_snap"]=np.where(r.snap.gt(0),r.is_rusher/r.snap,0.0)
    carry_den=r.groupby(gkeys)["is_rusher"].transform("sum")
    r["actual_role_share"]=np.where(carry_den.gt(0),r.is_rusher/carry_den,0.0)
    return r


def add_lags(role:pd.DataFrame)->pd.DataFrame:
    z=role.sort_values(["season","team","join_key","week"]).copy()
    out=[]
    for _,g in z.groupby(["season","team","join_key"],sort=False):
        g=g.sort_values("week").copy()
        for c in ROLE_BASE:
            s=num(g[c])
            g[f"prior1_{c}"]=s.shift(1)
            g[f"prior3_{c}"]=s.shift(1).rolling(3,min_periods=1).mean()
        g["prior_role_games"]=np.arange(len(g),dtype=float)
        out.append(g)
    return pd.concat(out,ignore_index=True) if out else z


def model_pipe():
    return Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",StandardScaler()),
        ("ridge",Ridge(alpha=ALPHA)),
    ])


def predict_role(role_lag:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    train=role_lag.loc[role_lag.season.eq(2024)&num(role_lag.prior_role_games).ge(1)&num(role_lag.actual_role_share).notna()].copy()
    test=role_lag.loc[role_lag.season.eq(2025)&num(role_lag.prior_role_games).ge(1)].copy()
    if len(train)<500 or test.empty: raise RuntimeError(f"insufficient STACK6 role rows train={len(train)} test={len(test)}")
    m=model_pipe(); m.fit(train[ROLE_FEATURES],num(train.actual_role_share).clip(0,1))
    for q,label in [(train,"train_2024"),(test,"test_2025")]:
        q["role_raw_pred"]=np.clip(m.predict(q[ROLE_FEATURES]),0,1)
        q["scope"]=label
    audit=pd.DataFrame([
        {"scope":"2024_fit_in_sample_diagnostic",**metric(train.actual_role_share,train.role_raw_pred)},
        {"scope":"2025_role_rows_raw_diagnostic",**metric(test.actual_role_share,test.role_raw_pred)},
    ])
    keep=KEYS+["player_name","actual_role_share","role_raw_pred","prior_role_games"]+ROLE_FEATURES
    return test[[c for c in keep if c in test.columns]].copy(),audit


def prep_parent(x:pd.DataFrame)->pd.DataFrame:
    z=x.copy(); z["season"]=num(z.get("season",2025)).fillna(2025).astype(int); z["week"]=num(z.week).astype(int); z["team"]=z.team.map(tm)
    if "player_clean_key" in z.columns: z["join_key"]=z.player_clean_key.astype(str).map(nk)
    else: z["join_key"]=z.player.astype(str).map(nk)
    z["baseline_att"]=num(z.enriched_att)
    z["stack_eff"]=num(z.stack_implied_ypc)
    z["parent_yards"]=np.where(z.week.eq(1),num(z.stack_yards),num(z.arch_enriched_opp_stack_eff_yards))
    z["parent_att"]=np.where(z.week.eq(1),num(z.stack_att),z.baseline_att)
    z["actual_carries"]=num(z.get("actual_carries",z.get("actual_rush_att")))
    z["actual_rush_yards"]=num(z.actual_rush_yards)
    # Frozen P3 high-workload risk state from STACK3.
    z["state_m95f_risk"]=num(z.state_m95f_risk).fillna(0).astype(int)
    rank=num(z.depth_rank)
    missing=num(z.get("depth_rank_missing",pd.Series(0,index=z.index))).fillna(0)
    z["secondary_defined"]=((rank.ge(2))&missing.eq(0)).astype(int)
    return z


def normalize_scores(x:pd.DataFrame,score:str,out:str):
    vals=pd.Series(np.nan,index=x.index,dtype=float)
    for _,g in x.groupby(["season","week","team"],sort=False):
        s=num(g[score]).fillna(0).clip(lower=0)
        tot=float(s.sum())
        if tot<=0: s=pd.Series(np.ones(len(g))/max(len(g),1),index=g.index)
        else: s=s/tot
        vals.loc[g.index]=s
    x[out]=vals


def add_arms(parent:pd.DataFrame,role_pred:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    x=parent.merge(role_pred.drop_duplicates(KEYS),on=KEYS,how="left",validate="one_to_one",suffixes=("","_role"))
    cov=pd.DataFrame([{
        "rows":len(x),"role_prediction_rows":int(num(x.role_raw_pred).notna().sum()),
        "role_prediction_coverage":float(num(x.role_raw_pred).notna().mean()),
        "weeks2_18_role_coverage":float(num(x.loc[x.week.ge(2),"role_raw_pred"]).notna().mean()),
        "secondary_nonrisk_rows":int((x.secondary_defined.eq(1)&x.state_m95f_risk.eq(0)&x.week.ge(2)).sum()),
        "secondary_nonrisk_role_rows":int((x.secondary_defined.eq(1)&x.state_m95f_risk.eq(0)&x.week.ge(2)&num(x.role_raw_pred).notna()).sum()),
    }])
    pool=x.groupby(["season","week","team"])["baseline_att"].transform("sum")
    x["team_rb_pool"]=pool
    x["baseline_share"]=np.where(pool.gt(0),x.baseline_att/pool,0.0)
    # Role-only diagnostic uses baseline fallback for rows lacking role history.
    x["role_score_fallback"]=num(x.role_raw_pred).where(num(x.role_raw_pred).notna(),x.baseline_share)
    normalize_scores(x,"role_score_fallback","role_share_normalized")
    x["role_only_att"]=x.role_share_normalized*pool
    x["role_only_yards"]=np.where(x.week.eq(1),x.parent_yards,x.role_only_att*x.stack_eff)

    # Diagnostic 25% role blend for every row with role history.
    avail=num(x.role_raw_pred).notna()&x.week.ge(2)
    x["all25_raw_share"]=x.baseline_share
    x.loc[avail,"all25_raw_share"]=(1-ROLE_WEIGHT)*x.loc[avail,"baseline_share"]+ROLE_WEIGHT*x.loc[avail,"role_share_normalized"]
    normalize_scores(x,"all25_raw_share","all25_share")
    x["all25_att"]=x.all25_share*pool
    x["all25_yards"]=np.where(x.week.eq(1),x.parent_yards,x.all25_att*x.stack_eff)

    # Primary frozen candidate: secondary, non-risk, valid-role rows only.
    route=x.secondary_defined.eq(1)&x.state_m95f_risk.eq(0)&x.week.ge(2)&num(x.role_raw_pred).notna()
    x["primary_route"]=route.astype(int)
    x["primary_raw_share"]=x.baseline_share
    x.loc[route,"primary_raw_share"]=(1-ROLE_WEIGHT)*x.loc[route,"baseline_share"]+ROLE_WEIGHT*x.loc[route,"role_share_normalized"]
    normalize_scores(x,"primary_raw_share","primary_share")
    x["primary_att"]=x.primary_share*pool
    x["primary_yards"]=np.where(x.week.eq(1),x.parent_yards,x.primary_att*x.stack_eff)

    # Pool conservation audit.
    audit=[]
    for arm in ["role_only_att","all25_att","primary_att"]:
        q=x.groupby(["season","week","team"],as_index=False).agg(pred_pool=(arm,"sum"),base_pool=("team_rb_pool","first"))
        q["diff"]=(q.pred_pool-q.base_pool).abs()
        audit.append({"arm":arm,"team_games":len(q),"max_abs_pool_diff":float(q["diff"].max()),"mean_abs_pool_diff":float(q["diff"].mean())})
    return x,pd.DataFrame(audit)


def masks(x):
    rank=num(x.depth_rank); risk=x.state_m95f_risk.eq(1); sec=x.secondary_defined.eq(1); nonrisk=~risk
    return {
        "all_rb":pd.Series(True,index=x.index),
        "weeks2_18":x.week.ge(2),
        "secondary_nonrisk":sec&nonrisk&x.week.ge(2),
        "depth_rank1":rank.eq(1),
        "depth_rank2":rank.eq(2),
        "depth_rank3plus":rank.ge(3),
        "m95f_risk":risk,
        "m95f_nonrisk":nonrisk,
        "primary_routed":x.primary_route.eq(1),
    }

ARMS={"P3_PARENT":("parent_att","parent_yards"),"ROLE_ONLY_DIAGNOSTIC":("role_only_att","role_only_yards"),"ROLE25_ALL_DIAGNOSTIC":("all25_att","all25_yards"),"ROLE25_SECONDARY_NONRISK":("primary_att","primary_yards")}

def score(x):
    rows=[]
    for sl,mask in masks(x).items():
        q=x.loc[mask]
        for arm,(ac,yc) in ARMS.items():
            rows.append({"target":"rush_att","slice":sl,"arm":arm,**metric(q.actual_carries,q[ac])})
            rows.append({"target":"rush_yards","slice":sl,"arm":arm,**metric(q.actual_rush_yards,q[yc])})
    return pd.DataFrame(rows)


def row(sc,target,sl,arm):
    q=sc.loc[sc.target.eq(target)&sc.slice.eq(sl)&sc.arm.eq(arm)]
    if q.empty: raise RuntimeError(f"missing metric {target}/{sl}/{arm}")
    return q.iloc[0]

def build_gate(sc):
    b_all=row(sc,"rush_yards","all_rb","P3_PARENT"); c_all=row(sc,"rush_yards","all_rb","ROLE25_SECONDARY_NONRISK")
    b_sec=row(sc,"rush_yards","secondary_nonrisk","P3_PARENT"); c_sec=row(sc,"rush_yards","secondary_nonrisk","ROLE25_SECONDARY_NONRISK")
    b_sec_c=row(sc,"rush_att","secondary_nonrisk","P3_PARENT"); c_sec_c=row(sc,"rush_att","secondary_nonrisk","ROLE25_SECONDARY_NONRISK")
    b_d1=row(sc,"rush_yards","depth_rank1","P3_PARENT"); c_d1=row(sc,"rush_yards","depth_rank1","ROLE25_SECONDARY_NONRISK")
    b_r=row(sc,"rush_yards","m95f_risk","P3_PARENT"); c_r=row(sc,"rush_yards","m95f_risk","ROLE25_SECONDARY_NONRISK")
    all_gain=float(b_all.mae-c_all.mae); sec_gain=float(b_sec.mae-c_sec.mae); sec_c_gain=float(b_sec_c.mae-c_sec_c.mae)
    d1_reg=float(c_d1.mae-b_d1.mae); risk_reg=float(c_r.mae-b_r.mae); rmse_reg=float(c_all.rmse-b_all.rmse)
    bias_det=float(abs(c_all.bias)-abs(b_all.bias))
    checks=[
        ("all_yard_mae_gain",all_gain,MIN_ALL_YARD_GAIN,">="),
        ("secondary_nonrisk_yard_mae_gain",sec_gain,MIN_SECONDARY_YARD_GAIN,">="),
        ("secondary_nonrisk_carry_mae_gain",sec_c_gain,MIN_SECONDARY_CARRY_GAIN,">="),
        ("depth1_yard_mae_regression",d1_reg,MAX_DEPTH1_YARD_REG,"<="),
        ("m95f_risk_yard_mae_regression",risk_reg,MAX_RISK_YARD_REG,"<="),
        ("all_rmse_regression",rmse_reg,MAX_ALL_RMSE_REG,"<="),
        ("abs_bias_deterioration",bias_det,MAX_ABS_BIAS_DETERIORATION,"<="),
    ]
    rows=[]
    for name,val,thr,op in checks:
        passed=(val>=thr) if op==">=" else (val<=thr)
        rows.append({"check":name,"value":val,"operator":op,"threshold":thr,"pass":int(passed)})
    g=pd.DataFrame(rows); return g,int(g["pass"].all())


def market_score(x,cb):
    c=cb.copy(); c["week"]=num(c.week).astype(int); c["team"]=c.team.map(tm); c["join_key"]=c.player.astype(str).map(nk)
    q=x.merge(c[["week","team","join_key","consensus_line"]].drop_duplicates(["week","team","join_key"]),on=["week","team","join_key"],how="inner",validate="one_to_one")
    rows=[]
    for arm,col in {"P3_PARENT":"parent_yards","ROLE25_SECONDARY_NONRISK":"primary_yards","VEGAS_CONSENSUS":"consensus_line"}.items(): rows.append({"arm":arm,**metric(q.actual_rush_yards,q[col])})
    edges=[]
    for arm,col in {"P3_PARENT":"parent_yards","ROLE25_SECONDARY_NONRISK":"primary_yards"}.items():
        z=q[["actual_rush_yards","consensus_line",col]].dropna().copy(); z["edge"]=num(z[col])-num(z.consensus_line); z["abs_edge"]=z.edge.abs(); z["bucket"]=pd.cut(z.abs_edge,[0,2.5,5,10,np.inf],labels=["0_2.5","2.5_5","5_10","10_plus"],right=False)
        z["closer"]=(np.abs(num(z[col])-num(z.actual_rush_yards))<np.abs(num(z.consensus_line)-num(z.actual_rush_yards))).astype(float)
        for b,g in z.groupby("bucket",observed=True): edges.append({"arm":arm,"edge_bucket":str(b),"n":len(g),"model_closer_rate":float(g.closer.mean()),"model_mae":float(np.abs(num(g[col])-num(g.actual_rush_yards)).mean()),"vegas_mae":float(np.abs(num(g.consensus_line)-num(g.actual_rush_yards).mean()) if False else np.abs(num(g.consensus_line)-num(g.actual_rush_yards)).mean()),"mean_edge":float(g.edge.mean())})
    return q,pd.DataFrame(rows),pd.DataFrame(edges)


def capability(sc,gate_pass):
    def gain(target,sl):
        b=row(sc,target,sl,"P3_PARENT"); c=row(sc,target,sl,"ROLE25_SECONDARY_NONRISK"); return float(b.mae-c.mae)
    return pd.DataFrame([
        {"module":"situational_role_participation","status":"RETAIN_PRIMARY_DEVELOPMENT" if gate_pass else "SCIENTIFIC_CLUE_ONLY_OR_REJECT_PRIMARY","detail":f"secondary_nonrisk carry MAE gain={gain('rush_att','secondary_nonrisk'):.6f}; yard MAE gain={gain('rush_yards','secondary_nonrisk'):.6f}"},
        {"module":"P3","status":"RETAIN_PARENT","detail":"Week1 full stack; Weeks2-18 enriched opportunity x full-stack efficiency"},
        {"module":"M95F_RISK","status":"PROTECTED_REGIME","detail":f"yard MAE gain under primary={gain('rush_yards','m95f_risk'):.6f}; no mean-router retune"},
    ])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--role-root",type=Path,required=True); ap.add_argument("--stack3-root",type=Path,required=True); ap.add_argument("--market-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    role=add_lags(prep_role(one(a.role_root,"stack6_target_game_role_observations_DIAGNOSTIC_ONLY.csv")))
    role_pred,role_audit=predict_role(role)
    parent=prep_parent(one(a.stack3_root,"stack3_2025_casebook.csv"))
    x,pool_audit=add_arms(parent,role_pred)
    if float(pool_audit.max_abs_pool_diff.max())>1e-6: raise RuntimeError(f"team RB pool conservation failed: {pool_audit.max_abs_pool_diff.max()}")
    sc=score(x); gate,passed=build_gate(sc)
    cb=one(a.market_root,"rb_market_casebook.csv"); mq,mm,edges=market_score(x,cb)
    cov=pd.DataFrame([{
        "stack3_rows":len(parent),"role_source_2024_rows":int(role.season.eq(2024).sum()),"role_source_2025_rows":int(role.season.eq(2025).sum()),
        "role_join_rows":int(num(x.role_raw_pred).notna().sum()),"role_join_coverage":float(num(x.role_raw_pred).notna().mean()),
        "primary_route_rows":int(x.primary_route.sum()),"market_rows":len(mq),"sportsbook_upstream":0,"target_game_participation_upstream":0,
    }])
    disp="STACK6_ROLE25_SECONDARY_NONRISK_RETAIN_DEVELOPMENT_FREEZE_FOR_2026" if passed else "STACK6_ROLE_STATE_PRIMARY_FAILED_NO_RETUNE"
    disposition=pd.DataFrame([{"disposition":disp,"gate_pass":passed,"role_model":"ridge_alpha_10_frozen","role_weight":ROLE_WEIGHT,"fit_season":2024,"evaluation_season":2025,"sportsbook_upstream":0,"market_used_for_selection":0,"validation_status":"2025_EXPOSED_DEVELOPMENT","next":"FREEZE_FOR_PROSPECTIVE_2026" if passed else "FAILURE_ATLAS_THEN_GENUINELY_DIFFERENT_FOOTBALL_MECHANISM"}])
    ledger=capability(sc,passed)
    for name,df in [
        ("stack6_role_model_audit.csv",role_audit),("stack6_coverage.csv",cov),("stack6_pool_conservation.csv",pool_audit),
        ("stack6_football_metrics.csv",sc),("stack6_primary_gate.csv",gate),("stack6_market_metrics_DOWNSTREAM.csv",mm),
        ("stack6_market_edges_DOWNSTREAM.csv",edges),("stack6_capability_ledger.csv",ledger),("stack6_disposition.csv",disposition),
        ("stack6_2025_casebook.csv",x),
    ]: df.to_csv(a.out_dir/name,index=False)
    print("=== COVERAGE ==="); print(cov.to_string(index=False))
    print("=== ROLE MODEL AUDIT ==="); print(role_audit.to_string(index=False))
    print("=== FOOTBALL HEADLINES ==="); print(sc.loc[sc.slice.isin(["all_rb","secondary_nonrisk","depth_rank1","depth_rank2","depth_rank3plus","m95f_risk"])].to_string(index=False))
    print("=== PRIMARY GATE ==="); print(gate.to_string(index=False))
    print("=== MARKET DOWNSTREAM ==="); print(mm.to_string(index=False))
    print("=== DISPOSITION ==="); print(disposition.to_string(index=False))

if __name__=="__main__": main()
