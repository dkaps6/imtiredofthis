#!/usr/bin/env python3
"""RB STACK6 / ND3: pregame-universe, as-of secondary-back situational role model.

Research/development only. Target rows come from frozen P3. Historical participation
supplies only completed-prior-game information. Sportsbook is downstream audit only.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHA = 10.0
START_WEEK = 6
MIN_TRAIN = 40
DELTA_CAP = 4.0
TEAM_ALIASES = {"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}

AGG_FEATURE_CANDIDATES = [
    "depth_rank", "depth_slot", "prior1_snap_pct", "prior3_snap_pct",
    "prior3_rb_share", "credible_competitors", "prior_backfield_hhi",
    "injury_reported", "injury_out_doubtful", "injury_questionable",
    "rookie_flag", "prior1_rb_share", "prior1_carries", "prior3_carries",
]
SITUATIONS = [
    "early_down", "third_down", "third_long", "two_minute", "short_yardage",
    "red_zone", "inside10", "inside5", "shotgun", "under_center",
]
PERSONNEL = ["11", "12", "21", "22"]


def num(s): return pd.to_numeric(s, errors="coerce")

def tm(v):
    s = "" if pd.isna(v) else str(v).strip().upper()
    return TEAM_ALIASES.get(s, s) if s not in {"", "NAN", "NONE", "<NA>"} else ""

def nk(v):
    s = "" if pd.isna(v) else str(v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", s)

def lower(x):
    z=x.copy(); z.columns=[str(c).strip().lower() for c in z.columns]; return z

def one(root:Path,name:str):
    hits=list(root.rglob(name))
    if len(hits)!=1: raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0],low_memory=False))

def prep_target(x):
    z=x.copy()
    z["season"]=num(z.get("season",2025)).fillna(2025).astype(int)
    z["week"]=num(z["week"]).astype(int); z["team"]=z["team"].map(tm)
    if "join_key" in z: z["player_key"]=z["join_key"].map(nk)
    elif "player_clean_key" in z: z["player_key"]=z["player_clean_key"].map(nk)
    elif "name_key" in z: z["player_key"]=z["name_key"].map(nk)
    else: z["player_key"]=z["player"].map(nk)
    z["target_order"]=z["season"]*100+z["week"]
    return z

def prep_roles(x):
    z=x.copy(); z["season"]=num(z.season).astype(int); z["week"]=num(z.week).astype(int); z["team"]=z.team.map(tm)
    if "player_clean_key" in z: z["player_key"]=z.player_clean_key.map(nk)
    elif "player_name" in z: z["player_key"]=z.player_name.map(nk)
    else: raise RuntimeError("STACK6 role history missing player identity")
    z["source_order"]=z.season*100+z.week
    return z

def metric(y,p):
    q=pd.DataFrame({"y":num(y),"p":num(p)}).dropna()
    if q.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan}
    e=q.p-q.y
    corr=q.y.corr(q.p) if len(q)>=3 and q.y.nunique()>1 and q.p.nunique()>1 else np.nan
    return {"n":int(len(q)),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(corr) if pd.notna(corr) else np.nan}

def ridge():
    return Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("model",Ridge(alpha=RIDGE_ALPHA))])


def add_role_ownership(r):
    z=r.copy()
    grp=["season","week","team"]
    def share(src,out):
        if src not in z: return
        den=z.groupby(grp)[src].transform("sum").replace(0,np.nan)
        z[out]=num(z[src])/den
    share("rb_onfield_plays","hist_rb_presence_share")
    share("rush_attempts_owned","hist_rush_share")
    for s in SITUATIONS:
        share(f"{s}_onfield_plays",f"hist_{s}_presence_share")
        if f"{s}_rushes_owned" in z: share(f"{s}_rushes_owned",f"hist_{s}_rush_share")
    for code in PERSONNEL:
        c=f"personnel_{code}_onfield_share"
        if c in z:
            z[f"_personnel_{code}_plays_proxy"]=num(z.rb_onfield_plays)*num(z[c])
            share(f"_personnel_{code}_plays_proxy",f"hist_personnel_{code}_presence_share")
    # Team-game concentration descriptors reconstructed from player shares.
    team=[]
    own=[c for c in z.columns if c.startswith("hist_") and c.endswith("_share")]
    for keys,g in z.groupby(grp,sort=True):
        rec={"season":keys[0],"week":keys[1],"team":keys[2],"source_order":keys[0]*100+keys[1],"team_role_rb_count":int(g.player_key.nunique())}
        for c in own:
            vals=num(g[c]).dropna(); rec[f"team_hhi_{c[5:-6]}"]=float(np.square(vals).sum()) if len(vals) else np.nan
        team.append(rec)
    return z,pd.DataFrame(team),own


def mean_last(g,cols,n):
    if g.empty:return {c:np.nan for c in cols}
    q=g.sort_values("source_order").tail(n)
    return {c:float(num(q[c]).mean()) if c in q and num(q[c]).notna().any() else np.nan for c in cols}


def asof_features(target,roles,team_hist,own_cols):
    z=target.copy()
    # Fixed situational source values: ownership shares plus player-role composition.
    comp=[c for c in roles.columns if c.endswith("_share_of_player_plays") or (c.startswith("personnel_") and c.endswith("_onfield_share"))]
    player_cols=own_cols+comp+[c for c in ["rush_attempts_owned","team_rb_count","team_rb_hhi","team_rb_rank"] if c in roles]
    team_cols=[c for c in team_hist.columns if c.startswith("team_hhi_") or c=="team_role_rb_count"]
    rows=[]
    source_keys=set(roles.player_key.dropna().astype(str))
    for idx,r in z.iterrows():
        order=int(r.target_order); pk=str(r.player_key); team=str(r.team)
        hp=roles.loc[roles.player_key.eq(pk)&roles.source_order.lt(order)].copy()
        hs=hp.loc[hp.team.eq(team)].copy()
        ht=team_hist.loc[team_hist.team.eq(team)&team_hist.source_order.lt(order)].copy()
        rec={"_idx":idx,"stack6_identity_match_anywhere":int(pk in source_keys),"stack6_history_games":int(len(hp)),"stack6_same_team_history_games":int(len(hs)),
             "stack6_player_max_source_order":float(hp.source_order.max()) if len(hp) else np.nan,
             "stack6_same_team_max_source_order":float(hs.source_order.max()) if len(hs) else np.nan,
             "stack6_team_max_source_order":float(ht.source_order.max()) if len(ht) else np.nan}
        for prefix,g in [("any",hp),("same",hs)]:
            for n in [1,3]:
                vals=mean_last(g,player_cols,n)
                rec.update({f"sr_{prefix}_p{n}_{k}":v for k,v in vals.items()})
        for n in [1,3]:
            vals=mean_last(ht,team_cols,n)
            rec.update({f"sr_team_p{n}_{k}":v for k,v in vals.items()})
        rows.append(rec)
    f=pd.DataFrame(rows).set_index("_idx")
    z=z.join(f,how="left")
    # Audit every actually used as-of source bound.
    safe=pd.Series(True,index=z.index)
    for c in ["stack6_player_max_source_order","stack6_same_team_max_source_order","stack6_team_max_source_order"]:
        v=num(z[c]); safe &= v.isna()|v.lt(num(z.target_order))
    z["stack6_asof_leakage_safe"]=safe.astype(int)
    sit=[c for c in z.columns if c.startswith("sr_")]
    return z,sit


def parent_fields(z):
    q=z.copy(); w=num(q.week)
    if "p3_parent" not in q: raise RuntimeError("STACK6 missing p3_parent")
    if "stack_att" not in q or "enriched_att" not in q: raise RuntimeError("STACK6 missing P3 opportunity components")
    q["parent_yards"]=num(q.p3_parent)
    q["parent_att"]=np.where(w.eq(1),num(q.stack_att),num(q.enriched_att))
    q["parent_ypc"]=np.where(num(q.parent_att).abs().gt(.20),num(q.parent_yards)/num(q.parent_att),np.nan)
    q["actual_rush_att"]=num(q.actual_rush_att);q["actual_rush_yards"]=num(q.actual_rush_yards)
    risk_col="state_m95f_risk_stack4" if "state_m95f_risk_stack4" in q else "state_m95f_risk"
    if risk_col in q:q["stack6_risk"]=num(q[risk_col]).fillna(0).eq(1)
    elif "cal_prob_20" in q and "m95f_p90" in q:q["stack6_risk"]=num(q.cal_prob_20).fillna(0).ge(.25)|num(q.m95f_p90).fillna(0).ge(20)
    else:raise RuntimeError("STACK6 missing frozen M95F risk state")
    if "depth_rank" not in q:raise RuntimeError("STACK6 missing pregame depth_rank")
    q["stack6_domain"]=num(q.week).ge(START_WEEK)&(~q.stack6_risk)&num(q.depth_rank).ge(2)
    q["stack6_model_eligible"]=q.stack6_domain&num(q.stack6_history_games).ge(1)&q.stack6_asof_leakage_safe.eq(1)
    q["carry_residual"]=q.actual_rush_att-q.parent_att
    return q


def available(z,cols,min_nonnull=30):
    return [c for c in cols if c in z and num(z[c]).notna().sum()>=min_nonnull]


def oof_predictions(z,feature_blocks):
    q=z.copy()
    for arm in feature_blocks:
        q[f"pred_att_{arm}"]=num(q.parent_att)
        q[f"pred_yards_{arm}"]=num(q.parent_yards)
        q[f"delta_{arm}"]=0.0
    fits=[]
    for week in range(START_WEEK,19):
        base_train=q.loc[num(q.week).lt(week)&q.stack6_model_eligible&q.carry_residual.notna()].copy()
        test=q.loc[num(q.week).eq(week)&q.stack6_model_eligible].copy()
        if test.empty:continue
        for arm,feats in feature_blocks.items():
            train=base_train.loc[base_train[feats].notna().any(axis=1)].copy()
            if len(train)<MIN_TRAIN:continue
            lo,hi=np.nanquantile(num(train.carry_residual),[.05,.95]); clo=max(float(lo),-DELTA_CAP); chi=min(float(hi),DELTA_CAP)
            model=ridge();model.fit(train[feats],num(train.carry_residual))
            d=np.clip(model.predict(test[feats]),clo,chi)
            att=np.clip(num(test.parent_att).to_numpy(dtype=float)+d,0,None)
            ypc=num(test.parent_ypc).to_numpy(dtype=float)
            yards=np.where(np.isfinite(ypc),att*ypc,num(test.parent_yards).to_numpy(dtype=float))
            q.loc[test.index,f"delta_{arm}"]=d;q.loc[test.index,f"pred_att_{arm}"]=att;q.loc[test.index,f"pred_yards_{arm}"]=yards
            fits.append({"week":week,"arm":arm,"train_n":len(train),"test_n":len(test),"clip_lo":clo,"clip_hi":chi,"mean_delta":float(np.mean(d)),"mean_abs_delta":float(np.mean(np.abs(d)))})
    return q,pd.DataFrame(fits)


def score_tables(z,arms):
    masks={
        "all_rb_w6_18":num(z.week).ge(START_WEEK),
        "eligible_w6_18":z.stack6_model_eligible,
        "eligible_w13_18":z.stack6_model_eligible&num(z.week).ge(13),
        "m95f_risk_w6_18":z.stack6_risk&num(z.week).ge(START_WEEK),
        "depth1_w6_18":num(z.depth_rank).eq(1)&num(z.week).ge(START_WEEK),
        "depth2_w6_18":num(z.depth_rank).eq(2)&num(z.week).ge(START_WEEK),
        "depth3plus_w6_18":num(z.depth_rank).ge(3)&num(z.week).ge(START_WEEK),
    }
    rows=[]
    cols={"P3_PARENT":("parent_att","parent_yards"),**{a:(f"pred_att_{a}",f"pred_yards_{a}") for a in arms}}
    for scope,m in masks.items():
        g=z.loc[m]
        for arm,(ac,yc) in cols.items():
            cm=metric(g.actual_rush_att,g[ac]);ym=metric(g.actual_rush_yards,g[yc])
            rows.append({"scope":scope,"arm":arm,"n":ym["n"],"carry_mae":cm["mae"],"carry_bias":cm["bias"],"yard_mae":ym["mae"],"yard_rmse":ym["rmse"],"yard_bias":ym["bias"],"yard_corr":ym["corr"]})
    return pd.DataFrame(rows)


def retention_gates(z,scores,feature_blocks):
    def row(scope,arm):
        q=scores.loc[scores.scope.eq(scope)&scores.arm.eq(arm)]
        if q.empty:raise RuntimeError(f"missing score {scope}/{arm}")
        return q.iloc[0]
    base=row("eligible_w6_18","P3_PARENT");allbase=row("all_rb_w6_18","P3_PARENT");latebase=row("eligible_w13_18","P3_PARENT")
    out=[]
    for arm,feats in feature_blocks.items():
        r=row("eligible_w6_18",arm); ar=row("all_rb_w6_18",arm); lr=row("eligible_w13_18",arm)
        carry_gain=float(base.carry_mae-r.carry_mae);yard_gain=float(base.yard_mae-r.yard_mae);late_gain=float(latebase.yard_mae-lr.yard_mae);all_reg=float(ar.yard_mae-allbase.yard_mae)
        bias_worsen=abs(float(r.carry_bias))-abs(float(base.carry_bias))
        risk_change=float(np.nanmax(np.abs(num(z.loc[z.stack6_risk,f"pred_yards_{arm}"])-num(z.loc[z.stack6_risk,"parent_yards"])))) if z.stack6_risk.any() else 0.0
        d1=num(z.depth_rank).eq(1);d1_change=float(np.nanmax(np.abs(num(z.loc[d1,f"pred_yards_{arm}"])-num(z.loc[d1,"parent_yards"])))) if d1.any() else 0.0
        passed=int(carry_gain>=.20 and yard_gain>=.15 and late_gain>0 and all_reg<=.05 and bias_worsen<=.25 and risk_change<=1e-9 and d1_change<=1e-9)
        out.append({"arm":arm,"feature_count":len(feats),"carry_mae_gain":carry_gain,"yard_mae_gain":yard_gain,"late_yard_mae_gain":late_gain,"all_rb_yard_mae_regression":all_reg,"carry_abs_bias_worsening":bias_worsen,"max_risk_yard_change":risk_change,"max_depth1_yard_change":d1_change,"gate_pass":passed})
    g=pd.DataFrame(out);passing=g.loc[g.gate_pass.eq(1)].copy();selected="NONE"
    if len(passing):
        best=float(passing.yard_mae_gain.max());pool=passing.loc[passing.yard_mae_gain.ge(best-.05)].sort_values(["feature_count","yard_mae_gain","arm"],ascending=[True,False,True]);selected=str(pool.iloc[0].arm)
    g["selected_arm"]=selected
    return g,selected


def market_audit(z,market,arms):
    cb=prep_target(market);keys=["season","week","team","player_key"]
    if "season" not in cb:cb["season"]=2025
    q=z.merge(cb[keys+["consensus_line"]].drop_duplicates(keys),on=keys,how="inner",validate="one_to_one")
    rows=[];details=[]
    cols={"P3_PARENT":"parent_yards",**{a:f"pred_yards_{a}" for a in arms},"VEGAS_CONSENSUS":"consensus_line"}
    strata={"all":pd.Series(True,index=q.index),"m95f_risk":q.stack6_risk,"m95f_nonrisk":~q.stack6_risk,"depth1":num(q.depth_rank).eq(1),"depth2":num(q.depth_rank).eq(2),"depth3plus":num(q.depth_rank).ge(3)}
    for arm,col in cols.items():
        for scope,m in strata.items():
            met=metric(q.loc[m,"actual_rush_yards"],q.loc[m,col]);rows.append({"scope":scope,"arm":arm,**met})
    # Downstream-only disagreement audit relative to Vegas.
    for arm,col in {k:v for k,v in cols.items() if k!="VEGAS_CONSENSUS"}.items():
        edge=num(q[col])-num(q.consensus_line);me=(num(q[col])-num(q.actual_rush_yards)).abs();ve=(num(q.consensus_line)-num(q.actual_rush_yards)).abs()
        buckets={"within5":edge.abs().lt(5),"above5_10":edge.between(5,10,inclusive="left"),"below5_10":edge.between(-10,-5,inclusive="right"),"above10":edge.ge(10),"below10":edge.le(-10)}
        for b,m in buckets.items():
            if not m.any():continue
            details.append({"arm":arm,"bucket":b,"n":int(m.sum()),"model_mae":float(me[m].mean()),"vegas_mae":float(ve[m].mean()),"model_closer_rate":float((me[m]<ve[m]).mean()),"mean_edge":float(edge[m].mean())})
    return q,pd.DataFrame(rows),pd.DataFrame(details)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stack4-root",type=Path,required=True);ap.add_argument("--source-root",type=Path,required=True);ap.add_argument("--market-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    target=prep_target(one(a.stack4_root,"stack4_2025_casebook.csv"));roles=prep_roles(one(a.source_root,"stack6_rb_player_game_roles.csv"));market=one(a.market_root,"rb_market_casebook.csv")
    roles,team_hist,own_cols=add_role_ownership(roles);x,sit=asof_features(target,roles,team_hist,own_cols);x=parent_fields(x)
    agg=available(x,AGG_FEATURE_CANDIDATES);sit=available(x,sit)
    if not agg:raise RuntimeError("STACK6 aggregate block empty")
    if len(sit)<10:raise RuntimeError(f"STACK6 situational block too small: {len(sit)}")
    blocks={"AGG_ROLE":agg,"SITUATIONAL_ROLE":sit,"AGG_PLUS_SITUATIONAL":agg+sit}
    # Source identity is audit-only; as-of history determines feature availability.
    identity_all=float(x.stack6_identity_match_anywhere.mean());x_market_keys=prep_target(market)[["season","week","team","player_key"]].drop_duplicates();xm=x.merge(x_market_keys,on=["season","week","team","player_key"],how="inner")
    identity_market=float(xm.stack6_identity_match_anywhere.mean()) if len(xm) else np.nan
    asof_cov=float(x.stack6_history_games.ge(1).mean());asof_market=float(xm.stack6_history_games.ge(1).mean()) if len(xm) else np.nan
    leak=float(x.stack6_asof_leakage_safe.mean())
    coverage=pd.DataFrame([{"target_rows":len(x),"market_join_rows_pre_audit":len(xm),"identity_match_anywhere_all":identity_all,"identity_match_anywhere_market":identity_market,"asof_history_coverage_all":asof_cov,"asof_history_coverage_market":asof_market,"asof_leakage_pass_rate":leak,"eligible_model_rows":int(x.stack6_model_eligible.sum()),"aggregate_feature_count":len(agg),"situational_feature_count":len(sit)}])
    if identity_market<.90:raise RuntimeError(f"STACK6 identity coverage below 90% on market universe: {identity_market:.4f}")
    if leak<1.0:raise RuntimeError(f"STACK6 as-of leakage audit failed: {leak:.6f}")
    pred,fits=oof_predictions(x,blocks);scores=score_tables(pred,blocks);gates,selected=retention_gates(pred,scores,blocks);mq,mm,edges=market_audit(pred,market,blocks)
    situational_pass=int(gates.loc[gates.arm.isin(["SITUATIONAL_ROLE","AGG_PLUS_SITUATIONAL"]),"gate_pass"].max())
    if selected!="NONE":
        disposition="STACK6_HISTORICAL_INFORMATION_FAMILY_SUPPORTED_REQUIRES_LIVE_SOURCE_AND_2026_CONFIRMATION" if selected in {"SITUATIONAL_ROLE","AGG_PLUS_SITUATIONAL"} else "STACK6_AGGREGATE_RECALIBRATION_ONLY_NO_NEW_ROLE_FAMILY_WIN"
    else:disposition="STACK6_NO_RETAINABLE_SECONDARY_ROLE_INCREMENT"
    disp=pd.DataFrame([{"selected_arm":selected,"situational_family_gate_pass":situational_pass,"disposition":disposition,"sportsbook_upstream":0,"model_fit":1,"hyperparameter_search":0,"feature_search":0,"threshold_search":0,"weight_search":0,"production_change":0,"validation_status":"2025_EXPOSED_RETROSPECTIVE_DEVELOPMENT","live_source_ready":0,"next":"FREEZE_FOR_2026_AND_FIND_LIVE_EQUIVALENT" if situational_pass else "FAILURE_ATLAS_OR_NEW_INFORMATION_FAMILY"}])
    feat_rows=[]
    for arm,fs in blocks.items():
        for f in fs:feat_rows.append({"arm":arm,"feature":f,"nonnull_rate":float(num(pred[f]).notna().mean()),"eligible_nonnull_rate":float(num(pred.loc[pred.stack6_model_eligible,f]).notna().mean()) if pred.stack6_model_eligible.any() else np.nan})
    outputs={"stack6_coverage.csv":coverage,"stack6_features.csv":pd.DataFrame(feat_rows),"stack6_weekly_fits.csv":fits,"stack6_score_table.csv":scores,"stack6_retention_gates.csv":gates,"stack6_market_metrics.csv":mm,"stack6_market_disagreement.csv":edges,"stack6_disposition.csv":disp,"stack6_2025_casebook.csv":pred,"stack6_market_casebook.csv":mq}
    for n,d in outputs.items():d.to_csv(a.out_dir/n,index=False)
    print("=== coverage ===");print(coverage.to_string(index=False));print("=== eligible football scores ===");print(scores.loc[scores.scope.isin(["eligible_w6_18","eligible_w13_18","all_rb_w6_18"])].to_string(index=False));print("=== gates ===");print(gates.to_string(index=False));print("=== market ===");print(mm.loc[mm.scope.isin(["all","m95f_nonrisk","depth2","depth3plus"])].to_string(index=False));print("=== disposition ===");print(disp.to_string(index=False))
    return 0

if __name__=="__main__":raise SystemExit(main())
