#!/usr/bin/env python3
"""RB receiving share-state coverage audit V1.

No fitting. Compares current ACT-only model RB-room target allocation with
strict-prior RB receiving identity state already defined in the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.rb_receiving_identity_runtime_v1 import (
    attach_identity,
    identity_atlas,
)

SEASON=2026
WEEKS=(1,2,3,4)
ROOM_FIELDS=[
    "prior_rb_room_share",
    "last8_rb_room_share",
    "prev_season_rb_room_share",
    "same_team_prior_rb_room_share",
]
SUPPORT_FIELDS=[
    "prior_targets_pg",
    "last8_targets_pg",
    "prev_season_targets_pg",
    "same_team_prior_targets_pg",
    "prior_target_share",
    "last8_target_share",
    "prev_season_target_share",
]
ALL_FIELDS=ROOM_FIELDS+SUPPORT_FIELDS
TOL=1e-10

def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def _spearman(a:pd.Series,b:pd.Series)->float:
    q=pd.DataFrame({"a":pd.to_numeric(a,errors="coerce"),"b":pd.to_numeric(b,errors="coerce")}).dropna()
    if len(q)<3 or q.a.nunique()<2 or q.b.nunique()<2:
        return np.nan
    return float(q.a.corr(q.b,method="spearman"))

def _normalize_room(g:pd.DataFrame,col:str)->pd.Series:
    x=pd.to_numeric(g[col],errors="coerce")
    if x.notna().sum()==0:
        return pd.Series(np.nan,index=g.index,dtype=float)
    x=x.fillna(0.0).clip(lower=0.0)
    s=float(x.sum())
    if not np.isfinite(s) or s<=0:
        return pd.Series(np.nan,index=g.index,dtype=float)
    return x/s

def _source_consumption_audit(repo_root:Path)->pd.DataFrame:
    files={
        "bayesian_v2": repo_root/"scripts/modeling/bayesian_v2.py",
        "simulation_rules": repo_root/"scripts/modeling/simulation_rules.py",
        "target_entitlement_v1": repo_root/"scripts/modeling/target_entitlement_v1.py",
        "te_r5p": repo_root/"scripts/modeling/te_r5p_entitlement_adapter_v1.py",
        "wr_r15": repo_root/"scripts/modeling/wr_r15_entitlement_adapter_v1.py",
        "rb_r26": repo_root/"scripts/modeling/rb_r26_receptions_production_adapter_v1.py",
        "rb_r22": repo_root/"scripts/modeling/rb_receiving_tail_production_adapter_v1.py",
    }
    text={k:p.read_text(encoding="utf-8") for k,p in files.items()}
    rows=[]
    for field in ALL_FIELDS:
        rows.append({
            "field":field,
            "generic_bayesian_mentions":field in text["bayesian_v2"],
            "generic_rules_mentions":field in text["simulation_rules"],
            "generic_entitlement_mentions":field in text["target_entitlement_v1"],
            "te_r5p_mentions":field in text["te_r5p"],
            "wr_r15_mentions":field in text["wr_r15"],
            "rb_r26_mentions":field in text["rb_r26"],
            "rb_r22_mentions":field in text["rb_r22"],
        })
    return pd.DataFrame(rows)

def run(*,rows_path:Path,out_dir:Path,repo_root:Path)->dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    x=_read(rows_path,"volume-share decomposition rows")
    required={
        "season","week","event_id","team","player","player_clean_key","position_family",
        "opportunity_type","actual_opportunities","final_player_probability",
        "actual_player_share","sportsbook_inputs_used_upstream",
    }
    miss=required-set(x.columns)
    if miss:
        raise RuntimeError(f"parent rows missing {sorted(miss)}")
    if x["sportsbook_inputs_used_upstream"].astype(bool).any():
        raise RuntimeError("sportsbook leakage in parent rows")
    x=x.loc[
        x["position_family"].astype(str).isin({"RB","FB"})
        & x["opportunity_type"].astype(str).eq("targets")
        & pd.to_numeric(x["week"],errors="coerce").isin(WEEKS)
    ].copy()
    if x.empty:
        raise RuntimeError("zero RB/FB target rows")

    states,prev=identity_atlas(2013,SEASON)
    pieces=[]
    for week,g in x.groupby("week",sort=True):
        q=g.copy()
        q["season"]=SEASON
        q["week"]=int(week)
        attached=attach_identity(q,SEASON,int(week),states,prev)
        pieces.append(attached)
    a=pd.concat(pieces,ignore_index=True,sort=False)
    for c in ALL_FIELDS:
        if c not in a.columns:
            a[c]=np.nan
        a[c]=pd.to_numeric(a[c],errors="coerce")

    # Within-current-room shares remove team volume and total RB target-pool size.
    a["model_rb_room_share"]=np.nan
    a["actual_rb_room_share"]=np.nan
    for _,idx in a.groupby(["week","event_id","team"],dropna=False).groups.items():
        g=a.loc[idx]
        pred=pd.to_numeric(g["final_player_probability"],errors="coerce").fillna(0).clip(lower=0)
        ps=float(pred.sum())
        if ps>0:
            a.loc[idx,"model_rb_room_share"]=pred/ps
        actual=pd.to_numeric(g["actual_opportunities"],errors="coerce").fillna(0).clip(lower=0)
        asum=float(actual.sum())
        if asum>0:
            a.loc[idx,"actual_rb_room_share"]=actual/asum
        for field in ROOM_FIELDS:
            a.loc[idx,f"{field}_room_normalized"]=_normalize_room(g,field)

    valid=a["actual_rb_room_share"].notna() & a["model_rb_room_share"].notna()
    scored=a.loc[valid].copy()
    if scored.empty:
        raise RuntimeError("zero scoreable RB room-share rows")
    scored["model_room_error"]=scored["model_rb_room_share"]-scored["actual_rb_room_share"]
    scored["model_room_abs_error"]=scored["model_room_error"].abs()

    feature_rows=[]
    proxy_rows=[]
    for field in ALL_FIELDS:
        cov=float(a[field].notna().mean())
        feature_rows.append({
            "field":field,
            "rows":int(len(a)),
            "available_rows":int(a[field].notna().sum()),
            "coverage_rate":cov,
            "spearman_vs_actual_rb_room_share":_spearman(scored[field],scored["actual_rb_room_share"]),
            "spearman_vs_model_room_residual":_spearman(scored[field],scored["model_room_error"]),
        })
    model_mae=float(scored["model_room_abs_error"].mean())
    model_bias=float(scored["model_room_error"].mean())
    for field in ROOM_FIELDS:
        col=f"{field}_room_normalized"
        q=scored.loc[scored[col].notna()].copy()
        if q.empty:
            proxy_rows.append({"field":field,"rows":0})
            continue
        err=q[col]-q["actual_rb_room_share"]
        current=q["model_rb_room_share"]-q["actual_rb_room_share"]
        proxy_rows.append({
            "field":field,
            "rows":int(len(q)),
            "proxy_mae":float(err.abs().mean()),
            "proxy_bias":float(err.mean()),
            "current_model_mae_same_rows":float(current.abs().mean()),
            "mae_improvement_current_minus_proxy":float(current.abs().mean()-err.abs().mean()),
            "proxy_spearman_vs_actual":_spearman(q[col],q["actual_rb_room_share"]),
        })

    # Leader diagnostics by team-room with actual target mass > 0.
    leader=[]
    for (week,event,team),g in scored.groupby(["week","event_id","team"],dropna=False):
        if g.empty:
            continue
        actual_max=float(g["actual_rb_room_share"].max())
        actual_leaders=set(g.loc[np.isclose(g["actual_rb_room_share"],actual_max,atol=TOL,rtol=0),"player_clean_key"].astype(str))
        model_max=float(g["model_rb_room_share"].max())
        model_leaders=set(g.loc[np.isclose(g["model_rb_room_share"],model_max,atol=TOL,rtol=0),"player_clean_key"].astype(str))
        rec={
            "week":int(week),"event_id":str(event),"team":str(team),
            "actual_leader_count":len(actual_leaders),
            "model_leader_match":bool(actual_leaders & model_leaders),
        }
        for field in ROOM_FIELDS:
            col=f"{field}_room_normalized"
            qq=g.loc[g[col].notna()]
            if qq.empty:
                rec[f"{field}_leader_match"]=np.nan
            else:
                m=float(qq[col].max())
                leaders=set(qq.loc[np.isclose(qq[col],m,atol=TOL,rtol=0),"player_clean_key"].astype(str))
                rec[f"{field}_leader_match"]=bool(actual_leaders & leaders)
        leader.append(rec)
    leader_df=pd.DataFrame(leader)

    consumption=_source_consumption_audit(repo_root)
    # Generic target path must not silently consume the richer identity fields.
    generic_cols=[
        "generic_bayesian_mentions","generic_rules_mentions","generic_entitlement_mentions",
        "te_r5p_mentions","wr_r15_mentions",
    ]
    generic_mentions=int(consumption[generic_cols].astype(bool).sum().sum())

    # Static canonical Bayesian target-share contract.
    bayes_path=repo_root/"scripts/modeling/bayesian_v2.py"
    bayes_text=bayes_path.read_text(encoding="utf-8")
    required_literals=[
        '"tgt_share": 3.0',
        '"tgt_share": 6.0',
        '"RB": 0.08',
        'empirical_bayes_position+player_prior+current',
    ]
    missing_literals=[z for z in required_literals if z not in bayes_text]
    if missing_literals:
        raise RuntimeError(f"Bayesian target-share contract drift: {missing_literals}")

    # Static specialist boundaries.
    r26_text=(repo_root/"scripts/modeling/rb_r26_receptions_production_adapter_v1.py").read_text(encoding="utf-8")
    r22_text=(repo_root/"scripts/modeling/rb_receiving_tail_production_adapter_v1.py").read_text(encoding="utf-8")
    if "qualified only for 2026 Week 1" not in r26_text:
        raise RuntimeError("R26 Week-1-only boundary not found")
    if "receptions_exact" not in r22_text or 'market == "receptions"' not in r22_text:
        raise RuntimeError("R22 receptions-invariance boundary not found")

    a.to_csv(out_dir/"rb_receiving_share_state_rows.csv",index=False)
    pd.DataFrame(feature_rows).to_csv(out_dir/"rb_receiving_share_state_feature_audit.csv",index=False)
    pd.DataFrame(proxy_rows).to_csv(out_dir/"rb_receiving_share_state_proxy_scoreboard.csv",index=False)
    leader_df.to_csv(out_dir/"rb_receiving_share_state_leader_audit.csv",index=False)
    consumption.to_csv(out_dir/"rb_receiving_share_state_consumption_audit.csv",index=False)

    payload={
        "version":"RB_RECEIVING_SHARE_STATE_COVERAGE_V1",
        "season":SEASON,
        "weeks":list(WEEKS),
        "rows":int(len(a)),
        "scoreable_rows":int(len(scored)),
        "team_rooms":int(a[["week","event_id","team"]].drop_duplicates().shape[0]),
        "scoreable_team_rooms":int(len(leader_df)),
        "current_model_room_share_mae":model_mae,
        "current_model_room_share_bias":model_bias,
        "feature_audit":feature_rows,
        "proxy_scoreboard":proxy_rows,
        "current_model_leader_match_rate":float(leader_df["model_leader_match"].mean()) if len(leader_df) else np.nan,
        "state_leader_match_rates":{
            field:float(pd.to_numeric(leader_df[f"{field}_leader_match"],errors="coerce").mean())
            for field in ROOM_FIELDS if f"{field}_leader_match" in leader_df.columns
        },
        "generic_identity_state_mentions":generic_mentions,
        "bayesian_target_share_group_strength":3.0,
        "bayesian_target_share_prior_player_cap":6.0,
        "bayesian_rb_default_target_share":0.08,
        "bayesian_method":"empirical_bayes_position+player_prior+current",
        "r26_week1_identity_state_specialist_exists":True,
        "r26_general_weeks2plus_target_allocator":False,
        "r22_changes_receptions_mean":False,
        "parameters_fit":0,
        "automatic_promotion":False,
        "sportsbook_inputs_used":False,
        "paid_odds_api_used":False,
        "disposition":"AUDIT_COMPLETE_RAW_RESULT_REQUIRES_INTERPRETATION",
    }
    (out_dir/"rb_receiving_share_state_summary.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,sort_keys=True,default=str))
    return payload

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--rows",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--repo-root",type=Path,default=Path("."))
    a=p.parse_args()
    run(rows_path=a.rows,out_dir=a.out_dir,repo_root=a.repo_root)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
