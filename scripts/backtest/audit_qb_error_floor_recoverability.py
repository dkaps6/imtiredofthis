#!/usr/bin/env python3
"""M86 forensic audit of frozen M82 full-stack QB errors.

Postgame PBP is descriptive only. No predictive fitting occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x


def canon_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().replace({"JAC":"JAX", "LA":"LAR", "OAK":"LV", "SD":"LAC", "STL":"LAR"})


def load_pbp(seasons=(2024,2025)) -> pd.DataFrame:
    import nflreadpy as nfl
    raw=nfl.load_pbp(seasons=list(seasons))
    x=raw.to_pandas() if hasattr(raw,"to_pandas") else pd.DataFrame(raw)
    x=lower(x)
    if "season_type" in x.columns:
        x=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x=x.loc[x["season"].isin(seasons)&x["week"].between(1,18)].copy()
    x["team"]=canon_team(x["posteam"])
    return x


def n(x: pd.DataFrame,col:str,default=0.0)->pd.Series:
    if col not in x.columns: return pd.Series(default,index=x.index,dtype=float)
    return pd.to_numeric(x[col],errors="coerce").fillna(default)


def aggregate_team_events(pbp: pd.DataFrame)->pd.DataFrame:
    x=pbp.copy()
    pass_attempt=n(x,"pass_attempt").eq(1)
    complete=n(x,"complete_pass").eq(1)
    pass_yards=n(x,"passing_yards")
    yac=n(x,"yards_after_catch",np.nan)
    x["completed_pass_gain"]=np.where(pass_attempt&complete,pass_yards,0.0)
    x["pass_40_plus"]=(pass_attempt&complete&pass_yards.ge(40)).astype(int)
    x["pass_60_plus"]=(pass_attempt&complete&pass_yards.ge(60)).astype(int)
    x["yac_30_plus"]=(pass_attempt&complete&yac.ge(30)).astype(int)
    x["int_thrown"]=n(x,"interception").eq(1).astype(int)
    x["sack_event"]=n(x,"sack").eq(1).astype(int)
    x["scramble_event"]=n(x,"qb_scramble").eq(1).astype(int)
    q=n(x,"qtr"); x["ot_play"]=q.ge(5).astype(int)
    x["fourth_down_attempt"]=((n(x,"down").eq(4)) & (pass_attempt | n(x,"rush_attempt").eq(1))).astype(int)
    x["fourth_down_conversion"]=(x["fourth_down_attempt"].eq(1)&n(x,"first_down").eq(1)).astype(int)
    fumble_lost=n(x,"fumble_lost").eq(1).astype(int)
    x["off_turnover"]=np.maximum(x["int_thrown"],fumble_lost)
    agg=x.groupby(["season","week","team"],as_index=False).agg(
        longest_completed_pass=("completed_pass_gain","max"),
        pass_40_plus=("pass_40_plus","sum"),
        pass_60_plus=("pass_60_plus","sum"),
        yac_30_plus=("yac_30_plus","sum"),
        max_yac=("yards_after_catch","max") if "yards_after_catch" in x.columns else ("completed_pass_gain","size"),
        interceptions=("int_thrown","sum"), sacks=("sack_event","sum"),
        qb_scrambles=("scramble_event","sum"), overtime=("ot_play","max"),
        fourth_down_attempts=("fourth_down_attempt","sum"),
        fourth_down_conversions=("fourth_down_conversion","sum"),
        offensive_turnovers=("off_turnover","sum"),
    )
    if "yards_after_catch" not in x.columns: agg["max_yac"]=np.nan
    return agg


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--m82-trace",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    q=lower(pd.read_csv(args.m82_trace,low_memory=False))
    if len(q)!=884: raise RuntimeError(f"M82 trace row drift: {len(q)}")
    required={"season","week","team","actual_pass_yards","actual_attempts","pred_attempts","actual_ypa","implied_pred_ypa","ensemble_proj","mc_proj","ml_proj","state_proj"}
    miss=required-set(q.columns)
    if miss: raise RuntimeError(f"M82 trace missing columns: {sorted(miss)}")
    q["team"]=canon_team(q["team"])
    for c in ["actual_pass_yards","actual_attempts","pred_attempts","actual_ypa","implied_pred_ypa","ensemble_proj","mc_proj","ml_proj","state_proj"]:
        q[c]=pd.to_numeric(q[c],errors="coerce")
    q["ensemble_error"]=q["ensemble_proj"]-q["actual_pass_yards"]
    q["ensemble_abs_error"]=q["ensemble_error"].abs()
    q["tail100"]=q["ensemble_abs_error"].ge(100)
    q["tail_direction"]=np.where(q["ensemble_error"].le(-100),"UNDERPROJECTED",np.where(q["ensemble_error"].ge(100),"OVERPROJECTED","NONTAIL"))
    q["attempt_resid"]=q["actual_attempts"]-q["pred_attempts"]
    q["ypa_resid"]=q["actual_ypa"]-q["implied_pred_ypa"]
    q["attempt_contribution_abs"]=(q["attempt_resid"]*q["implied_pred_ypa"]).abs()
    q["ypa_contribution_abs"]=(q["ypa_resid"]*q["pred_attempts"]).abs()
    q["component_class"]="MIXED"
    q.loc[q["attempt_contribution_abs"].ge(1.25*q["ypa_contribution_abs"]),"component_class"]="VOLUME_DOMINANT"
    q.loc[q["ypa_contribution_abs"].ge(1.25*q["attempt_contribution_abs"]),"component_class"]="EFFICIENCY_DOMINANT"

    events=aggregate_team_events(load_pbp())
    q=q.merge(events,on=["season","week","team"],how="left",validate="many_to_one")
    event_cols=["longest_completed_pass","pass_40_plus","pass_60_plus","yac_30_plus","max_yac","interceptions","sacks","qb_scrambles","overtime","fourth_down_attempts","fourth_down_conversions","offensive_turnovers"]
    for c in event_cols: q[c]=pd.to_numeric(q[c],errors="coerce")
    q["high_event_chaos"]=(
        q["pass_60_plus"].fillna(0).ge(1)|q["pass_40_plus"].fillna(0).ge(2)|q["yac_30_plus"].fillna(0).ge(1)|
        q["overtime"].fillna(0).ge(1)|q["sacks"].fillna(0).ge(4)|q["interceptions"].fillna(0).ge(2)|
        q["qb_scrambles"].fillna(0).ge(5)|q["fourth_down_attempts"].fillna(0).ge(4)
    )
    q["chaos_class"]=np.where(q["high_event_chaos"],"HIGH_EVENT_CHAOS","LOW_EVENT_CHAOS")

    model_cols={"MC":"mc_proj","ML":"ml_proj","STATE":"state_proj","ENSEMBLE":"ensemble_proj"}
    errs=pd.DataFrame({k:(q[v]-q["actual_pass_yards"]).abs() for k,v in model_cols.items()})
    q["hindsight_best_model"]=errs.idxmin(axis=1)
    q.to_csv(args.out_dir/"m86_qb_forensic_trace.csv",index=False)

    tails=q.loc[q["tail100"]].copy()
    if len(tails)!=123: raise RuntimeError(f"M82 ensemble tail-count drift: {len(tails)}")
    comp=tails.groupby(["component_class","tail_direction"],as_index=False).agg(n=("tail100","size"),mae=("ensemble_abs_error","mean"))
    comp.to_csv(args.out_dir/"m86_tail_component_classification.csv",index=False)
    chaos=tails.groupby(["chaos_class","component_class"],as_index=False).agg(n=("tail100","size"),mae=("ensemble_abs_error","mean"),mean_attempt_resid=("attempt_resid","mean"),mean_ypa_resid=("ypa_resid","mean"))
    chaos.to_csv(args.out_dir/"m86_tail_chaos_summary.csv",index=False)
    prev=pd.DataFrame([{ 
        "scope":scope,"n":len(part),"mae":float(part["ensemble_abs_error"].mean()),
        "high_event_chaos_rate":float(part["high_event_chaos"].mean()),
        "mean_longest_completion":float(part["longest_completed_pass"].mean()),
        "mean_sacks":float(part["sacks"].mean()),"mean_interceptions":float(part["interceptions"].mean()),
        "mean_scrambles":float(part["qb_scrambles"].mean()),"mean_fourth_down_attempts":float(part["fourth_down_attempts"].mean()),
    } for scope,part in [("TAIL100",tails),("NONTAIL",q.loc[~q["tail100"]]),("ALL",q)]])
    prev.to_csv(args.out_dir/"m86_event_prevalence.csv",index=False)
    low=tails.loc[~tails["high_event_chaos"]].copy()
    low.sort_values("ensemble_abs_error",ascending=False).to_csv(args.out_dir/"m86_low_event_chaos_tail_research_subset.csv",index=False)

    nonchaos=q.loc[~q["high_event_chaos"]]
    total_abs=float(q["ensemble_abs_error"].sum()); chaos_abs=float(q.loc[q["high_event_chaos"],"ensemble_abs_error"].sum())
    summary={
        "migration":"M86","rows":len(q),"baseline_mae":float(q["ensemble_abs_error"].mean()),"tail100":len(tails),
        "high_event_chaos_tail_count":int(tails["high_event_chaos"].sum()),"low_event_chaos_tail_count":int((~tails["high_event_chaos"]).sum()),
        "high_event_chaos_tail_share":float(tails["high_event_chaos"].mean()),
        "non_high_event_chaos_mae":float(nonchaos["ensemble_abs_error"].mean()),
        "high_event_chaos_share_of_total_absolute_error":chaos_abs/total_abs if total_abs else None,
        "model_library_oracle_floor":41.103131,
        "postgame_features_used_for_prediction":False,"sportsbook_features_used":False,"production_actionable":False,
    }
    (args.out_dir/"m86_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("[m86_summary]"); print(json.dumps(summary,indent=2))
    print("[m86_tail_components]"); print(comp.to_string(index=False))
    print("[m86_tail_chaos]"); print(chaos.to_string(index=False))
    print("[m86_event_prevalence]"); print(prev.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
