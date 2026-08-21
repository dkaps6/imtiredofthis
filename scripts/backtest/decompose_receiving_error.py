#!/usr/bin/env python3
"""Migration 34: decompose receiving error into opportunity, conversion and efficiency.
Diagnostic only; production receiving logic is unchanged.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules

def _read(p: Path):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def _opt(p: Path): return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
def _f(v,d=np.nan):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except Exception:return d
def _probs(shares):
    c=np.clip(np.nan_to_num(np.asarray(shares,float),nan=0.,posinf=0.,neginf=0.),0,.95); s=float(c.sum()); u=c.copy()
    if s>.95:u*=.95/s
    r=max(0.,1.-float(u.sum())); p=np.append(u,r); p/=p.sum(); return p[:-1]
def _prepared(b):
    m=cp.build_market_frame(b); m=apply_bayesian_to_metrics(m,build_bayesian_baseline(b.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(b.teams,b.players)):m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m.player_clean_key.fillna("").astype(str); k=["event_id","team","player_clean_key"]
    return m.sort_values(k).drop_duplicates(k,keep="last").copy()
def _score(a,p):
    z=pd.DataFrame({"a":pd.to_numeric(a,errors="coerce"),"p":pd.to_numeric(p,errors="coerce")}).dropna()
    if z.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"correlation":np.nan}
    e=z.p-z.a; return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(z.p.corr(z.a)) if len(z)>1 and z.p.nunique()>1 and z.a.nunique()>1 else np.nan}
def _bucket_summary(x,col):
    out=[]
    for bucket,g in x.groupby(col,dropna=False):
        for market,actual,pred in [("targets","actual_targets","pred_targets"),("receptions","actual_receptions","mc_receptions"),("rec_yards","actual_rec_yards","mc_rec_yards")]:out.append({"bucket_type":col,"bucket":str(bucket),"market":market,**_score(g[actual],g[pred])})
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,default=2025);p.add_argument("--prior-season",type=int,default=2024);p.add_argument("--weeks",default="1-18");p.add_argument("--iterations",type=int,default=2000);p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv"));p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv"));p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv"));p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe"));p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv"));p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv"));p.add_argument("--out-dir",type=Path,default=Path("data/backtests/receiving_error_decomposition"));a=p.parse_args()
    logs=_read(a.player_logs);team=_read(a.team_weekly);sched=_read(a.schedule);inj=_opt(a.injuries);weather=_opt(a.weather);all_rows=[]
    for w in _parse_weeks(a.weeks):
        u=_read(a.universe_dir/f"{a.season}_week_{w:02d}.csv");b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,w),weather=_exact_week(weather,a.season,w));m=_prepared(b);mc=cp.build_mc_predictions(b,iterations=a.iterations,seed=42+w);actual=cp.build_actual_rows(logs,a.season,w)
        ar=actual[actual.market.eq("receptions")][["team","player_clean_key","actual","actual_opportunities"]].rename(columns={"actual":"actual_receptions","actual_opportunities":"actual_targets"});ay=actual[actual.market.eq("rec_yards")][["team","player_clean_key","actual"]].rename(columns={"actual":"actual_rec_yards"});mr=mc[mc.market.eq("receptions")][["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"mc_receptions"});my=mc[mc.market.eq("rec_yards")][["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"mc_rec_yards"})
        rows=[]
        for (game,t),g in m.groupby(["event_id","team"],dropna=False):
            sh=[_f(r.get("rules_tgt_share",r.get("bayes_tgt_share",r.get("target_share",0.))),0.) for _,r in g.iterrows()];pr=_probs(sh);plays=float(np.mean([_f(v,64.) for v in g.get("rules_plays_est",pd.Series([64.]*len(g)))]));rate=float(np.mean([_f(v,.57) for v in g.get("rules_pass_rate",pd.Series([.57]*len(g)))]));team_opp=plays*rate
            for j,(_,r) in enumerate(g.iterrows()):
                catch=_f(r.get("rules_catch_rate",r.get("bayes_receptions_per_target",r.get("receptions_per_target",.64))),.64);ypt=_f(r.get("rules_ypt",r.get("bayes_ypt",r.get("ypt",7.5))),7.5);pt=team_opp*pr[j]
                rows.append({"week":w,"event_id":game,"team":t,"player_clean_key":r.player_clean_key,"player":r.get("player",""),"position":r.get("position",""),"target_share":sh[j],"pred_targets":pt,"pred_catch_rate":catch,"pred_ypt":ypt,"det_receptions":pt*catch,"det_rec_yards":pt*ypt})
        x=pd.DataFrame(rows).merge(ar,on=["team","player_clean_key"],how="left").merge(ay,on=["team","player_clean_key"],how="left").merge(mr,on=["event_id","team","player_clean_key"],how="left").merge(my,on=["event_id","team","player_clean_key"],how="left");x=x[x.mc_receptions.notna()|x.mc_rec_yards.notna()].copy();x["oracle_targets_receptions"]=x.actual_targets*x.pred_catch_rate;x["oracle_targets_rec_yards"]=x.actual_targets*x.pred_ypt;x["actual_catch_rate"]=np.where(x.actual_targets>0,x.actual_receptions/x.actual_targets,np.nan);x["actual_ypt"]=np.where(x.actual_targets>0,x.actual_rec_yards/x.actual_targets,np.nan);x["target_error"]=x.pred_targets-x.actual_targets;x["catch_rate_error"]=x.pred_catch_rate-x.actual_catch_rate;x["ypt_error"]=x.pred_ypt-x.actual_ypt;all_rows.append(x);print(f"[recv34] W{w:02d} matched={len(x)}")
    z=pd.concat(all_rows,ignore_index=True); stages=[]
    for market,actual,pairs in [("targets","actual_targets",[("pred_targets","pred_targets")]),("receptions","actual_receptions",[("deterministic","det_receptions"),("oracle_actual_targets","oracle_targets_receptions"),("canonical_mc","mc_receptions")]),("rec_yards","actual_rec_yards",[("deterministic","det_rec_yards"),("oracle_actual_targets","oracle_targets_rec_yards"),("canonical_mc","mc_rec_yards")])]:
        for stage,col in pairs:stages.append({"market":market,"stage":stage,**_score(z[actual],z[col])})
    z["target_share_tier"]=pd.cut(z.target_share,[-np.inf,.05,.10,.20,np.inf],labels=["<5%","5-10%","10-20%","20%+"]);z["pred_target_tier"]=pd.cut(z.pred_targets,[-np.inf,2,5,8,np.inf],labels=["<2","2-5","5-8","8+"]);z["week_phase"]=pd.cut(z.week,[0,4,9,13,18],labels=["W1-4","W5-9","W10-13","W14-18"]); buckets=[]
    for c in ["position","target_share_tier","pred_target_tier","week_phase"]:buckets.extend(_bucket_summary(z,c))
    drivers=[]
    for pos,g in z.groupby("position",dropna=False):drivers.append({"position":pos,"n":len(g),"target_mae":float(g.target_error.abs().mean()),"catch_rate_mae":float(g.catch_rate_error.abs().mean()),"ypt_mae":float(g.ypt_error.abs().mean()),"mean_target_error":float(g.target_error.mean()),"mean_catch_rate_error":float(g.catch_rate_error.mean()),"mean_ypt_error":float(g.ypt_error.mean())})
    a.out_dir.mkdir(parents=True,exist_ok=True);z.to_csv(a.out_dir/"receiving_error_player_decomposition.csv",index=False);pd.DataFrame(stages).to_csv(a.out_dir/"receiving_error_stage_summary.csv",index=False);pd.DataFrame(buckets).to_csv(a.out_dir/"receiving_error_bucket_summary.csv",index=False);pd.DataFrame(drivers).to_csv(a.out_dir/"receiving_error_driver_summary.csv",index=False);print(pd.DataFrame(stages).to_string(index=False));print(pd.DataFrame(drivers).to_string(index=False));return 0
if __name__=="__main__":raise SystemExit(main())
