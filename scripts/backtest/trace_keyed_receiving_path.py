#!/usr/bin/env python3
"""Migration 33: keyed receiving path + coverage reconciliation.
Diagnostic only; no production football changes.
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
from scripts import simulation_v2

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def opt(p): return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
def finite(v,d=0.):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except: return d
def probs(sh):
    c=np.clip(np.nan_to_num(np.asarray(sh,float),nan=0.,posinf=0.,neginf=0.),0,.95); s=float(c.sum()); u=c.copy()
    if s>.95: u*=.95/s
    r=max(0.,1.-float(u.sum())); p=np.append(u,r); p/=p.sum(); return p[:-1],s,float(p[-1])
def prepared(b):
    m=cp.build_market_frame(b); m=apply_bayesian_to_metrics(m,build_bayesian_baseline(b.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(b.teams,b.players)): m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m["player_clean_key"].fillna("").astype(str); k=["event_id","team","player_clean_key"]
    return m.sort_values(k).drop_duplicates(k,keep="last").copy()
def metrics(a,p):
    z=pd.DataFrame({"a":pd.to_numeric(a,errors="coerce"),"p":pd.to_numeric(p,errors="coerce")}).dropna()
    if z.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"correlation":np.nan}
    e=z.p-z.a; return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def main():
    q=argparse.ArgumentParser(); q.add_argument("--season",type=int,default=2025);q.add_argument("--prior-season",type=int,default=2024);q.add_argument("--weeks",default="1-18");q.add_argument("--iterations",type=int,default=2000);q.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv"));q.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv"));q.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv"));q.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe"));q.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv"));q.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv"));q.add_argument("--out-dir",type=Path,default=Path("data/backtests/keyed_receiving_trace"));a=q.parse_args()
    logs=read(a.player_logs); team=read(a.team_weekly); sched=read(a.schedule); inj=opt(a.injuries); weather=opt(a.weather); rows=[]
    for w in _parse_weeks(a.weeks):
        u=read(a.universe_dir/f"{a.season}_week_{w:02d}.csv"); b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,w),weather=_exact_week(weather,a.season,w)); m=prepared(b)
        sim=simulation_v2.simulate(m,iterations=a.iterations,seed=42+w); mc=cp.build_mc_predictions(b,iterations=a.iterations,seed=42+w); actual=cp.build_actual_rows(logs,a.season,w)
        ar=actual[actual.market.eq("receptions")][["team","player_clean_key","actual"]].rename(columns={"actual":"actual_receptions"}); ay=actual[actual.market.eq("rec_yards")][["team","player_clean_key","actual"]].rename(columns={"actual":"actual_rec_yards"}); mr=mc[mc.market.eq("receptions")][["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"component_mc_receptions"}); my=mc[mc.market.eq("rec_yards")][["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"component_mc_rec_yards"})
        wr=[]
        for (game,t),g in m.groupby(["event_id","team"],dropna=False):
            sh=[finite(r.get("rules_tgt_share",r.get("bayes_tgt_share",r.get("target_share",0.))),0.) for _,r in g.iterrows()]; pr,raw,res=probs(sh); plays=np.mean([finite(v,64.) for v in g.get("rules_plays_est",pd.Series([64.]*len(g)))]); rate=np.mean([finite(v,.57) for v in g.get("rules_pass_rate",pd.Series([.57]*len(g)))]); pa=plays*rate
            for j,(_,r) in enumerate(g.iterrows()):
                key=r.player_clean_key; catch=finite(r.get("rules_catch_rate",r.get("bayes_receptions_per_target",r.get("receptions_per_target",.64))),.64); ypt=finite(r.get("rules_ypt",r.get("bayes_ypt",r.get("ypt",7.5))),7.5); rv=sim.values.get((str(game),key,"receptions")); yv=sim.values.get((str(game),key,"rec_yards")); wr.append({"week":w,"event_id":game,"team":t,"player_clean_key":key,"player":r.get("player",""),"position":r.get("position",""),"raw_target_share":sh[j],"raw_team_target_share_sum":raw,"final_target_probability":pr[j],"residual_probability":res,"det_team_pass_attempts":pa,"det_expected_targets":pa*pr[j],"catch_rate":catch,"det_expected_receptions":pa*pr[j]*catch,"ypt":ypt,"det_expected_rec_yards":pa*pr[j]*ypt,"keyed_sim_receptions":float(np.mean(rv)) if rv is not None else np.nan,"keyed_sim_rec_yards":float(np.mean(yv)) if yv is not None else np.nan,"in_simulation_result":rv is not None and yv is not None})
        x=pd.DataFrame(wr).merge(mr,on=["event_id","team","player_clean_key"],how="outer").merge(my,on=["event_id","team","player_clean_key"],how="outer").merge(ar,on=["team","player_clean_key"],how="outer").merge(ay,on=["team","player_clean_key"],how="outer"); x["week"]=w; x["has_component_mc"]=x.component_mc_receptions.notna()|x.component_mc_rec_yards.notna(); x["has_actual"]=x.actual_receptions.notna()|x.actual_rec_yards.notna(); x["coverage_class"]=np.select([x.has_component_mc&x.has_actual,~x.has_component_mc&x.has_actual,x.has_component_mc&~x.has_actual],["mc_and_actual","actual_missing_mc","mc_no_actual"],default="neither"); rows.append(x); print(f"[recv33] W{w:02d} rows={len(x)} missing_mc_actual={(x.coverage_class=='actual_missing_mc').sum()}")
    z=pd.concat(rows,ignore_index=True); summaries=[]
    for market,actual_col,det_col,key_col,comp_col in [("receptions","actual_receptions","det_expected_receptions","keyed_sim_receptions","component_mc_receptions"),("rec_yards","actual_rec_yards","det_expected_rec_yards","keyed_sim_rec_yards","component_mc_rec_yards")]:
        for scope,g in [("all_actual",z[z[actual_col].notna()]),("component_mc_covered",z[z[actual_col].notna()&z[comp_col].notna()])]:
            for stage,col in [("deterministic",det_col),("keyed_simulation_result",key_col),("component_mc",comp_col)]: summaries.append({"market":market,"scope":scope,"stage":stage,**metrics(g[actual_col],g[col])})
    summary=pd.DataFrame(summaries); coverage=z.groupby(["coverage_class","position"],dropna=False).size().reset_index(name="n").sort_values(["coverage_class","n"],ascending=[True,False]); a.out_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.out_dir/"keyed_receiving_player_trace.csv",index=False); summary.to_csv(a.out_dir/"keyed_receiving_stage_summary.csv",index=False); coverage.to_csv(a.out_dir/"receiving_coverage_by_position.csv",index=False); print(summary.to_string(index=False)); print(coverage.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
