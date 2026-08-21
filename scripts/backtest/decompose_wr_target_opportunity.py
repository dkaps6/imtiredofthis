#!/usr/bin/env python3
"""Migration 35: decompose WR target-opportunity error.
Diagnostic only; production football logic is unchanged.
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

WR_POS={"WR","LWR","RWR","SWR"}
def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def opt(p): return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
def f(v,d=np.nan):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except Exception:return d
def probs(shares):
    c=np.clip(np.nan_to_num(np.asarray(shares,float),nan=0.,posinf=0.,neginf=0.),0,.95); s=float(c.sum()); u=c.copy()
    if s>.95:u*=.95/s
    r=max(0.,1.-float(u.sum())); p=np.append(u,r); p/=p.sum(); return p[:-1],s
def prepared(b):
    m=cp.build_market_frame(b); m=apply_bayesian_to_metrics(m,build_bayesian_baseline(b.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(b.teams,b.players)): m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m.player_clean_key.fillna("").astype(str); k=["event_id","team","player_clean_key"]
    return m.sort_values(k).drop_duplicates(k,keep="last").copy()
def score(a,p):
    z=pd.DataFrame({"a":pd.to_numeric(a,errors="coerce"),"p":pd.to_numeric(p,errors="coerce")}).dropna()
    if z.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"correlation":np.nan}
    e=z.p-z.a; return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(z.p.corr(z.a)) if len(z)>1 and z.a.nunique()>1 and z.p.nunique()>1 else np.nan}
def main():
    q=argparse.ArgumentParser();q.add_argument("--season",type=int,default=2025);q.add_argument("--prior-season",type=int,default=2024);q.add_argument("--weeks",default="1-18");q.add_argument("--iterations",type=int,default=2000);q.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv"));q.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv"));q.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv"));q.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe"));q.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv"));q.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv"));q.add_argument("--out-dir",type=Path,default=Path("data/backtests/wr_target_opportunity"));a=q.parse_args()
    logs=read(a.player_logs); team=read(a.team_weekly); sched=read(a.schedule); inj=opt(a.injuries); weather=opt(a.weather); out=[]
    for w in _parse_weeks(a.weeks):
        u=read(a.universe_dir/f"{a.season}_week_{w:02d}.csv"); b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,w),weather=_exact_week(weather,a.season,w)); m=prepared(b); mc=cp.build_mc_predictions(b,iterations=a.iterations,seed=42+w); actual=cp.build_actual_rows(logs,a.season,w)
        at=actual[actual.market.eq("receptions")][["team","player_clean_key","actual_opportunities"]].rename(columns={"actual_opportunities":"actual_targets"}); mcov=mc[mc.market.eq("receptions")][["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"mc_receptions"})
        rows=[]
        for (game,t),g in m.groupby(["event_id","team"],dropna=False):
            sh=np.array([f(r.get("rules_tgt_share",r.get("bayes_tgt_share",r.get("target_share",0.))),0.) for _,r in g.iterrows()]); pr,rawsum=probs(sh); plays=np.mean([f(v,64.) for v in g.get("rules_plays_est",pd.Series([64.]*len(g)))]); rate=np.mean([f(v,.57) for v in g.get("rules_pass_rate",pd.Series([.57]*len(g)))]); team_pred=float(plays*rate)
            for j,(_,r) in enumerate(g.iterrows()):
                pos=str(r.get("position","")).upper().strip()
                if pos not in WR_POS: continue
                rows.append({"week":w,"event_id":game,"team":t,"player_clean_key":r.player_clean_key,"player":r.get("player",""),"position":pos,"rules_tgt_share":f(r.get("rules_tgt_share")),"bayes_tgt_share":f(r.get("bayes_tgt_share")),"base_tgt_share":f(r.get("target_share",r.get("tgt_share"))),"allocator_target_prob":float(pr[j]),"raw_team_target_share_sum":rawsum,"pred_team_targets":team_pred,"pred_targets":team_pred*float(pr[j]),"matchup_available":int(f(r.get("matchup_available"),0)==1),"primary_cb":str(r.get("primary_cb","") or ""),"injury_available":int(f(r.get("injury_report_available"),0)==1),"coverage_man_rate":f(r.get("coverage_man_rate")),"coverage_zone_rate":f(r.get("coverage_zone_rate"))})
        x=pd.DataFrame(rows).merge(at,on=["team","player_clean_key"],how="left").merge(mcov,on=["event_id","team","player_clean_key"],how="left"); team_actual=at.groupby("team",as_index=False).actual_targets.sum().rename(columns={"actual_targets":"actual_team_targets"}); x=x.merge(team_actual,on="team",how="left"); x=x[x.mc_receptions.notna()&x.actual_targets.notna()].copy(); x["actual_target_share"]=np.where(x.actual_team_targets>0,x.actual_targets/x.actual_team_targets,np.nan); x["target_error"]=x.pred_targets-x.actual_targets; x["share_error"]=x.allocator_target_prob-x.actual_target_share; x["share_only_cf_targets"]=x.actual_team_targets*x.allocator_target_prob; x["volume_only_cf_targets"]=x.pred_team_targets*x.actual_target_share
        x["wr_rank"]=x.groupby(["event_id","team"])["allocator_target_prob"].rank(method="first",ascending=False).astype(int); x["wr_role"]=np.select([x.wr_rank.eq(1),x.wr_rank.eq(2),x.wr_rank.eq(3)],["WR1","WR2","WR3"],default="WR4+"); x["week_phase"]=pd.cut(x.week,[0,4,9,13,18],labels=["W1-4","W5-9","W10-13","W14-18"]); x["pred_target_tier"]=pd.cut(x.pred_targets,[-np.inf,3,6,9,np.inf],labels=["<3","3-6","6-9","9+"]); x["share_source"]=np.select([(x.rules_tgt_share-x.bayes_tgt_share).abs()<.005,(x.rules_tgt_share-x.base_tgt_share).abs()<.005],["bayes_like","base_like"],default="rules_adjusted"); out.append(x); print(f"[wr35] W{w:02d} WR matched={len(x)} target_bias={x.target_error.mean():.3f}")
    z=pd.concat(out,ignore_index=True); stages=[]
    for name,col in [("canonical_pred_targets","pred_targets"),("share_only_actual_team_volume","share_only_cf_targets"),("volume_only_actual_target_share","volume_only_cf_targets")]: stages.append({"stage":name,**score(z.actual_targets,z[col])})
    buckets=[]
    for c in ["wr_role","week_phase","pred_target_tier","share_source","matchup_available","injury_available"]:
        for b,g in z.groupby(c,dropna=False): buckets.append({"bucket_type":c,"bucket":str(b),"mean_target_error":float(g.target_error.mean()),"mean_share_error":float(g.share_error.mean()),"mean_pred_targets":float(g.pred_targets.mean()),"mean_actual_targets":float(g.actual_targets.mean()),**score(g.actual_targets,g.pred_targets)})
    role=z.groupby("wr_role",as_index=False).agg(n=("actual_targets","size"),pred_targets=("pred_targets","mean"),actual_targets=("actual_targets","mean"),target_bias=("target_error","mean"),pred_share=("allocator_target_prob","mean"),actual_share=("actual_target_share","mean"),share_bias=("share_error","mean"),pred_team_targets=("pred_team_targets","mean"),actual_team_targets=("actual_team_targets","mean"))
    a.out_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.out_dir/"wr_target_player_decomposition.csv",index=False); pd.DataFrame(stages).to_csv(a.out_dir/"wr_target_stage_summary.csv",index=False); pd.DataFrame(buckets).to_csv(a.out_dir/"wr_target_bucket_summary.csv",index=False); role.to_csv(a.out_dir/"wr_target_role_summary.csv",index=False); print("\n[wr35] stage summary\n",pd.DataFrame(stages).to_string(index=False)); print("\n[wr35] role summary\n",role.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
