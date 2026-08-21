#!/usr/bin/env python3
"""Migration 36: leakage-safe WR target-share hierarchy calibration.

Diagnostic/calibration only. Candidate transforms sharpen the WR hierarchy while
preserving each team's total WR target-share mass. Production target logic is
not changed by this script.
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

WR_POS={"WR","LWR","RWR","SWR"}
CANDIDATES={
    "current":("power",1.00),
    "power_110":("power",1.10),
    "power_120":("power",1.20),
    "power_130":("power",1.30),
    "power_140":("power",1.40),
    "rank_mild":("rank",(1.15,1.07,1.00,0.94)),
    "rank_medium":("rank",(1.25,1.10,0.97,0.88)),
    "rank_strong":("rank",(1.35,1.12,0.94,0.82)),
}

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def opt(p): return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
def f(v,d=np.nan):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except Exception:return d

def prepared(b):
    m=cp.build_market_frame(b); m=apply_bayesian_to_metrics(m,build_bayesian_baseline(b.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(b.teams,b.players)): m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m.player_clean_key.fillna("").astype(str); k=["event_id","team","player_clean_key"]
    return m.sort_values(k).drop_duplicates(k,keep="last").copy()

def sharpen(group, spec):
    g=group.copy(); pos=g.get("position",pd.Series("",index=g.index)).fillna("").astype(str).str.upper(); mask=pos.isin(WR_POS)
    raw=pd.to_numeric(g.get("rules_tgt_share"),errors="coerce").fillna(0.).clip(0.,.95)
    wr=raw.loc[mask].copy(); total=float(wr.sum())
    if len(wr)<=1 or total<=0 or spec[1]==1.00: return g
    kind,param=spec
    if kind=="power":
        x=np.power(np.maximum(wr.to_numpy(float),1e-9),float(param))
    else:
        order=np.argsort(-wr.to_numpy(float),kind="stable"); mult=np.ones(len(wr)); vals=list(param)
        for rank,idx in enumerate(order): mult[idx]=vals[min(rank,len(vals)-1)]
        x=wr.to_numpy(float)*mult
    if x.sum()>0: x=x*(total/x.sum())
    g.loc[wr.index,"rules_tgt_share"]=x
    return g

def allocator_probs(shares):
    c=np.clip(np.nan_to_num(np.asarray(shares,float),nan=0.,posinf=0.,neginf=0.),0,.95); s=float(c.sum()); u=c.copy()
    if s>.95:u*=.95/s
    r=max(0.,1.-float(u.sum())); p=np.append(u,r); p/=p.sum(); return p[:-1]

def score(a,p):
    z=pd.DataFrame({"a":pd.to_numeric(a,errors="coerce"),"p":pd.to_numeric(p,errors="coerce")}).dropna()
    if z.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"correlation":np.nan}
    e=z.p-z.a; return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(z.p.corr(z.a)) if len(z)>1 and z.a.nunique()>1 and z.p.nunique()>1 else np.nan}

def main():
    q=argparse.ArgumentParser(); q.add_argument("--season",type=int,default=2025); q.add_argument("--prior-season",type=int,default=2024); q.add_argument("--weeks",default="1-18"); q.add_argument("--iterations",type=int,default=2000); q.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); q.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); q.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); q.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); q.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); q.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); q.add_argument("--out-dir",type=Path,default=Path("data/backtests/wr_target_hierarchy_calibration")); a=q.parse_args()
    logs=read(a.player_logs); team=read(a.team_weekly); sched=read(a.schedule); inj=opt(a.injuries); weather=opt(a.weather); all_rows=[]; share_rows=[]
    for w in _parse_weeks(a.weeks):
        u=read(a.universe_dir/f"{a.season}_week_{w:02d}.csv"); b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,w),weather=_exact_week(weather,a.season,w)); base=prepared(b); actual=cp.build_actual_rows(logs,a.season,w)
        ar=actual[actual.market.eq("receptions")][["team","player_clean_key","actual","actual_opportunities"]].rename(columns={"actual":"actual_receptions","actual_opportunities":"actual_targets"}); ay=actual[actual.market.eq("rec_yards")][["team","player_clean_key","actual"]].rename(columns={"actual":"actual_rec_yards"})
        for name,spec in CANDIDATES.items():
            m=pd.concat([sharpen(g,spec) for _,g in base.groupby(["event_id","team"],dropna=False)],ignore_index=True)
            sim=simulation_v2.simulate(m,iterations=a.iterations,seed=4200+w)
            rows=[]
            for (game,t),g in m.groupby(["event_id","team"],dropna=False):
                shares=pd.to_numeric(g.rules_tgt_share,errors="coerce").fillna(0.).to_numpy(float); pr=allocator_probs(shares); plays=float(np.mean([f(v,64.) for v in g.get("rules_plays_est",pd.Series([64.]*len(g)))])); rate=float(np.mean([f(v,.57) for v in g.get("rules_pass_rate",pd.Series([.57]*len(g)))])); team_targets=plays*rate
                for j,(_,r) in enumerate(g.iterrows()):
                    pos=str(r.get("position","")).upper().strip()
                    if pos not in WR_POS: continue
                    key=str(r.player_clean_key); rec=sim.values.get((str(game),key,"receptions")); yards=sim.values.get((str(game),key,"rec_yards")); rows.append({"variant":name,"week":w,"event_id":game,"team":t,"player_clean_key":key,"player":r.get("player",""),"position":pos,"target_share":float(shares[j]),"allocator_probability":float(pr[j]),"pred_targets":team_targets*float(pr[j]),"mc_receptions":float(np.mean(rec)) if rec is not None else np.nan,"mc_rec_yards":float(np.mean(yards)) if yards is not None else np.nan})
            x=pd.DataFrame(rows).merge(ar,on=["team","player_clean_key"],how="inner").merge(ay,on=["team","player_clean_key"],how="inner"); all_rows.append(x)
            wrs=m[m.get("position","").fillna("").astype(str).str.upper().isin(WR_POS)].copy(); wrs["rank"]=wrs.groupby(["event_id","team"])["rules_tgt_share"].rank(method="first",ascending=False); s=wrs.groupby("rank").rules_tgt_share.mean(); share_rows.append({"variant":name,"week":w,"wr1_share":s.get(1,np.nan),"wr2_share":s.get(2,np.nan),"wr3_share":s.get(3,np.nan),"wr4_share":s.get(4,np.nan)})
        print(f"[wr36] W{w:02d} completed {len(CANDIDATES)} candidates")
    z=pd.concat(all_rows,ignore_index=True); summaries=[]
    for v,g in z.groupby("variant"):
        for market,a_col,p_col in [("targets","actual_targets","pred_targets"),("receptions","actual_receptions","mc_receptions"),("rec_yards","actual_rec_yards","mc_rec_yards")]: summaries.append({"variant":v,"market":market,**score(g[a_col],g[p_col])})
    summary=pd.DataFrame(summaries); wide=summary.pivot(index="variant",columns="market",values=["mae","rmse","bias","correlation"]); wide.columns=[f"{metric}_{market}" for metric,market in wide.columns]; wide=wide.reset_index(); wide["rank_score"]=wide[["mae_targets","mae_receptions","mae_rec_yards"]].rank(method="min").mean(axis=1); wide=wide.sort_values(["rank_score","mae_receptions","mae_rec_yards"])
    a.out_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.out_dir/"wr_target_hierarchy_player_predictions.csv",index=False); summary.to_csv(a.out_dir/"wr_target_hierarchy_market_summary.csv",index=False); wide.to_csv(a.out_dir/"wr_target_hierarchy_candidate_ranking.csv",index=False); pd.DataFrame(share_rows).groupby("variant",as_index=False).mean(numeric_only=True).to_csv(a.out_dir/"wr_target_hierarchy_share_summary.csv",index=False); print("\n[wr36] candidate ranking\n",wide.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
