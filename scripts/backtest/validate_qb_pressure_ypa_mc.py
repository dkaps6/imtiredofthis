#!/usr/bin/env python3
"""Migration 45: canonical Monte Carlo validation of pressure-based QB YPA candidates.
Diagnostic only; no production coefficient changes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def opt(p): return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
def met(a,p):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'p':pd.to_numeric(p,errors='coerce')}).dropna()
    if z.empty:return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan}
    e=z.p-z.a
    return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}

def candidate_apply_factory(original_apply, mode: str):
    def wrapped(metrics: pd.DataFrame) -> pd.DataFrame:
        out=original_apply(metrics)
        if mode=='current' or out.empty or 'market' not in out: return out
        _, players=simulation_rules.load_model_contexts()
        team_pressure={}
        for p in players:
            off=getattr(p,'offense',None)
            v=getattr(off,'pressure_rate_allowed',np.nan) if off is not None else np.nan
            try: v=float(v)
            except: v=np.nan
            if np.isfinite(v): team_pressure[str(p.team).upper().strip()]=v
        vals=np.asarray(list(team_pressure.values()),dtype=float)
        if len(vals)<4 or not np.isfinite(vals).all() or float(np.std(vals,ddof=1))<1e-9: return out
        mu=float(np.mean(vals)); sd=float(np.std(vals,ddof=1))
        mask=out['market'].astype(str).eq('pass_yards')
        z=out.loc[mask,'team'].astype(str).str.upper().str.strip().map(lambda t:(team_pressure.get(t,np.nan)-mu)/sd if t in team_pressure else np.nan)
        if mode=='replace_bayes': base=pd.to_numeric(out.loc[mask,'bayes_ypa'],errors='coerce').fillna(pd.to_numeric(out.loc[mask,'rules_ypa'],errors='coerce'))
        elif mode=='add_to_rules': base=pd.to_numeric(out.loc[mask,'rules_ypa'],errors='coerce')
        else: raise ValueError(mode)
        cand=base-0.20*z
        valid=cand.notna(); target=float(base[valid].mean()) if valid.any() else np.nan
        if np.isfinite(target): cand=cand-(float(cand[valid].mean())-target)
        out.loc[mask,'rules_ypa']=cand.clip(4.5,10.5)
        return out
    return wrapped

def main():
    q=argparse.ArgumentParser();q.add_argument('--season',type=int,default=2025);q.add_argument('--prior-season',type=int,default=2024);q.add_argument('--weeks',default='1-18');q.add_argument('--iterations',type=int,default=2000);q.add_argument('--player-logs',type=Path,default=Path('data/backtests/player_game_logs_history.csv'));q.add_argument('--team-weekly',type=Path,default=Path('data/backtests/team_weekly_history.csv'));q.add_argument('--schedule',type=Path,default=Path('data/backtests/schedule_history.csv'));q.add_argument('--universe-dir',type=Path,default=Path('data/backtests/pregame_universe'));q.add_argument('--injuries',type=Path,default=Path('data/backtests/injuries_history.csv'));q.add_argument('--weather',type=Path,default=Path('data/backtests/weather_history.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_pressure_ypa_mc'));a=q.parse_args()
    logs=read(a.player_logs);team=read(a.team_weekly);sched=read(a.schedule);inj=opt(a.injuries);weather=opt(a.weather);original=simulation_rules.apply_rules_to_metrics; rows=[]; traces=[]
    for w in _parse_weeks(a.weeks):
        u=read(a.universe_dir/f'{a.season}_week_{w:02d}.csv');b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,w),weather=_exact_week(weather,a.season,w));actual=build_actual_rows(logs,a.season,w)
        for mode in ['current','replace_bayes','add_to_rules']:
            fn=candidate_apply_factory(original,mode)
            with patch.object(simulation_rules,'apply_rules_to_metrics',side_effect=fn): mc=build_mc_predictions(b,iterations=a.iterations,seed=42+w)
            z=mc.merge(actual,on=['team','player_clean_key','market'],how='inner'); z['candidate']=mode; z['week']=w; traces.append(z[['candidate','week','team','player_clean_key','market','mc_proj','actual']])
            print(f'[m45] W{w:02d} {mode} pass_rows={(z.market=="pass_yards").sum()}')
    t=pd.concat(traces,ignore_index=True); summary=[]
    for cand,gc in t.groupby('candidate'):
        for market,gm in gc.groupby('market'):
            summary.append({'candidate':cand,'market':market,**met(gm.actual,gm.mc_proj)})
    s=pd.DataFrame(summary).sort_values(['market','mae','correlation'],ascending=[True,True,False]);a.out_dir.mkdir(parents=True,exist_ok=True);t.to_csv(a.out_dir/'qb_pressure_ypa_mc_trace.csv',index=False);s.to_csv(a.out_dir/'qb_pressure_ypa_mc_summary.csv',index=False);print(s.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
