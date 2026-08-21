#!/usr/bin/env python3
"""Migration 39: leakage-safe QB passing opportunity/efficiency decomposition.
Diagnostic only: no production football logic changes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.backtest.component_predictions import predict_week
from scripts.backtest.walk_forward import _exact_week, _parse_weeks

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def opt(p): return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
def met(a,p):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'p':pd.to_numeric(p,errors='coerce')}).dropna()
    if z.empty:return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan}
    e=z.p-z.a; return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def main():
    q=argparse.ArgumentParser(); q.add_argument('--season',type=int,default=2025);q.add_argument('--prior-season',type=int,default=2024);q.add_argument('--weeks',default='1-18');q.add_argument('--iterations',type=int,default=2000);q.add_argument('--player-logs',type=Path,default=Path('data/backtests/player_game_logs_history.csv'));q.add_argument('--team-weekly',type=Path,default=Path('data/backtests/team_weekly_history.csv'));q.add_argument('--schedule',type=Path,default=Path('data/backtests/schedule_history.csv'));q.add_argument('--universe-dir',type=Path,default=Path('data/backtests/pregame_universe'));q.add_argument('--injuries',type=Path,default=Path('data/backtests/injuries_history.csv'));q.add_argument('--weather',type=Path,default=Path('data/backtests/weather_history.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_passing_error'));a=q.parse_args()
    logs=read(a.player_logs); team=read(a.team_weekly); sched=read(a.schedule); inj=opt(a.injuries); weather=opt(a.weather); allrows=[]
    lc={str(c).lower():c for c in logs.columns}; attcol=lc.get('pass_att'); ydcol=lc.get('pass_yards');
    if not attcol or not ydcol: raise RuntimeError('historical player logs require pass_att and pass_yards')
    for w in _parse_weeks(a.weeks):
        u=read(a.universe_dir/f'{a.season}_week_{w:02d}.csv'); iw=_exact_week(inj,a.season,w); ww=_exact_week(weather,a.season,w)
        p=predict_week(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=iw,weather=ww,iterations=a.iterations,seed=42+w); p=p[p.market.eq('pass_yards')].copy()
        act=logs[(pd.to_numeric(logs[lc['season']],errors='coerce')==a.season)&(pd.to_numeric(logs[lc['week']],errors='coerce')==w)].copy(); act['player_clean_key']=act.get('player_clean_key',act[lc['player']]).astype(str).str.lower().str.replace(r'[^a-z0-9]','',regex=True); act['actual_pass_att']=pd.to_numeric(act[attcol],errors='coerce'); act['actual_pass_yards_raw']=pd.to_numeric(act[ydcol],errors='coerce'); act['actual_ypa']=act.actual_pass_yards_raw/act.actual_pass_att.replace(0,np.nan); act=act[[lc['team'],'player_clean_key','actual_pass_att','actual_pass_yards_raw','actual_ypa']].rename(columns={lc['team']:'team'}).drop_duplicates(['team','player_clean_key'])
        x=p.merge(act,on=['team','player_clean_key'],how='left'); x['week']=w; x['pred_attempts']=pd.to_numeric(x.get('mc_expected_pass_attempts'),errors='coerce'); x['pred_ypa']=pd.to_numeric(x.get('mc_rules_ypa'),errors='coerce'); x['oracle_attempts_pred_ypa']=x.actual_pass_att*x.pred_ypa; x['pred_attempts_oracle_ypa']=x.pred_attempts*x.actual_ypa; x['det_pass_yards']=x.pred_attempts*x.pred_ypa; x['attempt_error']=x.pred_attempts-x.actual_pass_att; x['ypa_error']=x.pred_ypa-x.actual_ypa; allrows.append(x)
    z=pd.concat(allrows,ignore_index=True); rows=[]
    for stage,col in [('canonical_mc','mc_proj'),('det_attempts_x_ypa','det_pass_yards'),('oracle_attempts_x_pred_ypa','oracle_attempts_pred_ypa'),('pred_attempts_x_oracle_ypa','pred_attempts_oracle_ypa')]: rows.append({'stage':stage,**met(z.actual_pass_yards_raw,z[col])})
    s=pd.DataFrame(rows); z['season_phase']=pd.cut(z.week,[0,4,9,13,18],labels=['W1-4','W5-9','W10-13','W14-18']); z['attempt_tier']=pd.qcut(z.pred_attempts.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4']) if len(z)>=4 else 'all'; group=[]
    for dim in ['season_phase','attempt_tier','mc_pass_attempt_rate_source','mc_qb_role_source']:
        if dim not in z: continue
        for val,g in z.groupby(dim,dropna=False,observed=True): group.append({'dimension':dim,'bucket':str(val),'mean_pred_attempts':pd.to_numeric(g.pred_attempts,errors='coerce').mean(),'mean_actual_attempts':pd.to_numeric(g.actual_pass_att,errors='coerce').mean(),'mean_attempt_error':pd.to_numeric(g.attempt_error,errors='coerce').mean(),'mean_pred_ypa':pd.to_numeric(g.pred_ypa,errors='coerce').mean(),'mean_actual_ypa':pd.to_numeric(g.actual_ypa,errors='coerce').mean(),'mean_ypa_error':pd.to_numeric(g.ypa_error,errors='coerce').mean(),**met(g.actual_pass_yards_raw,g.mc_proj)})
    a.out_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.out_dir/'qb_passing_player_trace.csv',index=False); s.to_csv(a.out_dir/'qb_passing_stage_summary.csv',index=False); pd.DataFrame(group).to_csv(a.out_dir/'qb_passing_bucket_summary.csv',index=False); print(s.to_string(index=False)); print(pd.DataFrame(group).to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())
