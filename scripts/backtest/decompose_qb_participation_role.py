#!/usr/bin/env python3
"""Migration 47: classify QB participation/starter-role disruptions.
Diagnostic only; no production football logic changes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def num(x): return pd.to_numeric(x,errors='coerce')
def met(a,p):
    z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna()
    if z.empty:return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan}
    e=z.p-z.a
    return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def key(s): return s.astype(str).str.lower().str.replace(r'[^a-z0-9]','',regex=True)
def main():
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_passing_error/qb_passing_player_trace.csv'));q.add_argument('--player-logs',type=Path,default=Path('data/backtests/player_game_logs_history.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_participation_role'));a=q.parse_args()
    x=read(a.trace); logs=read(a.player_logs); logs.columns=[str(c).lower() for c in logs.columns]
    required={'season','week','team','player','pass_att'}; missing=required-set(logs.columns)
    if missing: raise RuntimeError(f'player logs missing columns: {sorted(missing)}')
    logs['player_clean_key']=key(logs.get('player_clean_key',logs.player)); logs['pass_att_n']=num(logs.pass_att).fillna(0.0)
    teamtot=logs.groupby(['season','week','team'],as_index=False).pass_att_n.sum().rename(columns={'pass_att_n':'team_qb_pass_att'})
    logs=logs.merge(teamtot,on=['season','week','team'],how='left');logs['actual_qb_attempt_share']=logs.pass_att_n/logs.team_qb_pass_att.replace(0,np.nan)
    prim=logs.sort_values(['season','week','team','pass_att_n'],ascending=[True,True,True,False]).drop_duplicates(['season','week','team'])[['season','week','team','player_clean_key','pass_att_n']].rename(columns={'player_clean_key':'actual_primary_qb_key','pass_att_n':'actual_primary_qb_att'})
    y=logs[['season','week','team','player_clean_key','pass_att_n','team_qb_pass_att','actual_qb_attempt_share']].merge(prim,on=['season','week','team'],how='left')
    x['season']=num(x.get('season',2025)).fillna(2025).astype(int);x['week']=num(x.week).astype(int);x['player_clean_key']=key(x.player_clean_key)
    z=x.merge(y,on=['season','week','team','player_clean_key'],how='left');z['actual_pass_att']=num(z.get('actual_pass_att',z.pass_att_n));z['actual_qb_attempt_share']=num(z.actual_qb_attempt_share)
    z['is_actual_primary']=(z.player_clean_key==z.actual_primary_qb_key).astype(int);z['predicted_primary']=num(z.get('mc_qb_projection_eligible',1)).fillna(1).astype(int)
    z['participation_class']='stable_primary_80plus'
    z.loc[(z.is_actual_primary.eq(1)) & z.actual_qb_attempt_share.lt(.80),'participation_class']='primary_but_shared'
    z.loc[(z.is_actual_primary.eq(0)) & z.actual_qb_attempt_share.ge(.20),'participation_class']='secondary_meaningful'
    z.loc[(z.is_actual_primary.eq(0)) & z.actual_qb_attempt_share.lt(.20),'participation_class']='minimal_or_nonprimary'
    z.loc[z.actual_pass_att.le(1),'participation_class']='one_or_zero_attempts'
    z['role_mismatch']=(z.predicted_primary.eq(1)&z.is_actual_primary.eq(0)).astype(int)
    z['abs_pass_error']=(num(z.mc_proj)-num(z.actual_pass_yards_raw)).abs()
    rows=[]
    for label,g in [('all',z),('stable_primary_70plus',z[z.actual_qb_attempt_share.ge(.70)&z.is_actual_primary.eq(1)]),('stable_primary_80plus',z[z.actual_qb_attempt_share.ge(.80)&z.is_actual_primary.eq(1)]),('stable_primary_90plus',z[z.actual_qb_attempt_share.ge(.90)&z.is_actual_primary.eq(1)]),('role_mismatch',z[z.role_mismatch.eq(1)])]:
        rows.append({'slice':label,'mean_actual_attempt_share':num(g.actual_qb_attempt_share).mean(),'mean_actual_attempts':num(g.actual_pass_att).mean(),'mean_pred_attempts':num(g.pred_attempts).mean(),**met(g.actual_pass_yards_raw,g.mc_proj)})
    cls=[]
    for c,g in z.groupby('participation_class',dropna=False): cls.append({'participation_class':str(c),'n':len(g),'share_of_rows':len(g)/len(z),'mean_actual_attempt_share':num(g.actual_qb_attempt_share).mean(),'mean_actual_attempts':num(g.actual_pass_att).mean(),'mean_pred_attempts':num(g.pred_attempts).mean(),'mean_abs_pass_error':num(g.abs_pass_error).mean(),**met(g.actual_pass_yards_raw,g.mc_proj)})
    worst=z.sort_values('abs_pass_error',ascending=False).head(100)
    a.out_dir.mkdir(parents=True,exist_ok=True);z.to_csv(a.out_dir/'qb_participation_player_trace.csv',index=False);pd.DataFrame(rows).to_csv(a.out_dir/'qb_participation_slice_summary.csv',index=False);pd.DataFrame(cls).sort_values('n',ascending=False).to_csv(a.out_dir/'qb_participation_class_summary.csv',index=False);worst.to_csv(a.out_dir/'qb_participation_worst100.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(cls).sort_values('n',ascending=False).to_string(index=False));return 0
if __name__=='__main__': raise SystemExit(main())
