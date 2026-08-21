#!/usr/bin/env python3
"""Migration 41: identify leakage-safe pregame signals separating QB attempt tails.
Diagnostic only; no production model changes.
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
def corr(a,b):
    z=pd.DataFrame({'a':num(a),'b':num(b)}).dropna(); return float(z.a.corr(z.b)) if len(z)>2 else np.nan
def main():
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_attempt_opportunity/qb_attempt_player_trace.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_attempt_tail_context'));a=q.parse_args();x=read(a.trace)
    # Only pregame/canonical diagnostic columns; actual outcome is used solely as the target.
    candidates=[c for c in x.columns if c.startswith(('mc_','ctx_','hist')) and c not in {'mc_proj'}]
    # Add transparent derived pregame components already present in the canonical trace.
    derived={
      'pregame_team_expected_dropbacks':('mc_projected_plays','mc_dropback_rate','mul'),
      'pregame_team_expected_attempts':('mc_team_expected_pass_attempts',None,'copy'),
      'pregame_qb_attempt_share':('mc_qb_pass_att_share',None,'copy'),
      'pregame_pressure_mismatch':('mc_pressure_mismatch',None,'copy'),
    }
    for n,(c1,c2,op) in derived.items():
        if c1 in x: x[n]=num(x[c1]) if op=='copy' else num(x[c1])*num(x[c2]); candidates.append(n)
    rows=[]
    for c in dict.fromkeys(candidates):
        s=num(x[c]); z=pd.DataFrame({'v':s,'a':num(x.actual_pass_att)}).dropna()
        if len(z)<30 or z.v.nunique()<2: continue
        try: z['q']=pd.qcut(z.v.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
        except Exception: continue
        means=z.groupby('q',observed=True).agg(n=('a','size'),signal_mean=('v','mean'),actual_attempts=('a','mean')).reset_index(); q1=float(means.iloc[0].actual_attempts);q4=float(means.iloc[-1].actual_attempts)
        rows.append({'signal':c,'n':len(z),'corr_with_actual_attempts':corr(z.v,z.a),'q1_actual_attempts':q1,'q4_actual_attempts':q4,'q4_minus_q1_actual_attempts':q4-q1,'signal_std':float(z.v.std())})
    summary=pd.DataFrame(rows).sort_values(['corr_with_actual_attempts','q4_minus_q1_actual_attempts'],ascending=False)
    # Actual tails are descriptive labels only; compare pregame feature means for <=P20 vs >=P80 outcomes.
    aa=num(x.actual_pass_att);lo=aa.quantile(.2);hi=aa.quantile(.8); tail=[]
    for c in summary.signal.tolist():
        s=num(x[c]); low=s[aa<=lo].dropna();high=s[aa>=hi].dropna()
        if len(low)<10 or len(high)<10: continue
        pooled=np.sqrt((low.var()+high.var())/2); effect=(high.mean()-low.mean())/pooled if pooled and np.isfinite(pooled) else np.nan
        tail.append({'signal':c,'low_attempt_cutoff':lo,'high_attempt_cutoff':hi,'low_n':len(low),'high_n':len(high),'low_attempt_game_signal_mean':low.mean(),'high_attempt_game_signal_mean':high.mean(),'standardized_tail_effect':effect})
    tails=pd.DataFrame(tail).sort_values('standardized_tail_effect',key=lambda s:s.abs(),ascending=False)
    a.out_dir.mkdir(parents=True,exist_ok=True);summary.to_csv(a.out_dir/'qb_attempt_signal_ranking.csv',index=False);tails.to_csv(a.out_dir/'qb_attempt_tail_separation.csv',index=False);print('=== SIGNAL RANKING ===');print(summary.head(30).to_string(index=False));print('\n=== TAIL SEPARATION ===');print(tails.head(30).to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
