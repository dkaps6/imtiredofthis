#!/usr/bin/env python3
"""Migration 49: diagnose catastrophic passing errors among stable primary QBs.
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
def zcorr(a,b):
    z=pd.DataFrame({'a':num(a),'b':num(b)}).dropna(); return float(z.a.corr(z.b)) if len(z)>2 else np.nan

def main():
    q=argparse.ArgumentParser()
    q.add_argument('--participation-trace',type=Path,default=Path('data/backtests/qb_participation_role/qb_participation_player_trace.csv'))
    q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_stable_catastrophic'))
    a=q.parse_args(); x=read(a.participation_trace)
    share=num(x.get('actual_qb_attempt_share'))
    stable=x[(share>=.80)&(num(x.get('is_actual_primary')).fillna(0).eq(1))].copy()
    if stable.empty: raise RuntimeError('no stable >=80% primary QB rows')
    stable['attempt_error']=num(stable.pred_attempts)-num(stable.actual_pass_att)
    stable['ypa_error']=num(stable.mc_rules_ypa)-num(stable.actual_ypa)
    stable['pass_error']=num(stable.mc_proj)-num(stable.actual_pass_yards_raw)
    stable['abs_pass_error']=stable.pass_error.abs()
    stable['volume_contrib']=stable.attempt_error*num(stable.actual_ypa)
    stable['efficiency_contrib']=num(stable.actual_pass_att)*stable.ypa_error
    stable['interaction_contrib']=stable.attempt_error*stable.ypa_error
    stable['dominant_component']=np.select([
        stable.volume_contrib.abs().ge(stable.efficiency_contrib.abs()) & stable.volume_contrib.abs().ge(stable.interaction_contrib.abs()),
        stable.efficiency_contrib.abs().ge(stable.volume_contrib.abs()) & stable.efficiency_contrib.abs().ge(stable.interaction_contrib.abs())],['volume','efficiency'],default='interaction')
    bins=[-1,24.999,49.999,74.999,99.999,np.inf]; labels=['lt25','25_49','50_74','75_99','100plus']
    stable['error_bucket']=pd.cut(stable.abs_pass_error,bins=bins,labels=labels)
    rows=[]
    for b,g in stable.groupby('error_bucket',observed=True):
        rows.append({'error_bucket':str(b),'games':len(g),'share_of_stable':len(g)/len(stable),'mean_abs_pass_error':num(g.abs_pass_error).mean(),'mean_pass_error':num(g.pass_error).mean(),'mean_pred_attempts':num(g.pred_attempts).mean(),'mean_actual_attempts':num(g.actual_pass_att).mean(),'mean_abs_attempt_error':num(g.attempt_error).abs().mean(),'mean_pred_ypa':num(g.mc_rules_ypa).mean(),'mean_actual_ypa':num(g.actual_ypa).mean(),'mean_abs_ypa_error':num(g.ypa_error).abs().mean(),'mean_abs_volume_contrib':num(g.volume_contrib).abs().mean(),'mean_abs_efficiency_contrib':num(g.efficiency_contrib).abs().mean(),'mean_abs_interaction_contrib':num(g.interaction_contrib).abs().mean(),**met(g.actual_pass_yards_raw,g.mc_proj)})
    buckets=pd.DataFrame(rows)
    # Rank all available pregame/canonical numeric signals by association with absolute passing error.
    candidate_cols=[c for c in stable.columns if c.startswith(('mc_','ctx_','rules_','qb_'))]
    sig=[]
    for c in candidate_cols:
        s=num(stable[c])
        if s.notna().sum()<30 or s.nunique(dropna=True)<2: continue
        sig.append({'signal':c,'n':int(s.notna().sum()),'corr_with_abs_pass_error':zcorr(s,stable.abs_pass_error),'corr_with_signed_pass_error':zcorr(s,stable.pass_error),'mean_in_lt50':s[stable.abs_pass_error<50].mean(),'mean_in_100plus':s[stable.abs_pass_error>=100].mean()})
    signals=pd.DataFrame(sig)
    if not signals.empty: signals=signals.sort_values('corr_with_abs_pass_error',key=lambda s:s.abs(),ascending=False)
    # Sign pattern and component dominance by severity.
    stable['sign_combo']=np.select([(stable.attempt_error>=0)&(stable.ypa_error>=0),(stable.attempt_error<0)&(stable.ypa_error<0),(stable.attempt_error>=0)&(stable.ypa_error<0)],['both_high','both_low','attempt_high_ypa_low'],default='attempt_low_ypa_high')
    combos=stable.groupby(['error_bucket','sign_combo'],observed=True).agg(games=('pass_error','size'),mean_abs_error=('abs_pass_error','mean')).reset_index()
    dom=stable.groupby(['error_bucket','dominant_component'],observed=True).agg(games=('pass_error','size'),mean_abs_error=('abs_pass_error','mean')).reset_index()
    worst=stable.sort_values('abs_pass_error',ascending=False).head(100)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    stable.to_csv(a.out_dir/'qb_stable_player_trace.csv',index=False)
    buckets.to_csv(a.out_dir/'qb_stable_error_bucket_summary.csv',index=False)
    signals.to_csv(a.out_dir/'qb_stable_pregame_signal_ranking.csv',index=False)
    combos.to_csv(a.out_dir/'qb_stable_sign_combo_summary.csv',index=False)
    dom.to_csv(a.out_dir/'qb_stable_dominant_component_summary.csv',index=False)
    worst.to_csv(a.out_dir/'qb_stable_worst100.csv',index=False)
    print('=== STABLE ERROR BUCKETS ===');print(buckets.to_string(index=False));print('\n=== TOP PREGAME SIGNALS ===');print(signals.head(40).to_string(index=False) if not signals.empty else 'none');print('\n=== DOMINANT COMPONENT ===');print(dom.to_string(index=False));return 0
if __name__=='__main__': raise SystemExit(main())
