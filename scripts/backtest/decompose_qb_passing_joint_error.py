#!/usr/bin/env python3
"""Migration 46: exact QB passing joint-error / interaction decomposition.
Diagnostic only; no production football changes.
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
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_passing_error/qb_passing_player_trace.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_passing_joint_error'));a=q.parse_args();x=read(a.trace)
    req=['pred_attempts','actual_pass_att','pred_ypa','actual_ypa','actual_pass_yards_raw','mc_proj']
    miss=[c for c in req if c not in x.columns]
    if miss: raise RuntimeError(f'missing trace columns: {miss}')
    for c in req: x[c]=num(x[c])
    x=x.dropna(subset=req).copy()
    # Exact product-error identity for deterministic passing yards:
    # pA*pY - A*Y = (pA-A)*Y + A*(pY-Y) + (pA-A)*(pY-Y)
    x['attempt_error']=x.pred_attempts-x.actual_pass_att
    x['ypa_error']=x.pred_ypa-x.actual_ypa
    x['det_pass_yards']=x.pred_attempts*x.pred_ypa
    x['det_yard_error']=x.det_pass_yards-x.actual_pass_yards_raw
    x['mc_yard_error']=x.mc_proj-x.actual_pass_yards_raw
    x['volume_contribution']=x.attempt_error*x.actual_ypa
    x['efficiency_contribution']=x.actual_pass_att*x.ypa_error
    x['interaction_contribution']=x.attempt_error*x.ypa_error
    x['reconstructed_error']=x.volume_contribution+x.efficiency_contribution+x.interaction_contribution
    x['reconstruction_residual']=x.det_yard_error-x.reconstructed_error
    x['attempt_abs_error']=x.attempt_error.abs();x['ypa_abs_error']=x.ypa_error.abs();x['yard_abs_error']=x.mc_yard_error.abs()
    x['error_sign_combo']=np.select([
      (x.attempt_error>0)&(x.ypa_error>0),(x.attempt_error<0)&(x.ypa_error<0),(x.attempt_error>0)&(x.ypa_error<0),(x.attempt_error<0)&(x.ypa_error>0)
    ],['both_high','both_low','attempt_high_ypa_low','attempt_low_ypa_high'],default='mixed_zero')
    # Dominant component based on absolute exact contribution.
    comps=x[['volume_contribution','efficiency_contribution','interaction_contribution']].abs()
    x['dominant_component']=comps.idxmax(axis=1).str.replace('_contribution','',regex=False)
    for t in [50,75,100]: x[f'miss_{t}plus']=x.yard_abs_error.ge(t)
    overall=pd.DataFrame([{
      'n':len(x),'attempt_error_corr_with_ypa_error':corr(x.attempt_error,x.ypa_error),
      'attempt_abs_error_corr_with_yard_abs_error':corr(x.attempt_abs_error,x.yard_abs_error),
      'ypa_abs_error_corr_with_yard_abs_error':corr(x.ypa_abs_error,x.yard_abs_error),
      'mean_abs_volume_contribution':x.volume_contribution.abs().mean(),
      'mean_abs_efficiency_contribution':x.efficiency_contribution.abs().mean(),
      'mean_abs_interaction_contribution':x.interaction_contribution.abs().mean(),
      'mean_reconstruction_residual':x.reconstruction_residual.abs().mean(),
      'mc_mae':x.yard_abs_error.mean(),
    }])
    groups=[]
    for dim in ['error_sign_combo','dominant_component']:
        for b,g in x.groupby(dim,dropna=False):
            groups.append({'dimension':dim,'bucket':str(b),'n':len(g),'share':len(g)/len(x),'mc_mae':g.yard_abs_error.mean(),'mean_attempt_error':g.attempt_error.mean(),'mean_ypa_error':g.ypa_error.mean(),'mean_abs_volume_contribution':g.volume_contribution.abs().mean(),'mean_abs_efficiency_contribution':g.efficiency_contribution.abs().mean(),'mean_abs_interaction_contribution':g.interaction_contribution.abs().mean()})
    for t in [50,75,100]:
        g=x[x[f'miss_{t}plus']]
        groups.append({'dimension':'yard_miss_threshold','bucket':f'{t}+','n':len(g),'share':len(g)/len(x),'mc_mae':g.yard_abs_error.mean(),'mean_attempt_error':g.attempt_error.mean(),'mean_ypa_error':g.ypa_error.mean(),'mean_abs_volume_contribution':g.volume_contribution.abs().mean(),'mean_abs_efficiency_contribution':g.efficiency_contribution.abs().mean(),'mean_abs_interaction_contribution':g.interaction_contribution.abs().mean()})
    # Worst misses for direct football inspection.
    cols=[c for c in ['week','team','player','player_clean_key','actual_pass_att','pred_attempts','attempt_error','actual_ypa','pred_ypa','ypa_error','actual_pass_yards_raw','mc_proj','mc_yard_error','volume_contribution','efficiency_contribution','interaction_contribution','error_sign_combo','dominant_component'] if c in x.columns]
    worst=x.reindex(x.yard_abs_error.sort_values(ascending=False).index).head(100)[cols]
    a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/'qb_passing_joint_error_trace.csv',index=False);overall.to_csv(a.out_dir/'qb_passing_joint_error_summary.csv',index=False);pd.DataFrame(groups).to_csv(a.out_dir/'qb_passing_joint_error_groups.csv',index=False);worst.to_csv(a.out_dir/'qb_passing_worst_100_misses.csv',index=False);print(overall.to_string(index=False));print(pd.DataFrame(groups).to_string(index=False));print('\n=== WORST 25 ===');print(worst.head(25).to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
