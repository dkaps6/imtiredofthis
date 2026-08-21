#!/usr/bin/env python3
"""Migration 44: leakage-safe QB YPA context calibration.
Diagnostic/calibration only. No production coefficients are changed.
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
    e=z.p-z.a; return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def zscore(s):
    s=num(s); sd=s.std(); return (s-s.mean())/sd if pd.notna(sd) and sd>1e-9 else pd.Series(0.,index=s.index)
def main():
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_passing_error/qb_passing_player_trace.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_ypa_context_calibration'));a=q.parse_args();x=read(a.trace)
    base=num(x.get('mc_bayes_ypa',x.get('mc_rules_ypa'))).fillna(num(x.mc_rules_ypa)); actual=num(x.actual_ypa); attempts=num(x.pred_attempts); actual_yards=num(x.actual_pass_yards_raw)
    off_pressure=zscore(x.mc_off_pressure_allowed) if 'mc_off_pressure_allowed' in x else pd.Series(0.,index=x.index)
    def_pressure=zscore(x.mc_def_pressure_generated) if 'mc_def_pressure_generated' in x else pd.Series(0.,index=x.index)
    mismatch=zscore(x.mc_pressure_mismatch) if 'mc_pressure_mismatch' in x else def_pressure-off_pressure
    explosive=zscore(x.get('rules_explosive_play_rate_allowed',pd.Series(np.nan,index=x.index)))
    # Signs are football-informed from Migration 43 diagnostics: more pressure allowed hurts YPA; more defensive pressure hurts YPA.
    signals={
      'pressure_allowed':-off_pressure,
      'pressure_mismatch':-mismatch,
      'pressure_blend':-(0.65*off_pressure+0.35*def_pressure),
    }
    if explosive.notna().sum()>30: signals['pressure_explosive_blend']=-(0.55*off_pressure+0.30*def_pressure)+0.15*explosive.fillna(0)
    rows=[]; traces=[]
    # Current baseline plus conservative additive YPA adjustments, all recentered to preserve mean YPA.
    candidates=[('current',None,0.)]
    for name in signals:
        for strength in [0.10,0.20,0.30,0.40,0.50,0.65,0.80]: candidates.append((f'{name}_{strength:.2f}',name,strength))
    target_mean=float(base.mean())
    for cname,sname,strength in candidates:
        cand=base.copy() if sname is None else base + strength*signals[sname]
        cand=cand-(cand.mean()-target_mean); cand=cand.clip(4.5,10.5)
        pass_yards=attempts*cand
        r={'candidate':cname,'ypa_mean':cand.mean(),'ypa_std':cand.std(),**{f'ypa_{k}':v for k,v in met(actual,cand).items()},**{f'pass_{k}':v for k,v in met(actual_yards,pass_yards).items()}}
        rows.append(r)
        traces.append(pd.DataFrame({'candidate':cname,'team':x.team,'player_clean_key':x.player_clean_key,'week':x.week,'candidate_ypa':cand,'candidate_pass_yards':pass_yards,'actual_ypa':actual,'actual_pass_yards':actual_yards}))
    s=pd.DataFrame(rows).sort_values(['pass_mae','ypa_mae','pass_correlation'],ascending=[True,True,False]);t=pd.concat(traces,ignore_index=True)
    a.out_dir.mkdir(parents=True,exist_ok=True);s.to_csv(a.out_dir/'qb_ypa_context_candidate_summary.csv',index=False);t.to_csv(a.out_dir/'qb_ypa_context_candidate_trace.csv',index=False);print(s.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
