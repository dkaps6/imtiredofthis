#!/usr/bin/env python3
"""Migration 42: leakage-safe QB dropback-rate calibration around the 57% anchor.
Diagnostic only. Candidate rates are recentered to preserve the 0.57 league anchor.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ANCHOR=0.57

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def num(x): return pd.to_numeric(x,errors='coerce')
def metrics(a,p):
    z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna()
    if z.empty:return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan}
    e=z.p-z.a
    return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def recenter(rate: pd.Series) -> pd.Series:
    r=num(rate).copy(); m=r.mean()
    if np.isfinite(m): r=r+(ANCHOR-m)
    return r.clip(.45,.68)
def main():
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_attempt_opportunity/qb_attempt_player_trace.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_dropback_anchor_calibration'));a=q.parse_args();x=read(a.trace)
    required=['mc_projected_plays','mc_pass_attempts_per_dropback','mc_qb_pass_att_share','actual_pass_att','actual_pass_yards_raw','pred_ypa','hist3_pass_rate','hist5_pass_rate']
    missing=[c for c in required if c not in x.columns]
    if missing: raise RuntimeError(f'missing required trace columns: {missing}')
    plays=num(x.mc_projected_plays); conv=num(x.mc_pass_attempts_per_dropback).fillna(1.0); share=num(x.mc_qb_pass_att_share).fillna(1.0); ypa=num(x.pred_ypa)
    h3=num(x.hist3_pass_rate); h5=num(x.hist5_pass_rate); blend=(h3+h5)/2.0
    cands={'current':pd.Series(ANCHOR,index=x.index,dtype=float)}
    for w in [.25,.50,.75,1.00,1.25]:
        cands[f'h3_w{w:.2f}']=recenter(ANCHOR+w*(h3-ANCHOR))
        cands[f'h5_w{w:.2f}']=recenter(ANCHOR+w*(h5-ANCHOR))
        cands[f'blend35_w{w:.2f}']=recenter(ANCHOR+w*(blend-ANCHOR))
    rows=[]; trace=[]
    for name,rate in cands.items():
        att=plays*rate*conv*share; py=att*ypa
        am=metrics(x.actual_pass_att,att); pm=metrics(x.actual_pass_yards_raw,py)
        rows.append({'candidate':name,'mean_dropback_rate':float(num(rate).mean()),'dropback_rate_std':float(num(rate).std()),'attempt_mae':am['mae'],'attempt_rmse':am['rmse'],'attempt_bias':am['bias'],'attempt_corr':am['correlation'],'pass_yards_mae':pm['mae'],'pass_yards_rmse':pm['rmse'],'pass_yards_bias':pm['bias'],'pass_yards_corr':pm['correlation']})
        t=pd.DataFrame({'week':x.get('week'),'team':x.get('team'),'player':x.get('player'),'candidate':name,'candidate_dropback_rate':rate,'candidate_attempts':att,'actual_attempts':x.actual_pass_att,'candidate_pass_yards':py,'actual_pass_yards':x.actual_pass_yards_raw});trace.append(t)
    s=pd.DataFrame(rows).sort_values(['attempt_mae','pass_yards_mae','attempt_corr'],ascending=[True,True,False]); tr=pd.concat(trace,ignore_index=True)
    a.out_dir.mkdir(parents=True,exist_ok=True);s.to_csv(a.out_dir/'qb_dropback_candidate_summary.csv',index=False);tr.to_csv(a.out_dir/'qb_dropback_candidate_trace.csv',index=False)
    print(s.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
