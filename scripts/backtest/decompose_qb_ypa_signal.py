#!/usr/bin/env python3
"""Migration 43: leakage-safe QB YPA / passing-efficiency decomposition.
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
    e=z.p-z.a; return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def corr(a,b):
    z=pd.DataFrame({'a':num(a),'b':num(b)}).dropna(); return float(z.a.corr(z.b)) if len(z)>2 else np.nan
def main():
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_passing_error/qb_passing_player_trace.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_ypa_signal'));a=q.parse_args();x=read(a.trace)
    x['actual_ypa']=num(x.actual_ypa);x['pred_ypa']=num(x.pred_ypa);x['ypa_error']=x.pred_ypa-x.actual_ypa
    # Compare canonical efficiency stages and quantify compression.
    stage=[]
    for name,col in [('base_ypa','mc_base_ypa'),('bayes_ypa','mc_bayes_ypa'),('rules_ypa','mc_rules_ypa')]:
        if col in x: stage.append({'stage':name,**met(x.actual_ypa,x[col])})
    stages=pd.DataFrame(stage)
    disp=[]
    for name,col in [('actual_ypa','actual_ypa'),('pred_ypa','pred_ypa'),('base_ypa','mc_base_ypa'),('bayes_ypa','mc_bayes_ypa'),('rules_ypa','mc_rules_ypa')]:
        if col not in x: continue
        s=num(x[col]).dropna();
        if s.empty: continue
        disp.append({'series':name,'mean':s.mean(),'std':s.std(),'p10':s.quantile(.1),'p90':s.quantile(.9)})
    dispersion=pd.DataFrame(disp)
    # Rank every available leakage-safe pregame numeric context by relation to actual YPA.
    candidates=[c for c in x.columns if c.startswith(('mc_','ctx_','rules_','qb_','hist')) and c not in {'mc_proj'}]
    rows=[]
    for c in dict.fromkeys(candidates):
        s=num(x[c]);z=pd.DataFrame({'v':s,'a':x.actual_ypa}).dropna()
        if len(z)<30 or z.v.nunique()<2: continue
        try:z['q']=pd.qcut(z.v.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
        except Exception:continue
        means=z.groupby('q',observed=True).agg(n=('a','size'),signal_mean=('v','mean'),actual_ypa=('a','mean')).reset_index();rows.append({'signal':c,'n':len(z),'corr_with_actual_ypa':corr(z.v,z.a),'q1_actual_ypa':float(means.iloc[0].actual_ypa),'q4_actual_ypa':float(means.iloc[-1].actual_ypa),'q4_minus_q1_actual_ypa':float(means.iloc[-1].actual_ypa-means.iloc[0].actual_ypa),'signal_std':float(z.v.std())})
    ranking=pd.DataFrame(rows).sort_values(['corr_with_actual_ypa','q4_minus_q1_actual_ypa'],ascending=False)
    # Tail comparison: realized bottom/top 20% YPA games, describing pregame context only.
    lo=x.actual_ypa.quantile(.2);hi=x.actual_ypa.quantile(.8);tails=[]
    for c in ranking.signal.tolist():
        s=num(x[c]);low=s[x.actual_ypa<=lo].dropna();high=s[x.actual_ypa>=hi].dropna()
        if len(low)<10 or len(high)<10: continue
        pooled=np.sqrt((low.var()+high.var())/2);effect=(high.mean()-low.mean())/pooled if pooled and np.isfinite(pooled) else np.nan
        tails.append({'signal':c,'low_ypa_cutoff':lo,'high_ypa_cutoff':hi,'low_n':len(low),'high_n':len(high),'low_ypa_game_signal_mean':low.mean(),'high_ypa_game_signal_mean':high.mean(),'standardized_tail_effect':effect})
    tails=pd.DataFrame(tails).sort_values('standardized_tail_effect',key=lambda s:s.abs(),ascending=False)
    # Bucket canonical YPA to see whether its ranking signal is monotonic.
    bx=x[['pred_ypa','actual_ypa','actual_pass_att','actual_pass_yards_raw']].dropna().copy();bx['pred_ypa_quartile']=pd.qcut(bx.pred_ypa.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4']);bucket=bx.groupby('pred_ypa_quartile',observed=True).agg(n=('actual_ypa','size'),pred_ypa=('pred_ypa','mean'),actual_ypa=('actual_ypa','mean'),actual_pass_att=('actual_pass_att','mean'),actual_pass_yards=('actual_pass_yards_raw','mean')).reset_index()
    a.out_dir.mkdir(parents=True,exist_ok=True);stages.to_csv(a.out_dir/'qb_ypa_stage_summary.csv',index=False);dispersion.to_csv(a.out_dir/'qb_ypa_dispersion.csv',index=False);ranking.to_csv(a.out_dir/'qb_ypa_signal_ranking.csv',index=False);tails.to_csv(a.out_dir/'qb_ypa_tail_separation.csv',index=False);bucket.to_csv(a.out_dir/'qb_ypa_bucket_summary.csv',index=False);x.to_csv(a.out_dir/'qb_ypa_player_trace.csv',index=False);print('=== YPA STAGES ===');print(stages.to_string(index=False));print('\n=== DISPERSION ===');print(dispersion.to_string(index=False));print('\n=== SIGNAL RANKING ===');print(ranking.head(30).to_string(index=False));print('\n=== TAIL SEPARATION ===');print(tails.head(30).to_string(index=False));print('\n=== CANONICAL YPA QUARTILES ===');print(bucket.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
