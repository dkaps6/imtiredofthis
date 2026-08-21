#!/usr/bin/env python3
"""Migration 40: decompose QB pass-attempt opportunity signal.
Diagnostic only. Reuses Migration 39 trace and historical team-week context.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p)
def num(s): return pd.to_numeric(s,errors='coerce')
def met(a,p):
    z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna()
    if z.empty:return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan}
    e=z.p-z.a; return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def main():
    q=argparse.ArgumentParser();q.add_argument('--trace',type=Path,default=Path('data/backtests/qb_passing_error/qb_passing_player_trace.csv'));q.add_argument('--team-weekly',type=Path,default=Path('data/backtests/team_weekly_history.csv'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_attempt_opportunity'));a=q.parse_args()
    z=read(a.trace); tw=read(a.team_weekly); lc={str(c).lower():c for c in tw.columns}
    # Candidate historical team signals are joined only from rows strictly before each target week.
    plays=next((lc[x] for x in ['plays','off_plays','total_plays'] if x in lc),None); patt=next((lc[x] for x in ['pass_att','attempts','pass_attempts'] if x in lc),None)
    if not plays or not patt: raise RuntimeError(f'team_weekly missing plays/pass attempts; columns={list(tw.columns)}')
    season_col=lc.get('season'); week_col=lc.get('week'); team_col=lc.get('team') or lc.get('recent_team')
    if not all([season_col,week_col,team_col]): raise RuntimeError('team_weekly missing season/week/team')
    tw['_plays']=num(tw[plays]);tw['_pass_att']=num(tw[patt]);tw['_pass_rate']=tw._pass_att/tw._plays.replace(0,np.nan)
    out=[]
    for _,r in z.iterrows():
        season=int(r.get('season',2025)); week=int(r.week); team=str(r.team)
        h=tw[(num(tw[season_col])==season)&(num(tw[week_col])<week)&(tw[team_col].astype(str)==team)].sort_values(week_col)
        prior=tw[(num(tw[season_col])==season-1)&(tw[team_col].astype(str)==team)].sort_values(week_col)
        base=h if len(h) else prior
        rec=r.to_dict()
        for n in [1,3,5,8]:
            g=base.tail(n); rec[f'hist{n}_plays']=g._plays.mean();rec[f'hist{n}_pass_rate']=g._pass_rate.mean();rec[f'hist{n}_team_pass_att']=g._pass_att.mean()
        rec['actual_team_pass_att_proxy']=r.actual_pass_att
        out.append(rec)
    x=pd.DataFrame(out); rows=[]
    for col in ['pred_attempts','hist1_team_pass_att','hist3_team_pass_att','hist5_team_pass_att','hist8_team_pass_att']:
        rows.append({'candidate':col,**met(x.actual_pass_att,x[col])})
    # Diagnose compression: compare projected-vs-actual dispersion and quartile separation.
    summary=pd.DataFrame(rows); dispersion=pd.DataFrame([{'series':'actual_attempts','mean':num(x.actual_pass_att).mean(),'std':num(x.actual_pass_att).std(),'p10':num(x.actual_pass_att).quantile(.1),'p90':num(x.actual_pass_att).quantile(.9)},{'series':'pred_attempts','mean':num(x.pred_attempts).mean(),'std':num(x.pred_attempts).std(),'p10':num(x.pred_attempts).quantile(.1),'p90':num(x.pred_attempts).quantile(.9)}])
    x['pred_attempt_quartile']=pd.qcut(num(x.pred_attempts).rank(method='first'),4,labels=['Q1','Q2','Q3','Q4']); buckets=[]
    for b,g in x.groupby('pred_attempt_quartile',observed=True): buckets.append({'bucket':str(b),'n':len(g),'pred_attempts':num(g.pred_attempts).mean(),'actual_attempts':num(g.actual_pass_att).mean(),'attempt_bias':(num(g.pred_attempts)-num(g.actual_pass_att)).mean(),'hist3_pass_rate':num(g.hist3_pass_rate).mean(),'hist3_plays':num(g.hist3_plays).mean()})
    a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/'qb_attempt_player_trace.csv',index=False);summary.to_csv(a.out_dir/'qb_attempt_candidate_summary.csv',index=False);dispersion.to_csv(a.out_dir/'qb_attempt_dispersion.csv',index=False);pd.DataFrame(buckets).to_csv(a.out_dir/'qb_attempt_bucket_summary.csv',index=False);print(summary.to_string(index=False));print(dispersion.to_string(index=False));print(pd.DataFrame(buckets).to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
