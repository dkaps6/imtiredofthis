#!/usr/bin/env python3
"""Migration 48: audit whether QB role disruptions were knowable pregame.
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
def key(x): return ''.join(ch.lower() for ch in str(x or '') if ch.isalnum())
def met(a,p):
    z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna()
    if z.empty:return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan}
    e=z.p-z.a; return {'n':len(z),'mae':float(e.abs().mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>1 else np.nan}
def norm_status(v):
    s=str(v or '').upper()
    for t in ['OUT','DOUBTFUL','QUESTIONABLE','LIMITED','IR','PUP']:
        if t in s:return t
    return s.strip() or 'NONE'
def main():
    q=argparse.ArgumentParser();q.add_argument('--participation-trace',type=Path,default=Path('data/backtests/qb_participation/qb_participation_player_trace.csv'));q.add_argument('--injuries',type=Path,default=Path('data/backtests/injuries_history.csv'));q.add_argument('--universe-dir',type=Path,default=Path('data/backtests/pregame_universe'));q.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_pregame_availability'));a=q.parse_args();x=read(a.participation_trace);inj=read(a.injuries) if a.injuries.exists() else pd.DataFrame()
    if 'season' not in x:x['season']=2025
    rows=[]
    for _,r in x.iterrows():
        season=int(r.season);week=int(r.week);team=str(r.team);pk=str(r.player_clean_key)
        u=read(a.universe_dir/f'{season}_week_{week:02d}.csv');u.columns=[str(c).lower() for c in u.columns];u['player_clean_key']=u.get('player_clean_key',u.get('player','')).map(key);uu=u[(u.team.astype(str)==team)&(u.player_clean_key==pk)]
        rec=r.to_dict();rec['pregame_in_universe']=int(len(uu)>0);rec['pregame_position']=str(uu.iloc[0].get('position','')) if len(uu) else '';rec['pregame_role']=str(uu.iloc[0].get('role','')) if len(uu) else ''
        ii=pd.DataFrame()
        if not inj.empty:
            z=inj.copy();z.columns=[str(c).lower() for c in z.columns];z['player_clean_key']=z.get('player_clean_key',z.get('player','')).map(key);ii=z[(num(z.get('season'))==season)&(num(z.get('week'))==week)&(z.get('team','').astype(str)==team)&(z.player_clean_key==pk)]
        status='NONE';designation='';report=0
        if len(ii):
            report=1;status=norm_status(ii.iloc[0].get('status',ii.iloc[0].get('injury_status','')));designation=str(ii.iloc[0].get('designation',ii.iloc[0].get('injury_designation','')))
        rec['pregame_injury_report']=report;rec['pregame_injury_status']=status;rec['pregame_injury_designation']=designation
        source=str(r.get('mc_qb_role_source',r.get('qb_role_source','')));rec['pregame_qb_role_source']=source
        share=float(num(pd.Series([r.get('actual_qb_share')])).fillna(0).iloc[0]);pclass=str(r.get('participation_class',''))
        mismatch=bool(r.get('role_mismatch',False)) or (pclass not in {'stable_primary','primary_but_shared'} and share<.8)
        risky=status in {'OUT','DOUBTFUL','QUESTIONABLE','IR','PUP'} or source=='prior_history' or rec['pregame_in_universe']==0
        rec['pregame_risk_flag']=int(risky);rec['audit_class']='pregame_flagged_disruption' if mismatch and risky else ('unflagged_disruption' if mismatch else 'stable_or_shared')
        rows.append(rec)
    z=pd.DataFrame(rows);summ=[]
    for name,g in [('all',z),('stable_or_shared',z[z.audit_class=='stable_or_shared']),('pregame_flagged_disruption',z[z.audit_class=='pregame_flagged_disruption']),('unflagged_disruption',z[z.audit_class=='unflagged_disruption'])]:
        if g.empty: continue
        d={'scope':name,'games':len(g),'mean_actual_qb_share':num(g.get('actual_qb_share')).mean(),'mean_actual_attempts':num(g.get('actual_pass_att')).mean(),'mean_pred_attempts':num(g.get('pred_attempts')).mean(),'risk_rate':num(g.get('pregame_risk_flag')).mean()};d.update(met(g.actual_pass_yards_raw,g.mc_proj));summ.append(d)
    s=pd.DataFrame(summ);status=z.groupby(['audit_class','pregame_injury_status','pregame_qb_role_source'],dropna=False).size().reset_index(name='games').sort_values('games',ascending=False)
    worst=z.assign(abs_miss=(num(z.mc_proj)-num(z.actual_pass_yards_raw)).abs()).sort_values('abs_miss',ascending=False).head(100)
    a.out_dir.mkdir(parents=True,exist_ok=True);z.to_csv(a.out_dir/'qb_pregame_availability_trace.csv',index=False);s.to_csv(a.out_dir/'qb_pregame_availability_summary.csv',index=False);status.to_csv(a.out_dir/'qb_pregame_signal_summary.csv',index=False);worst.to_csv(a.out_dir/'qb_pregame_worst100.csv',index=False);print(s.to_string(index=False));print(status.head(50).to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
