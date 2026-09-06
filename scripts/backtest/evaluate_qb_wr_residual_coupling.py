#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd

def one(root,name):
 h=list(Path(root).rglob(name));
 if len(h)!=1: raise RuntimeError(f'expected one {name}, got {len(h)}')
 return h[0]
def num(s): return pd.to_numeric(s,errors='coerce')
def team(v):
 r=str(v or '').strip().upper();return {'JAC':'JAX','LA':'LAR'}.get(r,r)
def corr(x,y,method):
 z=pd.DataFrame({'x':num(x),'y':num(y)}).dropna();return float(z.x.corr(z.y,method=method)) if len(z)>2 else np.nan
def score(x,col):
 g=x.loc[num(x[col]).notna() & num(x.qb_residual).notna()].copy();v=num(g[col]);q=num(g.qb_residual);q1=float(v.quantile(.25));q4=float(v.quantile(.75));gap=float(q.loc[v.ge(q4)].mean()-q.loc[v.le(q1)].mean());nz=(v.ne(0)&q.ne(0));same=float((np.sign(v.loc[nz])==np.sign(q.loc[nz])).mean()) if nz.any() else np.nan
 out={'signal':col,'n':len(g),'pearson':corr(v,q,'pearson'),'spearman':corr(v,q,'spearman'),'same_sign_rate':same,'q4_minus_q1_qb_residual_gap':gap}
 for s in [2024,2025]:
  z=g.loc[g.season.eq(s)];out[f'pearson_{s}']=corr(z[col],z.qb_residual,'pearson');out[f'spearman_{s}']=corr(z[col],z.qb_residual,'spearman')
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--m89-root',type=Path,required=True);ap.add_argument('--wr-r1-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 qb=pd.read_csv(one(a.m89_root,'m89_2024_2025_synthesis_trace.csv'),low_memory=False);qb.columns=[str(c).strip().lower() for c in qb.columns];qb=qb.loc[num(qb.season).isin([2024,2025])].copy();qb['season']=num(qb.season).astype(int);qb['week']=num(qb.week).astype(int);qb['team']=qb.team.map(team);qb['qb_residual']=num(qb.actual_pass_yards)-num(qb.football_synthesis)
 if qb.duplicated(['season','week','team']).any(): raise RuntimeError('duplicate QB team-week rows')
 p=pd.read_csv(one(a.wr_r1_root,'wr_r1_paired_wr_casebook.csv'),low_memory=False);p.columns=[str(c).strip().lower() for c in p.columns];w=p.loc[p.market.astype(str).str.lower().eq('rec_yards') & p.position.astype(str).str.upper().eq('WR') & num(p.season).isin([2024,2025])].copy();w['season']=num(w.season).astype(int);w['week']=num(w.week).astype(int);w['team']=w.team.map(team);w['actual']=num(w.actual_m38);w['proj']=num(w.mc_proj_m38);w['resid']=w.actual-w.proj
 rows=[]
 for k,g in w.groupby(['season','week','team'],sort=False):
  gg=g.sort_values(['proj','player'],ascending=[False,True],kind='stable');rows.append({'season':k[0],'week':k[1],'team':k[2],'wr_count':len(g),'wr_all_residual':float(g.resid.sum()),'wr_top1_projected_residual':float(gg.head(1).resid.sum()),'wr_top2_projected_residual':float(gg.head(2).resid.sum()),'wr_all_actual':float(g.actual.sum()),'wr_all_proj':float(g.proj.sum())})
 wr=pd.DataFrame(rows);x=qb.merge(wr,on=['season','week','team'],how='inner',validate='one_to_one');scores=pd.DataFrame([score(x,c) for c in ['wr_all_residual','wr_top2_projected_residual','wr_top1_projected_residual']]);r=scores.loc[scores.signal.eq('wr_all_residual')].iloc[0]
 gates={'aligned_ge700':bool(len(x)>=700),'pearson_ge_0_35':bool(r.pearson>=.35),'spearman_ge_0_30':bool(r.spearman>=.30),'same_sign_ge_0_60':bool(r.same_sign_rate>=.60),'quartile_gap_ge30':bool(r.q4_minus_q1_qb_residual_gap>=30),'positive_both_seasons':bool(r.pearson_2024>0 and r.pearson_2025>0)};disp='STRONG_QB_WR_ERROR_COUPLING' if all(gates.values()) else 'NO_STRONG_QB_WR_ERROR_COUPLING'
 out={'migration':'QB_WR_RESIDUAL_COUPLING','qb_rows':int(len(qb)),'wr_team_week_rows':int(len(wr)),'aligned_rows':int(len(x)),'gates':gates,'sportsbook_inputs_used':False,'production_changed':False,'m72_status':'aggregate pregame explosive-weapon x defense signal screen previously failed','disposition':disp}
 a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/'qb_wr_residual_coupling_casebook.csv',index=False);scores.to_csv(a.out_dir/'qb_wr_residual_coupling_metrics.csv',index=False);(a.out_dir/'qb_wr_residual_coupling_result.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2,sort_keys=True));print(scores.to_string(index=False))
if __name__=='__main__':main()
