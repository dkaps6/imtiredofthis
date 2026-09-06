#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd
MIN_PRIOR=8;SHRINK=8.0;CAP=20.0;EXPECTED=12396

def one(root,name):
 h=list(Path(root).rglob(name));
 if len(h)!=1: raise RuntimeError(f'expected one {name}, got {len(h)}')
 return h[0]
def key(v): return re.sub(r'[^a-z0-9]','',str(v or '').lower())
def num(s): return pd.to_numeric(s,errors='coerce')
def metrics(a,p):
 z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna();e=z.p-z.a;ae=e.abs()
 return {'n':len(z),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'median_ae':float(ae.median()),'p90_ae':float(np.quantile(ae,.9)),'miss20':float(ae.ge(20).mean()),'miss30':float(ae.ge(30).mean()),'miss40':float(ae.ge(40).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wr-r1-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 p=pd.read_csv(one(a.wr_r1_root,'wr_r1_paired_wr_casebook.csv'),low_memory=False);p.columns=[str(c).strip().lower() for c in p.columns]
 q=p.loc[p.market.astype(str).str.lower().eq('rec_yards') & p.position.astype(str).str.upper().eq('WR')].copy()
 if len(q)!=EXPECTED: raise RuntimeError(f'row drift {len(q)}')
 q['season']=num(q.season).astype(int);q['week']=num(q.week).astype(int);q['player_key']=q.player_clean_key.map(key);q['actual']=num(q.actual_m38);q['base']=num(q.mc_proj_m38)
 q=q.sort_values(['player_key','season','week']).reset_index(drop=True);q['prior_n']=0;q['candidate']=np.nan;q['prior_mean_error']=np.nan;q['shrunk_bias']=np.nan
 for k,g in q.groupby('player_key',sort=False):
  errs=[]
  for idx,r in g.iterrows():
   n=len(errs);q.at[idx,'prior_n']=n
   if n>=MIN_PRIOR:
    m=float(np.mean(errs));b=float(np.clip(m*n/(n+SHRINK),-CAP,CAP));q.at[idx,'prior_mean_error']=m;q.at[idx,'shrunk_bias']=b;q.at[idx,'candidate']=float(r.base)-b
   errs.append(float(r.base-r.actual))
 e=q.loc[q.prior_n.ge(MIN_PRIOR)&q.candidate.notna()].copy();base=metrics(e.actual,e.base);cand=metrics(e.actual,e.candidate)
 ss=[]
 for s in range(2020,2026):
  g=e.loc[e.season.eq(s)];bm=metrics(g.actual,g.base);cm=metrics(g.actual,g.candidate);ss.append({'season':s,'n':len(g),'base_mae':bm['mae'],'candidate_mae':cm['mae'],'mae_delta':cm['mae']-bm['mae']})
 ss=pd.DataFrame(ss);ps=[]
 for k,g in e.groupby('player_key'):
  if len(g)>=8:
   bm=metrics(g.actual,g.base);cm=metrics(g.actual,g.candidate);ps.append({'player_key':k,'games':len(g),'base_mae':bm['mae'],'candidate_mae':cm['mae'],'mae_delta':cm['mae']-bm['mae']})
 ps=pd.DataFrame(ps);imp=(base['mae']-cand['mae'])/base['mae']
 gates={'eligible_ge6000':len(e)>=6000,'mae_improve_ge1pct':imp>=.01,'rmse_nonworse':cand['rmse']<=base['rmse']+1e-12,'median_nonworse':cand['median_ae']<=base['median_ae']+1e-12,'p90_nonworse':cand['p90_ae']<=base['p90_ae']+1e-12,'miss20_nonworse':cand['miss20']<=base['miss20']+1e-12,'miss30_nonworse':cand['miss30']<=base['miss30']+1e-12,'miss40_nonworse':cand['miss40']<=base['miss40']+1e-12,'four_of_six_seasons':int((ss.mae_delta<0).sum())>=4,'2024_improve':float(ss.loc[ss.season.eq(2024),'mae_delta'].iloc[0])<0,'2025_improve':float(ss.loc[ss.season.eq(2025),'mae_delta'].iloc[0])<0,'median_player_delta_negative':bool(len(ps) and ps.mae_delta.median()<0)}
 disp='WR_PLAYER_BIAS_PERSISTENCE_SIGNAL' if all(gates.values()) else 'NO_ACTIONABLE_WR_PLAYER_BIAS_PERSISTENCE'
 out={'migration':'WR_R3_PLAYER_BIAS_PERSISTENCE','source_rows':len(q),'eligible_rows':len(e),'players':int(q.player_key.nunique()),'qualifying_players':len(ps),'base':base,'candidate':cand,'mae_improvement_fraction':float(imp),'gates':gates,'sportsbook_inputs_used':False,'production_changed':False,'disposition':disp}
 a.out_dir.mkdir(parents=True,exist_ok=True);e.to_csv(a.out_dir/'wr_r3_player_bias_casebook.csv',index=False);ss.to_csv(a.out_dir/'wr_r3_season_metrics.csv',index=False);ps.to_csv(a.out_dir/'wr_r3_player_metrics.csv',index=False);(a.out_dir/'wr_r3_result.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2,sort_keys=True));print(ss.to_string(index=False))
if __name__=='__main__':main()
