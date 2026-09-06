#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

MIN_PRIOR=6; SHRINK=8.0; CAP=25.0; EXPECTED=884

def one(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name}, got {len(h)}")
    return h[0]
def key(v): return re.sub(r'[^a-z0-9]','',str(v or '').lower())
def num(s): return pd.to_numeric(s,errors='coerce')
def metrics(a,p):
    z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna(); e=z.p-z.a; ae=e.abs()
    return {'n':int(len(z)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'median_ae':float(ae.median()),'p90_ae':float(np.quantile(ae,.9)),'miss30':float(ae.ge(30).mean()),'miss50':float(ae.ge(50).mean()),'miss75':float(ae.ge(75).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--m89-root',required=True,type=Path); ap.add_argument('--out-dir',required=True,type=Path); a=ap.parse_args()
    x=pd.read_csv(one(a.m89_root,'m89_2024_2025_synthesis_trace.csv'),low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]
    req={'season','week','player_clean_key','actual_pass_yards','football_synthesis'}
    if req-set(x.columns): raise RuntimeError(f"missing {sorted(req-set(x.columns))}")
    x=x.loc[num(x.season).isin([2024,2025])].copy()
    if len(x)!=EXPECTED: raise RuntimeError(f"row drift {len(x)}")
    x['season']=num(x.season).astype(int); x['week']=num(x.week).astype(int); x['player_key']=x.player_clean_key.map(key); x['actual']=num(x.actual_pass_yards); x['base']=num(x.football_synthesis)
    x=x.sort_values(['player_key','season','week']).reset_index(drop=True)
    x['prior_n']=0; x['prior_mean_error']=np.nan; x['shrunk_bias']=np.nan; x['candidate']=np.nan
    for k,g in x.groupby('player_key',sort=False):
        errs=[]
        for idx,r in g.iterrows():
            n=len(errs); x.at[idx,'prior_n']=n
            if n>=MIN_PRIOR:
                m=float(np.mean(errs)); b=m*n/(n+SHRINK); b=float(np.clip(b,-CAP,CAP))
                x.at[idx,'prior_mean_error']=m; x.at[idx,'shrunk_bias']=b; x.at[idx,'candidate']=float(r.base)-b
            errs.append(float(r.base-r.actual))
    e=x.loc[x.prior_n.ge(MIN_PRIOR)&x.candidate.notna()].copy()
    base=metrics(e.actual,e.base); cand=metrics(e.actual,e.candidate)
    season_rows=[]
    for s in [2024,2025]:
        g=e.loc[e.season.eq(s)]; bm=metrics(g.actual,g.base); cm=metrics(g.actual,g.candidate)
        season_rows.append({'season':s,'n':len(g),'base_mae':bm['mae'],'candidate_mae':cm['mae'],'mae_delta':cm['mae']-bm['mae']})
    p_rows=[]
    for k,g in e.groupby('player_key'):
        if len(g)>=8:
            bm=metrics(g.actual,g.base); cm=metrics(g.actual,g.candidate); p_rows.append({'player_key':k,'games':len(g),'base_mae':bm['mae'],'candidate_mae':cm['mae'],'mae_delta':cm['mae']-bm['mae']})
    ps=pd.DataFrame(p_rows); ss=pd.DataFrame(season_rows)
    improve=(base['mae']-cand['mae'])/base['mae'] if base['mae'] else -np.inf
    gates={
      'eligible_ge400':len(e)>=400,
      'mae_improve_ge1pct':improve>=.01,
      'rmse_nonworse':cand['rmse']<=base['rmse']+1e-12,
      'median_nonworse':cand['median_ae']<=base['median_ae']+1e-12,
      'p90_nonworse':cand['p90_ae']<=base['p90_ae']+1e-12,
      'miss30_nonworse':cand['miss30']<=base['miss30']+1e-12,
      'miss50_nonworse':cand['miss50']<=base['miss50']+1e-12,
      'both_seasons_improve':bool((ss.mae_delta<0).all()),
      'median_player_delta_negative':bool(len(ps) and ps.mae_delta.median()<0),
    }
    disp='PERSISTENT_PLAYER_BIAS_EVIDENCE' if all(gates.values()) else 'NO_ACTIONABLE_PLAYER_BIAS_PERSISTENCE'
    out={'migration':'QB_PLAYER_BIAS_PERSISTENCE','rows_source':len(x),'eligible_rows':len(e),'players_source':int(x.player_key.nunique()),'qualifying_player_profiles':int(len(ps)),'base':base,'candidate':cand,'mae_improvement_fraction':float(improve),'gates':gates,'sportsbook_inputs_used':False,'production_changed':False,'disposition':disp}
    a.out_dir.mkdir(parents=True,exist_ok=True); e.to_csv(a.out_dir/'qb_player_bias_casebook.csv',index=False); ss.to_csv(a.out_dir/'qb_player_bias_season_metrics.csv',index=False); ps.to_csv(a.out_dir/'qb_player_bias_player_metrics.csv',index=False); (a.out_dir/'qb_player_bias_result.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    print(json.dumps(out,indent=2,sort_keys=True)); print(ss.to_string(index=False)); print(ps.sort_values('mae_delta').head(20).to_string(index=False) if len(ps) else 'no player rows')
if __name__=='__main__': main()
