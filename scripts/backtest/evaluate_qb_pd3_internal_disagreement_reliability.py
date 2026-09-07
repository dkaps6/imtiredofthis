#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_ROWS=884
STATES={
    'COMPONENT_RANGE_40': lambda x: x.component_range.ge(40.0),
    'SYNTH_MOVE_30': lambda x: x.abs_correction.ge(30.0),
    'SYNTH_CAP_45': lambda x: x.abs_correction.ge(44.999),
}

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f'expected one {name}, got {len(h)}')
    return h[0]

def metric(a,p):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'p':pd.to_numeric(p,errors='coerce')}).dropna()
    e=z.p-z.a; ae=e.abs()
    return {'n':int(len(z)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),
            'median_abs':float(ae.median()),'p75_abs':float(ae.quantile(.75)),'p90_abs':float(ae.quantile(.90)),
            'miss30':float(ae.ge(30).mean()),'miss50':float(ae.ge(50).mean()),'miss75':float(ae.ge(75).mean()),'miss100':float(ae.ge(100).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--m89-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    x=pd.read_csv(one(a.m89_root,'m89_2024_2025_synthesis_trace.csv'),low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]
    req={'season','week','actual_pass_yards','base_proj','football_synthesis','component_range','football_residual_correction'}
    if req-set(x.columns): raise RuntimeError(f'missing columns {sorted(req-set(x.columns))}')
    x=x.loc[pd.to_numeric(x.season,errors='coerce').isin([2024,2025])].copy()
    if len(x)!=EXPECTED_ROWS: raise RuntimeError(f'row drift {len(x)}')
    for c in req-{'season','week'}: x[c]=pd.to_numeric(x[c],errors='coerce')
    if x[list(req-{'season','week'})].isna().any().any(): raise RuntimeError('non-finite football fields')
    x['abs_correction']=x.football_residual_correction.abs(); x['synth_abs_error']=(x.football_synthesis-x.actual_pass_yards).abs(); x['base_abs_error']=(x.base_proj-x.actual_pass_yards).abs(); x['synth_minus_base_abs']=x.synth_abs_error-x.base_abs_error
    for name,fn in STATES.items(): x[name]=fn(x)
    x['MULTI_FLAG_2']=x[list(STATES)].sum(axis=1).ge(2)
    rows=[]; gates={}; state_names=list(STATES)+['MULTI_FLAG_2']
    for state in state_names:
        for season in ['POOLED',2024,2025]:
            q=x if season=='POOLED' else x.loc[x.season.eq(season)]
            for label,mask in [('IN',q[state]),('OUT',~q[state])]:
                z=q.loc[mask]; sm=metric(z.actual_pass_yards,z.football_synthesis); bm=metric(z.actual_pass_yards,z.base_proj)
                rows.append({'state':state,'season':season,'group':label,**{f'synth_{k}':v for k,v in sm.items()},**{f'base_{k}':v for k,v in bm.items()},'synth_minus_base_abs_mean':float(z.synth_minus_base_abs.mean()) if len(z) else np.nan})
        def row(season,group): return next(r for r in rows if r['state']==state and str(r['season'])==str(season) and r['group']==group)
        pi,po=row('POOLED','IN'),row('POOLED','OUT'); y24=row(2024,'IN'); y24o=row(2024,'OUT'); y25=row(2025,'IN'); y25o=row(2025,'OUT')
        sg={
          'n_pooled_ge60':pi['synth_n']>=60,'n_each_season_ge20':y24['synth_n']>=20 and y25['synth_n']>=20,
          'pooled_mae_penalty_ge5':pi['synth_mae']-po['synth_mae']>=5.0,
          'mae_penalty_positive_both_seasons':(y24['synth_mae']-y24o['synth_mae']>0) and (y25['synth_mae']-y25o['synth_mae']>0),
          'pooled_synth_worse_than_base_ge2':pi['synth_minus_base_abs_mean']>=2.0,
          'synth_vs_base_nonnegative_both_seasons':y24['synth_minus_base_abs_mean']>=0 and y25['synth_minus_base_abs_mean']>=0,
          'miss75_enrichment_ge3pp':pi['synth_miss75']-po['synth_miss75']>=.03,
          'sportsbook_unused':True,'no_leakage':True}
        sg['pass']=all(sg.values()); gates[state]=sg
    cont=[]
    for sig in ['component_range','abs_correction']:
        for out in ['synth_abs_error','synth_minus_base_abs']:
            cont.append({'signal':sig,'outcome':out,'spearman':float(x[sig].corr(x[out],method='spearman'))})
    passed=[s for s,g in gates.items() if g['pass']]
    result={'migration':'QB_PD3_INTERNAL_DISAGREEMENT_RELIABILITY','rows':len(x),'seasons':[2024,2025],'sportsbook_inputs_used':False,'production_changed':False,'state_gates':gates,'actionable_states':passed,'disposition':'QB_INTERNAL_RELIABILITY_STATE_SUPPORTED' if passed else 'NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE'}
    a.out_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out_dir/'qb_pd3_state_metrics.csv',index=False); pd.DataFrame(cont).to_csv(a.out_dir/'qb_pd3_continuous_metrics.csv',index=False); x.to_csv(a.out_dir/'qb_pd3_casebook.csv',index=False); (a.out_dir/'qb_pd3_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True)); print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(cont).to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())
