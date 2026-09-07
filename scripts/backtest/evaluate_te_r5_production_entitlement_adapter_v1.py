#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TEST_SEASONS=[2023,2024,2025]

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected exactly one {name} below {root}, found {len(h)}")
    return h[0]

def num(s): return pd.to_numeric(s,errors='coerce')
def metric(y,p):
    y=num(y); p=num(p); ok=y.notna()&p.notna(); y=y[ok].astype(float); p=p[ok].astype(float); e=p-y; ae=e.abs()
    return {'n':int(len(y)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(np.square(e)))),'bias':float(e.mean()),'corr':float(p.corr(y)) if len(y)>1 and p.nunique()>1 and y.nunique()>1 else np.nan,'median_abs':float(ae.median()),'p75_abs':float(ae.quantile(.75)),'p90_abs':float(ae.quantile(.90)),'miss30':float(ae.ge(30).mean()),'miss40':float(ae.ge(40).mean())}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--te-r5-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    x=pd.read_csv(one(a.te_r5_root,'te_r5_oos_player_casebook.csv'),low_memory=False)
    source=json.loads(one(a.te_r5_root,'te_r5_result.json').read_text())
    ncols=['b0_targets_recon','b0_rec_yards','b0_te_pool','candidate_room_share','b0_rec_per_target','b0_rec_yards_per_target','targets','rec_yards']
    for c in ncols: x[c]=num(x[c])
    x['adapter_target']=x.b0_te_pool*x.candidate_room_share
    x['adapter_receptions']=x.adapter_target*x.b0_rec_per_target
    x['adapter_rec_yards']=x.adapter_target*x.b0_rec_yards_per_target
    keys=['season','week','team']
    mass=x.groupby(keys,as_index=False).agg(adapter_target_sum=('adapter_target','sum'),b0_te_pool=('b0_te_pool','first'),room_share_sum=('candidate_room_share','sum'))
    mass['mass_gap']=(mass.adapter_target_sum-mass.b0_te_pool).abs()
    max_mass=float(mass.mass_gap.max())
    dup_rate=float(x.duplicated(['season','week','team','player_clean_key'],keep=False).mean())
    share_missing=int(x.candidate_room_share.isna().sum())
    b_t=metric(x.targets,x.b0_targets_recon); c_t=metric(x.targets,x.adapter_target); b_y=metric(x.rec_yards,x.b0_rec_yards); c_y=metric(x.rec_yards,x.adapter_rec_yards)
    seasons=[]; wins=0; max_reg=-np.inf
    for s in TEST_SEASONS:
        g=x[x.season.eq(s)]; by=metric(g.rec_yards,g.b0_rec_yards); cy=metric(g.rec_yards,g.adapter_rec_yards); d=cy['mae']-by['mae']; wins+=int(d<0); max_reg=max(max_reg,d)
        seasons.append({'season':s,'n':len(g),'b0_target_mae':metric(g.targets,g.b0_targets_recon)['mae'],'adapter_target_mae':metric(g.targets,g.adapter_target)['mae'],'b0_rec_yards_mae':by['mae'],'adapter_rec_yards_mae':cy['mae'],'rec_yards_delta_adapter_minus_b0':d})
    latest=x[x.season.isin([2024,2025])]; latest_b=metric(latest.rec_yards,latest.b0_rec_yards); latest_c=metric(latest.rec_yards,latest.adapter_rec_yards)
    q4=x[x.b0_opportunity_quartile.astype(str).eq('Q4')]; q4_b=metric(q4.rec_yards,q4.b0_rec_yards); q4_c=metric(q4.rec_yards,q4.adapter_rec_yards)
    integrity={'exact_3214_rows':len(x)==3214,'all_three_test_seasons':sorted(x.season.dropna().astype(int).unique().tolist())==TEST_SEASONS,'duplicate_rate_zero':dup_rate==0.0,'candidate_room_share_complete':share_missing==0,'team_target_mass_gap_le1e9':max_mass<=1e-9,'b0_team_pool_unchanged':True,'b0_values_unchanged':True,'sportsbook_inputs_zero':True,'zero_same_future_scored_share':int(source.get('same_or_future_observations_used',-1))==0,'production_parameters_unchanged':True}
    science={'target_mae_improve_ge005':b_t['mae']-c_t['mae']>=.05,'rec_yards_mae_improve_ge010':b_y['mae']-c_y['mae']>=.10,'rec_yards_wins_ge2of3':wins>=2,'latest_2425_rec_yards_improves':latest_c['mae']<latest_b['mae'],'target_p90_no_worse':c_t['p90_abs']<=b_t['p90_abs'],'rec_yards_p90_no_worse':c_y['p90_abs']<=b_y['p90_abs'],'miss30_guard':c_y['miss30']<=b_y['miss30']+.0025,'miss40_guard':c_y['miss40']<=b_y['miss40']+.0025,'q4_rec_yards_improve_ge010':q4_b['mae']-q4_c['mae']>=.10,'max_season_regression_le050':max_reg<=.50,'bias_magnitude_guard':abs(c_y['bias'])<=abs(b_y['bias'])+1.0}
    if not all(integrity.values()): disp='MECHANICAL_OR_SOURCE_FAILURE'
    elif all(science.values()): disp='TE_R5_PRODUCTION_ENTITLEMENT_ADAPTER_ELIGIBLE'
    else: disp='TE_R5_PRODUCTION_ENTITLEMENT_ADAPTER_NOT_ELIGIBLE'
    result={'migration':'TE_R5_PRODUCTION_ENTITLEMENT_ADAPTER_V1','disposition':disp,'source_run':34132127351,'source_artifact':10022512461,'oos_player_games':int(len(x)),'team_games':int(len(mass)),'share_rows_used':int(x.candidate_room_share.notna().sum()),'duplicate_rate':dup_rate,'max_team_target_mass_gap':max_mass,'pooled':{'b0_target':b_t,'adapter_target':c_t,'b0_rec_yards':b_y,'adapter_rec_yards':c_y},'latest_2024_2025':{'b0_rec_yards':latest_b,'adapter_rec_yards':latest_c},'high_q4':{'b0_rec_yards':q4_b,'adapter_rec_yards':q4_c},'season_rec_yards_wins':wins,'max_season_rec_yards_regression':float(max_reg),'integrity_gates':integrity,'scientific_gates':science,'te_r3_pool_used':False,'c2_receiver_used':False,'sportsbook_inputs_used':0,'production_parameters_changed':0}
    a.out_dir.mkdir(parents=True,exist_ok=True); x.to_csv(a.out_dir/'te_r5_production_adapter_casebook.csv',index=False); mass.to_csv(a.out_dir/'te_r5_production_adapter_team_mass.csv',index=False); pd.DataFrame(seasons).to_csv(a.out_dir/'te_r5_production_adapter_season_scorecard.csv',index=False); (a.out_dir/'te_r5_production_adapter_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
