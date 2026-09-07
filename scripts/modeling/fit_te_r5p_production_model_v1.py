#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS=.02; ALPHA=20.0
FEATURES=['b0_te_room_share','log_b0_te_pool','pool_ratio','room_size','prior1_same_team_offense_pct','prior1_same_team_offense_snaps','prior1_anyteam_offense_pct','prior3_anyteam_offense_pct','prior1_anyteam_offense_snaps','prior3_anyteam_offense_snaps','log1p_prior_count_same_team','log1p_prior_count_anyteam','prior1_same_team_available','prior3_same_team_available','snap_share_prior1_same_team','snap_share_prior3_anyteam']

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f'expected one {name} below {root}, found {len(h)}')
    return h[0]
def num(s): return pd.to_numeric(s,errors='coerce')

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--te-r3-root',type=Path,required=True); ap.add_argument('--te-r4-root',type=Path,required=True); ap.add_argument('--te-r5p-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    upstream=json.loads(one(a.te_r5p_root,'te_r5p_result.json').read_text())
    if upstream.get('disposition')!='TE_R5P_PRODUCTION_CONTRACT_REFIT_ELIGIBLE': raise RuntimeError('upstream TE-R5P not eligible')
    r3=pd.read_csv(one(a.te_r3_root,'te_r3_oos_player_casebook.csv'),low_memory=False); r4=pd.read_csv(one(a.te_r4_root,'te_r4_prior_participation_casebook.csv'),low_memory=False); r4res=json.loads(one(a.te_r4_root,'te_r4_result.json').read_text())
    r3['player_key']=r3.player_clean_key.fillna('').astype(str); r4['player_key']=r4.player_key.fillna('').astype(str)
    x=r3.merge(r4,on=['season','week','team','player_key'],how='left',suffixes=('','_r4'),indicator=True,validate='one_to_one')
    for c in ['prior1_same_team_offense_pct','prior1_same_team_offense_snaps','prior1_anyteam_offense_pct','prior3_anyteam_offense_pct','prior1_anyteam_offense_snaps','prior3_anyteam_offense_snaps','prior_count_same_team','prior_count_anyteam','b0_te_room_share','b0_te_pool','actual_te_pool','targets']:
        x[c]=num(x[c])
    x['prior1_same_team_available']=x.prior1_same_team.fillna(False).astype(float); x['prior3_same_team_available']=x.prior3_same_team.fillna(False).astype(float)
    x['log1p_prior_count_same_team']=np.log1p(x.prior_count_same_team.fillna(0).clip(lower=0)); x['log1p_prior_count_anyteam']=np.log1p(x.prior_count_anyteam.fillna(0).clip(lower=0)); x['log_b0_te_pool']=np.log1p(x.b0_te_pool.clip(lower=0)); x['pool_ratio']=1.0; x['room_size']=x.groupby(['season','week','team']).player_key.transform('count').astype(float)
    for src,dst in [('prior1_same_team_offense_pct','snap_share_prior1_same_team'),('prior3_anyteam_offense_pct','snap_share_prior3_anyteam')]:
        z=x[src].fillna(0).clip(lower=0); den=z.groupby([x.season,x.week,x.team]).transform('sum'); x[dst]=np.where(den>0,z/den,0.0)
    for c in FEATURES: x[c]=num(x[c]).fillna(0.0)
    x['actual_room_share']=np.where(x.actual_te_pool.gt(0),x.targets/x.actual_te_pool,0.0); x['target']=(np.log(x.actual_room_share+EPS)-np.log(x.b0_te_room_share.clip(lower=0)+EPS)).clip(-2,2)
    tr=x[x.season.between(2022,2025)&x.actual_te_pool.gt(0)&x._merge.eq('both')].copy()
    model=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA)); model.fit(tr[FEATURES],tr.target); sc=model.named_steps['standardscaler']; rg=model.named_steps['ridge']
    seasons=sorted(tr.season.unique().astype(int).tolist()); dup=float(tr.duplicated(['season','week','team','player_key'],keep=False).mean())
    finite=bool(np.isfinite(sc.mean_).all() and np.isfinite(sc.scale_).all() and np.isfinite(rg.coef_).all() and np.isfinite(rg.intercept_)); integrity={'upstream_eligible':True,'training_rows_gt0':len(tr)>0,'training_seasons_exact_2022_2025':seasons==[2022,2023,2024,2025],'duplicate_rate_zero':dup==0.0,'zero_same_future_participation':int(r4res.get('same_or_future_observations_used',-1))==0,'pool_ratio_exactly_one':bool(np.allclose(tr.pool_ratio.to_numpy(float),1.0,rtol=0,atol=0)),'te_r3_candidate_pool_used':False,'sportsbook_inputs_zero':True,'no_2026_outcomes':int(tr.season.max())<=2025,'parameter_lengths_16':len(sc.mean_)==len(sc.scale_)==len(rg.coef_)==16,'parameters_finite':finite}
    contract={'model_version':'TE_R5P_PRODUCTION_MODEL_V1','authorized_by_run':34152797603,'authorized_by_artifact':10029942404,'source_te_r3_run':34126813280,'source_te_r4_run':34127474412,'training_seasons':seasons,'training_rows':int(len(tr)),'features':FEATURES,'eps':EPS,'ridge_alpha':ALPHA,'training_target_clip':[-2.0,2.0],'prediction_clip':[-1.0,1.0],'pool_ratio_contract':1.0,'candidate_team_pool':'b0_te_pool','scaler_mean':[float(v) for v in sc.mean_],'scaler_scale':[float(v) for v in sc.scale_],'ridge_coef':[float(v) for v in rg.coef_],'ridge_intercept':float(rg.intercept_),'feature_construction':{'log_b0_te_pool':'log1p(max(b0_te_pool,0))','room_size':'count of current team-game TE rows','availability_flags':'strict-prior TE-R4 booleans as 0/1','log_prior_counts':'log1p(strict-prior counts)','snap_share_prior1_same_team':'prior1 same-team offense_pct normalized within current TE room; 0 if denominator 0','snap_share_prior3_anyteam':'prior3 any-team offense_pct normalized within current TE room; 0 if denominator 0'},'allocation':'clip model residual [-1,1]; score=log(b0_te_room_share+0.02)+residual; team softmax; candidate_targets=b0_te_pool*share','integrity':integrity,'sportsbook_inputs_used':0}
    a.out_dir.mkdir(parents=True,exist_ok=True); (a.out_dir/'te_r5p_production_model_v1.json').write_text(json.dumps(contract,indent=2,sort_keys=True)); tr[['season','week','team','player_key','b0_te_room_share','b0_te_pool','actual_te_pool','targets','target']+FEATURES].to_csv(a.out_dir/'te_r5p_final_fit_training_audit.csv',index=False); pd.DataFrame({'feature':FEATURES,'scaler_mean':sc.mean_,'scaler_scale':sc.scale_,'ridge_coef':rg.coef_}).assign(ridge_intercept=float(rg.intercept_)).to_csv(a.out_dir/'te_r5p_final_fit_parameters.csv',index=False); result={'disposition':'TE_R5P_FINAL_PRODUCTION_MODEL_FIT_COMPLETE' if all(integrity.values()) else 'MECHANICAL_OR_SOURCE_FAILURE','training_rows':int(len(tr)),'training_seasons':seasons,'integrity':integrity}; (a.out_dir/'te_r5p_final_fit_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(contract,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
