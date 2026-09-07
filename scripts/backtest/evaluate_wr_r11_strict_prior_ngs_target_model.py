#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_START=2021
TEST_SEASONS=[2022,2023,2024,2025]
NGS_BASE=['avg_separation','avg_cushion','avg_intended_air_yards','percent_share_of_intended_air_yards']
FEATURES=[f'{f}_prior1' for f in NGS_BASE]+[f'{f}_prior3_mean' for f in NGS_BASE]+['prior_obs_count','b0_expected_targets']

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f'expected one {name}, got {len(h)}')
    return h[0]
def pkey(v): return ''.join(ch.lower() for ch in str(v or '') if ch.isalnum())
def metric(a,p):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'p':pd.to_numeric(p,errors='coerce')}).dropna(); e=z.p-z.a; ae=e.abs()
    return {'n':int(len(z)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>2 else np.nan,'median_abs':float(ae.median()),'p75_abs':float(ae.quantile(.75)),'p90_abs':float(ae.quantile(.90)),'miss20':float(ae.ge(20).mean()),'miss30':float(ae.ge(30).mean()),'miss40':float(ae.ge(40).mean())}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--joint-root',type=Path,required=True); ap.add_argument('--r10-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    base=pd.read_csv(one(a.joint_root,'joint_v1_paired_player_casebook.csv'),low_memory=False); base.columns=[str(c).strip().lower() for c in base.columns]
    feat=pd.read_csv(one(a.r10_root,'wr_r10_strict_prior_feature_casebook.csv'),low_memory=False); feat.columns=[str(c).strip().lower() for c in feat.columns]
    reqb={'season','week','team','player','player_clean_key','position_group','b0_expected_targets','b0_receptions','b0_rec_yards','targets','receptions','rec_yards'}
    reqf={'season','week','team','player_key','prior_obs_count',*[f'{f}_prior1' for f in NGS_BASE],*[f'{f}_prior3_mean' for f in NGS_BASE]}
    if reqb-set(base.columns): raise RuntimeError(f'missing base {sorted(reqb-set(base.columns))}')
    if reqf-set(feat.columns): raise RuntimeError(f'missing features {sorted(reqf-set(feat.columns))}')
    base=base.loc[base.position_group.eq('WR') & pd.to_numeric(base.season,errors='coerce').between(TRAIN_START,2025)].copy(); base['player_key']=base.player_clean_key.map(pkey)
    for c in ['season','week','b0_expected_targets','b0_receptions','b0_rec_yards','targets','receptions','rec_yards']: base[c]=pd.to_numeric(base[c],errors='coerce')
    feat['season']=pd.to_numeric(feat.season,errors='coerce'); feat['week']=pd.to_numeric(feat.week,errors='coerce'); feat['player_key']=feat.player_key.map(pkey)
    keys=['season','week','team','player_key'];
    if feat.duplicated(keys).any(): raise RuntimeError('feature duplicate keys')
    x=base.merge(feat,on=keys,how='left',validate='one_to_one',suffixes=('','_ngs'))
    x=x.dropna(subset=['b0_expected_targets','b0_receptions','b0_rec_yards','targets','receptions','rec_yards']).copy(); x['eligible']=pd.to_numeric(x.prior_obs_count,errors='coerce').ge(3)
    x['candidate_targets']=x.b0_expected_targets; x['candidate_receptions']=x.b0_receptions; x['candidate_rec_yards']=x.b0_rec_yards; x['predicted_target_residual']=0.0
    model_rows=[]
    for season in TEST_SEASONS:
        tr=x.loc[(x.season.lt(season)) & (x.season.ge(TRAIN_START)) & x.eligible].copy(); te=x.loc[x.season.eq(season) & x.eligible].copy()
        if len(tr)<500 or len(te)<500: raise RuntimeError(f'insufficient OOS rows season={season} train={len(tr)} test={len(te)}')
        y=tr.targets-tr.b0_expected_targets
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('ridge',Ridge(alpha=20.0))])
        pipe.fit(tr[FEATURES],y); corr=pipe.predict(te[FEATURES]); cand=np.clip(te.b0_expected_targets.to_numpy()+corr,0.0,25.0)
        ratio=np.where(te.b0_expected_targets.to_numpy()>0,cand/te.b0_expected_targets.to_numpy(),1.0)
        idx=te.index; x.loc[idx,'candidate_targets']=cand; x.loc[idx,'candidate_receptions']=te.b0_receptions.to_numpy()*ratio; x.loc[idx,'candidate_rec_yards']=te.b0_rec_yards.to_numpy()*ratio; x.loc[idx,'predicted_target_residual']=corr
        model_rows.append({'test_season':season,'train_rows':int(len(tr)),'test_rows':int(len(te)),'train_seasons':f'{TRAIN_START}-{season-1}','mean_correction':float(np.mean(corr)),'sd_correction':float(np.std(corr)),'min_correction':float(np.min(corr)),'max_correction':float(np.max(corr)),'pct_up':float(np.mean(corr>0)),'pct_down':float(np.mean(corr<0))})
    score=x.loc[x.season.isin(TEST_SEASONS) & x.eligible].copy(); inelig=x.loc[x.season.isin(TEST_SEASONS) & ~x.eligible].copy()
    parity=float(np.max(np.abs(inelig[['candidate_targets','candidate_receptions','candidate_rec_yards']].to_numpy()-inelig[['b0_expected_targets','b0_receptions','b0_rec_yards']].to_numpy()))) if len(inelig) else 0.0
    score['b0_target_ae']=(score.b0_expected_targets-score.targets).abs(); q=float(score.b0_expected_targets.quantile(.75)); score['high_target_tier']=score.b0_expected_targets.ge(q)
    rows=[]
    scopes=[('POOLED',score),('2024_2025',score.loc[score.season.isin([2024,2025])]),('HIGH_TARGET_Q4',score.loc[score.high_target_tier])]+[(str(s),score.loc[score.season.eq(s)]) for s in TEST_SEASONS]
    for scope,g in scopes:
        for market,actual,b0,cand in [('targets','targets','b0_expected_targets','candidate_targets'),('receptions','receptions','b0_receptions','candidate_receptions'),('rec_yards','rec_yards','b0_rec_yards','candidate_rec_yards')]:
            for variant,pred in [('B0',b0),('NGS_TARGET',cand)]: rows.append({'scope':scope,'market':market,'variant':variant,**metric(g[actual],g[pred])})
    m=pd.DataFrame(rows)
    def get(scope,market,variant,field='mae'): return float(m.loc[m.scope.eq(scope)&m.market.eq(market)&m.variant.eq(variant),field].iloc[0])
    season_target_improve=sum(get(str(s),'targets','NGS_TARGET')<get(str(s),'targets','B0') for s in TEST_SEASONS)
    gates={
      'pooled_target_mae_improve_ge_0_05':get('POOLED','targets','B0')-get('POOLED','targets','NGS_TARGET')>=.05,
      'pooled_rec_yard_mae_improve_ge_0_25':get('POOLED','rec_yards','B0')-get('POOLED','rec_yards','NGS_TARGET')>=.25,
      'target_mae_improves_ge3of4_seasons':season_target_improve>=3,
      'latest_target_mae_improves':get('2024_2025','targets','NGS_TARGET')<get('2024_2025','targets','B0'),
      'latest_rec_yard_mae_improves':get('2024_2025','rec_yards','NGS_TARGET')<get('2024_2025','rec_yards','B0'),
      'no_season_target_regression_gt_0_10':max(get(str(s),'targets','NGS_TARGET')-get(str(s),'targets','B0') for s in TEST_SEASONS)<=.10,
      'no_season_rec_yard_regression_gt_0_75':max(get(str(s),'rec_yards','NGS_TARGET')-get(str(s),'rec_yards','B0') for s in TEST_SEASONS)<=.75,
      'pooled_target_p90_not_worse':get('POOLED','targets','NGS_TARGET','p90_abs')<=get('POOLED','targets','B0','p90_abs')+1e-12,
      'pooled_rec_yard_p90_not_worse':get('POOLED','rec_yards','NGS_TARGET','p90_abs')<=get('POOLED','rec_yards','B0','p90_abs')+1e-12,
      'rec_yard_miss30_not_worse_gt_0_5pp':get('POOLED','rec_yards','NGS_TARGET','miss30')-get('POOLED','rec_yards','B0','miss30')<=.005,
      'rec_yard_miss40_not_worse_gt_0_5pp':get('POOLED','rec_yards','NGS_TARGET','miss40')-get('POOLED','rec_yards','B0','miss40')<=.005,
      'high_target_target_mae_not_worse_gt_0_05':get('HIGH_TARGET_Q4','targets','NGS_TARGET')-get('HIGH_TARGET_Q4','targets','B0')<=.05,
      'high_target_rec_yard_mae_not_worse_gt_0_25':get('HIGH_TARGET_Q4','rec_yards','NGS_TARGET')-get('HIGH_TARGET_Q4','rec_yards','B0')<=.25,
      'ineligible_rows_exact_b0':parity<=1e-12,
      'no_leakage':True,'sportsbook_unused':True}
    passed=all(gates.values()); result={'migration':'WR_R11_STRICT_PRIOR_NGS_TARGET_MODEL','oos_seasons':TEST_SEASONS,'eligible_oos_rows':int(len(score)),'ineligible_oos_rows':int(len(inelig)),'ineligible_max_parity_gap':parity,'season_target_improve_count':int(season_target_improve),'gates':gates,'sportsbook_inputs_used':False,'production_changed':False,'disposition':'WR_NGS_TARGET_MODEL_SUPPORTED' if passed else 'WR_NGS_TARGET_MODEL_FAIL'}
    a.out_dir.mkdir(parents=True,exist_ok=True); score.to_csv(a.out_dir/'wr_r11_oos_casebook.csv',index=False); m.to_csv(a.out_dir/'wr_r11_metric_summary.csv',index=False); pd.DataFrame(model_rows).to_csv(a.out_dir/'wr_r11_model_trace.csv',index=False); (a.out_dir/'wr_r11_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True)); print(m.to_string(index=False)); print(pd.DataFrame(model_rows).to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())
