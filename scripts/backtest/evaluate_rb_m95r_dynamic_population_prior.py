#!/usr/bin/env python3
"""M95R: precommitted dynamic population-prior adjustment for stable-workhorse 20+ carries."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

FEATURES = [
    'league_std_lead20_rate',
    'league_l4_lead20_rate',
    'team_l4_lead20',
    'team_l4_lead25',
    'team_l4_lead_rb_carries',
]
TARGET_SEASONS = [2023, 2024, 2025]
TRAIN_SEASONS = [2020, 2021, 2022, 2023, 2024]
RIDGE_LAMBDA = 10.0
DELTA_CAP = 0.75
EPS = 1e-6


def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-30,30)
    return 1/(1+np.exp(-x))

def metrics(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
    brier=float(np.mean((p-y)**2))
    ll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    actual=float(y.mean()) if len(y) else np.nan
    meanp=float(p.mean()) if len(p) else np.nan
    gap=actual-meanp
    return {'n':len(y),'events':int(y.sum()),'actual_rate':actual,'mean_prob':meanp,
            'calibration_gap_actual_minus_pred':gap,'abs_calibration_gap':abs(gap),
            'brier':brier,'logloss':ll,'auc':auc}

def prep_fit(train):
    med=train[FEATURES].median(numeric_only=True)
    X=train[FEATURES].fillna(med).astype(float)
    mu=X.mean(); sd=X.std(ddof=0).replace(0,1.0)
    Z=((X-mu)/sd).to_numpy(float)
    y=train['actual_20plus'].to_numpy(int)
    offset=logit(train['p20_base'].to_numpy(float))
    def objective(theta):
        eta=offset+theta[0]+Z.dot(theta[1:])
        p=sigmoid(eta)
        nll=-np.sum(y*np.log(np.clip(p,EPS,1-EPS))+(1-y)*np.log(np.clip(1-p,EPS,1-EPS)))
        penalty=0.5*RIDGE_LAMBDA*np.sum(theta[1:]**2)
        return nll+penalty
    res=minimize(objective,np.zeros(len(FEATURES)+1),method='L-BFGS-B')
    if not res.success:
        raise RuntimeError(f'optimizer failed: {res.message}')
    return {'theta':res.x,'median':med,'mean':mu,'std':sd,'objective':float(res.fun)}

def predict(frame,fit):
    X=frame[FEATURES].fillna(fit['median']).astype(float)
    Z=((X-fit['mean'])/fit['std']).to_numpy(float)
    raw_delta=fit['theta'][0]+Z.dot(fit['theta'][1:])
    delta=np.clip(raw_delta,-DELTA_CAP,DELTA_CAP)
    cand=sigmoid(logit(frame['p20_base'].to_numpy(float))+delta)
    return cand,delta,raw_delta

def load_panel(qdir:Path,pdir:Path):
    q=pd.read_csv(qdir/'m95q_enriched_holdouts.csv')
    q=q.loc[q['season'].isin([2020,2021,2022]) & q['stable_workhorse_m95k'].eq(1)].copy()
    q=q.rename(columns={'cal_prob_20':'p20_base'})
    q['source_trace']='m95q_reconstructed'
    census=pd.read_csv(pdir/'m95p_team_week_broad_census_2018_2025.csv')
    join_cols=['season','week','team']+FEATURES
    q=q.merge(census[join_cols].drop_duplicates(['season','week','team']),on=['season','week','team'],how='left',validate='many_to_one')
    if q[FEATURES].isna().all(axis=1).any():
        raise RuntimeError('M95Q rows missing all M95P pregame regime features')
    keep=['season','week','team','player_clean_key','player','actual_carries','actual_20plus','actual_25plus','p20_base','source_trace']+FEATURES
    q=q[keep]

    p=pd.read_csv(pdir/'m95p_exact_stable_workhorse_regime_trace.csv')
    p=p.loc[p['season'].isin([2023,2024,2025])].copy()
    if 'player' not in p.columns: p['player']=p['player_clean_key']
    if 'actual_25plus' not in p.columns: p['actual_25plus']=(p['actual_carries']>=25).astype(int)
    p=p[keep]
    panel=pd.concat([q,p],ignore_index=True,sort=False)
    panel['season']=panel['season'].astype(int); panel['week']=panel['week'].astype(int)
    if panel.duplicated(['season','week','team','player_clean_key']).any():
        raise RuntimeError('duplicate stable player-week rows in M95R panel')
    return panel.sort_values(['season','week','team','player_clean_key']).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--m95q-dir',type=Path,required=True)
    ap.add_argument('--m95p-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    panel=load_panel(args.m95q_dir,args.m95p_dir)
    panel.to_csv(args.out_dir/'m95r_exact_panel.csv',index=False)

    primary=panel.loc[panel['week'].between(13,18)].copy()
    preds=[]; coeff=[]
    for target in TARGET_SEASONS:
        train=primary.loc[(primary['season']<target) & primary['season'].isin(TRAIN_SEASONS)].copy()
        test=primary.loc[primary['season'].eq(target)].copy()
        if len(train)<100 or len(test)<20:
            raise RuntimeError(f'insufficient rolling sample target={target}: train={len(train)} test={len(test)}')
        fit=prep_fit(train)
        cand,delta,raw=predict(test,fit)
        test['p20_candidate']=cand; test['candidate_logodds_delta']=delta; test['candidate_raw_delta']=raw
        test['train_rows']=len(train); test['train_seasons']=','.join(str(x) for x in sorted(train.season.unique()))
        preds.append(test)
        row={'target_season':target,'train_rows':len(train),'train_seasons':test['train_seasons'].iloc[0],
             'ridge_lambda':RIDGE_LAMBDA,'delta_cap':DELTA_CAP,'objective':fit['objective'],'intercept':fit['theta'][0]}
        row.update({f'beta_{f}':b for f,b in zip(FEATURES,fit['theta'][1:])}); coeff.append(row)
    oof=pd.concat(preds,ignore_index=True)
    oof.to_csv(args.out_dir/'m95r_rolling_oof_trace.csv',index=False)
    pd.DataFrame(coeff).to_csv(args.out_dir/'m95r_rolling_coefficients.csv',index=False)

    rows=[]
    for season,g in oof.groupby('season'):
        b=metrics(g.actual_20plus,g.p20_base); c=metrics(g.actual_20plus,g.p20_candidate)
        row={'scope':'late_primary','season':int(season),**{f'base_{k}':v for k,v in b.items()},**{f'cand_{k}':v for k,v in c.items()}}
        row['brier_gain']=b['brier']-c['brier']; row['logloss_gain']=b['logloss']-c['logloss']; row['auc_gain']=c['auc']-b['auc']; row['abs_gap_gain']=b['abs_calibration_gap']-c['abs_calibration_gap']
        rows.append(row)
    b=metrics(oof.actual_20plus,oof.p20_base); c=metrics(oof.actual_20plus,oof.p20_candidate)
    pooled={'scope':'late_primary','season':'POOLED',**{f'base_{k}':v for k,v in b.items()},**{f'cand_{k}':v for k,v in c.items()}}
    pooled['brier_gain']=b['brier']-c['brier']; pooled['logloss_gain']=b['logloss']-c['logloss']; pooled['auc_gain']=c['auc']-b['auc']; pooled['abs_gap_gain']=b['abs_calibration_gap']-c['abs_calibration_gap']; rows.append(pooled)

    tr=primary.loc[primary['season']<=2024].copy(); te=panel.loc[panel['season'].eq(2025)].copy(); fit=prep_fit(tr)
    cand,delta,raw=predict(te,fit); te['p20_candidate']=cand; te['candidate_logodds_delta']=delta; te['candidate_raw_delta']=raw
    te.to_csv(args.out_dir/'m95r_2025_full_secondary_trace.csv',index=False)
    b25=metrics(te.actual_20plus,te.p20_base); c25=metrics(te.actual_20plus,te.p20_candidate)
    row={'scope':'2025_full_secondary','season':2025,**{f'base_{k}':v for k,v in b25.items()},**{f'cand_{k}':v for k,v in c25.items()}}
    row['brier_gain']=b25['brier']-c25['brier']; row['logloss_gain']=b25['logloss']-c25['logloss']; row['auc_gain']=c25['auc']-b25['auc']; row['abs_gap_gain']=b25['abs_calibration_gap']-c25['abs_calibration_gap']; rows.append(row)
    metrics_df=pd.DataFrame(rows); metrics_df.to_csv(args.out_dir/'m95r_metrics.csv',index=False)

    reg=[]
    for season,g in oof.groupby('season'):
        gg=g.copy(); gg['delta_quartile']=pd.qcut(gg['candidate_logodds_delta'].rank(method='first'),4,labels=['Q1_low','Q2','Q3','Q4_high'])
        for qn,h in gg.groupby('delta_quartile',observed=True):
            bm=metrics(h.actual_20plus,h.p20_base); cm=metrics(h.actual_20plus,h.p20_candidate)
            reg.append({'season':int(season),'delta_quartile':str(qn),'n':len(h),'events':int(h.actual_20plus.sum()),'mean_delta':h.candidate_logodds_delta.mean(),'actual_rate':h.actual_20plus.mean(),'base_mean':h.p20_base.mean(),'cand_mean':h.p20_candidate.mean(),'base_brier':bm['brier'],'cand_brier':cm['brier']})
    pd.DataFrame(reg).to_csv(args.out_dir/'m95r_regime_slices.csv',index=False)

    case=oof.assign(abs_shift=(oof.p20_candidate-oof.p20_base).abs(),prob_shift=oof.p20_candidate-oof.p20_base).sort_values('abs_shift',ascending=False).head(30)
    case[['season','week','team','player','player_clean_key','actual_carries','actual_20plus','p20_base','p20_candidate','prob_shift','candidate_logodds_delta']+FEATURES].to_csv(args.out_dir/'m95r_casebook.csv',index=False)

    d25=[]
    for season,g in primary.groupby('season'):
        d25.append({'season':int(season),'n':len(g),'events25':int(g.actual_25plus.sum()),'actual25_rate':g.actual_25plus.mean()})
    pd.DataFrame(d25).to_csv(args.out_dir/'m95r_25plus_event_audit.csv',index=False)

    season_rows=metrics_df[(metrics_df.scope=='late_primary') & (metrics_df.season!='POOLED')].copy()
    pooled_row=metrics_df[(metrics_df.scope=='late_primary') & (metrics_df.season=='POOLED')].iloc[0]
    brier_wins=int((season_rows.brier_gain>0).sum()); gap_wins=int((season_rows.abs_gap_gain>0).sum())
    max_brier_reg=float(np.maximum(-season_rows.brier_gain,0).max()); max_gap_reg=float(np.maximum(-season_rows.abs_gap_gain,0).max())
    gates={
        'pooled_brier_improves':int(pooled_row.brier_gain>0),
        'pooled_logloss_improves':int(pooled_row.logloss_gain>0),
        'pooled_auc_guard':int(pooled_row.auc_gain>=-0.02),
        'season_brier_wins_ge2':int(brier_wins>=2),
        'season_abs_gap_wins_ge2':int(gap_wins>=2),
        'max_season_brier_regression_le_0p01':int(max_brier_reg<=0.01),
        'max_season_abs_gap_regression_le_0p025':int(max_gap_reg<=0.025),
    }
    passed=all(gates.values())
    disp={'m95r_role':'dynamic_population_prior_candidate','primary_target':'stable_workhorse_20plus','rolling_targets':'2023,2024,2025','feature_search':0,'hyperparameter_search':0,'coefficient_fit':1,'sportsbook_inputs':0,'production_change':0,'ridge_lambda':RIDGE_LAMBDA,'delta_cap':DELTA_CAP,'season_brier_wins':brier_wins,'season_abs_gap_wins':gap_wins,'max_season_brier_regression':max_brier_reg,'max_season_abs_gap_regression':max_gap_reg,**gates,'disposition':'M95R_ADVANCE_TO_PROSPECTIVE_CONFIRMATION' if passed else 'M95R_RETAIN_DIAGNOSTIC_DO_NOT_PROMOTE'}
    pd.DataFrame([disp]).to_csv(args.out_dir/'m95r_disposition.csv',index=False)
    pd.DataFrame([{'candidate':'M95F logit backbone + additive ridge population-prior adjustment','features':'|'.join(FEATURES),'ridge_lambda':RIDGE_LAMBDA,'delta_cap':DELTA_CAP,'training':'strict earlier-season expanding window; 2020-22 seed 2023','primary_scope':'W13-18 stable workhorses','feature_search':0,'hyperparameter_search':0,'sportsbook_inputs':0,'target_week_postgame_inputs':0,'m94c_central_change':0,'nonstable_change':0,'vacancy_change':0}]).to_csv(args.out_dir/'m95r_method_audit.csv',index=False)

    print('\n[M95R] metrics'); print(metrics_df.to_string(index=False))
    print('\n[M95R] disposition'); print(pd.DataFrame([disp]).to_string(index=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
