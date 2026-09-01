#!/usr/bin/env python3
"""M95S: diagnostic decomposition of population mass vs player ranking for RB 20+ tails."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

EPS=1e-6

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-30,30); return 1/(1+np.exp(-x))

def prob_metrics(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return {
        'n':len(y),'events':int(y.sum()),'actual_rate':float(y.mean()),'mean_prob':float(p.mean()),
        'gap_actual_minus_pred':float(y.mean()-p.mean()),
        'brier':float(np.mean((p-y)**2)),
        'logloss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
    }

def mass_match(p,target_mean):
    z=logit(p)
    target=float(np.clip(target_mean,EPS,1-EPS))
    f=lambda a: float(sigmoid(z+a).mean()-target)
    a=brentq(f,-15,15)
    return sigmoid(z+a),a

def rolling_census(census):
    c=census.sort_values(['season','team','week']).copy()
    weekly=c.groupby(['season','week']).agg(
        league_week_lead20=('lead20','mean'), league_week_lead25=('lead25','mean'),
        league_week_mean_lead_carries=('lead_rb_carries','mean')
    ).reset_index().sort_values(['season','week'])
    for col in ['league_week_lead20','league_week_lead25','league_week_mean_lead_carries']:
        for w in [1,2,4]:
            weekly[f'{col}_prior{w}']=weekly.groupby('season')[col].transform(lambda s: s.shift(1).rolling(w,min_periods=1).mean())
        weekly[f'{col}_std']=weekly.groupby('season')[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    for base in ['lead20','lead25','lead_rb_carries']:
        for w in [1,2,4]:
            c[f'team_{base}_prior{w}']=c.groupby(['season','team'])[base].transform(lambda s: s.shift(1).rolling(w,min_periods=1).mean())
    return weekly,c

def corr_rows(df,scope):
    feats=[
        'league_week_lead20_prior1','league_week_lead20_prior2','league_week_lead20_prior4','league_week_lead20_std',
        'team_lead20_prior1','team_lead20_prior2','team_lead20_prior4',
        'team_lead_rb_carries_prior1','team_lead_rb_carries_prior2','team_lead_rb_carries_prior4'
    ]
    out=[]
    for f in feats:
        g=df.dropna(subset=[f,'actual_20plus','p20_base']).copy()
        if len(g)<20: continue
        resid=g.actual_20plus-g.p20_base
        rr,rp=spearmanr(g[f],resid); ar,ap=spearmanr(g[f],g.actual_20plus)
        out.append({'scope':scope,'feature':f,'n':len(g),'rho_residual':rr,'p_residual':rp,'rho_actual':ar,'p_actual':ap})
    return out

def ranking_block(frame,season,base_col,cand_col,label):
    g=frame.copy()
    b=prob_metrics(g.actual_20plus,g[base_col]); c=prob_metrics(g.actual_20plus,g[cand_col])
    pm,a=mass_match(g[cand_col],b['mean_prob']); m=prob_metrics(g.actual_20plus,pm)
    return {
        'season':season,'trace':label,'n':len(g),'events':int(g.actual_20plus.sum()),
        'base_mean':b['mean_prob'],'frozen_mean':c['mean_prob'],'massnorm_mean':m['mean_prob'],'massnorm_logit_shift':a,
        'base_auc':b['auc'],'frozen_auc':c['auc'],'massnorm_auc':m['auc'],
        'auc_gain_frozen':c['auc']-b['auc'],'auc_gain_massnorm':m['auc']-b['auc'],
        'base_brier':b['brier'],'frozen_brier':c['brier'],'massnorm_brier':m['brier'],
        'base_logloss':b['logloss'],'frozen_logloss':c['logloss'],'massnorm_logloss':m['logloss'],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--m95r-dir',type=Path,required=True)
    ap.add_argument('--m95p-dir',type=Path,required=True)
    ap.add_argument('--m95k-dir',type=Path,required=True)
    ap.add_argument('--m95l-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    panel=pd.read_csv(a.m95r_dir/'m95r_exact_panel.csv')
    census=pd.read_csv(a.m95p_dir/'m95p_team_week_broad_census_2018_2025.csv')
    weekly,c=rolling_census(census)
    panel=panel.merge(weekly,on=['season','week'],how='left',validate='many_to_one')
    tcols=['season','week','team']+[f'team_{b}_prior{w}' for b in ['lead20','lead25','lead_rb_carries'] for w in [1,2,4]]
    panel=panel.merge(c[tcols],on=['season','week','team'],how='left',validate='many_to_one')
    panel['m95f_residual20']=panel.actual_20plus-panel.p20_base

    # Season and weekly population-mass diagnostics.
    mass=[]
    for season,g in panel.groupby('season'):
        m=prob_metrics(g.actual_20plus,g.p20_base)
        mass.append({'scope':'season_available','season':int(season),'week':'ALL',**m})
    for (season,week),g in panel.groupby(['season','week']):
        m=prob_metrics(g.actual_20plus,g.p20_base)
        mass.append({'scope':'weekly','season':int(season),'week':int(week),**m})
    pd.DataFrame(mass).to_csv(a.out_dir/'m95s_population_mass_gaps.csv',index=False)

    corrs=[]
    corrs += corr_rows(panel,'exact_2020_2025_available')
    corrs += corr_rows(panel[panel.week.between(13,18)],'late_2020_2025')
    corrs += corr_rows(panel[panel.season.eq(2025)],'2025_full')
    corrdf=pd.DataFrame(corrs); corrdf.to_csv(a.out_dir/'m95s_pregame_anchor_correlations.csv',index=False)

    # 2025: was R's upward mass shift contradicted by contemporaneous information?
    r25=pd.read_csv(a.m95r_dir/'m95r_2025_full_secondary_trace.csv')
    mergecols=['season','week','team','player_clean_key','p20_candidate','candidate_logodds_delta']
    d25=panel[panel.season.eq(2025)].merge(r25[mergecols],on=['season','week','team','player_clean_key'],how='left',validate='one_to_one')
    w25=d25.groupby('week').agg(
        n=('actual_20plus','size'),events=('actual_20plus','sum'),actual_rate=('actual_20plus','mean'),
        m95f_mean=('p20_base','mean'),m95r_mean=('p20_candidate','mean'),m95r_mean_logodds_delta=('candidate_logodds_delta','mean'),
        league_prior1=('league_week_lead20_prior1','first'),league_prior2=('league_week_lead20_prior2','first'),
        league_prior4=('league_week_lead20_prior4','first'),league_std=('league_week_lead20_std','first')
    ).reset_index()
    w25['m95f_gap_actual_minus_pred']=w25.actual_rate-w25.m95f_mean
    w25['m95r_gap_actual_minus_pred']=w25.actual_rate-w25.m95r_mean
    w25['r_mass_increase']=w25.m95r_mean-w25.m95f_mean
    w25.to_csv(a.out_dir/'m95s_2025_weekly_mass_response.csv',index=False)

    # Frozen ranking decomposition in authoritative 2023/2025 traces.
    k=pd.read_csv(a.m95k_dir/'m95k_2025_trace.csv')
    k=k[k.stable_workhorse_m95k.eq(1)].copy()
    kr=ranking_block(k,2025,'p20_base','p20_m95k','M95K_authoritative')
    lpath=a.m95l_dir/'rb_m95l'/'m95l_2023_confirmation_trace.csv'
    if not lpath.exists(): lpath=a.m95l_dir/'m95l_2023_confirmation_trace.csv'
    l=pd.read_csv(lpath)
    # M95L candidate is the frozen M95K architecture on 2023.
    stable_col='stable_workhorse_m95k' if 'stable_workhorse_m95k' in l.columns else 'stable_workhorse'
    l=l[l[stable_col].eq(1)].copy()
    base_col='p20_base' if 'p20_base' in l.columns else 'cal_prob_20'
    lr=ranking_block(l,2023,base_col,'p20_m95l','M95L_frozen_M95K_on_2023')
    rdf=pd.DataFrame([lr,kr]); rdf.to_csv(a.out_dir/'m95s_mass_normalized_ranking_audit.csv',index=False)

    # 25+ sparse-event audit.
    p25=[]
    for season,g in panel.groupby('season'):
        p25.append({'season':int(season),'n':len(g),'events25':int(g.actual_25plus.sum()),'rate25':float(g.actual_25plus.mean())})
    pd.DataFrame(p25).to_csv(a.out_dir/'m95s_25plus_event_audit.csv',index=False)

    early=w25[w25.week.between(2,9)]
    early_base_over=float((early.m95f_mean-early.actual_rate).mean())
    early_r_increase=float(early.r_mass_increase.mean())
    early_l4=float(early.league_prior4.mean())
    late=w25[w25.week.between(13,18)]
    late_base_abs=float((late.actual_rate-late.m95f_mean).abs().mean())
    ranking_flip=int(np.sign(lr['auc_gain_massnorm'])!=np.sign(kr['auc_gain_massnorm']))
    r_contradicted=int((early_base_over>0.05) and (early_r_increase>0.08) and (early_l4<0.20))
    fast_anchor_signal=int(corrdf[(corrdf.scope=='exact_2020_2025_available') & corrdf.feature.isin(['league_week_lead20_prior1','league_week_lead20_prior4','league_week_lead20_std'])].rho_residual.max()>0.07)
    decomposition=int(r_contradicted and ranking_flip)
    disp={
        'm95s_role':'diagnostic_mass_ranking_decomposition','primary_target':'stable_workhorse_20plus',
        'model_fit':0,'feature_search':0,'coefficient_search':0,'sportsbook_inputs':0,'production_change':0,
        'early_2025_m95f_overprediction_mean':early_base_over,'early_2025_r_mass_increase_mean':early_r_increase,
        'early_2025_league_l4_mean':early_l4,'late_2025_m95f_weekly_abs_gap_mean':late_base_abs,
        'ranking_direction_flips_2023_vs_2025':ranking_flip,'r2025_overcorrection_visible_pregame':r_contradicted,
        'pregame_fast_population_anchor_signal_detected':fast_anchor_signal,'mass_vs_ranking_decomposition_supported':decomposition,
        'disposition':'M95S_DECOMPOSITION_SUPPORTED_ADVANCE_TO_CONSTRAINED_M95T' if decomposition else 'M95S_DIAGNOSTIC_INCONCLUSIVE_STOP_NEW_TAIL_CANDIDATES'
    }
    pd.DataFrame([disp]).to_csv(a.out_dir/'m95s_disposition.csv',index=False)
    pd.DataFrame([{
        'audit':'M95S population mass vs player ranking','anchor_grid':'league/team prior1|prior2|prior4|season_to_date',
        'ranking_mass_normalization':'common logit intercept to M95F scope mean','target_week_postgame_inputs':0,
        'model_fit':0,'feature_search':0,'coefficient_search':0,'hyperparameter_search':0,'sportsbook_inputs':0,
        'm94c_central_change':0,'m95f_baseline_change':0,'production_change':0
    }]).to_csv(a.out_dir/'m95s_method_audit.csv',index=False)

    print('\n[M95S] disposition'); print(pd.DataFrame([disp]).to_string(index=False))
    print('\n[M95S] ranking audit'); print(rdf.to_string(index=False))
    print('\n[M95S] anchor correlations'); print(corrdf.to_string(index=False))
    print('\n[M95S] 2025 weekly'); print(w25.to_string(index=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
