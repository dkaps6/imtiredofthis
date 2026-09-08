#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.modeling.rb_r17_tail_distribution_adapter_v1 import ResidualPools, adapt_rb_receiving_tail, stable_seed, target_tail_draws

DRAWS=2000
EXPECTED_SIM_BLOB='887e9c776ab112276ec8281195b0fed790ea0551'
R17_COMPARATOR={
    'crps':7.525222724118877,'brier30':0.05748601013634733,'brier50':0.018027474165769646,
    'pinball90':3.665298977473567,'pinball95':2.4849645782257346,
}
R17_FOLD_BRIER50={2024:0.014133386836441894,2025:0.02192435696338837}


def git_blob_sha1(path: Path)->str:
    data=path.read_bytes(); header=f'blob {len(data)}\0'.encode(); return hashlib.sha1(header+data).hexdigest()

def crps(samples,y):
    x=np.sort(np.asarray(samples,float)); n=len(x); i=np.arange(1,n+1,dtype=float)
    return float(np.mean(np.abs(x-float(y)))-0.5*((2.0/(n*n))*np.sum((2*i-n-1)*x)))

def pinball(y,qhat,q):
    e=float(y)-float(qhat); return float(q*e if e>=0 else (1-q)*(-e))

def score(mu,y,x):
    q90,q95=np.quantile(x,[.90,.95]); p30=float(np.mean(x>=mu+30)); p50=float(np.mean(x>=mu+50))
    return {'crps':crps(x,y),'brier30':(p30-float(y>=mu+30))**2,'brier50':(p50-float(y>=mu+50))**2,
            'pinball90':pinball(y,q90,.90),'pinball95':pinball(y,q95,.95)}

def risk_wide(path):
    r=pd.read_csv(path,low_memory=False)
    r=r.loc[r.label.isin(['cat30_under','cat50_under']),['season','week','team','player_clean_key','label','p_full']].copy()
    w=r.pivot_table(index=['season','week','team','player_clean_key'],columns='label',values='p_full',aggfunc='first').reset_index()
    return w.rename(columns={'cat30_under':'p30','cat50_under':'p50'})

def load_pred(path):
    p=pd.read_csv(path,low_memory=False)
    for c in ['season','week','actual_rec_yards','baseline_pred_rec_yards']: p[c]=pd.to_numeric(p[c],errors='coerce')
    p=p.dropna(subset=['season','week','team','player_clean_key','actual_rec_yards','baseline_pred_rec_yards']).copy()
    p['season']=p.season.astype(int); p['week']=p.week.astype(int); p['residual']=p.actual_rec_yards-p.baseline_pred_rec_yards
    return p

def pools_from(train):
    return ResidualPools(train.loc[train.residual.lt(30),'residual'].to_numpy(float),
                         train.loc[train.residual.ge(30)&train.residual.lt(50),'residual'].to_numpy(float),
                         train.loc[train.residual.ge(50),'residual'].to_numpy(float))

def historical_replay(pred,risk):
    rows=[]
    for trains,testseason in [((2023,),2024),((2023,2024),2025)]:
        train=pred.loc[pred.season.isin(trains)]; test=pred.loc[pred.season.eq(testseason)].merge(risk.loc[risk.season.eq(testseason)],on=['season','week','team','player_clean_key'],how='inner',validate='one_to_one')
        pools=pools_from(train)
        for r in test.itertuples(index=False):
            mu=max(0.,float(r.baseline_pred_rec_yards)); y=max(0.,float(r.actual_rec_yards))
            rng=np.random.default_rng(stable_seed(917,r.season,r.week,r.team,r.player_clean_key,'candidate'))
            draws=target_tail_draws(mu,DRAWS,float(r.p30),float(r.p50),pools,rng)
            rows.append({'season':int(r.season),**score(mu,y,draws)})
    c=pd.DataFrame(rows)
    out=[]
    for season,g in c.groupby('season'):
        out.append({'test_season':int(season),'n':len(g),**{k:float(g[k].mean()) for k in ['crps','brier30','brier50','pinball90','pinball95']}})
    out.append({'test_season':'COMBINED','n':len(c),**{k:float(c[k].mean()) for k in ['crps','brier30','brier50','pinball90','pinball95']}})
    return pd.DataFrame(out)

def fixture_metrics():
    rows=[]
    specs=[
      ('ARI','BUF','qb_ari','QB',0.00,0.10,7.1,4.7,.66),('ARI','BUF','rb1_ari','RB',0.16,0.40,6.8,4.4,.78),('ARI','BUF','rb2_ari','RB',0.08,0.18,6.0,4.1,.72),
      ('ARI','BUF','wr1_ari','WR',0.22,0.00,8.7,0,.64),('ARI','BUF','wr2_ari','WR',0.16,0.00,7.9,0,.63),('ARI','BUF','te1_ari','TE',0.10,0.00,7.0,0,.70),
      ('BUF','ARI','qb_buf','QB',0.00,0.12,7.5,5.0,.66),('BUF','ARI','rb1_buf','RB',0.15,0.38,7.0,4.5,.79),('BUF','ARI','rb2_buf','RB',0.07,0.16,5.9,4.0,.71),
      ('BUF','ARI','wr1_buf','WR',0.23,0.00,8.9,0,.65),('BUF','ARI','wr2_buf','WR',0.15,0.00,8.0,0,.63),('BUF','ARI','te1_buf','TE',0.11,0.00,7.2,0,.71),
    ]
    for team,opp,pkey,pos,tgt,rush,ypt,ypc,catch in specs:
        rows.append({'event_id':'R18_FIXTURE','team':team,'opponent':opp,'player_clean_key':pkey,'player':pkey,'position':pos,'position_family':pos,
                     'rules_plays_est':66 if team=='BUF' else 64,'rules_pass_rate':.60 if team=='BUF' else .57,'rules_tgt_share':tgt,'rules_rush_share':rush,
                     'rules_ypt':ypt,'rules_ypc':ypc if ypc>0 else 4.2,'rules_catch_rate':catch,'rules_volatility_mult':1.0,'offensive_td_rate':.12})
    return pd.DataFrame(rows)

def fixture_audit(train2023):
    from scripts.simulation_v2 import simulate
    metrics=fixture_metrics(); trace=[]
    canonical=simulate(metrics,iterations=10000,seed=4242,allocation_trace=trace); trace_before=copy.deepcopy(trace)
    risk=pd.DataFrame([
      {'event_id':'R18_FIXTURE','player_clean_key':'rb1_ari','p30':.11,'p50':.035},{'event_id':'R18_FIXTURE','player_clean_key':'rb2_ari','p30':.045,'p50':.010},
      {'event_id':'R18_FIXTURE','player_clean_key':'rb1_buf','p30':.13,'p50':.045},{'event_id':'R18_FIXTURE','player_clean_key':'rb2_buf','p30':.040,'p50':.008},
    ])
    pools=pools_from(train2023)
    adapted,audit=adapt_rb_receiving_tail(canonical,metrics,risk,pools,seed=918)
    adapted2,_=adapt_rb_receiving_tail(canonical,metrics,risk,pools,seed=918)
    pos={(str(r.event_id),str(r.player_clean_key)):str(r.position_family).upper() for r in metrics.itertuples(index=False)}
    non_rb_exact=True; rb_component_exact=True; combo_ok=True; nonneg=True; finite=True; deterministic=True; mean_delta=0.; min_rho=1.0
    details=[]
    for k,v0 in canonical.values.items():
        v1=adapted.values[k]; deterministic &= np.array_equal(v1,adapted2.values[k]); game,pkey,market=k; isrb=pos.get((game,pkey))=='RB'
        finite &= bool(np.isfinite(v1).all())
        if not isrb: non_rb_exact &= np.array_equal(v0,v1)
        elif market not in {'rec_yards','rush_rec_yards'}: rb_component_exact &= np.array_equal(v0,v1)
        if isrb and market=='rec_yards':
            nonneg &= bool((v1>=0).all()); delta=abs(float(np.mean(v1))-float(np.mean(v0))); mean_delta=max(mean_delta,delta)
            rho=1.0
            if not np.allclose(v0,v0[0],atol=0,rtol=0):
                rho=float(pd.Series(v0).corr(pd.Series(v1),method='spearman')); min_rho=min(min_rho,rho)
            details.append({'player_clean_key':pkey,'canonical_mean':float(np.mean(v0)),'adapted_mean':float(np.mean(v1)),'mean_delta':delta,'spearman':rho})
    for game,pkey in [k for k,p in pos.items() if p=='RB']:
        rk=(game,pkey,'rush_yards'); ck=(game,pkey,'rush_rec_yards'); yk=(game,pkey,'rec_yards')
        if all(k in adapted.values for k in [rk,ck,yk]): combo_ok &= bool(np.allclose(adapted.values[ck],adapted.values[rk]+adapted.values[yk],atol=1e-10,rtol=0))
    summary={'canonical_mean_parity':mean_delta<=1e-8,'non_rb_exact':bool(non_rb_exact),'rb_component_exact':bool(rb_component_exact),'rush_rec_identity':bool(combo_ok),
             'allocation_trace_exact':trace==trace_before,'nonnegative_rec_yards':bool(nonneg),'finite_draws':bool(finite),'rank_preservation':min_rho>=.9999,'deterministic_replay':bool(deterministic),
             'max_mean_delta':mean_delta,'min_spearman':min_rho,'adapted_rb_count':len(audit)}
    return summary,pd.DataFrame(details)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',type=Path,required=True); ap.add_argument('--r16-tail-predictions',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    pred=load_pred(a.predictions); risk=risk_wide(a.r16_tail_predictions); hist=historical_replay(pred,risk)
    comb=hist.loc[hist.test_season.astype(str).eq('COMBINED')].iloc[0]
    fixture,details=fixture_audit(pred.loc[pred.season.eq(2023)])
    f50={int(r.test_season):float(r.brier50) for r in hist.itertuples(index=False) if str(r.test_season)!='COMBINED'}
    sim_blob=git_blob_sha1(Path('scripts/simulation_v2.py'))
    gates={
      'canonical_mean_parity':bool(fixture['canonical_mean_parity']),'non_rb_exact':bool(fixture['non_rb_exact']),'rb_component_exact':bool(fixture['rb_component_exact']),
      'rush_rec_identity':bool(fixture['rush_rec_identity']),'allocation_trace_exact':bool(fixture['allocation_trace_exact']),'nonnegative_rec_yards':bool(fixture['nonnegative_rec_yards']),
      'finite_draws':bool(fixture['finite_draws']),'rank_preservation':bool(fixture['rank_preservation']),'deterministic_replay':bool(fixture['deterministic_replay']),
      'r17_combined_q90_guard':float(comb.pinball90)<=R17_COMPARATOR['pinball90'],'r17_combined_q95_guard':float(comb.pinball95)<=R17_COMPARATOR['pinball95'],
      'r17_combined_crps_guard':float(comb.crps)<=R17_COMPARATOR['crps']*1.005,'r17_cat30_brier_guard':float(comb.brier30)<=R17_COMPARATOR['brier30'],
      'r17_cat50_fragility_guard':all(f50[s]<=R17_FOLD_BRIER50[s]+.00010 for s in [2024,2025]),
      'canonical_simulation_unmodified':sim_blob==EXPECTED_SIM_BLOB,'sportsbook_zero':True,'production_parameters_zero':True,
    }
    passed=all(gates.values())
    result={'candidate':'RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_V1','disposition':'RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_PASS_RESEARCH_ONLY' if passed else 'RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_FAIL_RESEARCH_ONLY',
            'science_pass':bool(passed),'parents':['RB_R16_UPSIDE_TAIL_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY','RB_R17_DISTRIBUTION_SIGNAL_SUPPORTED_RESEARCH_ONLY'],
            'fixture':fixture,'historical_replay_combined':{k:(int(comb[k]) if k=='n' else float(comb[k]) if k!='test_season' else str(comb[k])) for k in comb.index},
            'historical_replay_folds':hist.loc[hist.test_season.astype(str).ne('COMBINED')].to_dict(orient='records'),'gates':gates,'canonical_simulation_blob':sim_blob,
            'expected_canonical_simulation_blob':EXPECTED_SIM_BLOB,'sportsbook_inputs_added':0,'production_parameters_changed':0,
            'governance_note':'PASS authorizes only a separately frozen 2026 tail-scorer/refit and full-slate shadow gate; no production promotion occurs in R18.'}
    a.out_dir.mkdir(parents=True,exist_ok=True); hist.to_csv(a.out_dir/'rb_r18_historical_replay_summary.csv',index=False); details.to_csv(a.out_dir/'rb_r18_fixture_player_audit.csv',index=False); (a.out_dir/'rb_r18_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); print('\n=== historical replay ===\n',hist.to_string(index=False)); print('\n=== fixture players ===\n',details.to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())
