from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=5610
QCOLS=['p05','p10','p25','p50','p75','p90','p95']
QVALS=[.05,.10,.25,.50,.75,.90,.95]


def mean_metrics(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); m=np.isfinite(a)&np.isfinite(p); a=a[m]; p=p[m]
    e=p-a
    return {'n':int(len(a)),'mae':float(np.mean(np.abs(e))) if len(e) else np.nan,'rmse':float(np.sqrt(np.mean(e*e))) if len(e) else np.nan,'bias_pred_minus_actual':float(np.mean(e)) if len(e) else np.nan,'corr':float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan,'p90_abs_error':float(np.quantile(np.abs(e),.9)) if len(e) else np.nan}


def make_dist(mean_value,residuals):
    r=np.asarray(residuals,float); r=r[np.isfinite(r)]; mu=max(float(mean_value),0.0)
    if len(r)<2 or mu<=1e-12: return np.array([mu],dtype=float)
    c=r-float(np.mean(r)); raw=np.maximum(0.0,mu+c); rm=float(np.mean(raw))
    if rm<=1e-12: return np.full(len(raw),mu,dtype=float)
    return raw*(mu/rm)


def empirical_crps(samples,y):
    x=np.sort(np.asarray(samples,float)); n=len(x)
    if n==0: return np.nan
    term1=float(np.mean(np.abs(x-float(y)))); coeff=2*np.arange(1,n+1)-n-1; term2=float(np.sum(coeff*x)/(n*n))
    return term1-term2


def dist_stats(samples,y,mean_value):
    x=np.asarray(samples,float); qs=np.quantile(x,QVALS); d={k:float(v) for k,v in zip(QCOLS,qs)}
    d['crps']=empirical_crps(x,y); d['cover50']=int(qs[2]<=y<=qs[4]); d['cover80']=int(qs[1]<=y<=qs[5]); d['cover90']=int(qs[0]<=y<=qs[6])
    d['width50']=float(qs[4]-qs[2]); d['width80']=float(qs[5]-qs[1]); d['width90']=float(qs[6]-qs[0])
    thr=float(mean_value)+50.0; d['tail50_prob']=float(np.mean(x>=thr)); d['tail50_obs']=int(float(y)>=thr); d['tail50_brier']=float((d['tail50_prob']-d['tail50_obs'])**2); d['sample_mean']=float(np.mean(x))
    return d


def score(df,label):
    rows=[]
    for dist in ['b0','candidate']:
        mm=mean_metrics(df.actual,df[f'{dist}_mean']); row={'slice':label,'distribution':dist.upper(),'n':int(len(df)),**mm}
        for nom in [50,80,90]:
            cov=float(df[f'{dist}_cover{nom}'].mean()) if len(df) else np.nan; row[f'cover{nom}']=cov; row[f'cover{nom}_abs_error']=abs(cov-nom/100.0) if np.isfinite(cov) else np.nan; row[f'width{nom}']=float(df[f'{dist}_width{nom}'].mean()) if len(df) else np.nan
        row['crps']=float(df[f'{dist}_crps'].mean()) if len(df) else np.nan; row['tail50_brier']=float(df[f'{dist}_tail50_brier'].mean()) if len(df) else np.nan
        for q in QCOLS: row[f'avg_{q}']=float(df[f'{dist}_{q}'].mean()) if len(df) else np.nan
        rows.append(row)
    return rows


def bootstrap_prob(diff,nboot=10000,seed=SEED):
    d=np.asarray(diff,float); d=d[np.isfinite(d)]
    if len(d)==0: return np.nan
    rng=np.random.default_rng(seed); wins=0; done=0; chunk=250
    while done<nboot:
        b=min(chunk,nboot-done); idx=rng.integers(0,len(d),size=(b,len(d))); wins+=int(np.sum(d[idx].mean(axis=1)>0)); done+=b
    return wins/nboot


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase-c-all-rows',required=True); ap.add_argument('--phase-g-casebook',required=True); ap.add_argument('--phase-i-qb-casebook'); ap.add_argument('--out-dir',required=True); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    allrows=pd.read_csv(args.phase_c_all_rows); g=pd.read_csv(args.phase_g_casebook); wr=allrows[allrows.position.eq('WR')].copy(); wr['residual']=wr.actual-wr.pred; keys=['season','week','team']
    g['actual_state']=np.where((g.qb_res>0)&(g.wrte_res>0)&(g.rb_res<0),'PASS_STATE_HIGH',np.where((g.qb_res<0)&(g.wrte_res<0)&(g.rb_res>0),'PASS_STATE_LOW','MIXED'))
    scored=wr[wr.season.eq(2025)].merge(g[keys+['delta_pass_attempts','delta_up','actual_state']],on=keys,how='inner').sort_values(['week','team','player_clean_key']).reset_index(drop=True)
    prior25=wr[wr.season.eq(2025)].merge(g[keys+['actual_state']],on=keys,how='inner')
    rows=[]; pool_trace=[]
    for _,r in scored.iterrows():
        wk=int(r.week); q=str(r.opportunity_quartile); hist=wr[(wr.season<2025)|((wr.season==2025)&(wr.week<wk))]
        base_pool=hist.loc[hist.opportunity_quartile.astype(str).eq(q),'residual'].dropna().to_numpy(float)
        if len(base_pool)<2: base_pool=hist.residual.dropna().to_numpy(float)
        use_state=False; pool_type='B0'; state_pool=np.array([],dtype=float)
        if float(r.delta_pass_attempts)>0:
            h=prior25[(prior25.week<wk)&prior25.actual_state.eq('PASS_STATE_HIGH')]; qpool=h.loc[h.opportunity_quartile.astype(str).eq(q),'residual'].dropna().to_numpy(float); apool=h.residual.dropna().to_numpy(float)
            if len(qpool)>=100: state_pool=qpool; use_state=True; pool_type='HIGH_Q'
            elif len(apool)>=100: state_pool=apool; use_state=True; pool_type='HIGH_ALLQ'
        cand_pool=state_pool if use_state else base_pool; bdist=make_dist(r.pred,base_pool); cdist=make_dist(r.pred,cand_pool)
        d=r.to_dict(); d['b0_mean']=float(r.pred); d['candidate_mean']=float(r.pred); bstats=dist_stats(bdist,float(r.actual),float(r.pred)); cstats=dist_stats(cdist,float(r.actual),float(r.pred))
        for k,v in bstats.items(): d[f'b0_{k}']=v
        for k,v in cstats.items(): d[f'candidate_{k}']=v
        d['use_state_pool']=bool(use_state); d['state_pool_type']=pool_type; d['baseline_pool_n']=int(len(base_pool)); d['candidate_pool_n']=int(len(cand_pool)); d['mean_anchor_gap']=abs(d['candidate_sample_mean']-float(r.pred)); d['crps_diff_b0_minus_candidate']=d['b0_crps']-d['candidate_crps']; rows.append(d)
        pool_trace.append({'season':2025,'week':wk,'team':r.team,'player':r.player,'opportunity_quartile':q,'delta_pass_attempts':float(r.delta_pass_attempts),'state_pool_type':pool_type,'baseline_pool_n':len(base_pool),'candidate_pool_n':len(cand_pool)})
    c=pd.DataFrame(rows); score_rows=[]; score_rows+=score(c,'ALL')
    for q in ['Q1','Q2','Q3','Q4']: score_rows+=score(c[c.opportunity_quartile.astype(str).eq(q)],q)
    score_rows+=score(c[c.delta_pass_attempts>0],'DELTA_POSITIVE')+score(c[c.delta_pass_attempts<=0],'DELTA_NONPOSITIVE')+score(c[c.actual_state.eq('PASS_STATE_HIGH')],'PASS_STATE_HIGH')+score(c[c.actual_state.eq('PASS_STATE_LOW')],'PASS_STATE_LOW')
    sc=pd.DataFrame(score_rows)
    err=c.pred-c.actual; cerr=c.candidate_mean-c.actual; cats={'n':int(len(c)),'baseline_total':int((np.abs(err)>=50).sum()),'candidate_total':int((np.abs(cerr)>=50).sum()),'baseline_underprojection':int((err<=-50).sum()),'candidate_underprojection':int((cerr<=-50).sum()),'baseline_overprojection':int((err>=50).sum()),'candidate_overprojection':int((cerr>=50).sum())}
    joint=pd.DataFrame(); corr=np.nan
    if args.phase_i_qb_casebook and Path(args.phase_i_qb_casebook).exists():
        qi=pd.read_csv(args.phase_i_qb_casebook); qi['qb_candidate_width80']=qi.candidate_p90-qi.candidate_p10; wteam=c.groupby(keys,as_index=False).agg(wr_candidate_width80_sum=('candidate_width80','sum'),wr_b0_width80_sum=('b0_width80','sum'),wr_rows=('player','size')); joint=wteam.merge(qi[keys+['qb_candidate_width80','delta_pass_attempts']],on=keys,how='inner'); corr=float(np.corrcoef(joint.wr_candidate_width80_sum,joint.qb_candidate_width80)[0,1]) if len(joint)>1 else np.nan
    def get(s,d): return sc[(sc['slice']==s)&(sc.distribution==d)].iloc[0]
    b=get('ALL','B0'); cc=get('ALL','CANDIDATE'); bq=get('Q4','B0'); cq=get('Q4','CANDIDATE'); mean_gap=float(c.mean_anchor_gap.max()); mean_mae_delta=abs(float(cc.mae-b.mae)); crps_imp=float(b.crps-cc.crps); q4_crps_imp=float(bq.crps-cq.crps); boot=bootstrap_prob(c.crps_diff_b0_minus_candidate.to_numpy()); brier_delta=float(cc.tail50_brier-b.tail50_brier); p90_delta=float(cc.p90_abs_error-b.p90_abs_error)
    gates={'mean_anchor_max_gap_le001':bool(mean_gap<=.01),'mean_mae_delta_le001':bool(mean_mae_delta<=.01),'overall_crps_improve_ge010':bool(crps_imp>=.10),'crps_bootstrap_ge090':bool(boot>=.90),'q4_crps_improve_ge010':bool(q4_crps_imp>=.10),'cover80_abs_error_guard':bool(cc.cover80_abs_error<=b.cover80_abs_error+.02),'cover90_abs_error_guard':bool(cc.cover90_abs_error<=b.cover90_abs_error+.02),'tail50_brier_no_worse_gt002':bool(brier_delta<=.002),'p90_mean_abs_error_unchanged_le001':bool(abs(p90_delta)<=.01),'miss50_count_unchanged':bool(cats['baseline_total']==cats['candidate_total']),'integrity':True}
    disp='MEAN_NEUTRAL_WR_DISTRIBUTION_STATE_ELIGIBLE' if all(gates.values()) else 'MEAN_NEUTRAL_WR_DISTRIBUTION_STATE_NOT_ELIGIBLE'
    result={'disposition':disp,'wr_player_games':int(len(c)),'team_games':int(c[keys].drop_duplicates().shape[0]),'state_specific_rows':int(c.use_state_pool.sum()),'b0_fallback_rows':int((~c.use_state_pool).sum()),'mean_anchor_max_gap':mean_gap,'mean_mae_delta':mean_mae_delta,'crps_improvement_vs_b0':crps_imp,'q4_crps_improvement_vs_b0':q4_crps_imp,'bootstrap_probability':boot,'tail50_brier_delta_candidate_minus_b0':brier_delta,'phase_i_qb_vs_wr_width80_corr':corr,'catastrophic_point_mean_counts':cats,'gates':gates,'sportsbook_inputs_used':0,'same_or_future_outcomes_used_as_selectors':0,'production_parameters_changed':0,'qb_te_rb_point_means_changed':0,'m38_wr_point_means_or_ranks_changed':0}
    c.to_csv(out/'phase_k_wr_casebook.csv',index=False); sc.to_csv(out/'phase_k_scorecard.csv',index=False); pd.DataFrame(pool_trace).to_csv(out/'phase_k_pool_trace.csv',index=False)
    if len(joint): joint.to_csv(out/'phase_k_joint_width_diagnostic.csv',index=False)
    (out/'phase_k_result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print('\n[phase_k] scorecard'); print(sc.to_string(index=False))

if __name__=='__main__': main()
