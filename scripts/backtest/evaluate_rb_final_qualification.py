#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TOL=1e-8
EXPECTED={
    'football_rows':1393,
    'p3_all_yards_mae':19.94952397834036,
    'p3_all_yards_rmse':28.86651928636813,
    'market_rows':899,
    'p3_market_mae':24.315798244183124,
    'vegas_market_mae':23.701890989988875,
    'p3_market_rmse':33.34291378702183,
    'vegas_market_rmse':32.493543467503315,
}

def num(s): return pd.to_numeric(s,errors='coerce')
def read_one(root:Path,name:str):
    hits=list(root.rglob(name))
    if len(hits)!=1: raise RuntimeError(f'expected one {name} under {root}, found {len(hits)}')
    d=pd.read_csv(hits[0],low_memory=False); d.columns=[str(c).strip().lower() for c in d.columns]; return d

def metric(y,p):
    y=num(y);p=num(p);ok=y.notna()&p.notna();y=y[ok].astype(float);p=p[ok].astype(float)
    if len(y)==0: return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'corr':np.nan,'median_abs_error':np.nan,'p75_abs_error':np.nan,'p90_abs_error':np.nan,'over_rate':np.nan,'under_rate':np.nan,'exact_rate':np.nan}
    e=p-y;ae=e.abs()
    return {'n':int(len(y)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.square(e).mean())),'bias':float(e.mean()),'corr':float(np.corrcoef(p,y)[0,1]) if len(y)>1 and y.std()>0 and p.std()>0 else np.nan,'median_abs_error':float(ae.median()),'p75_abs_error':float(ae.quantile(.75)),'p90_abs_error':float(ae.quantile(.90)),'over_rate':float((e>0).mean()),'under_rate':float((e<0).mean()),'exact_rate':float((e.abs()<1e-12).mean())}

def add_p3(x):
    z=x.copy();w=num(z['week']);z['p3_yards']=np.where(w.eq(1),num(z['stack_yards']),num(z['arch_enriched_opp_stack_eff_yards']));z['p3_att']=np.where(w.eq(1),num(z['stack_att']),num(z['enriched_att']));return z
ARMS={'M94C':('m94c_att','m94c_yards'),'FULL_STACK':('stack_att','stack_yards'),'STACK2_ENRICHED':('enriched_att','arch_enriched_opp_stack_eff_yards'),'P3':('p3_att','p3_yards')}

def time_masks(x):
    w=num(x.week);return {'ALL':pd.Series(True,index=x.index),'W1':w.eq(1),'W2_5':w.between(2,5),'W6_12':w.between(6,12),'W13_18':w.between(13,18)}
def grade_masks(x):
    a=num(x.actual_rush_att);y=num(x.actual_rush_yards);out=time_masks(x);out.update({'ACT_CARRY_GE10':a.ge(10),'ACT_CARRY_GE15':a.ge(15),'ACT_CARRY_GE20':a.ge(20),'ACT_CARRY_GE25':a.ge(25),'ACT_YARDS_GE50':y.ge(50),'ACT_YARDS_GE75':y.ge(75),'ACT_YARDS_GE100':y.ge(100)});return out

def score_players(x):
    rows=[]
    for sl,mask in grade_masks(x).items():
        q=x.loc[mask]
        for arm,(ac,yc) in ARMS.items():
            rows.append({'slice':sl,'arm':arm,'stat':'carries',**metric(q.actual_rush_att,q[ac])});rows.append({'slice':sl,'arm':arm,'stat':'rush_yards',**metric(q.actual_rush_yards,q[yc])})
    return pd.DataFrame(rows)

def depth_scores(x):
    cov=float(num(x.depth_rank).notna().mean()) if 'depth_rank' in x else 0.0;rows=[]
    if 'depth_rank' not in x or cov<.80:return pd.DataFrame([{'depth_slice':'DEPTH_SCORING_NOT_RUN','depth_coverage':cov}])
    d=num(x.depth_rank);masks={'RB1':d.eq(1),'RB2':d.eq(2),'SECONDARY':d.ge(3),'DEPTH_UNKNOWN':d.isna()}
    for sl,mask in masks.items():
        q=x.loc[mask]
        for arm,(ac,yc) in ARMS.items():
            rows.append({'depth_slice':sl,'depth_coverage':cov,'arm':arm,'stat':'carries',**metric(q.actual_rush_att,q[ac])});rows.append({'depth_slice':sl,'depth_coverage':cov,'arm':arm,'stat':'rush_yards',**metric(q.actual_rush_yards,q[yc])})
    return pd.DataFrame(rows)

def state_scores(x):
    rows=[]
    for state in ['state_m95f_risk','state_vacancy','state_m95i_tail']:
        if state not in x:continue
        v=num(x[state]).fillna(0)
        for val,label in [(1,'ON'),(0,'OFF')]:
            q=x.loc[v.eq(val)]
            for arm,(ac,yc) in ARMS.items():
                rows.append({'state':state,'state_value':label,'arm':arm,'stat':'carries',**metric(q.actual_rush_att,q[ac])});rows.append({'state':state,'state_value':label,'arm':arm,'stat':'rush_yards',**metric(q.actual_rush_yards,q[yc])})
    return pd.DataFrame(rows)

def market_eval(m):
    z=m.copy();z['p3_yards_formula']=np.where(num(z.week).eq(1),num(z.stack_yards),num(z.arch_enriched_opp_stack_eff_yards));z['p3_att_formula']=np.where(num(z.week).eq(1),num(z.stack_att),num(z.enriched_att));z['p3_yards']=num(z.parent_yards);z['p3_att']=num(z.parent_att);z['vegas']=num(z.consensus_line);z['actual']=num(z.actual_rush_yards)
    z['p3_abs_err']=(z.p3_yards-z.actual).abs();z['vegas_abs_err']=(z.vegas-z.actual).abs();z['p3_minus_vegas']=z.p3_yards-z.vegas;z['abs_disagreement']=z.p3_minus_vegas.abs();z['closer_outcome']=np.select([z.p3_abs_err<z.vegas_abs_err,z.p3_abs_err>z.vegas_abs_err],['P3','VEGAS'],default='TIE');z['directional_eligible']=z.p3_minus_vegas.abs().gt(1e-12);z['directional_success']=np.where(z.p3_minus_vegas.gt(0),z.actual.gt(z.vegas),np.where(z.p3_minus_vegas.lt(0),z.actual.lt(z.vegas),np.nan))
    bins=[-np.inf,2.5,5,7.5,10,np.inf];labels=['LT2_5','2_5_TO_5','5_TO_7_5','7_5_TO_10','GE10'];z['edge_bin']=pd.cut(z.abs_disagreement,bins=bins,labels=labels,right=False)
    overall=[]
    for arm,col in [('P3','p3_yards'),('VEGAS','vegas')]:overall.append({'arm':arm,**metric(z.actual,z[col])})
    o=pd.DataFrame(overall);closer={'p3_closer_n':int((z.closer_outcome=='P3').sum()),'vegas_closer_n':int((z.closer_outcome=='VEGAS').sum()),'tie_n':int((z.closer_outcome=='TIE').sum()),'p3_closer_rate':float((z.closer_outcome=='P3').mean()),'vegas_closer_rate':float((z.closer_outcome=='VEGAS').mean()),'tie_rate':float((z.closer_outcome=='TIE').mean()),'directional_n':int(z.directional_eligible.sum()),'directional_accuracy':float(z.loc[z.directional_eligible,'directional_success'].astype(float).mean()),'zero_disagreement_n':int((~z.directional_eligible).sum())}
    for k,v in closer.items():o[k]=v
    br=[]
    for b,g in z.groupby('edge_bin',observed=False):
        pm=metric(g.actual,g.p3_yards);vm=metric(g.actual,g.vegas);eligible=g.directional_eligible;br.append({'edge_bin':str(b),'n':len(g),'p3_mae':pm['mae'],'vegas_mae':vm['mae'],'mae_diff_vegas_minus_p3':vm['mae']-pm['mae'],'p3_strict_closer_rate':float((g.closer_outcome=='P3').mean()) if len(g) else np.nan,'tie_rate':float((g.closer_outcome=='TIE').mean()) if len(g) else np.nan,'directional_n':int(eligible.sum()),'directional_accuracy':float(g.loc[eligible,'directional_success'].astype(float).mean()) if eligible.any() else np.nan,'mean_signed_disagreement':float(g.p3_minus_vegas.mean()) if len(g) else np.nan})
    tr=[];w=num(z.week)
    for sl,mask in {'W1':w.eq(1),'W2_12':w.between(2,12),'W13_18':w.between(13,18)}.items():
        g=z.loc[mask];eligible=g.directional_eligible;pm=metric(g.actual,g.p3_yards);vm=metric(g.actual,g.vegas);tr.append({'slice':sl,'n':len(g),'p3_mae':pm['mae'],'vegas_mae':vm['mae'],'p3_rmse':pm['rmse'],'vegas_rmse':vm['rmse'],'p3_closer_rate':float((g.closer_outcome=='P3').mean()) if len(g) else np.nan,'tie_rate':float((g.closer_outcome=='TIE').mean()) if len(g) else np.nan,'directional_n':int(eligible.sum()),'directional_accuracy':float(g.loc[eligible,'directional_success'].astype(float).mean()) if eligible.any() else np.nan})
    return z,o,pd.DataFrame(br),pd.DataFrame(tr)

def getrow(df,sl,arm,stat):return df[(df.slice==sl)&(df.arm==arm)&(df.stat==stat)].iloc[0]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stack3-root',type=Path,required=True);ap.add_argument('--stack5-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    x=add_p3(read_one(a.stack3_root,'stack3_2025_casebook.csv'));m=read_one(a.stack5_root,'stack5_899_casebook.csv');key=[c for c in ['season','week','team','player_clean_key'] if c in x.columns];mkey=[c for c in ['season','week','team','player_clean_key'] if c in m.columns];dup=int(x.duplicated(key).sum());mdup=int(m.duplicated(mkey).sum());base_y=metric(x.actual_rush_yards,x.p3_yards);mz,mo,mb,mt=market_eval(m);p3o=mo[mo.arm=='P3'].iloc[0];vego=mo[mo.arm=='VEGAS'].iloc[0];formula_y_diff=float((num(m.parent_yards)-num(mz.p3_yards_formula)).abs().max());formula_a_diff=float((num(m.parent_att)-num(mz.p3_att_formula)).abs().max())
    checks={'football_rows_pass':len(x)==EXPECTED['football_rows'],'p3_all_yards_mae_pass':abs(base_y['mae']-EXPECTED['p3_all_yards_mae'])<=TOL,'p3_all_yards_rmse_pass':abs(base_y['rmse']-EXPECTED['p3_all_yards_rmse'])<=TOL,'market_rows_pass':len(m)==EXPECTED['market_rows'],'p3_market_mae_pass':abs(float(p3o.mae)-EXPECTED['p3_market_mae'])<=TOL,'vegas_market_mae_pass':abs(float(vego.mae)-EXPECTED['vegas_market_mae'])<=TOL,'p3_market_rmse_pass':abs(float(p3o.rmse)-EXPECTED['p3_market_rmse'])<=TOL,'vegas_market_rmse_pass':abs(float(vego.rmse)-EXPECTED['vegas_market_rmse'])<=TOL,'football_unique_pass':dup==0,'market_unique_pass':mdup==0,'market_p3_yards_formula_pass':formula_y_diff<=TOL,'market_p3_att_formula_pass':formula_a_diff<=TOL};integrity_pass=all(checks.values())
    integ=pd.DataFrame([{**{'football_rows':len(x),'market_rows':len(m),'football_duplicate_rows':dup,'market_duplicate_rows':mdup,'p3_all_yards_mae':base_y['mae'],'p3_all_yards_rmse':base_y['rmse'],'p3_market_mae':float(p3o.mae),'vegas_market_mae':float(vego.mae),'p3_market_rmse':float(p3o.rmse),'vegas_market_rmse':float(vego.rmse),'market_p3_yards_formula_max_abs_diff':formula_y_diff,'market_p3_att_formula_max_abs_diff':formula_a_diff},**{k:int(v) for k,v in checks.items()},'integrity_pass':int(integrity_pass)}])
    if not integrity_pass:
        disp=pd.DataFrame([{'football_status':'INTEGRITY_FAILURE','market_status':'MARKET_INTEGRITY_FAILURE','combined':'INTEGRITY_FAILURE','sportsbook_upstream':0,'production_promotion_authorized':0}]);integ.to_csv(a.out_dir/'rb_final_integrity.csv',index=False);disp.to_csv(a.out_dir/'rb_final_disposition.csv',index=False);print(integ.to_string(index=False));print(disp.to_string(index=False));raise SystemExit(2)
    pm=score_players(x);ds=depth_scores(x);ss=state_scores(x);ed=pm[['slice','arm','stat','n','median_abs_error','p75_abs_error','p90_abs_error','over_rate','under_rate','exact_rate']].copy();p3y=getrow(pm,'ALL','P3','rush_yards');m94y=getrow(pm,'ALL','M94C','rush_yards');p3a=getrow(pm,'ALL','P3','carries');m94a=getrow(pm,'ALL','M94C','carries');latep=getrow(pm,'W13_18','P3','rush_yards');latem=getrow(pm,'W13_18','M94C','rush_yards');tail20p=getrow(pm,'ACT_CARRY_GE20','P3','rush_yards');tail20m=getrow(pm,'ACT_CARRY_GE20','M94C','rush_yards');y100p=getrow(pm,'ACT_YARDS_GE100','P3','rush_yards');y100m=getrow(pm,'ACT_YARDS_GE100','M94C','rush_yards')
    gates=[('p3_yard_mae_gain_ge_0_50',float(m94y.mae-p3y.mae),m94y.mae-p3y.mae>=.50),('p3_yard_rmse_gain_gt_0',float(m94y.rmse-p3y.rmse),m94y.rmse-p3y.rmse>0),('p3_yard_corr_delta_ge_neg_0_01',float(p3y['corr']-m94y['corr']),p3y['corr']-m94y['corr']>=-.01),('p3_carry_mae_gain_ge_neg_0_10',float(m94a.mae-p3a.mae),m94a.mae-p3a.mae>=-.10),('p3_carry_rmse_gain_ge_neg_0_10',float(m94a.rmse-p3a.rmse),m94a.rmse-p3a.rmse>=-.10),('p3_w13_18_yard_mae_better',float(latem.mae-latep.mae),latep.mae<latem.mae),('not_both_tail_slices_worse',float((tail20m.mae-tail20p.mae)+(y100m.mae-y100p.mae)),not (tail20p.mae>tail20m.mae and y100p.mae>y100m.mae))];gd=pd.DataFrame([{'gate':n,'value':v,'pass':int(p)} for n,v,p in gates]);football_ok=bool(gd['pass'].all());market_ok=float(p3o.mae)<=float(vego.mae);fstatus='FOOTBALL_QUALIFIED' if football_ok else 'FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED';mstatus='AGGREGATE_VEGAS_CLEARED' if market_ok else 'AGGREGATE_VEGAS_NOT_CLEARED';disp=pd.DataFrame([{'football_status':fstatus,'market_status':mstatus,'football_gates_passed':int(gd['pass'].sum()),'football_gate_count':len(gd),'sportsbook_upstream':0,'market_status_changes_football_qualification':0,'production_promotion_authorized':int(football_ok),'combined':f'{fstatus}__{mstatus}'}])
    keep=[c for c in ['player','player_clean_key','team','opponent','season','week','position','depth_rank','actual_rush_att','actual_rush_yards','state_m95f_risk','state_vacancy','state_m95i_tail'] if c in x.columns];finalcb=x[keep+['m94c_att','m94c_yards','stack_att','stack_yards','enriched_att','arch_enriched_opp_stack_eff_yards','p3_att','p3_yards']].copy()
    for name,df in [('rb_final_integrity.csv',integ),('rb_final_player_metrics.csv',pm),('rb_final_error_distribution.csv',ed),('rb_final_depth_metrics.csv',ds),('rb_final_state_diagnostics.csv',ss),('rb_final_market_overall.csv',mo),('rb_final_market_bins.csv',mb),('rb_final_market_time_slices.csv',mt),('rb_final_gates.csv',gd),('rb_final_disposition.csv',disp),('rb_final_casebook.csv',finalcb),('rb_final_market_casebook.csv',mz)]:df.to_csv(a.out_dir/name,index=False)
    print('=== integrity ===');print(integ.to_string(index=False));print('=== football gates ===');print(gd.to_string(index=False));print('=== disposition ===');print(disp.to_string(index=False));print('=== market overall ===');print(mo.to_string(index=False));print('=== market bins ===');print(mb.to_string(index=False));print('=== market time ===');print(mt.to_string(index=False));print('=== core player metrics ===');print(pm[pm['slice'].isin(['ALL','W1','W13_18','ACT_CARRY_GE20','ACT_YARDS_GE100'])].to_string(index=False))
if __name__=='__main__':main()
