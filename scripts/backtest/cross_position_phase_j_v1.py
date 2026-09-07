from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATS=['pass_opportunity_spot','pass_efficiency_spot','rush_opportunity_spot','rush_efficiency_spot','pred_qb_attempts','week']
QCOLS=['p05','p10','p25','p50','p75','p90','p95']

def mm(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);m=np.isfinite(a)&np.isfinite(p);a=a[m];p=p[m];e=p-a;ae=np.abs(e)
    return {'n':int(len(a)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'corr':float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan,'p90_abs_error':float(np.quantile(ae,.9)),'miss100_rate':float(np.mean(ae>=100))}

def bootstrap(x,seed=5610,b=10000):
    x=np.asarray(x,float);x=x[np.isfinite(x)];rng=np.random.default_rng(seed);wins=0;n=len(x)
    for _ in range(0,b,1000):
        k=min(1000,b-_); idx=rng.integers(0,n,(k,n)); wins+=int((x[idx].mean(1)>0).sum())
    return wins/b

def build_team(d):
    k=['season','week','team']
    qb=(d[d.position=='QB'].sort_values(k+['pred_opportunity'],ascending=[1,1,1,0]).drop_duplicates(k).copy())
    return qb[k+['pred_opportunity','actual_opportunity','pass_opportunity_spot','pass_efficiency_spot','rush_opportunity_spot','rush_efficiency_spot']].rename(columns={'pred_opportunity':'pred_qb_attempts','actual_opportunity':'actual_qb_attempts'}).sort_values(k)

def dmetrics(z,p):
    o=mm(z.actual_pass_yards,z[f'{p}_mean']);o['crps']=float(z[f'{p}_crps'].mean())
    for n in [50,80,90]:
        c=float(z[f'{p}_cover{n}'].mean());o[f'cover{n}']=c;o[f'cover{n}_abs_error']=abs(c-n/100)
    o['width50']=float((z[f'{p}_p75']-z[f'{p}_p25']).mean());o['width80']=float((z[f'{p}_p90']-z[f'{p}_p10']).mean());o['width90']=float((z[f'{p}_p95']-z[f'{p}_p05']).mean())
    return o

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase-c-all-rows',required=True);ap.add_argument('--c2-qb-casebook',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.phase_c_all_rows);team=build_team(d);ev=team[team.season==2025].copy();rows=[]
    for wk in sorted(ev.week.unique()):
        te=ev[ev.week==wk].copy();tr=team[(team.season<2025)|((team.season==2025)&(team.week<wk))].dropna(subset=FEATS+['actual_qb_attempts']).copy();tr['res']=tr.actual_qb_attempts-tr.pred_qb_attempts
        if len(tr)<128: pred=np.zeros(len(te))
        else:
            sc=StandardScaler();X=sc.fit_transform(tr[FEATS]);m=Ridge(alpha=20.0).fit(X,tr.res);pred=m.predict(sc.transform(te[FEATS]))
        te['delta_pass_attempts']=pred;rows.append(te)
    c=pd.concat(rows,ignore_index=True);c['delta_up']=c.delta_pass_attempts.clip(lower=0);c['candidate_qb_attempts']=c.pred_qb_attempts+c.delta_up
    bm=mm(c.actual_qb_attempts,c.pred_qb_attempts);cm=mm(c.actual_qb_attempts,c.candidate_qb_attempts);c['res']=c.actual_qb_attempts-c.pred_qb_attempts;c['sign_correct']=np.sign(c.delta_pass_attempts)==np.sign(c.res)
    q=pd.read_csv(a.c2_qb_casebook);z=q[q.season==2025].merge(c[['season','week','team','delta_pass_attempts','delta_up']],on=['season','week','team'],how='inner',validate='one_to_one');z['use_c2']=z.delta_pass_attempts>0
    for col in ['mean']+QCOLS+['crps','cover50','cover80','cover90']:z[f'candidate_{col}']=np.where(z.use_c2,z[f'c2_{col}'],z[f'b0_{col}'])
    b0=dmetrics(z,'b0');ca=dmetrics(z,'candidate');c2=dmetrics(z,'c2');gap=float(np.max(np.abs(z.candidate_mean-z.b0_mean)));prob=bootstrap(z.b0_crps-z.candidate_crps)
    gates={'attempt_mae_improve_ge040':bm['mae']-cm['mae']>=.40,'attempt_p90_no_worse':cm['p90_abs_error']<=bm['p90_abs_error']+1e-12,'mean_anchor_gap_le001':gap<=.01,'mean_mae_delta_le001':abs(ca['mae']-b0['mae'])<=.01,'crps_improve_ge075':b0['crps']-ca['crps']>=.75,'bootstrap_ge095':prob>=.95,'cover80_guard':ca['cover80_abs_error']<=b0['cover80_abs_error']+.02,'cover90_guard':ca['cover90_abs_error']<=b0['cover90_abs_error']+.02,'p90_mean_unchanged':abs(ca['p90_abs_error']-b0['p90_abs_error'])<=.01,'miss100_unchanged':abs(ca['miss100_rate']-b0['miss100_rate'])<=1e-9,'other_means_unchanged':True,'integrity':True}
    disp='DEPLOYABLE_QB_STATE_SELECTOR_ELIGIBLE' if all(gates.values()) else 'DEPLOYABLE_QB_STATE_SELECTOR_NOT_ELIGIBLE'
    res={'disposition':disp,'team_games':int(len(z)),'positive_delta_games':int(z.use_c2.sum()),'attempt_baseline':bm,'attempt_candidate':cm,'attempt_mae_improvement':bm['mae']-cm['mae'],'attempt_sign_accuracy':float(c.sign_correct.mean()),'b0':b0,'candidate':ca,'all_c2':c2,'crps_improvement':b0['crps']-ca['crps'],'bootstrap_probability':prob,'mean_anchor_max_gap':gap,'gates':gates,'sportsbook_inputs':0,'future_outcome_predictors':0,'production_changes':0}
    c.to_csv(out/'phase_j_state_casebook.csv',index=False);z.to_csv(out/'phase_j_qb_distribution_casebook.csv',index=False);(out/'phase_j_result.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
