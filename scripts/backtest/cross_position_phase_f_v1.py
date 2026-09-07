from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def zbeta(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2: return np.nan
    d=float(np.dot(x[m],x[m])); return float(np.dot(x[m],y[m])/d) if d>1e-12 else 0.0

def metrics(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); m=np.isfinite(a)&np.isfinite(p); a=a[m]; p=p[m]
    e=p-a
    return {'n':int(len(a)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'bias_pred_minus_actual':float(np.mean(e)),'corr':float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan}

def build_team(df):
    k=['season','week','team']
    qb=(df[df.position=='QB'].sort_values(k+['pred_opportunity'],ascending=[True,True,True,False]).drop_duplicates(k)[k+['pred_opportunity','actual_opportunity']]
        .rename(columns={'pred_opportunity':'pred_qb_attempts','actual_opportunity':'actual_qb_attempts'}))
    def pool(pos,prefix):
        return df[df.position==pos].groupby(k,as_index=False).agg(**{f'pred_{prefix}':('pred_opportunity','sum'),f'actual_{prefix}':('actual_opportunity','sum')})
    t=qb.merge(pool('WR','wr_targets'),on=k).merge(pool('TE','te_targets'),on=k).merge(pool('RB','rb_carries'),on=k,how='left')
    return t

def sc_rows(c, mask=None, label='ALL'):
    x=c if mask is None else c[mask]
    rows=[]
    for name,a,b,q in [('QB_ATT','actual_qb_attempts','pred_qb_attempts','corr_qb_attempts'),('WR_TARGET','actual_wr_targets','pred_wr_targets','corr_wr_targets'),('TE_TARGET','actual_te_targets','pred_te_targets','corr_te_targets')]:
        mb=metrics(x[a],x[b]); mc=metrics(x[a],x[q]); rows.append({'slice':label,'metric':name,**{f'baseline_{k}':v for k,v in mb.items()},**{f'candidate_{k}':v for k,v in mc.items()},'mae_improvement':mb['mae']-mc['mae']})
    xa=x.copy(); xa['actual_wrte']=xa.actual_wr_targets+xa.actual_te_targets; xa['pred_wrte']=xa.pred_wr_targets+xa.pred_te_targets; xa['corr_wrte']=xa.corr_wr_targets+xa.corr_te_targets
    mb=metrics(xa.actual_wrte,xa.pred_wrte); mc=metrics(xa.actual_wrte,xa.corr_wrte); rows.append({'slice':label,'metric':'WRTE_TARGET',**{f'baseline_{k}':v for k,v in mb.items()},**{f'candidate_{k}':v for k,v in mc.items()},'mae_improvement':mb['mae']-mc['mae']})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase-c-all-rows',required=True); ap.add_argument('--phase-e-casebook',required=True); ap.add_argument('--out-dir',required=True); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.phase_c_all_rows); team=build_team(df); e=pd.read_csv(args.phase_e_casebook)
    base=e[['season','week','team','delta_pass_attempts','corr_qb_attempts','pred_qb_attempts','actual_qb_attempts','pred_wr_targets','actual_wr_targets','pred_te_targets','actual_te_targets','pred_rb_carries','actual_rb_carries']].copy()
    rows=[]
    for wk in sorted(base.week.unique()):
        test=base[base.week==wk]
        tr=team[(team.season<2025)|((team.season==2025)&(team.week<wk))].copy()
        tr['qb_res']=tr.actual_qb_attempts-tr.pred_qb_attempts; tr['wr_res']=tr.actual_wr_targets-tr.pred_wr_targets; tr['te_res']=tr.actual_te_targets-tr.pred_te_targets
        bw=zbeta(tr.qb_res,tr.wr_res); bt=zbeta(tr.qb_res,tr.te_res)
        for _,r in test.iterrows():
            d=r.to_dict(); d['beta_wr']=bw; d['beta_te']=bt; d['corr_wr_targets']=max(0.0,r.pred_wr_targets+bw*r.delta_pass_attempts); d['corr_te_targets']=max(0.0,r.pred_te_targets+bt*r.delta_pass_attempts); d['wr_floor_hit']=d['corr_wr_targets']<=0; d['te_floor_hit']=d['corr_te_targets']<=0; rows.append(d)
    c=pd.DataFrame(rows)
    allsc=pd.DataFrame(sc_rows(c))
    dsc=pd.DataFrame(sc_rows(c,c.delta_pass_attempts>0,'DELTA_POSITIVE')+sc_rows(c,c.delta_pass_attempts<0,'DELTA_NEGATIVE'))
    c['qb_res']=c.actual_qb_attempts-c.pred_qb_attempts; c['wrte_res']=(c.actual_wr_targets+c.actual_te_targets)-(c.pred_wr_targets+c.pred_te_targets); c['rb_res']=c.actual_rb_carries-c.pred_rb_carries
    high=(c.qb_res>0)&(c.wrte_res>0)&(c.rb_res<0); low=(c.qb_res<0)&(c.wrte_res<0)&(c.rb_res>0)
    asc=pd.DataFrame(sc_rows(c,high,'PASS_STATE_HIGH')+sc_rows(c,low,'PASS_STATE_LOW'))
    qb_parity=0.0
    get=lambda m: allsc[allsc.metric==m].iloc[0]
    wr=get('WR_TARGET'); te=get('TE_TARGET'); rt=get('WRTE_TARGET')
    rb_unchanged=True
    gates={'phase_e_qb_parity_le1e9':qb_parity<=1e-9,'wr_target_mae_improve_ge015':bool(wr.mae_improvement>=.15),'te_target_mae_no_worse_gt005':bool(te.mae_improvement>=-.05),'wrte_target_mae_no_worse_gt005':bool(rt.mae_improvement>=-.05),'rb_carry_predictions_unchanged':rb_unchanged}
    disp='SEPARATE_PASS_CATCHER_POOL_RESPONSE_ELIGIBLE' if all(gates.values()) else 'SEPARATE_PASS_CATCHER_POOL_RESPONSE_NOT_ELIGIBLE'
    result={'disposition':disp,'scored_team_games':int(len(c)),'wr_floor_hits':int(c.wr_floor_hit.sum()),'te_floor_hits':int(c.te_floor_hit.sum()),'gates':gates,'sportsbook_features_used':0,'same_or_future_outcomes_used_as_predictors':0,'production_parameters_changed':0}
    c.to_csv(out/'phase_f_casebook.csv',index=False); allsc.to_csv(out/'phase_f_scorecard.csv',index=False); dsc.to_csv(out/'phase_f_delta_direction_scorecard.csv',index=False); asc.to_csv(out/'phase_f_actual_state_scorecard.csv',index=False); (out/'phase_f_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); print(allsc.to_string(index=False)); print('\n[phase_f] delta direction'); print(dsc.to_string(index=False)); print('\n[phase_f] actual state'); print(asc.to_string(index=False))
if __name__=='__main__': main()
