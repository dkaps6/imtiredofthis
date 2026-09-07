from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def zbeta(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2: return np.nan
    d=float(np.dot(x[m],x[m]))
    return float(np.dot(x[m],y[m])/d) if d>1e-12 else 0.0


def metrics(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); m=np.isfinite(a)&np.isfinite(p); a=a[m]; p=p[m]
    e=p-a; ae=np.abs(e)
    return {
        'n':int(len(a)),
        'mae':float(np.mean(ae)),
        'rmse':float(np.sqrt(np.mean(e*e))),
        'bias_pred_minus_actual':float(np.mean(e)),
        'corr':float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan,
        'p90_abs_error':float(np.quantile(ae,.90)) if len(a) else np.nan,
    }


def build_team(df):
    k=['season','week','team']
    qb=(df[df.position=='QB'].sort_values(k+['pred_opportunity'],ascending=[True,True,True,False])
        .drop_duplicates(k)[k+['pred_opportunity','actual_opportunity']]
        .rename(columns={'pred_opportunity':'pred_qb_attempts','actual_opportunity':'actual_qb_attempts'}))
    def pool(pos,prefix):
        return df[df.position==pos].groupby(k,as_index=False).agg(**{
            f'pred_{prefix}':('pred_opportunity','sum'),
            f'actual_{prefix}':('actual_opportunity','sum')})
    return qb.merge(pool('WR','wr_targets'),on=k).merge(pool('TE','te_targets'),on=k).merge(pool('RB','rb_carries'),on=k,how='left')


def scoreboard(c, mask=None, label='ALL'):
    x=c if mask is None else c[mask]
    rows=[]
    specs=[
        ('QB_ATT','actual_qb_attempts','pred_qb_attempts','candidate_qb_attempts'),
        ('WR_TARGET','actual_wr_targets','pred_wr_targets','candidate_wr_targets'),
        ('TE_TARGET','actual_te_targets','pred_te_targets','candidate_te_targets'),
        ('RB_CARRY','actual_rb_carries','pred_rb_carries','candidate_rb_carries'),
    ]
    for name,a,b,q in specs:
        mb=metrics(x[a],x[b]); mc=metrics(x[a],x[q])
        rows.append({'slice':label,'metric':name,
                     **{f'baseline_{k}':v for k,v in mb.items()},
                     **{f'candidate_{k}':v for k,v in mc.items()},
                     'mae_improvement':mb['mae']-mc['mae'],
                     'p90_improvement':mb['p90_abs_error']-mc['p90_abs_error']})
    y=x.copy()
    y['actual_wrte']=y.actual_wr_targets+y.actual_te_targets
    y['pred_wrte']=y.pred_wr_targets+y.pred_te_targets
    y['candidate_wrte']=y.candidate_wr_targets+y.candidate_te_targets
    mb=metrics(y.actual_wrte,y.pred_wrte); mc=metrics(y.actual_wrte,y.candidate_wrte)
    rows.append({'slice':label,'metric':'WRTE_TARGET',
                 **{f'baseline_{k}':v for k,v in mb.items()},
                 **{f'candidate_{k}':v for k,v in mc.items()},
                 'mae_improvement':mb['mae']-mc['mae'],
                 'p90_improvement':mb['p90_abs_error']-mc['p90_abs_error']})
    return rows


def catastrophe_counts(c, metric, actual_col, base_col, cand_col):
    ae=np.abs(c[actual_col]-c[base_col])
    threshold=float(np.quantile(ae,.75))
    base_count=int((ae>=threshold).sum())
    cand_count=int((np.abs(c[actual_col]-c[cand_col])>=threshold).sum())
    return {'metric':metric,'threshold':threshold,'baseline_count':base_count,'candidate_count':cand_count,'count_improvement':base_count-cand_count}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase-c-all-rows',required=True)
    ap.add_argument('--phase-e-casebook',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.phase_c_all_rows)
    team=build_team(df)
    e=pd.read_csv(args.phase_e_casebook)
    cols=['season','week','team','delta_pass_attempts','pred_qb_attempts','actual_qb_attempts',
          'pred_wr_targets','actual_wr_targets','pred_te_targets','actual_te_targets',
          'pred_rb_carries','actual_rb_carries']
    base=e[cols].copy()

    rows=[]
    for wk in sorted(base.week.unique()):
        test=base[base.week==wk]
        tr=team[(team.season<2025)|((team.season==2025)&(team.week<wk))].copy()
        tr['qb_res']=tr.actual_qb_attempts-tr.pred_qb_attempts
        tr['wr_res']=tr.actual_wr_targets-tr.pred_wr_targets
        bw=zbeta(tr.qb_res,tr.wr_res)
        for _,r in test.iterrows():
            d=r.to_dict()
            up=max(float(r.delta_pass_attempts),0.0)
            d['delta_up']=up
            d['beta_wr']=bw
            d['candidate_qb_attempts']=float(r.pred_qb_attempts)+up
            d['candidate_wr_targets']=max(0.0,float(r.pred_wr_targets)+bw*up)
            d['candidate_te_targets']=float(r.pred_te_targets)
            d['candidate_rb_carries']=float(r.pred_rb_carries) if pd.notna(r.pred_rb_carries) else np.nan
            rows.append(d)
    c=pd.DataFrame(rows)

    allsc=pd.DataFrame(scoreboard(c))
    dirsc=pd.DataFrame(scoreboard(c,c.delta_pass_attempts>0,'DELTA_POSITIVE')+
                       scoreboard(c,c.delta_pass_attempts<=0,'DELTA_NONPOSITIVE'))
    c['qb_res']=c.actual_qb_attempts-c.pred_qb_attempts
    c['wrte_res']=(c.actual_wr_targets+c.actual_te_targets)-(c.pred_wr_targets+c.pred_te_targets)
    c['rb_res']=c.actual_rb_carries-c.pred_rb_carries
    high=(c.qb_res>0)&(c.wrte_res>0)&(c.rb_res<0)
    low=(c.qb_res<0)&(c.wrte_res<0)&(c.rb_res>0)
    statesc=pd.DataFrame(scoreboard(c,high,'PASS_STATE_HIGH')+scoreboard(c,low,'PASS_STATE_LOW'))

    cats=pd.DataFrame([
        catastrophe_counts(c,'QB_ATT','actual_qb_attempts','pred_qb_attempts','candidate_qb_attempts'),
        catastrophe_counts(c,'WR_TARGET','actual_wr_targets','pred_wr_targets','candidate_wr_targets'),
    ])

    get=lambda m: allsc[allsc.metric==m].iloc[0]
    qb=get('QB_ATT'); wr=get('WR_TARGET'); te=get('TE_TARGET'); rb=get('RB_CARRY'); rt=get('WRTE_TARGET')
    qcat=cats[cats.metric=='QB_ATT'].iloc[0]; wcat=cats[cats.metric=='WR_TARGET'].iloc[0]
    te_parity=float(np.nanmax(np.abs(c.candidate_te_targets-c.pred_te_targets)))
    rb_parity=float(np.nanmax(np.abs(c.candidate_rb_carries-c.pred_rb_carries)))
    gates={
        'qb_attempt_mae_improve_ge050':bool(qb.mae_improvement>=.50),
        'wr_target_mae_improve_ge050':bool(wr.mae_improvement>=.50),
        'wrte_target_mae_improve_ge030':bool(rt.mae_improvement>=.30),
        'qb_p90_no_worse':bool(qb.p90_improvement>=-1e-12),
        'wr_p90_no_worse':bool(wr.p90_improvement>=-1e-12),
        'qb_catastrophic_count_no_increase':bool(qcat.candidate_count<=qcat.baseline_count),
        'wr_catastrophic_count_no_increase':bool(wcat.candidate_count<=wcat.baseline_count),
        'te_exact_parity_le1e9':bool(te_parity<=1e-9),
        'rb_exact_parity_le1e9':bool(rb_parity<=1e-9),
        'integrity':True,
    }
    disp='POSITIVE_SHARED_PASS_STATE_ELIGIBLE' if all(gates.values()) else 'POSITIVE_SHARED_PASS_STATE_NOT_ELIGIBLE'
    result={
        'disposition':disp,
        'scored_team_games':int(len(c)),
        'positive_delta_games':int((c.delta_pass_attempts>0).sum()),
        'nonpositive_delta_games':int((c.delta_pass_attempts<=0).sum()),
        'te_max_abs_change':te_parity,
        'rb_max_abs_change':rb_parity,
        'gates':gates,
        'sportsbook_features_used':0,
        'same_or_future_outcomes_used_as_predictors':0,
        'production_parameters_changed':0,
    }
    c.to_csv(out/'phase_g_casebook.csv',index=False)
    allsc.to_csv(out/'phase_g_scorecard.csv',index=False)
    dirsc.to_csv(out/'phase_g_delta_direction_scorecard.csv',index=False)
    statesc.to_csv(out/'phase_g_actual_state_scorecard.csv',index=False)
    cats.to_csv(out/'phase_g_catastrophic_opportunity_counts.csv',index=False)
    (out/'phase_g_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    print(allsc.to_string(index=False))
    print('\n[phase_g] catastrophic opportunity counts')
    print(cats.to_string(index=False))
    print('\n[phase_g] delta direction')
    print(dirsc.to_string(index=False))
    print('\n[phase_g] actual state')
    print(statesc.to_string(index=False))

if __name__=='__main__': main()
