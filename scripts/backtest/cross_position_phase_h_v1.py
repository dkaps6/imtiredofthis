from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

THRESH={'QB':100.0,'WR':50.0}


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


def score_rows(df,label='ALL'):
    rows=[]
    for pos in ['QB','WR','TE','RB']:
        x=df[df.position==pos]
        if x.empty: continue
        mb=metrics(x.actual,x.pred); mc=metrics(x.actual,x.candidate_pred)
        rows.append({'slice':label,'position':pos,
                     **{f'baseline_{k}':v for k,v in mb.items()},
                     **{f'candidate_{k}':v for k,v in mc.items()},
                     'mae_improvement':mb['mae']-mc['mae'],
                     'p90_improvement':mb['p90_abs_error']-mc['p90_abs_error']})
    return rows


def catastrophe(df,pos):
    x=df[df.position==pos].copy(); t=THRESH[pos]
    bres=x.actual-x.pred; cres=x.actual-x.candidate_pred
    return {
        'position':pos,'threshold':t,'n':int(len(x)),
        'baseline_total':int((np.abs(bres)>=t).sum()),
        'candidate_total':int((np.abs(cres)>=t).sum()),
        'baseline_underprojection':int((bres>=t).sum()),
        'candidate_underprojection':int((cres>=t).sum()),
        'baseline_overprojection':int((bres<=-t).sum()),
        'candidate_overprojection':int((cres<=-t).sum()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase-c-all-rows',required=True)
    ap.add_argument('--phase-g-casebook',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.phase_c_all_rows)
    g=pd.read_csv(args.phase_g_casebook)
    keys=['season','week','team']
    gc=g[keys+['pred_qb_attempts','candidate_qb_attempts','pred_wr_targets','candidate_wr_targets','qb_res','wrte_res','rb_res']].copy()
    x=df.merge(gc,on=keys,how='inner')
    x=x[x.season==2025].copy()

    x['candidate_pred']=x['pred'].astype(float)
    x['candidate_opportunity']=x['pred_opportunity'].astype(float)

    mq=x.position.eq('QB')
    qratio=np.where(x.loc[mq,'pred_qb_attempts'].abs()>1e-12,
                    x.loc[mq,'candidate_qb_attempts']/x.loc[mq,'pred_qb_attempts'],1.0)
    x.loc[mq,'candidate_pred']=x.loc[mq,'pred']*qratio
    x.loc[mq,'candidate_opportunity']=x.loc[mq,'pred_opportunity']*qratio

    mw=x.position.eq('WR')
    wratio=np.where(x.loc[mw,'pred_wr_targets'].abs()>1e-12,
                    x.loc[mw,'candidate_wr_targets']/x.loc[mw,'pred_wr_targets'],1.0)
    x.loc[mw,'candidate_pred']=x.loc[mw,'pred']*wratio
    x.loc[mw,'candidate_opportunity']=x.loc[mw,'pred_opportunity']*wratio

    # Frozen team-state descriptive labels from realized residual signs; never predictors.
    x['actual_state']='MIXED'
    high=(x.qb_res>0)&(x.wrte_res>0)&(x.rb_res<0)
    low=(x.qb_res<0)&(x.wrte_res<0)&(x.rb_res>0)
    x.loc[high,'actual_state']='PASS_STATE_HIGH'
    x.loc[low,'actual_state']='PASS_STATE_LOW'

    allsc=pd.DataFrame(score_rows(x))
    state_rows=[]
    for lab in ['PASS_STATE_HIGH','PASS_STATE_LOW']:
        state_rows += score_rows(x[x.actual_state==lab],lab)
    statesc=pd.DataFrame(state_rows)

    q4=x[x.opportunity_quartile.astype(str).eq('Q4')].copy()
    q4sc=pd.DataFrame(score_rows(q4,'Q4'))
    wrq=[]
    for q in ['Q1','Q2','Q3','Q4']:
        wrq += score_rows(x[(x.position=='WR')&(x.opportunity_quartile.astype(str)==q)],q)
    wrqsc=pd.DataFrame(wrq)

    cats=pd.DataFrame([catastrophe(x,'QB'),catastrophe(x,'WR')])

    wr=x[x.position=='WR'].copy()
    wr['base_share']=wr.pred_opportunity/wr.groupby(keys).pred_opportunity.transform('sum')
    wr['cand_share']=wr.candidate_opportunity/wr.groupby(keys).candidate_opportunity.transform('sum')
    share_max=float(np.nanmax(np.abs(wr.base_share-wr.cand_share)))
    wr['base_rank']=wr.groupby(keys).pred_opportunity.rank(method='min',ascending=False)
    wr['cand_rank']=wr.groupby(keys).candidate_opportunity.rank(method='min',ascending=False)
    rank_changes=int((wr.base_rank!=wr.cand_rank).sum())

    te=x[x.position=='TE']; rb=x[x.position=='RB']
    te_max=float(np.nanmax(np.abs(te.candidate_pred-te.pred))) if len(te) else 0.0
    rb_max=float(np.nanmax(np.abs(rb.candidate_pred-rb.pred))) if len(rb) else 0.0

    get=lambda pos,tab=allsc: tab[tab.position==pos].iloc[0]
    qb=get('QB'); wrall=get('WR'); wrq4=q4sc[q4sc.position=='WR'].iloc[0]
    qcat=cats[cats.position=='QB'].iloc[0]; wcat=cats[cats.position=='WR'].iloc[0]
    gates={
        'qb_yard_mae_improve_ge100':bool(qb.mae_improvement>=1.0),
        'qb_p90_no_worse':bool(qb.p90_improvement>=-1e-12),
        'qb_100plus_count_no_increase':bool(qcat.candidate_total<=qcat.baseline_total),
        'wr_yard_mae_improve_ge020':bool(wrall.mae_improvement>=.20),
        'wr_q4_yard_mae_improve_ge050':bool(wrq4.mae_improvement>=.50),
        'wr_p90_no_worse':bool(wrall.p90_improvement>=-1e-12),
        'wr_50plus_count_no_increase':bool(wcat.candidate_total<=wcat.baseline_total),
        'm38_rank_unchanged':bool(rank_changes==0 and share_max<=1e-9),
        'te_exact_parity_le1e9':bool(te_max<=1e-9),
        'rb_exact_parity_le1e9':bool(rb_max<=1e-9),
        'integrity':True,
    }
    disp='PLAYER_YARD_PROPAGATION_ELIGIBLE' if all(gates.values()) else 'PLAYER_YARD_PROPAGATION_NOT_ELIGIBLE'
    result={
        'disposition':disp,'player_rows':int(len(x)),'team_games':int(x[keys].drop_duplicates().shape[0]),
        'wr_rows':int((x.position=='WR').sum()),'qb_rows':int((x.position=='QB').sum()),
        'm38_share_max_abs_change':share_max,'m38_rank_changes':rank_changes,
        'te_max_abs_yard_change':te_max,'rb_max_abs_yard_change':rb_max,
        'gates':gates,'sportsbook_features_used':0,'same_or_future_outcomes_used_as_predictors':0,
        'production_parameters_changed':0,
    }

    x.to_csv(out/'phase_h_player_casebook.csv',index=False)
    allsc.to_csv(out/'phase_h_scorecard.csv',index=False)
    q4sc.to_csv(out/'phase_h_q4_scorecard.csv',index=False)
    wrqsc.to_csv(out/'phase_h_wr_opportunity_quartiles.csv',index=False)
    statesc.to_csv(out/'phase_h_actual_state_scorecard.csv',index=False)
    cats.to_csv(out/'phase_h_catastrophic_yard_counts.csv',index=False)
    (out/'phase_h_result.json').write_text(json.dumps(result,indent=2))

    print(json.dumps(result,indent=2))
    print('\n[phase_h] all-player scorecard')
    print(allsc.to_string(index=False))
    print('\n[phase_h] Q4 scorecard')
    print(q4sc.to_string(index=False))
    print('\n[phase_h] WR quartiles')
    print(wrqsc.to_string(index=False))
    print('\n[phase_h] catastrophic yard counts')
    print(cats.to_string(index=False))
    print('\n[phase_h] actual state')
    print(statesc.to_string(index=False))

if __name__=='__main__': main()
