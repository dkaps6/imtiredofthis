#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    'player_air_yards_per_target_prior','player_yac_per_reception_prior','player_screen_target_rate_prior',
    'player_explosive20_target_rate_prior','team_rb_targets_per_official_pass_attempt_prior',
    'team_rb_air_yards_per_target_prior','team_rb_yac_per_reception_prior','team_rb_screen_target_rate_prior',
    'opp_rb_air_yards_allowed_per_target_prior','opp_rb_yac_allowed_per_reception_prior',
    'opp_rb_catch_rate_allowed_prior','opp_rb_explosive20_allowed_per_target_prior','opp_rb_screen_target_rate_faced_prior',
    'production_ypt','production_catch_rate','baseline_targets','candidate_targets','target_delta','ypt_corr'
]
ABLS = {
    'player_shape':'ablation_player_shape_rec_yards',
    'team_qb_environment':'ablation_team_qb_environment_rec_yards',
    'opponent_context':'ablation_opponent_context_rec_yards',
}

def q(x,p):
    return float(x.quantile(p)) if len(x) else None

def base_metrics(g):
    out={'n':int(len(g))}
    if not len(g): return out
    out['actual_rec_yards_mean']=float(g.actual_rec_yards.mean())
    out['actual_targets_mean']=float(g.actual_targets.mean())
    out['actual_receptions_mean']=float(g.actual_receptions.mean())
    for p in ['b0','b1','c1']:
        err=g[f'{p}_rec_yards']-g.actual_rec_yards
        ae=err.abs()
        out.update({
            f'{p}_mae':float(ae.mean()),
            f'{p}_rmse':float(np.sqrt(np.mean(err**2))),
            f'{p}_bias':float(err.mean()),
            f'{p}_p90_ae':q(ae,.90),
            f'{p}_miss30_rate':float((ae>=30).mean()),
            f'{p}_underprediction_rate':float((err<0).mean()),
            f'{p}_pred_mean':float(g[f'{p}_rec_yards'].mean()),
        })
    out.update({
        'opp_ae_delta_mean':float(g.opp_ae_delta.mean()),
        'opp_ae_delta_median':float(g.opp_ae_delta.median()),
        'opp_ae_delta_p75':q(g.opp_ae_delta,.75),
        'opp_ae_delta_p90':q(g.opp_ae_delta,.90),
        'opp_share_worsened':float((g.opp_ae_delta>0).mean()),
        'ctx_ae_delta_mean':float(g.ctx_ae_delta.mean()),
        'ctx_ae_delta_median':float(g.ctx_ae_delta.median()),
        'ctx_ae_delta_p75':q(g.ctx_ae_delta,.75),
        'ctx_ae_delta_p90':q(g.ctx_ae_delta,.90),
        'ctx_share_worsened':float((g.ctx_ae_delta>0).mean()),
        'target_delta_mean':float(g.target_delta.mean()),
        'candidate_targets_mean':float(g.candidate_targets.mean()),
        'production_ypt_mean':float(g.production_ypt.mean()),
        'ypt_corr_mean':float(g.ypt_corr.mean()),
    })
    h=g[g.actual_targets>=1].copy()
    if len(h):
        actual_cr=h.actual_receptions/h.actual_targets
        resid=h.actual_ypt-h.production_ypt
        out.update({
            'targeted_n':int(len(h)),
            'actual_ypt_mean':float(h.actual_ypt.mean()),
            'realized_ypt_resid_mean':float(resid.mean()),
            'production_catch_rate_targeted_mean':float(h.production_catch_rate.mean()),
            'actual_catch_rate_mean':float(actual_cr.mean()),
            'catch_rate_resid_mean':float((actual_cr-h.production_catch_rate).mean()),
            'correction_sign_agreement':float((np.sign(h.ypt_corr)==np.sign(resid)).mean()),
            'correction_resid_correlation':float(h[['ypt_corr','realized_ypt_resid']].corr().iloc[0,1]) if h['ypt_corr'].std() and h['realized_ypt_resid'].std() else None,
            'correction_magnitude_mae_vs_realized_resid':float((h.ypt_corr-resid).abs().mean()),
        })
    return out

def cohort_masks(df):
    rb1=(df.vacancy_active.eq(1)&df.vacancy_incumbent.eq(1)&df.role.eq('RB1'))
    rb2=(df.vacancy_active.eq(1)&df.vacancy_incumbent.eq(1)&df.role.eq('RB2+'))
    vac=df.vacancy_active.eq(1)
    return {
        'VACANCY_ACTIVE':vac,
        'VACANCY_RB1_INCUMBENT':rb1,
        'VACANCY_RB2PLUS_INCUMBENT':rb2,
        '2023_VACANCY_ACTIVE':vac&df.season.eq(2023),
        '2023_VACANCY_RB1_INCUMBENT':rb1&df.season.eq(2023),
        'NON2023_VACANCY_RB1_INCUMBENT':rb1&df.season.ne(2023),
        'WEEK1_VACANCY_ACTIVE':vac&df.week.eq(1),
    }

def bin_rows(df, cohort_name, mask, var, cats):
    rows=[]
    g=df[mask]
    for cat in cats:
        h=g[g[var].astype(str).eq(str(cat))]
        m=base_metrics(h)
        m.update({'cohort':cohort_name,'bin_variable':var,'bin':str(cat)})
        rows.append(m)
    return rows

def crossing_rows(g, cohort):
    b=g.b1_ae.ge(30); c=g.c1_ae.ge(30)
    sets={'INTO_30PLUS':g[~b & c],'OUT_OF_30PLUS':g[b & ~c],'BOTH_30PLUS':g[b & c]}
    rows=[]
    for name,h in sets.items():
        row={'cohort':cohort,'crossing':name,'n':int(len(h))}
        if len(h):
            row.update({
                'target_delta_mean':float(h.target_delta.mean()),
                'ypt_corr_mean':float(h.ypt_corr.mean()),
                'actual_targets_mean':float(h.actual_targets.mean()),
                'actual_ypt_mean':float(h.actual_ypt.mean(skipna=True)),
                'b1_error_mean':float(h.b1_err.mean()),
                'c1_error_mean':float(h.c1_err.mean()),
                'rb1_share':float(h.role.eq('RB1').mean()),
                'season_2023_share':float(h.season.eq(2023).mean()),
                'under_b1_share':float(h.b1_err.lt(0).mean()),
                'under_c1_share':float(h.c1_err.lt(0).mean()),
            })
        rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.predictions)
    required=['actual_rec_yards','actual_targets','actual_receptions','b0_rec_yards','b1_rec_yards','c1_rec_yards','baseline_targets','candidate_targets','production_ypt','c1_ypt']+FEATURES[:-4]+list(ABLS.values())
    miss=[x for x in required if x not in df.columns]
    if miss: raise RuntimeError(f'missing required columns: {miss}')
    df=df[df.actual_rec_yards.notna()].copy()
    for p in ['b0','b1','c1']:
        df[f'{p}_err']=df[f'{p}_rec_yards']-df.actual_rec_yards
        df[f'{p}_ae']=df[f'{p}_err'].abs()
    df['opp_ae_delta']=df.b1_ae-df.b0_ae
    df['ctx_ae_delta']=df.c1_ae-df.b1_ae
    df['target_delta']=df.candidate_targets-df.baseline_targets
    df['ypt_corr']=df.c1_ypt-df.production_ypt
    df['realized_ypt_resid']=df.actual_ypt-df.production_ypt

    df['target_delta_bin']=np.select([df.target_delta<=-1,(df.target_delta>-1)&(df.target_delta<=-.25),(df.target_delta>-.25)&(df.target_delta<.25),(df.target_delta>=.25)&(df.target_delta<1),df.target_delta>=1],['<=-1.0','(-1.0,-0.25]','(-0.25,0.25)','[0.25,1.0)','>=1.0'],default='UNBINNED')
    df['candidate_targets_bin']=pd.cut(df.candidate_targets,[-np.inf,2,4,6,np.inf],right=False,labels=['<2','[2,4)','[4,6)','>=6'])
    df['production_ypt_bin']=pd.cut(df.production_ypt,[-np.inf,5,6.5,8,np.inf],right=False,labels=['<5.0','[5.0,6.5)','[6.5,8.0)','>=8.0'])
    df['ypt_corr_bin']=np.select([df.ypt_corr<=-1,(df.ypt_corr>-1)&(df.ypt_corr<=-.25),(df.ypt_corr>-.25)&(df.ypt_corr<.25),(df.ypt_corr>=.25)&(df.ypt_corr<1),df.ypt_corr>=1],['<=-1.0','(-1.0,-0.25]','(-0.25,0.25)','[0.25,1.0)','>=1.0'],default='UNBINNED')
    df['actual_targets_bin']=pd.cut(df.actual_targets,[-.001,.999,2.999,4.999,6.999,np.inf],labels=['0','1-2','3-4','5-6','7+'],include_lowest=True)
    df['actual_ypt_bin']=pd.cut(df.actual_ypt,[-np.inf,3,6,9,12,np.inf],right=False,labels=['<3','[3,6)','[6,9)','[9,12)','>=12'])

    masks=cohort_masks(df)
    cohort_rows=[]
    for name,mask in masks.items():
        r=base_metrics(df[mask]); r['cohort']=name; cohort_rows.append(r)
    pd.DataFrame(cohort_rows).to_csv(out/'r27c_cohort_metrics.csv',index=False)

    bins=[]
    bin_defs={
        'target_delta_bin':['<=-1.0','(-1.0,-0.25]','(-0.25,0.25)','[0.25,1.0)','>=1.0'],
        'candidate_targets_bin':['<2','[2,4)','[4,6)','>=6'],
        'production_ypt_bin':['<5.0','[5.0,6.5)','[6.5,8.0)','>=8.0'],
        'ypt_corr_bin':['<=-1.0','(-1.0,-0.25]','(-0.25,0.25)','[0.25,1.0)','>=1.0'],
        'actual_targets_bin':['0','1-2','3-4','5-6','7+'],
        'actual_ypt_bin':['<3','[3,6)','[6,9)','[9,12)','>=12'],
    }
    for cname in ['VACANCY_RB1_INCUMBENT','VACANCY_RB2PLUS_INCUMBENT','2023_VACANCY_RB1_INCUMBENT','NON2023_VACANCY_RB1_INCUMBENT']:
        for var,cats in bin_defs.items(): bins.extend(bin_rows(df,cname,masks[cname],var,cats))
    pd.DataFrame(bins).to_csv(out/'r27c_bin_metrics.csv',index=False)

    frows=[]
    for cname in ['VACANCY_RB1_INCUMBENT','VACANCY_RB2PLUS_INCUMBENT','2023_VACANCY_RB1_INCUMBENT','NON2023_VACANCY_RB1_INCUMBENT']:
        g=df[masks[cname]]
        for f in FEATURES:
            s=g[f].dropna()
            frows.append({'cohort':cname,'feature':f,'n':int(s.size),'mean':float(s.mean()) if len(s) else None,'median':float(s.median()) if len(s) else None,'p25':q(s,.25),'p75':q(s,.75)})
    pd.DataFrame(frows).to_csv(out/'r27c_feature_distributions.csv',index=False)

    arows=[]
    for cname,mask in masks.items():
        g=df[mask]
        b1_mae=float(g.b1_ae.mean()) if len(g) else np.nan
        for aname,col in ABLS.items():
            ae=(g[col]-g.actual_rec_yards).abs()
            mae=float(ae.mean()) if len(g) else np.nan
            arows.append({'cohort':cname,'ablation':aname,'n':int(len(g)),'b1_mae':b1_mae,'ablation_mae':mae,'mae_delta_vs_b1':mae-b1_mae,'mae_pct_vs_b1':(mae/b1_mae-1)*100 if b1_mae else np.nan})
    pd.DataFrame(arows).to_csv(out/'r27c_ablation_metrics.csv',index=False)

    crosses=[]
    for cname in ['VACANCY_ACTIVE','VACANCY_RB1_INCUMBENT']:
        crosses.extend(crossing_rows(df[masks[cname]],cname))
    pd.DataFrame(crosses).to_csv(out/'r27c_30plus_crossings.csv',index=False)

    dist=[]
    for cname in ['2023_VACANCY_ACTIVE','2023_VACANCY_RB1_INCUMBENT','NON2023_VACANCY_RB1_INCUMBENT']:
        g=df[masks[cname]]
        dist.append({
            'cohort':cname,'n':int(len(g)),
            'opp_ae_delta_mean':float(g.opp_ae_delta.mean()),'opp_ae_delta_median':float(g.opp_ae_delta.median()),
            'opp_ae_delta_p75':q(g.opp_ae_delta,.75),'opp_ae_delta_p90':q(g.opp_ae_delta,.90),'opp_share_worsened':float((g.opp_ae_delta>0).mean()),
            'ctx_ae_delta_mean':float(g.ctx_ae_delta.mean()),'ctx_ae_delta_median':float(g.ctx_ae_delta.median()),
            'ctx_ae_delta_p75':q(g.ctx_ae_delta,.75),'ctx_ae_delta_p90':q(g.ctx_ae_delta,.90),'ctx_share_worsened':float((g.ctx_ae_delta>0).mean()),
        })
    pd.DataFrame(dist).to_csv(out/'r27c_2023_distribution.csv',index=False)

    vac=df[masks['VACANCY_ACTIVE']]; rb1=df[masks['VACANCY_RB1_INCUMBENT']]; rb2=df[masks['VACANCY_RB2PLUS_INCUMBENT']]
    checks={
        'evaluable_n_8429':len(df)==8429,
        'vacancy_n_1761':len(vac)==1761,
        'rb1_n_503':len(rb1)==503,
        'rb2plus_n_941':len(rb2)==941,
        '2023_vacancy_n_259':int((masks['2023_VACANCY_ACTIVE']).sum())==259,
        'parent_vacancy_b1_mae_match':abs(vac.b1_ae.mean()-11.15926768815846)<1e-10,
        'parent_vacancy_c1_mae_match':abs(vac.c1_ae.mean()-11.126700786821461)<1e-10,
        'parent_rb1_b1_mae_match':abs(rb1.b1_ae.mean()-14.709394943903904)<1e-10,
        'parent_rb1_c1_mae_match':abs(rb1.c1_ae.mean()-14.666495644919056)<1e-10,
    }
    integrity=all(checks.values())

    summary={
        'study':'RB_R27C_RB1_2023_FORENSIC_V1',
        'diagnostic_only':True,
        'new_model_fit':False,
        'new_candidate_created':False,
        'production_changed':False,
        'r26_changed':False,
        'r22_changed':False,
        'sportsbook_inputs':0,
        'integrity_checks':{k:bool(v) for k,v in checks.items()},
        'integrity_pass':bool(integrity),
        'cohorts':{name:base_metrics(df[mask]) for name,mask in masks.items()},
        '30plus_crossings':crosses,
        'disposition':'R27C_FORENSIC_COMPLETE_NO_SINGLE_MECHANISM_IDENTIFIED' if integrity else 'R27C_MECHANICAL_OR_INTEGRITY_FAILURE_NO_FORENSIC_CONCLUSION'
    }
    (out/'r27c_forensic_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps({'integrity_pass':bool(integrity),'disposition':summary['disposition'],'out_dir':str(out)},indent=2))
    if not integrity: raise SystemExit(2)

if __name__=='__main__': main()
