#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd

SEASONS=[2020,2021,2022,2023,2024,2025]
FACTORS=['TARGETS','CATCH_RATE','YPR']

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f'expected exactly one {name} below {root}, got {len(h)}')
    return h[0]

def metric(a,p):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'p':pd.to_numeric(p,errors='coerce')}).dropna(); e=z.p-z.a; ae=e.abs()
    return {'n':int(len(z)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'correlation':float(z.p.corr(z.a)) if len(z)>2 else np.nan,'median_abs':float(ae.median()),'p75_abs':float(ae.quantile(.75)),'p90_abs':float(ae.quantile(.90)),'miss20':float(ae.ge(20).mean()),'miss30':float(ae.ge(30).mean()),'miss40':float(ae.ge(40).mean())}

def shapley(row):
    p={'TARGETS':row.pred_targets,'CATCH_RATE':row.pred_catch_rate,'YPR':row.pred_ypr}; a={'TARGETS':row.actual_targets,'CATCH_RATE':row.actual_catch_rate,'YPR':row.actual_ypr}
    def f(v): return v['TARGETS']*v['CATCH_RATE']*v['YPR']
    out={k:0.0 for k in FACTORS}
    for perm in itertools.permutations(FACTORS):
        cur=p.copy(); before=f(cur)
        for k in perm:
            cur[k]=a[k]; after=f(cur); out[k]+=after-before; before=after
    return {k:out[k]/6.0 for k in FACTORS}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--joint-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    x=pd.read_csv(one(a.joint_root,'joint_v1_paired_player_casebook.csv'),low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]
    req={'season','week','team','player','player_clean_key','position_group','b0_expected_targets','b0_receptions','b0_rec_yards','targets','receptions','rec_yards'}
    if req-set(x.columns): raise RuntimeError(f'missing columns {sorted(req-set(x.columns))}')
    x=x.loc[x.position_group.eq('TE') & pd.to_numeric(x.season,errors='coerce').isin(SEASONS)].copy()
    for c in ['b0_expected_targets','b0_receptions','b0_rec_yards','targets','receptions','rec_yards']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=['b0_expected_targets','b0_receptions','b0_rec_yards','targets','receptions','rec_yards']).copy()
    x['pred_targets']=x.b0_expected_targets.clip(lower=0); x['pred_catch_rate']=np.where(x.pred_targets>0,x.b0_receptions/x.pred_targets,0.0); x['pred_ypr']=np.where(x.b0_receptions>0,x.b0_rec_yards/x.b0_receptions,0.0)
    x['actual_targets']=x.targets.clip(lower=0); x['actual_catch_rate']=np.where(x.actual_targets>0,x.receptions/x.actual_targets,0.0); x['actual_ypr']=np.where(x.receptions>0,x.rec_yards/x.receptions,x.pred_ypr)
    pred_recon=x.pred_targets*x.pred_catch_rate*x.pred_ypr; actual_recon=x.actual_targets*x.actual_catch_rate*x.actual_ypr
    pred_gap=(pred_recon-x.b0_rec_yards).abs(); actual_gap=(actual_recon-x.rec_yards).abs()
    if float(pred_gap.max())>1e-6 or float(actual_gap.max())>1e-6: raise RuntimeError(f'identity reconstruction failed pred={pred_gap.max()} actual={actual_gap.max()}')
    vals=[shapley(r) for r in x[['pred_targets','pred_catch_rate','pred_ypr','actual_targets','actual_catch_rate','actual_ypr']].itertuples(index=False)]
    for k in FACTORS: x[f'{k.lower()}_component']=[v[k] for v in vals]
    x['residual_actual_minus_pred']=x.rec_yards-x.b0_rec_yards; x['shapley_sum']=x[[f'{k.lower()}_component' for k in FACTORS]].sum(axis=1); x['shapley_recon_gap']=(x.shapley_sum-x.residual_actual_minus_pred).abs(); x['abs_error']=x.residual_actual_minus_pred.abs()
    metrics=[]
    for season,g in [('POOLED',x)]+[(str(s),x.loc[x.season.eq(s)]) for s in SEASONS]:
        for market,actual,pred in [('targets','targets','b0_expected_targets'),('receptions','receptions','b0_receptions'),('rec_yards','rec_yards','b0_rec_yards')]: metrics.append({'season':season,'market':market,**metric(g[actual],g[pred])})
    abs_mass={k:float(x[f'{k.lower()}_component'].abs().sum()) for k in FACTORS}; total=sum(abs_mass.values()); shares={k:(abs_mass[k]/total if total else np.nan) for k in FACTORS}
    qcut=x.abs_error.quantile(.75); high=x.loc[x.abs_error.ge(qcut)].copy(); comps=[f'{k.lower()}_component' for k in FACTORS]; high['game_dominant']=high[comps].abs().idxmax(axis=1).str.replace('_component','',regex=False).str.upper(); high_dom=high.game_dominant.value_counts().to_dict()
    profiles=[]
    for key,g in x.groupby('player_clean_key'):
        if len(g)<20: continue
        masses={k:float(g[f'{k.lower()}_component'].abs().sum()) for k in FACTORS}; tm=sum(masses.values()); frac={k:(masses[k]/tm if tm else 0.0) for k in FACTORS}; dom=max(frac,key=frac.get) if max(frac.values())>=.45 else 'MIXED'; m=metric(g.rec_yards,g.b0_rec_yards)
        profiles.append({'player_clean_key':key,'player':g.player.mode().iloc[0] if len(g.player.mode()) else key,'games':len(g),'seasons':int(g.season.nunique()),'dominant_mechanism':dom,**{f'{k.lower()}_abs_mass_share':frac[k] for k in FACTORS},**{f'{k.lower()}_mean_abs_component':float(g[f'{k.lower()}_component'].abs().mean()) for k in FACTORS},'rec_yards_mae':m['mae'],'rec_yards_p90_abs':m['p90_abs'],'miss30':m['miss30'],'miss40':m['miss40']})
    prof=pd.DataFrame(profiles); season_counts={str(s):int(x.season.eq(s).sum()) for s in SEASONS}; sorted_shares=sorted(shares.values(),reverse=True)
    gates={'scoreable_ge4000':len(x)>=4000,'each_season_ge500':all(v>=500 for v in season_counts.values()),'shapley_reconstruction_le1e_6':float(x.shapley_recon_gap.max())<=1e-6,'players20_ge40':len(prof)>=40,'mechanism_concentration':max(shares.values())>=.45 or sum(sorted_shares[:2])>=.75}
    result={'migration':'TE_R1_INDIVIDUAL_MECHANISM_DECOMPOSITION','source_run':34081764151,'source_artifact':10004223287,'rows':len(x),'season_counts':season_counts,'players_ge20':len(prof),'pooled_absolute_mechanism_mass':abs_mass,'pooled_absolute_mechanism_share':shares,'highest_error_quartile_game_dominant_counts':high_dom,'max_shapley_reconstruction_gap':float(x.shapley_recon_gap.max()),'gates':gates,'sportsbook_inputs_used':False,'production_changed':False,'disposition':'TE_MECHANISM_DECOMPOSITION_ACTIONABLE' if all(gates.values()) else 'TE_MECHANISM_DECOMPOSITION_TOO_DIFFUSE'}
    a.out_dir.mkdir(parents=True,exist_ok=True); x.to_csv(a.out_dir/'te_r1_casebook.csv',index=False); pd.DataFrame(metrics).to_csv(a.out_dir/'te_r1_metric_summary.csv',index=False); prof.sort_values(['rec_yards_mae','games'],ascending=[False,False]).to_csv(a.out_dir/'te_r1_player_profiles.csv',index=False); (a.out_dir/'te_r1_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True)); print(pd.DataFrame(metrics).to_string(index=False)); print(prof.dominant_mechanism.value_counts(dropna=False).to_string() if len(prof) else 'no profiles'); return 0
if __name__=='__main__': raise SystemExit(main())
