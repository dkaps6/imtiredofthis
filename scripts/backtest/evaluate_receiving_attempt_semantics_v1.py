#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
SEASONS=[2020,2021,2022,2023,2024,2025];POS=["WR","TE","RB"];GROUPS=["WR","TE","RB_FB"]
EXPECTED_N=4647;EXPECTED_MAE=17.099904733366

def num(s):return pd.to_numeric(s,errors="coerce")
def metric(a,p):
 z=pd.DataFrame({'a':num(a),'p':num(p)}).dropna();e=z.p-z.a
 return {'n':len(z),'mae':float(e.abs().mean()) if len(z) else np.nan,'rmse':float(np.sqrt(np.mean(e*e))) if len(z) else np.nan,'bias':float(e.mean()) if len(z) else np.nan,'correlation':float(z.p.corr(z.a)) if len(z)>2 else np.nan}
def load(root,s,name):
 p=root/str(s)/'trace'/name
 if not p.exists() or not p.stat().st_size:raise RuntimeError(f'missing {p}')
 x=pd.read_csv(p,low_memory=False);x.columns=[str(c).strip().lower() for c in x.columns];
 if 'season' in x:x['season']=num(x.season).astype(int)
 if 'week' in x:x['week']=num(x.week).astype(int)
 return x
def casebook(root):
 out=[]
 for s in SEASONS:
  p=load(root,s,'attempt_v1_player_trace.csv');a=load(root,s,'attempt_v1_actual_usage.csv');k=['season','week','team','join_key'];out.append(p.merge(a[k+['targets','receptions','rec_yards']],on=k,how='inner',validate='one_to_one'))
 return pd.concat(out,ignore_index=True)
def player_metrics(x):
 rows=[]
 for pos in POS:
  z=x.loc[x.position_group.eq(pos)]
  for sl,g in [('POOLED',z)]+[(str(s),z.loc[z.season.eq(s)]) for s in SEASONS]+[('2024_2025',z.loc[z.season.isin([2024,2025])])]:
   for v in ['B0','C4']:
    pre=v.lower()
    for market,suf in [('targets','expected_targets'),('receptions','receptions'),('rec_yards','rec_yards')]:rows.append({'season':sl,'position_group':pos,'variant':v,'market':market,**metric(g[market],g[f'{pre}_{suf}'])})
 return pd.DataFrame(rows)
def getmae(pm,sl,pos,v,m):return float(pm.loc[(pm.season.eq(sl))&pm.position_group.eq(pos)&pm.variant.eq(v)&pm.market.eq(m),'mae'].iloc[0])
def team_target_case(root):
 out=[]
 for s in SEASONS:
  p=load(root,s,'attempt_v1_player_trace.csv');a=load(root,s,'attempt_v1_actual_usage.csv');a=a.loc[a.mass_group.isin(GROUPS)].groupby(['season','week','team'],as_index=False).targets.sum().rename(columns={'targets':'actual_modeled_targets'});q=p.groupby(['season','week','team'],as_index=False).agg(b0_modeled_targets=('b0_expected_targets','sum'),c4_modeled_targets=('c4_expected_targets','sum'));out.append(a.merge(q,on=['season','week','team'],how='inner',validate='one_to_one'))
 return pd.concat(out,ignore_index=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args();x=casebook(a.root);pm=player_metrics(x);tt=team_target_case(a.root);teams=pd.concat([load(a.root,s,'attempt_v1_team_trace.csv') for s in SEASONS],ignore_index=True)
 parity=metric(x.loc[x.season.eq(2025),'rec_yards'],x.loc[x.season.eq(2025),'b0_rec_yards']);valid_cov=float(teams.attempt_rate_source.eq('historical_pregame_pbp').mean());team_b0=metric(tt.actual_modeled_targets,tt.b0_modeled_targets);team_c4=metric(tt.actual_modeled_targets,tt.c4_modeled_targets)
 ptarget_b=np.mean([getmae(pm,'POOLED',p,'B0','targets') for p in POS]);ptarget_c=np.mean([getmae(pm,'POOLED',p,'C4','targets') for p in POS]);pyard_b=np.mean([getmae(pm,'POOLED',p,'B0','rec_yards') for p in POS]);pyard_c=np.mean([getmae(pm,'POOLED',p,'C4','rec_yards') for p in POS]);years=sum(np.mean([getmae(pm,str(s),p,'C4','rec_yards') for p in POS])<np.mean([getmae(pm,str(s),p,'B0','rec_yards') for p in POS]) for s in SEASONS);latest_c=np.mean([getmae(pm,'2024_2025',p,'C4','rec_yards') for p in POS]);latest_b=np.mean([getmae(pm,'2024_2025',p,'B0','rec_yards') for p in POS]);season_pos=[getmae(pm,str(s),p,'C4','rec_yards')-getmae(pm,str(s),p,'B0','rec_yards') for s in SEASONS for p in POS]
 integrity={'b0_n_exact':parity['n']==EXPECTED_N,'b0_mae_within_0_05':abs(parity['mae']-EXPECTED_MAE)<=.05,'attempt_source_coverage_ge_0_95':valid_cov>=.95,'team_games_all_accounted':len(teams)>0 and teams.attempt_rate_source.isin(['historical_pregame_pbp','fallback_1.0']).all(),'sportsbook_inputs_used':False};iok=all(bool(v) for k,v in integrity.items() if k!='sportsbook_inputs_used')
 gates={'team_modeled_target_mae_improves_ge_0_50':team_b0['mae']-team_c4['mae']>=.50,'macro_player_target_mae_improves_ge_0_03':ptarget_b-ptarget_c>=.03,'no_position_target_mae_worse_gt_0_03':all(getmae(pm,'POOLED',p,'C4','targets')-getmae(pm,'POOLED',p,'B0','targets')<=.03 for p in POS),'no_position_reception_mae_worse_gt_0_03':all(getmae(pm,'POOLED',p,'C4','receptions')-getmae(pm,'POOLED',p,'B0','receptions')<=.03 for p in POS),'no_position_rec_yard_mae_worse_gt_0_50':all(getmae(pm,'POOLED',p,'C4','rec_yards')-getmae(pm,'POOLED',p,'B0','rec_yards')<=.50 for p in POS),'macro_rec_yard_mae_improves':pyard_c<pyard_b,'macro_rec_yard_improves_ge4_of6_seasons':years>=4,'latest_2024_2025_macro_rec_yard_improves':latest_c<latest_b,'no_season_position_rec_yard_regression_gt_1_50':max(season_pos)<=1.50};passed=iok and all(gates.values());disp='ATTEMPT_SEMANTICS_CANDIDATE_PASS' if passed else ('BASELINE_OR_INTEGRITY_FAIL' if not iok else 'ATTEMPT_SEMANTICS_CANDIDATE_FAIL')
 result={'migration':'RECEIVING_ATTEMPT_SEMANTICS_V1','seasons':SEASONS,'iterations':2000,'production_changed':False,'sportsbook_inputs_used':False,'baseline_parity':parity,'attempt_source_coverage':valid_cov,'team_target_b0':team_b0,'team_target_c4':team_c4,'macro_player_target_mae_b0':ptarget_b,'macro_player_target_mae_c4':ptarget_c,'macro_player_rec_yard_mae_b0':pyard_b,'macro_player_rec_yard_mae_c4':pyard_c,'yearly_macro_rec_yard_improve_count':int(years),'integrity_gates':integrity,'scientific_gates':gates,'disposition':disp};a.out_dir.mkdir(parents=True,exist_ok=True);pm.to_csv(a.out_dir/'attempt_v1_player_metric_summary.csv',index=False);tt.to_csv(a.out_dir/'attempt_v1_team_target_casebook.csv',index=False);teams.to_csv(a.out_dir/'attempt_v1_team_conversion_casebook.csv',index=False);(a.out_dir/'attempt_v1_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
