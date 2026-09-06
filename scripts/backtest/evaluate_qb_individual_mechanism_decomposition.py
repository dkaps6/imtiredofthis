#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def one(root,name):
 h=list(Path(root).rglob(name))
 if len(h)!=1: raise RuntimeError(f'expected one {name}, got {len(h)}')
 return h[0]
def n(s): return pd.to_numeric(s,errors='coerce')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--m89-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 r=pd.read_csv(one(a.m89_root,'m89_corrected_qb_common_trace.csv'),low_memory=False);s=pd.read_csv(one(a.m89_root,'m89_2024_2025_synthesis_trace.csv'),low_memory=False)
 r.columns=[str(c).strip().lower() for c in r.columns];s.columns=[str(c).strip().lower() for c in s.columns]
 k=['season','week','team','player_clean_key'];cols=k+['actual_attempts','actual_ypa','actual_pass_yards','pred_attempts','pred_ypa']
 x=s[k+['base_proj','football_synthesis','actual_pass_yards']].rename(columns={'actual_pass_yards':'synth_actual'}).merge(r[cols],on=k,how='inner',validate='one_to_one')
 if len(x)!=884: raise RuntimeError(f'aligned row drift {len(x)}')
 for c in ['actual_attempts','actual_ypa','actual_pass_yards','pred_attempts','pred_ypa','base_proj','football_synthesis','synth_actual']:x[c]=n(x[c])
 x['mechanics_proj']=x.pred_attempts*x.pred_ypa;x['attempt_component']=(x.actual_attempts-x.pred_attempts)*(x.actual_ypa+x.pred_ypa)/2;x['ypa_component']=(x.actual_ypa-x.pred_ypa)*(x.actual_attempts+x.pred_attempts)/2;x['stack_adjustment']=x.base_proj-x.mechanics_proj;x['synthesis_adjustment']=x.football_synthesis-x.base_proj;x['final_residual']=x.actual_pass_yards-x.football_synthesis;x['decomp_reconstructed']=x.attempt_component+x.ypa_component-x.stack_adjustment-x.synthesis_adjustment;x['final_abs_error']=x.final_residual.abs()
 actual_recon=float((x.actual_pass_yards-x.synth_actual).abs().max());decomp_err=float((x.final_residual-x.decomp_reconstructed).abs().max())
 if actual_recon>1e-6 or decomp_err>1e-6: raise RuntimeError(f'integrity actual={actual_recon} decomp={decomp_err}')
 def summarize(g):
  return {'rows':len(g),'mae':float(g.final_abs_error.mean()),'bias_actual_minus_proj':float(g.final_residual.mean()),'att_mean':float(g.attempt_component.mean()),'att_abs':float(g.attempt_component.abs().mean()),'ypa_mean':float(g.ypa_component.mean()),'ypa_abs':float(g.ypa_component.abs().mean()),'stack_mean':float(g.stack_adjustment.mean()),'stack_abs':float(g.stack_adjustment.abs().mean()),'synth_mean':float(g.synthesis_adjustment.mean()),'synth_abs':float(g.synthesis_adjustment.abs().mean())}
 season=[]
 for yr,g in x.groupby('season'):season.append({'season':int(yr),**summarize(g)})
 prof=[]
 for p,g in x.groupby('player_clean_key'):
  if len(g)<8:continue
  vals={'ATTEMPTS':float(g.attempt_component.abs().mean()),'YPA':float(g.ypa_component.abs().mean()),'STACK':float(g.stack_adjustment.abs().mean()),'SYNTHESIS':float(g.synthesis_adjustment.abs().mean())};prof.append({'player_key':p,'games':len(g),'mae':float(g.final_abs_error.mean()),'bias_actual_minus_proj':float(g.final_residual.mean()),'miss30':float(g.final_abs_error.ge(30).mean()),'miss50':float(g.final_abs_error.ge(50).mean()),'miss75':float(g.final_abs_error.ge(75).mean()),'attempt_mean':float(g.attempt_component.mean()),'attempt_abs':vals['ATTEMPTS'],'ypa_mean':float(g.ypa_component.mean()),'ypa_abs':vals['YPA'],'stack_mean':float(g.stack_adjustment.mean()),'stack_abs':vals['STACK'],'synthesis_mean':float(g.synthesis_adjustment.mean()),'synthesis_abs':vals['SYNTHESIS'],'dominant_component':max(vals,key=vals.get)})
 ps=pd.DataFrame(prof);ss=pd.DataFrame(season);out={'migration':'QB_INDIVIDUAL_MECHANISM_DECOMPOSITION','rows':len(x),'players':int(x.player_clean_key.nunique()),'qualifying_players':len(ps),'actual_reconciliation_max_abs_error':actual_recon,'decomposition_max_abs_error':decomp_err,'aggregate':summarize(x),'sportsbook_inputs_used':False,'model_fitting_used':False,'production_changed':False,'disposition':'QB_INDIVIDUAL_MECHANISMS_MAPPED'}
 a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/'qb_mechanism_casebook.csv',index=False);ps.sort_values('mae',ascending=False).to_csv(a.out_dir/'qb_individual_mechanisms.csv',index=False);ss.to_csv(a.out_dir/'qb_mechanism_seasons.csv',index=False);(a.out_dir/'qb_mechanism_result.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2,sort_keys=True));print(ps.sort_values('mae',ascending=False).head(25).to_string(index=False))
if __name__=='__main__':main()
