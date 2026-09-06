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
def rate(y,o):
 y=n(y);o=n(o);r=pd.Series(np.nan,index=y.index,dtype=float);r.loc[o.gt(0)]=y.loc[o.gt(0)]/o.loc[o.gt(0)];r.loc[o.eq(0)&y.eq(0)]=0.0;return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wr-r1-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 p=pd.read_csv(one(a.wr_r1_root,'wr_r1_paired_wr_casebook.csv'),low_memory=False);p.columns=[str(c).strip().lower() for c in p.columns];k=['season','week','team','player_clean_key','player','position']
 y=p.loc[p.market.astype(str).str.lower().eq('rec_yards'),k+['actual_m38','mc_proj_m38']].rename(columns={'actual_m38':'actual_yards','mc_proj_m38':'proj_yards'});r=p.loc[p.market.astype(str).str.lower().eq('receptions'),k+['actual_m38','mc_proj_m38']].rename(columns={'actual_m38':'actual_rec','mc_proj_m38':'proj_rec'})
 if len(y)!=12396 or len(r)!=12396: raise RuntimeError(f'source drift yards={len(y)} rec={len(r)}')
 x=y.merge(r,on=k,how='inner',validate='one_to_one');x['actual_yards']=n(x.actual_yards);x['proj_yards']=n(x.proj_yards);x['actual_rec']=n(x.actual_rec);x['proj_rec']=n(x.proj_rec);x['actual_ypr']=rate(x.actual_yards,x.actual_rec);x['proj_ypr']=rate(x.proj_yards,x.proj_rec);x['scoreable']=x.actual_ypr.notna()&x.proj_ypr.notna()
 z=x.loc[x.scoreable].copy();z['rec_component']=(z.actual_rec-z.proj_rec)*(z.actual_ypr+z.proj_ypr)/2;z['ypr_component']=(z.actual_ypr-z.proj_ypr)*(z.actual_rec+z.proj_rec)/2;z['yard_residual']=z.actual_yards-z.proj_yards;z['recon']=z.rec_component+z.ypr_component;z['yard_ae']=z.yard_residual.abs();z['rec_error']=z.proj_rec-z.actual_rec
 err=float((z.yard_residual-z.recon).abs().max());
 if err>1e-6: raise RuntimeError(f'decomp error {err}')
 prof=[]
 for pk,g in z.groupby('player_clean_key'):
  if len(g)<8:continue
  ra=float(g.rec_component.abs().mean());ya=float(g.ypr_component.abs().mean());dom='RECEPTIONS' if ra>=1.25*ya else ('YPR' if ya>=1.25*ra else 'MIXED');prof.append({'player_key':pk,'player':g.player.iloc[0],'games':len(g),'yard_mae':float(g.yard_ae.mean()),'yard_bias_actual_minus_proj':float(g.yard_residual.mean()),'yard_miss20':float(g.yard_ae.ge(20).mean()),'yard_miss30':float(g.yard_ae.ge(30).mean()),'yard_miss40':float(g.yard_ae.ge(40).mean()),'reception_mae':float(g.rec_error.abs().mean()),'reception_bias_proj_minus_actual':float(g.rec_error.mean()),'rec_component_mean':float(g.rec_component.mean()),'rec_component_abs':ra,'ypr_component_mean':float(g.ypr_component.mean()),'ypr_component_abs':ya,'rec_component_share':ra/(ra+ya) if ra+ya else np.nan,'ypr_component_share':ya/(ra+ya) if ra+ya else np.nan,'dominant_mechanism':dom})
 ps=pd.DataFrame(prof);ss=[]
 for yr,g in z.groupby('season'):ss.append({'season':int(yr),'rows':len(g),'yard_mae':float(g.yard_ae.mean()),'rec_component_abs':float(g.rec_component.abs().mean()),'ypr_component_abs':float(g.ypr_component.abs().mean()),'yard_bias_actual_minus_proj':float(g.yard_residual.mean())})
 ss=pd.DataFrame(ss);counts=ps.dominant_mechanism.value_counts().to_dict() if len(ps) else {};out={'migration':'WR_R4_INDIVIDUAL_MECHANISM_DECOMPOSITION','source_yard_rows':len(y),'paired_rows':len(x),'scoreable_rows':len(z),'inconsistent_rows':int((~x.scoreable).sum()),'players':int(x.player_clean_key.nunique()),'qualifying_players':len(ps),'dominant_mechanism_counts':{str(k):int(v) for k,v in counts.items()},'decomposition_max_abs_error':err,'sportsbook_inputs_used':False,'model_fitting_used':False,'production_changed':False,'disposition':'WR_INDIVIDUAL_MECHANISMS_MAPPED'}
 a.out_dir.mkdir(parents=True,exist_ok=True);z.to_csv(a.out_dir/'wr_r4_mechanism_casebook.csv',index=False);ps.sort_values('yard_mae',ascending=False).to_csv(a.out_dir/'wr_r4_individual_mechanisms.csv',index=False);ss.to_csv(a.out_dir/'wr_r4_season_mechanisms.csv',index=False);(a.out_dir/'wr_r4_result.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2,sort_keys=True));print(ps.sort_values('yard_mae',ascending=False).head(30).to_string(index=False))
if __name__=='__main__':main()
