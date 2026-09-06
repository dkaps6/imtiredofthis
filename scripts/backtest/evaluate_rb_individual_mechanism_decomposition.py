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
 ap=argparse.ArgumentParser();ap.add_argument('--rb-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 x=pd.read_csv(one(a.rb_root,'rb_role_transition_casebook.csv'),low_memory=False);x.columns=[str(c).strip().lower() for c in x.columns]
 if len(x)!=1393: raise RuntimeError(f'row drift {len(x)}')
 for c in ['pred_att','actual_att','pred_yards','actual_yards']:x[c]=n(x[c])
 x['actual_ypc']=rate(x.actual_yards,x.actual_att);x['proj_ypc']=rate(x.pred_yards,x.pred_att);x['scoreable']=x.actual_ypc.notna()&x.proj_ypc.notna();z=x.loc[x.scoreable].copy();z['carry_component']=(z.actual_att-z.pred_att)*(z.actual_ypc+z.proj_ypc)/2;z['ypc_component']=(z.actual_ypc-z.proj_ypc)*(z.actual_att+z.pred_att)/2;z['yard_residual']=z.actual_yards-z.pred_yards;z['recon']=z.carry_component+z.ypc_component;z['yard_ae']=z.yard_residual.abs();z['carry_error']=z.pred_att-z.actual_att
 err=float((z.yard_residual-z.recon).abs().max());
 if err>1e-6: raise RuntimeError(f'decomp error {err}')
 prof=[]
 for pk,g in z.groupby('player_key'):
  if len(g)<8:continue
  ca=float(g.carry_component.abs().mean());ya=float(g.ypc_component.abs().mean());dom='CARRIES' if ca>=1.25*ya else ('YPC' if ya>=1.25*ca else 'MIXED');prof.append({'player_key':pk,'player':g.player.iloc[0],'games':len(g),'carry_mae':float(g.carry_error.abs().mean()),'carry_bias_proj_minus_actual':float(g.carry_error.mean()),'yard_mae':float(g.yard_ae.mean()),'yard_bias_actual_minus_proj':float(g.yard_residual.mean()),'yard_miss20':float(g.yard_ae.ge(20).mean()),'yard_miss30':float(g.yard_ae.ge(30).mean()),'yard_miss40':float(g.yard_ae.ge(40).mean()),'yard_miss50':float(g.yard_ae.ge(50).mean()),'carry_component_mean':float(g.carry_component.mean()),'carry_component_abs':ca,'ypc_component_mean':float(g.ypc_component.mean()),'ypc_component_abs':ya,'carry_component_share':ca/(ca+ya) if ca+ya else np.nan,'ypc_component_share':ya/(ca+ya) if ca+ya else np.nan,'dominant_mechanism':dom})
 ps=pd.DataFrame(prof)
 def sm(name,mask):
  g=z.loc[mask];return {'slice':name,'rows':len(g),'yard_mae':float(g.yard_ae.mean()) if len(g) else np.nan,'carry_component_abs':float(g.carry_component.abs().mean()) if len(g) else np.nan,'ypc_component_abs':float(g.ypc_component.abs().mean()) if len(g) else np.nan,'yard_bias_actual_minus_proj':float(g.yard_residual.mean()) if len(g) else np.nan}
 slices=pd.DataFrame([sm('ALL',pd.Series(True,index=z.index)),sm('ROOKIE',z.state_rookie.eq(1)),sm('MISMATCH_ROOKIE',z.state_mismatch_and_rookie.eq(1)),sm('DEPTH_RB1',n(z.depth_rank).eq(1))]);counts=ps.dominant_mechanism.value_counts().to_dict() if len(ps) else {};out={'migration':'RB_INDIVIDUAL_MECHANISM_DECOMPOSITION','source_rows':len(x),'scoreable_rows':len(z),'inconsistent_rows':int((~x.scoreable).sum()),'players':int(x.player_key.nunique()),'qualifying_players':len(ps),'dominant_mechanism_counts':{str(k):int(v) for k,v in counts.items()},'decomposition_max_abs_error':err,'sportsbook_inputs_used':False,'model_fitting_used':False,'production_changed':False,'disposition':'RB_INDIVIDUAL_MECHANISMS_MAPPED'}
 a.out_dir.mkdir(parents=True,exist_ok=True);z.to_csv(a.out_dir/'rb_mechanism_casebook.csv',index=False);ps.sort_values('yard_mae',ascending=False).to_csv(a.out_dir/'rb_individual_mechanisms.csv',index=False);slices.to_csv(a.out_dir/'rb_mechanism_slices.csv',index=False);(a.out_dir/'rb_mechanism_result.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2,sort_keys=True));print(slices.to_string(index=False));print(ps.sort_values('yard_mae',ascending=False).head(30).to_string(index=False))
if __name__=='__main__':main()
