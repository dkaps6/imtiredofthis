#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.backtest.evaluate_rb_individual_error_role_transition import _read,_one,_num,build_stack1_wide,attach_metadata,add_states

def m(a,p):
 z=pd.DataFrame({'a':_num(a),'p':_num(p)}).dropna();e=z.p-z.a;ae=e.abs();return {'n':len(z),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'median':float(ae.median()),'p90':float(np.quantile(ae,.9))}
def apply(x):
 y=x.copy();y['cand_att']=_num(y.pred_att);y['base_ypc']=np.where(_num(y.pred_att)>0,_num(y.pred_yards)/_num(y.pred_att),np.nan)
 for _,g in y.groupby(['season','week','team'],sort=False):
  k=g.loc[g.position.isin({'RB','HB'}) & _num(g.depth_rank).notna()].copy()
  if len(k)<2: continue
  order=k.sort_values(['depth_rank','player_key'],ascending=[True,True],kind='stable').index.tolist();vals=sorted(_num(k.pred_att).astype(float).tolist(),reverse=True)
  y.loc[order,'cand_att']=vals
 y['cand_yards']=np.where(_num(y.pred_att)>0,_num(y.cand_att)*_num(y.base_ypc),_num(y.pred_yards))
 return y
def subset(x,name,mask):
 g=x.loc[mask].copy();ba=m(g.actual_att,g.pred_att);ca=m(g.actual_att,g.cand_att);by=m(g.actual_yards,g.pred_yards);cy=m(g.actual_yards,g.cand_yards);return {'slice':name,'n':len(g),'base_carry_mae':ba['mae'],'cand_carry_mae':ca['mae'],'carry_delta':ca['mae']-ba['mae'],'base_yards_mae':by['mae'],'cand_yards_mae':cy['mae'],'yards_delta':cy['mae']-by['mae']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stack1-root',type=Path,required=True);ap.add_argument('--stack2-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 s1=_read(_one(a.stack1_root,'stack1_2025_rb_trace.csv'));s2=_read(_one(a.stack2_root,'stack2_2025_casebook.csv'));cov=_read(_one(a.stack2_root,'stack2_coverage.csv'));x=add_states(attach_metadata(build_stack1_wide(s1),s2,cov));y=apply(x)
 mass=(y.groupby(['season','week','team'])[['pred_att','cand_att']].sum());maxdiff=float((mass.pred_att-mass.cand_att).abs().max())
 base=m(y.actual_att,y.pred_att);cand=m(y.actual_att,y.cand_att);by=m(y.actual_yards,y.pred_yards);cy=m(y.actual_yards,y.cand_yards)
 slices=[];slices.append(subset(y,'W1',y.week.eq(1)));slices.append(subset(y,'W2_18',y.week.between(2,18)));slices.append(subset(y,'W13_18',y.week.between(13,18)));slices.append(subset(y,'DEPTH_RB1',_num(y.depth_rank).eq(1)));slices.append(subset(y,'MISMATCH_ROOKIE',y.state_mismatch_and_rookie.eq(1)));sl=pd.DataFrame(slices)
 ps=[]
 for k,g in y.groupby('player_key'):
  if len(g)>=8:
   bm=m(g.actual_att,g.pred_att);cm=m(g.actual_att,g.cand_att);ps.append({'player_key':k,'games':len(g),'base_mae':bm['mae'],'cand_mae':cm['mae'],'delta':cm['mae']-bm['mae']})
 ps=pd.DataFrame(ps);imp=(base['mae']-cand['mae'])/base['mae'];yimp=(by['mae']-cy['mae'])/by['mae']
 get=lambda n:float(sl.loc[sl.slice.eq(n),'carry_delta'].iloc[0])
 gates={'row_parity':len(y)==1393,'depth_coverage':abs(float(_num(y.depth_present).fillna(0).gt(0).mean())-0.949748743718593)<=1e-12,'team_mass':maxdiff<=1e-9,'carry_mae_improve_ge1pct':imp>=.01,'yards_mae_improve_ge0_5pct':yimp>=.005,'carry_rmse_nonworse':cand['rmse']<=base['rmse']+1e-12,'yards_rmse_nonworse':cy['rmse']<=by['rmse']+1e-12,'carry_bias_guard':abs(cand['bias'])<=abs(base['bias'])+.10,'W1_improve':get('W1')<0,'W2_18_improve':get('W2_18')<0,'W13_18_improve':get('W13_18')<0,'RB1_improve':get('DEPTH_RB1')<0,'mismatch_rookie_improve':get('MISMATCH_ROOKIE')<0,'median_player_delta_negative':bool(len(ps) and ps.delta.median()<0)}
 disp='ROLE_ORDER_REMAP_V2_FULL_STACK_AUTHORIZED' if all(gates.values()) else 'ROLE_ORDER_REMAP_V2_REJECTED';out={'migration':'RB_ROLE_ORDER_REMAP_V2','base_carries':base,'candidate_carries':cand,'base_yards':by,'candidate_yards':cy,'carry_improvement_fraction':float(imp),'yards_improvement_fraction':float(yimp),'max_team_carry_mass_diff':maxdiff,'gates':gates,'sportsbook_inputs_used':False,'production_changed':False,'disposition':disp}
 a.out_dir.mkdir(parents=True,exist_ok=True);y.to_csv(a.out_dir/'rb_role_order_remap_v2_casebook.csv',index=False);sl.to_csv(a.out_dir/'rb_role_order_remap_v2_slices.csv',index=False);ps.to_csv(a.out_dir/'rb_role_order_remap_v2_player_metrics.csv',index=False);(a.out_dir/'rb_role_order_remap_v2_result.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2,sort_keys=True));print(sl.to_string(index=False))
if __name__=='__main__':main()
