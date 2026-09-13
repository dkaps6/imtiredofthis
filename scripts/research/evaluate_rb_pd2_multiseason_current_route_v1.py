#!/usr/bin/env python3
"""Frozen RB-PD2 2021-2024 replication on current production-equivalent mean route.

Research only. No sportsbook inputs, no production changes, and no uncertainty-width
candidate is implemented here. A replicated difficulty signal can only authorize a
separately preregistered width study. The literal PD6 P3-equivalent blocker remains open.
"""
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd
from scripts.modeling.ensemble_v2 import apply_ensemble, fit_market_weights
COMP=('mc_proj','ml_proj','state_proj'); RB_POS={'RB','HB','FB'}; MARKETS={'rush_att','rush_yards'}
TARGET_SEASONS=[2021,2022,2023,2024]; HIST=8; MIN_PRIOR=4; MIN_ROWS=700
TEAM_ALIAS={'JAC':'JAX','JAX':'JAX','LA':'LAR','LAR':'LAR','STL':'LAR','OAK':'LV','SD':'LAC','ARZ':'ARI'}
def key(v): return re.sub(r'[^a-z0-9]','',str(v or '').lower())
def team(v):
 s=str(v or '').strip().upper(); return TEAM_ALIAS.get(s,s)
def num(s): return pd.to_numeric(s,errors='coerce')
def read(p):
 x=pd.read_csv(p,low_memory=False); x.columns=[str(c).lower().strip() for c in x.columns]; return x

def verify_m95q_parity(root):
 disp=read(root/'m95q_disposition.csv')
 m91=read(root/'m95q_2024_m91_universe_parity.csv')
 par=read(root/'m95q_2024_parity_audit.csv')
 if len(disp)!=1 or str(disp.iloc[0].get('disposition',''))!='M95Q_EXPANDED_PANEL_READY':
  raise RuntimeError('M95Q source disposition is not M95Q_EXPANDED_PANEL_READY')
 if int(num(pd.Series([m91.iloc[0].get('m91_universe_parity_pass',0)])).fillna(0).iloc[0])!=1:
  raise RuntimeError('M95Q 2024 M91 universe parity did not pass')
 if int(num(pd.Series([par.iloc[0].get('parity_pass',0)])).fillna(0).iloc[0])!=1:
  raise RuntimeError('M95Q downstream 2024 parity did not pass')
 return {'m95q_disposition':'M95Q_EXPANDED_PANEL_READY','m91_universe_2024_pass':True,'downstream_2024_parity_pass':True}

def build_panel(root):
 src={s:read(root/str(s)/'component_predictions.csv') for s in range(2020,2025)}
 for season,x in src.items():
  req={'season','week','team','player','position','market','actual','mc_proj','ml_proj','state_proj'}
  miss=req-set(x.columns)
  if miss: raise RuntimeError(f'season {season} missing required columns {sorted(miss)}')
  observed=set(num(x.season).dropna().astype(int).unique().tolist())
  if observed!={season}: raise RuntimeError(f'source season drift expected={season} observed={sorted(observed)}')
 panels=[]; weight_rows=[]
 for s in TARGET_SEASONS:
  w=fit_market_weights(src[s-1]); w['fit_season']=s-1; w['target_season']=s; weight_rows.append(w)
  for market in MARKETS:
   if not w.market.astype(str).str.lower().eq(market).any(): raise RuntimeError(f'missing {market} weights target={s}')
  q=apply_ensemble(src[s], weights=w)
  q['position']=q.position.fillna('').astype(str).str.upper().str.strip(); q['market']=q.market.astype(str).str.lower().str.strip()
  q=q.loc[q.position.isin(RB_POS)&q.market.isin(MARKETS)&num(q.week).between(1,18)].copy()
  q['team']=q.team.map(team); q['player_key']=q.get('player_clean_key',q.player).map(key); q['season']=num(q.season).astype(int); q['week']=num(q.week).astype(int)
  if q.player_key.eq('').any(): raise RuntimeError(f'empty player key target={s}')
  keys=['season','week','team','player_key']
  if q.duplicated(keys+['market']).any(): raise RuntimeError(f'duplicate RB market rows target={s}')
  msets=q.groupby(keys,sort=False)['market'].agg(lambda z: tuple(sorted(z.tolist())))
  bad=msets.map(lambda z: tuple(z)!=('rush_att','rush_yards'))
  if bad.any(): raise RuntimeError(f'missing RB market pair target={s} bad_groups={int(bad.sum())}')
  rows=[]
  for k,g in q.groupby(keys,sort=False):
   d=dict(zip(keys,k)); d['player']=g.iloc[0].get('player','')
   for market,suf in [('rush_att','carry'),('rush_yards','yard')]:
    z=g.loc[g.market.eq(market)]
    rr=z.iloc[0]; pred=float(num(pd.Series([rr.ensemble_proj])).iloc[0]); actual=float(num(pd.Series([rr.actual])).iloc[0])
    if not np.isfinite(pred) or not np.isfinite(actual): raise RuntimeError(f'non-finite {market} target={s} identity={k}')
    d[f'pred_{suf}']=pred; d[f'actual_{suf}']=actual
   rows.append(d)
  panels.append(pd.DataFrame(rows))
 panel=pd.concat(panels,ignore_index=True).sort_values(['season','week','player_key'],kind='stable')
 if set(panel.season.unique().tolist())!=set(TARGET_SEASONS): raise RuntimeError(f'target season set drift {sorted(panel.season.unique().tolist())}')
 if panel.season.eq(2025).any(): raise RuntimeError('forbidden 2025 row entered replication')
 if panel.duplicated(['season','week','team','player_key']).any(): raise RuntimeError('identity duplicate')
 return panel,pd.concat(weight_rows,ignore_index=True)
def build_wf(x):
 hist={}; rows=[]
 for r in x.sort_values(['season','week','player_key'],kind='stable').itertuples(index=False):
  ce=r.pred_carry-r.actual_carry; ca=abs(ce); ye=r.pred_yard-r.actual_yard; ya=abs(ye)
  h=hist.get(r.player_key,[])[-HIST:]
  rec={'season':r.season,'week':r.week,'team':r.team,'player':r.player,'player_key':r.player_key,'target_carry_error':ce,'target_carry_abs_error':ca,'target_yard_error':ye,'target_yard_abs_error':ya,'prior_games':len(h)}
  if h:
   d=pd.DataFrame(h); rec.update(prior8_carry_bias=d.carry_error.mean(),prior8_carry_mae=d.carry_abs.mean(),prior8_yard_bias=d.yard_error.mean(),prior8_yard_mae=d.yard_abs.mean(),last_prior_ord=int(d.iloc[-1].ord))
  else:
   rec.update(prior8_carry_bias=np.nan,prior8_carry_mae=np.nan,prior8_yard_bias=np.nan,prior8_yard_mae=np.nan,last_prior_ord=np.nan)
  rows.append(rec); hist.setdefault(r.player_key,[]).append({'ord':r.season*100+r.week,'carry_error':ce,'carry_abs':ca,'yard_error':ye,'yard_abs':ya})
 out=pd.DataFrame(rows); cur=out.season*100+out.week
 if ((num(out.last_prior_ord)>=cur)&out.last_prior_ord.notna()).any(): raise RuntimeError('leakage')
 return out
def gap(g,feat,outcome):
 f=num(g[feat]); y=num(g[outcome]); return float(y[f>=f.quantile(.75)].mean()-y[f<=f.quantile(.25)].mean())
def slice_gap(g,feat,outcome,lo,hi):
 q=g[g.week.between(lo,hi)]; return gap(q,feat,outcome) if len(q)>=100 and num(q[feat]).nunique()>=4 else np.nan
def diag_metrics(g,feat,outcome,kind,signmin):
 sp=float(num(g[feat]).corr(num(g[outcome]),method='spearman')); gp=gap(g,feat,outcome)
 if kind=='sign':
  q=g[num(g[feat]).abs()>=signmin]; sign=float((np.sign(num(q[feat]))==np.sign(num(q[outcome]))).mean()) if len(q) else np.nan
 else: sign=np.nan
 return sp,gp,sign,slice_gap(g,feat,outcome,5,12),slice_gap(g,feat,outcome,13,18)
def score(wf):
 g=wf[wf.prior_games>=MIN_PRIOR].copy()
 specs=[('CARRY_DIRECTIONAL_PERSISTENCE','prior8_carry_bias','target_carry_error',1.,'sign',.5),('CARRY_DIFFICULTY_PERSISTENCE','prior8_carry_mae','target_carry_abs_error',.75,'nosign',0),('YARD_DIRECTIONAL_PERSISTENCE','prior8_yard_bias','target_yard_error',6.,'sign',3.),('YARD_DIFFICULTY_PERSISTENCE','prior8_yard_mae','target_yard_abs_error',5.,'nosign',0)]
 rows=[]; season_rows=[]
 for name,feat,outcome,mingap,kind,signmin in specs:
  sp,gp,sign,early,late=diag_metrics(g,feat,outcome,kind,signmin)
  signok=(not kind=='sign') or (np.isfinite(sign) and sign>=.55)
  pooled=bool(len(g)>=MIN_ROWS and sp>=.08 and gp>=mingap and signok and np.isfinite(early) and early>0 and np.isfinite(late) and late>0)
  pos=0; recent_bad=False
  for s in TARGET_SEASONS:
   q=g[g.season==s]; ssp=float(num(q[feat]).corr(num(q[outcome]),method='spearman')) if len(q)>2 else np.nan; sg=gap(q,feat,outcome) if len(q)>10 else np.nan
   if np.isfinite(ssp) and np.isfinite(sg) and ssp>0 and sg>0: pos+=1
   if s in (2023,2024) and np.isfinite(ssp) and np.isfinite(sg) and ssp<=0 and sg<=0: recent_bad=True
   season_rows.append({'diagnostic':name,'season':s,'rows':len(q),'spearman':ssp,'quartile_gap':sg})
  repl=bool(pooled and pos>=3 and not recent_bad)
  rows.append({'diagnostic':name,'rows':len(g),'spearman':sp,'quartile_gap':gp,'sign_agreement':sign,'gap_weeks5_12':early,'gap_weeks13_18':late,'original_pooled_gate_pass':pooled,'positive_seasons':pos,'recent_joint_negative':recent_bad,'replicated':repl})
 m=pd.DataFrame(rows); sd=pd.DataFrame(season_rows)
 winners=m.loc[m.replicated,'diagnostic'].tolist()
 res={'migration':'RB_PD2_MULTISEASON_CURRENT_PRODUCTION_ROUTE_REPLICATION_V1','source_rows':len(wf),'scoreable_rows':len(g),'players':int(wf.player_key.nunique()),'target_seasons':TARGET_SEASONS,'history_window':HIST,'minimum_prior_games':MIN_PRIOR,'sportsbook_inputs_used':False,'production_changed':False,'literal_pd6_p3_equivalent_blocker_resolved':False,'replicated_diagnostics':winners,'carry_width_unlocked':'CARRY_DIFFICULTY_PERSISTENCE' in winners,'yard_width_unlocked':'YARD_DIFFICULTY_PERSISTENCE' in winners,'disposition':'MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE_REPRODUCED' if winners else 'NO_MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE'}
 return m,sd,res
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--parity-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
 parity=verify_m95q_parity(a.parity_root); panel,w=build_panel(a.root); wf=build_wf(panel); m,sd,res=score(wf); res['source_parity']=parity
 panel.to_csv(a.out_dir/'rb_pd2_multiseason_parent_panel.csv',index=False); w.to_csv(a.out_dir/'rb_pd2_multiseason_weights.csv',index=False); wf.to_csv(a.out_dir/'rb_pd2_multiseason_walkforward_casebook.csv',index=False); m.to_csv(a.out_dir/'rb_pd2_multiseason_metrics.csv',index=False); sd.to_csv(a.out_dir/'rb_pd2_multiseason_by_season.csv',index=False); (a.out_dir/'rb_pd2_multiseason_result.json').write_text(json.dumps(res,indent=2,sort_keys=True)); print(m.to_string(index=False)); print(json.dumps(res,indent=2,sort_keys=True))
