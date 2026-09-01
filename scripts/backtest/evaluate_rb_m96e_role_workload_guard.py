"""M96E final retrospective RB efficiency router audit."""
from __future__ import annotations
import argparse,re,unicodedata
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
KEYS=['season','week','team','player_join_key']; ALIASES={'audricestime':'audricestim'}
def num(s): return pd.to_numeric(s,errors='coerce')
def team(x):
 s='' if pd.isna(x) else str(x).upper().strip(); return {'OAK':'LV','SD':'LAC','STL':'LA','JAX':'JAC'}.get(s,s)
def key(x):
 s='' if pd.isna(x) else str(x); s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower(); s=re.sub(r'[^a-z0-9]','',s); return ALIASES.get(s,s)
def find(root,n):
 h=list(root.rglob(n));
 if len(h)!=1: raise RuntimeError(f'expected one {n}, found {len(h)}')
 return h[0]
def prep(x):
 z=x.copy(); z.columns=[str(c).lower() for c in z.columns]; z['season']=num(z.season).astype(int); z['week']=num(z.week).astype(int); z['team']=z.team.map(team)
 if 'player_join_key' not in z:
  if 'player_clean_key' not in z: raise RuntimeError('missing player key')
  z['player_join_key']=z.player_clean_key.map(key)
 else: z['player_join_key']=z.player_join_key.map(key)
 return z
def metrics(a,p):
 q=pd.DataFrame({'a':num(a),'p':num(p)}).dropna(); e=q.p-q.a
 return dict(n=len(q),mae=float(e.abs().mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),corr=float(q.a.corr(q.p)))
def auc(y,s):
 q=pd.DataFrame({'y':num(y),'s':num(s)}).dropna(); return float(roc_auc_score(q.y.astype(int),q.s)) if q.y.nunique()>1 else np.nan
def load(droot,froot,iroot):
 d=prep(pd.read_csv(find(droot,'m96d_router_trace.csv'),low_memory=False)); f=prep(pd.read_csv(find(froot,'m95f_2025_rb_trace.csv'),low_memory=False)); i=prep(pd.read_csv(find(iroot,'m95i_2025_trace.csv'),low_memory=False))
 fk=KEYS+['cal_prob_20','m95f_p90']; ik=KEYS+['prior_top1_unavailable']
 for n,x in [('m96d',d),('m95f',f),('m95i',i)]:
  if x.duplicated(KEYS).any(): raise RuntimeError(f'duplicate {n} keys')
 j=d.merge(f[fk],on=KEYS,how='left',validate='one_to_one',suffixes=('','_f')).merge(i[ik],on=KEYS,how='left',validate='one_to_one')
 covf=float(j.cal_prob_20.notna().mean()); covi=float(j.prior_top1_unavailable.notna().mean())
 if covf<.98 or covi<.98: raise RuntimeError(f'guard coverage low f={covf} i={covi}')
 return j,pd.DataFrame([{'rows':len(j),'m95f_coverage':covf,'m95i_coverage':covi}])
def arms(j):
 q=j.copy(); entren=num(q.role_is_workhorse).fillna(0).eq(1)&num(q.rb_rb_share_avg5).ge(.65); w=num(q.cal_prob_20).ge(.25)|num(q.m95f_p90).ge(20); v=num(q.prior_top1_unavailable).fillna(0).eq(1)
 q['entrenched']=entren; q['w_guard']=w; q['v_guard']=v; q['workload_risk']=w|v
 q['pred_C']=num(q.candidate_rush_yards); d=num(q.pred_d)
 q['pred_PRIMARY']=np.where((~entren)&(~w)&(~v),d,q.pred_C); q['pred_ROLE_ONLY']=np.where(~entren,d,q.pred_C); q['pred_ROLE_W_ONLY']=np.where((~entren)&(~w),d,q.pred_C); q['pred_ROLE_V_ONLY']=np.where((~entren)&(~v),d,q.pred_C)
 q['actual_75']=num(q.actual_rush_yards_m94c).ge(75).astype(int); q['actual_100']=num(q.actual_rush_yards_m94c).ge(100).astype(int); return q
def slices(q):
 a=num(q.actual_rush_att); return {'all':pd.Series(True,index=q.index),'0_5':a.between(0,5),'6_10':a.between(6,10),'11_14':a.between(11,14),'15_19':a.between(15,19),'20p':a.ge(20),'25p':a.ge(25)}
def evaluate(q):
 arms=['C','PRIMARY','ROLE_ONLY','ROLE_W_ONLY','ROLE_V_ONLY']; rows=[]; tails=[]; x=q[num(q.week).ge(6)].copy()
 for scope,sm in {'w6_18':pd.Series(True,index=x.index),'w13_18':num(x.week).ge(13)}.items():
  s=x[sm]
  for sl,m in slices(s).items():
   g=s[m]
   for a in arms: rows.append({'scope':scope,'slice':sl,'arm':a,**metrics(g.actual_rush_yards_m94c,g[f'pred_{a}'])})
  for th in [75,100]:
   for a in arms: tails.append({'scope':scope,'threshold':th,'arm':a,'auc':auc(s[f'actual_{th}'],s[f'pred_{a}'])})
 return pd.DataFrame(rows),pd.DataFrame(tails)
def gate(pt,tl):
 def r(sc,sl,a): return pt[(pt.scope==sc)&(pt.slice==sl)&(pt.arm==a)].iloc[0]
 b=r('w6_18','all','C'); p=r('w6_18','all','PRIMARY'); lb=r('w13_18','all','C'); lp=r('w13_18','all','PRIMARY')
 regs={s:r('w6_18',s,'PRIMARY').mae-r('w6_18',s,'C').mae for s in ['15_19','20p','25p']}; ag={}
 for th in [75,100]:
  bb=tl[(tl.scope=='w6_18')&(tl.threshold==th)&(tl.arm=='C')].iloc[0]; pp=tl[(tl.scope=='w6_18')&(tl.threshold==th)&(tl.arm=='PRIMARY')].iloc[0]; ag[th]=pp.auc-bb.auc
 checks=[b.mae-p.mae>=.15,p.rmse-b.rmse<=.10,abs(p.bias)-abs(b.bias)<=1,*[regs[s]<=.50 for s in ['15_19','20p','25p']],ag[75]>=-.005,ag[100]>=-.005,lb.mae-lp.mae>=0]
 return pd.DataFrame([{'retention_pass':int(all(checks)),'mae_gain':b.mae-p.mae,'rmse_gain':b.rmse-p.rmse,'late_mae_gain':lb.mae-lp.mae,'reg_15_19':regs['15_19'],'reg_20p':regs['20p'],'reg_25p':regs['25p'],'auc75_gain':ag[75],'auc100_gain':ag[100],**{f'check_{i+1}':int(v) for i,v in enumerate(checks)}}])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--m96d-root',type=Path,required=True); ap.add_argument('--m95f-root',type=Path,required=True); ap.add_argument('--m95i-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
 q,audit=load(a.m96d_root,a.m95f_root,a.m95i_root); q=arms(q); pt,tl=evaluate(q); gt=gate(pt,tl); passed=bool(gt.retention_pass.iloc[0]); x=q[num(q.week).ge(6)]
 act=pd.DataFrame([{'n':len(x),'primary_activation':float(((~x.entrenched)&(~x.workload_risk)).mean()),'w_guard_rate':float(x.w_guard.mean()),'v_guard_rate':float(x.v_guard.mean()),'protected_actual20_by_any_guard':int((x.workload_risk&num(x.actual_rush_att).ge(20)).sum()),'actual20_total':int(num(x.actual_rush_att).ge(20).sum()),'protected_actual25_by_any_guard':int((x.workload_risk&num(x.actual_rush_att).ge(25)).sum()),'actual25_total':int(num(x.actual_rush_att).ge(25).sum())}])
 disp=pd.DataFrame([{'selected_arm':'PRIMARY' if passed else 'C_M94C','retention_pass':int(passed),'disposition':'M96E_ROLE_WORKLOAD_GUARD_RETAIN_RESEARCH_ONLY' if passed else 'M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP','next_step':'PROSPECTIVE_2026_CONFIRMATION_ONLY','model_fit':0,'threshold_search':0,'feature_search':0,'sportsbook_inputs':0,'production_change':0}])
 for n,d in [('source_audit',audit),('router_trace',q),('point_metrics',pt),('tail_auc',tl),('retention_gate',gt),('activation',act),('disposition',disp)]: d.to_csv(a.out_dir/f'm96e_{n}.csv',index=False)
 print('=== gate ===');print(gt.to_string(index=False));print('=== activation ===');print(act.to_string(index=False));print('=== all arms ===');print(pt[(pt.scope=='w6_18')&(pt.slice=='all')].to_string(index=False));print('=== disposition ===');print(disp.to_string(index=False))
if __name__=='__main__': main()
