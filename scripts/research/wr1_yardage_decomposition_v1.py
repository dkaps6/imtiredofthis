#!/usr/bin/env python3
from __future__ import annotations
import argparse,itertools
from pathlib import Path
import numpy as np,pandas as pd

VAR='WR_R15_WR1_ANCHORED_PARTICIPATION'; KEYS=['season','week','team','player_clean_key']
def n(x): return pd.to_numeric(x,errors='coerce')
def wb(w): return 'W1-4' if w<=4 else 'W5-9' if w<=9 else 'W10-13' if w<=13 else 'W14-18'
def sumscore(g,arm,group,market):
 d=g[g.win_loss_push.isin(['WIN','LOSS'])]
 return dict(arm=arm,group=group,market=market,rows=len(g),wins=int(d.win_loss_push.eq('WIN').sum()),losses=int(d.win_loss_push.eq('LOSS').sum()),win_rate=float(d.win_loss_push.eq('WIN').mean()),roi=float(n(d.unit_result).mean()),model_mae=float(n(g.abs_model_error).mean()),vegas_mae=float(n(g.abs_vegas_error).mean()),bias=float((n(g.pregame_projection)-n(g.actual_result)).mean()))
def shap(r):
 a={'t':float(r['at']),'c':float(r['acr']),'p':float(r['aypr'])}; p={'t':float(r['pt']),'c':float(r['pcr']),'p':float(r['pypr'])}
 if not all(np.isfinite(list(a.values())+list(p.values()))): return (np.nan,)*3
 out={k:0. for k in a}
 for perm in itertools.permutations(a):
  x=a.copy(); prev=x['t']*x['c']*x['p']
  for k in perm:
   x[k]=p[k]; cur=x['t']*x['c']*x['p']; out[k]+=cur-prev; prev=cur
 return out['t']/6,out['c']/6,out['p']/6
def dsum(g,label):
 q=g.dropna(subset=['st','sc','sp']); den=sum(abs(q[c]).sum() for c in ['st','sc','sp']); den2=abs(g.t2).sum()+abs(g.e2).sum(); dec=g[g.result.isin(['WIN','LOSS'])]
 return dict(slice=label,rows=len(g),win_rate=float(dec.result.eq('WIN').mean()) if len(dec) else np.nan,bias=float((g['py']-g['ay']).mean()),mae=float(abs(g['py']-g['ay']).mean()),target_error=float((g['pt']-g['at']).mean()),catch_error=float((g['pcr']-g['acr']).mean()),ypr_error=float((g['pypr']-g['aypr']).mean()),ypt_error=float((g['pypt']-g['aypt']).mean()),target_share=float(abs(q.st).sum()/den) if den else np.nan,catch_share=float(abs(q.sc).sum()/den) if den else np.nan,ypr_share=float(abs(q.sp).sum()/den) if den else np.nan,target_2share=float(abs(g.t2).sum()/den2) if den2 else np.nan,ypt_2share=float(abs(g.e2).sum()/den2) if den2 else np.nan,actual_100_rate=float(g['ay'].ge(100).mean()))
def main():
 ap=argparse.ArgumentParser();
 for x in ['wr_authority','authority_scoreboard','production_scoreboard','component_2024']: ap.add_argument('--'+x.replace('_','-'),required=True,type=Path)
 ap.add_argument('--out-dir',required=True,type=Path); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
 wr=pd.read_csv(a.wr_authority,low_memory=False); au=pd.read_csv(a.authority_scoreboard,low_memory=False); pr=pd.read_csv(a.production_scoreboard,low_memory=False); comp=pd.read_csv(a.component_2024,low_memory=False)
 cand=wr[(wr.variant==VAR)&(wr.season==2024)].copy()
 if len(cand)!=2117 or cand.duplicated(KEYS).any(): raise RuntimeError('2024 WR authority drift')
 top=cand[cand.wr_rank==1]
 if top.duplicated(['season','week','team']).any(): raise RuntimeError('WR1 not unique')
 roles=comp[(comp.season==2024)&(comp.position=='WR')][KEYS+['role']].drop_duplicates(KEYS); lit=roles[roles.role=='WR1']; lc=lit.groupby(['season','week','team']).size()
 pd.DataFrame([['authority_rows',len(cand)],['model_wr1_rows',len(top)],['literal_wr1_rows',len(lit)],['literal_wr1_max_per_teamgame',lc.max()],['literal_wr1_teamgames_2plus',(lc>=2).sum()]],columns=['metric','value']).to_csv(a.out_dir/'wr1_cohort_audit.csv',index=False)
 rank=cand[KEYS+['wr_rank','pred_targets','actual_targets']]
 arms={}; out=[]
 for name,df in [('AUTHORITY_OOS',au),('CURRENT_PRODUCTION_ORDER',pr)]:
  x=df[(df.season==2024)&(df.position=='WR')&df.market.isin(['rec_yards','receptions'])].merge(rank,on=KEYS,how='inner'); arms[name]=x
  for grp,m in [('WR1',x.wr_rank==1),('WR2PLUS',x.wr_rank>1)]:
   for market in ['rec_yards','receptions']: out.append(sumscore(x[m&(x.market==market)],name,grp,market))
 pd.DataFrame(out).to_csv(a.out_dir/'wr1_vs_wr2plus_scoreboard.csv',index=False)
 cur=arms['CURRENT_PRODUCTION_ORDER']; y=cur[(cur.wr_rank==1)&(cur.market=='rec_yards')].copy(); r=cur[(cur.wr_rank==1)&(cur.market=='receptions')][KEYS+['pregame_projection','actual_result']].rename(columns={'pregame_projection':'pr','actual_result':'ar'})
 d=y.merge(r,on=KEYS,how='left').rename(columns={'pred_targets':'pt','actual_targets':'at','pregame_projection':'py','actual_result':'ay','directional_pick':'pick','win_loss_push':'result'})
 for c in ['pt','at','pr','ar','py','ay']: d[c]=n(d[c])
 d['pcr']=d['pr']/d['pt'].replace(0,np.nan); d['acr']=d['ar']/d['at'].replace(0,np.nan); d['pypr']=d['py']/d['pr'].replace(0,np.nan); d['aypr']=d['ay']/d['ar'].replace(0,np.nan); d['pypt']=d['py']/d['pt'].replace(0,np.nan); d['aypt']=d['ay']/d['at'].replace(0,np.nan)
 s=d.apply(shap,axis=1,result_type='expand'); s.columns=['st','sc','sp']; d=pd.concat([d,s],axis=1); d['t2']=.5*(d['pt']-d['at'])*(d['aypt']+d['pypt']); d['e2']=.5*(d['pypt']-d['aypt'])*(d['at']+d['pt'])
 if np.nanmax(abs(d['t2']+d['e2']-(d['py']-d['ay'])))>1e-8: raise RuntimeError('decomposition identity failed')
 d.to_csv(a.out_dir/'wr1_error_decomposition_rows.csv',index=False)
 masks={'ALL':pd.Series(True,index=d.index),'WIN':d.result=='WIN','LOSS':d.result=='LOSS','UNDER':d.pick=='UNDER','OVER':d.pick=='OVER','UNDER_LOSS':(d.pick=='UNDER')&(d.result=='LOSS'),'OVER_LOSS':(d.pick=='OVER')&(d.result=='LOSS')}
 pd.DataFrame([dsum(d[m],k) for k,m in masks.items()]).to_csv(a.out_dir/'wr1_error_decomposition_summary.csv',index=False)
 tail=[]
 for label,g in [('ACTUAL_100_PLUS',d[d['ay']>=100]),('ABS_ERROR_30_PLUS',d[abs(d['py']-d['ay'])>=30])]: tail.append(dsum(g,label))
 for col,prefix in [('aypt','ACTUAL_YPT'),('pypt','PRED_YPT')]:
  q=d[col].quantile([.25,.5,.75]).tolist(); bounds=[(-np.inf,q[0]),(q[0],q[1]),(q[1],q[2]),(q[2],np.inf)]
  for i,(lo,hi) in enumerate(bounds,1): tail.append(dsum(d[(d[col]>lo)&(d[col]<=hi)],f'{prefix}_Q{i}'))
 pd.DataFrame(tail).to_csv(a.out_dir/'wr1_tail_failure_summary.csv',index=False)
 m=cur[cur.market=='rec_yards'].copy(); m['wb']=m.week.map(wb)
 for c in ['pred_targets','vegas_line']:
  mu=n(m[c]).mean(); sd=n(m[c]).std(ddof=0); m[c+'z']=(n(m[c])-mu)/(sd or 1)
 t=m[m.wr_rank==1]; c=m[m.wr_rank>1]; pairs=[]
 for ix,row in t.iterrows():
  z=c[c.wb==row.wb].copy(); z['dist']=(z.pred_targetsz-row.pred_targetsz)**2+(z.vegas_linez-row.vegas_linez)**2; q=z.sort_values(['dist','season','week','team','player_clean_key']).iloc[0]; pairs.append((ix,q.name,q.dist))
 p=pd.DataFrame(pairs,columns=['ti','ci','dist']); tw=t.loc[p.ti].reset_index(drop=True); cw=c.loc[p.ci].reset_index(drop=True); ms=[sumscore(tw,'MATCHED','WR1','rec_yards'),sumscore(cw,'MATCHED','WR2PLUS','rec_yards')]; pd.DataFrame(ms).to_csv(a.out_dir/'wr1_matched_control_summary.csv',index=False)
 aa=arms['AUTHORITY_OOS'][(arms['AUTHORITY_OOS'].wr_rank==1)&(arms['AUTHORITY_OOS'].market=='rec_yards')]; pp=cur[(cur.wr_rank==1)&(cur.market=='rec_yards')]; keep=KEYS+['directional_pick','win_loss_push','pregame_projection']; z=aa[keep].merge(pp[keep],on=KEYS,suffixes=('_a','_p')); z['flip']=z.directional_pick_a!=z.directional_pick_p; z['delta']=z.pregame_projection_p-z.pregame_projection_a
 dr=[]
 for label,g in [('ALL',z),('SIDE_FLIP',z[z.flip]),('OVER_TO_UNDER',z[(z.directional_pick_a=='OVER')&(z.directional_pick_p=='UNDER')]),('UNDER_TO_OVER',z[(z.directional_pick_a=='UNDER')&(z.directional_pick_p=='OVER')])]: dr.append(dict(slice=label,rows=len(g),authority_wins=int(g.win_loss_push_a.eq('WIN').sum()),production_wins=int(g.win_loss_push_p.eq('WIN').sum()),net_wins=int(g.win_loss_push_p.eq('WIN').sum()-g.win_loss_push_a.eq('WIN').sum()),mean_delta=float(g.delta.mean()) if len(g) else np.nan))
 pd.DataFrame(dr).to_csv(a.out_dir/'wr1_production_order_drift.csv',index=False)
 return 0
if __name__=='__main__': raise SystemExit(main())
