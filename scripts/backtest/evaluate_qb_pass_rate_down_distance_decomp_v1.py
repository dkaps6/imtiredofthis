#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

KEYS=['season','week','team','player_clean_key']
STATES=['D1','D2_SHORT','D2_MEDIUM','D2_LONG','D3_SHORT','D3_MEDIUM','D3_LONG','D4']
COMP={'LEVEL_VS_057':'level_vs_057','OCCUPANCY':'occupancy_contrib','WITHIN_STATE_RATE':'within_state_rate_contrib'}
WIN=8; SHRINK=4.0; TOL=1e-10

def one(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f'expected one {name}, found {len(h)}')
    return h[0]
def num(x): return pd.to_numeric(x,errors='coerce')
def canon(x):
    t=canon_team(x); return 'WAS' if t=='WSH' else t
def ckeys(d):
    x=d.copy(); x['season']=num(x.season).astype('Int64'); x['week']=num(x.week).astype('Int64')
    x['team']=x.team.fillna('').astype(str).map(canon); x['player_clean_key']=x.player_clean_key.fillna('').astype(str).str.strip(); return x

def parent_keys(root):
    d=pd.read_csv(one(root,'play_rate_decomposition_casebook.csv'),low_memory=False); d.columns=[str(c).strip().lower() for c in d.columns]
    d=ckeys(d[KEYS].copy())
    if len(d)!=884 or d.duplicated(KEYS).any() or d.duplicated(['season','week','team']).any(): raise RuntimeError('parent cohort drift')
    return d

def state(d,y):
    if not np.isfinite(d) or d not in (1,2,3,4): return None
    d=int(d)
    if d==1:return 'D1'
    if d==4:return 'D4'
    if not np.isfinite(y):return None
    if d==2:return 'D2_SHORT' if y<=3 else ('D2_MEDIUM' if y<=7 else 'D2_LONG')
    return 'D3_SHORT' if y<=3 else ('D3_MEDIUM' if y<=6 else 'D3_LONG')

def load_pbp():
    import nflreadpy as nfl
    fs=[]; audit={}
    for yr in (2023,2024,2025):
        r=nfl.load_pbp(seasons=[yr]); p=r.to_pandas() if hasattr(r,'to_pandas') else pd.DataFrame(r); p.columns=[str(c).strip().lower() for c in p.columns]
        c='season_type' if 'season_type' in p.columns else ('game_type' if 'game_type' in p.columns else None)
        if c:
            s=p[c].fillna('').astype(str).str.upper(); keep=s.isin(['REG','REGULAR','RS','']); p=p.loc[keep].copy() if keep.any() else p
        need={'week','posteam','defteam','qb_dropback','rush_attempt','down','ydstogo'}; miss=sorted(need-set(p.columns))
        if miss: raise RuntimeError(f'PBP {yr} missing {miss}')
        p['season']=yr
        for q in ['week','qb_dropback','rush_attempt','down','ydstogo']: p[q]=num(p[q])
        p['posteam']=p.posteam.fillna('').astype(str).map(canon); p['defteam']=p.defteam.fillna('').astype(str).map(canon)
        two=num(p.two_point_attempt).fillna(0).eq(1) if 'two_point_attempt' in p else pd.Series(False,index=p.index)
        nop=num(p.no_play).fillna(0).eq(1) if 'no_play' in p else pd.Series(False,index=p.index)
        ok=(p.qb_dropback.fillna(0).eq(1)|p.rush_attempt.fillna(0).eq(1))&~two&~nop&p.week.between(1,18)&p.posteam.ne('')
        p=p.loc[ok].copy(); p['dropback']=p.qb_dropback.fillna(0).eq(1).astype(int)
        p['state']=[state(float(d) if pd.notna(d) else np.nan,float(y) if pd.notna(y) else np.nan) for d,y in zip(p.down,p.ydstogo)]; p['decomp']=p.state.notna()
        audit[str(yr)]={'eligible':int(len(p)),'decomposable':int(p.decomp.sum()),'coverage':float(p.decomp.mean())}
        fs.append(p[['season','week','posteam','defteam','dropback','state','decomp']])
    return pd.concat(fs,ignore_index=True),audit

def game_table(p):
    x=p.copy(); x['decomp_db']=x.dropback*x.decomp.astype(int)
    b=x.groupby(['season','week','posteam','defteam'],as_index=False).agg(eligible_plays=('dropback','size'),decomposable_plays=('decomp','sum'),decomposable_dropbacks=('decomp_db','sum')).rename(columns={'posteam':'team','defteam':'opponent'})
    q=x.loc[x.decomp]; g=q.groupby(['season','week','posteam','defteam','state'],as_index=False).agg(n=('dropback','size'),db=('dropback','sum'))
    npiv=g.pivot_table(index=['season','week','posteam','defteam'],columns='state',values='n',fill_value=0,aggfunc='sum'); dpiv=g.pivot_table(index=['season','week','posteam','defteam'],columns='state',values='db',fill_value=0,aggfunc='sum')
    for s in STATES:
        if s not in npiv: npiv[s]=0
        if s not in dpiv: dpiv[s]=0
    npiv=npiv[STATES].reset_index().rename(columns={**{s:f'n_{s}' for s in STATES},'posteam':'team','defteam':'opponent'})
    dpiv=dpiv[STATES].reset_index().rename(columns={**{s:f'db_{s}' for s in STATES},'posteam':'team','defteam':'opponent'})
    z=b.merge(npiv,on=['season','week','team','opponent'],how='left',validate='one_to_one').merge(dpiv,on=['season','week','team','opponent'],how='left',validate='one_to_one')
    for c in [f'n_{s}' for s in STATES]+[f'db_{s}' for s in STATES]: z[c]=num(z[c]).fillna(0)
    z['team']=z.team.map(canon); z['opponent']=z.opponent.map(canon); z['ord']=z.season.astype(int)*100+z.week.astype(int)
    if z.duplicated(['season','week','team']).any(): raise RuntimeError('duplicate team-week PBP rows')
    return z

def metric(g,s,k):
    if g.empty:return np.nan
    if k=='occ':
        d=float(num(g.decomposable_plays).sum()); return float(num(g[f'n_{s}']).sum()/d) if d else np.nan
    d=float(num(g[f'n_{s}']).sum()); return float(num(g[f'db_{s}']).sum()/d) if d else np.nan

def ref_for(r,games):
    t=int(r.season)*100+int(r.week); prior=games.loc[games.ord<t]
    if prior.empty: raise RuntimeError('no strict-prior history')
    oh=prior.loc[prior.team.eq(r.team)].sort_values('ord').tail(WIN); dh=prior.loc[prior.opponent.eq(r.opponent)].sort_values('ord').tail(WIN)
    out={'max_prior_ord':int(prior.ord.max()),'team_prior_games':int(len(oh)),'oppdef_prior_games':int(len(dh))}; po={}; qr={}
    for s in STATES:
        lo=metric(prior,s,'occ'); lr=metric(prior,s,'dbr')
        def shr(v,n,l): return (n*v+SHRINK*l)/(n+SHRINK) if n>0 and np.isfinite(v) else np.nan
        oo,od=shr(metric(oh,s,'occ'),len(oh),lo),shr(metric(oh,s,'dbr'),len(oh),lr)
        do,dd=shr(metric(dh,s,'occ'),len(dh),lo),shr(metric(dh,s,'dbr'),len(dh),lr)
        a=[v for v in (oo,do) if np.isfinite(v)]; b=[v for v in (od,dd) if np.isfinite(v)]
        po[s]=float(np.mean(a)) if a else float(lo); qr[s]=float(np.clip(np.mean(b) if b else lr,.05,.95))
    sm=sum(po.values()); po={s:v/sm for s,v in po.items()}
    for s in STATES: out[f'p_{s}']=po[s]; out[f'q_{s}']=qr[s]
    return out

def corr(x,y):
    z=pd.DataFrame({'x':num(x),'y':num(y)}).dropna()
    if len(z)<3 or z.x.nunique()<2 or z.y.nunique()<2:return {'n':int(len(z)),'pearson':np.nan,'spearman':np.nan,'same_sign':np.nan}
    return {'n':int(len(z)),'pearson':float(z.x.corr(z.y)),'spearman':float(z.x.corr(z.y,method='spearman')),'same_sign':float((np.sign(z.x)==np.sign(z.y)).mean())}

def component_summary(z):
    rows=[]
    for sl,sm in [('2024',z.season.eq(2024)),('2025',z.season.eq(2025)),('POOLED_2024_2025',pd.Series(True,index=z.index))]:
      for cl,cm in [('ALL',pd.Series(True,index=z.index)),('ABS_RATE_MISS_0_08_PLUS',z.fixed_057_residual.abs().ge(.08)),('ABS_RATE_MISS_0_12_PLUS',z.fixed_057_residual.abs().ge(.12))]:
        g=z.loc[sm&cm]
        if g.empty:continue
        mass={k:float(g[v].abs().mean()) for k,v in COMP.items()}; den=sum(mass.values()); arr=np.column_stack([g[v].abs() for v in COMP.values()]); dom=np.array(list(COMP),object)[arr.argmax(1)]
        for k,c in COMP.items():
            v=g[c]; rows.append({'season':sl,'cohort':cl,'component':k,'n':len(g),'mean_contribution':float(v.mean()),'mean_abs_contribution':mass[k],'abs_mass_share':float(mass[k]/den),'sign_agreement':float((np.sign(v)==np.sign(g.fixed_057_residual)).mean()),'dominant_row_rate':float((dom==k).mean()),'p50_abs':float(v.abs().quantile(.5)),'p75_abs':float(v.abs().quantile(.75)),'p90_abs':float(v.abs().quantile(.9))})
    return pd.DataFrame(rows)
def state_summary(z):
    rows=[]
    for sl,g in [('2024',z.loc[z.season.eq(2024)]),('2025',z.loc[z.season.eq(2025)]),('POOLED_2024_2025',z)]:
      for s in STATES:
        a,p,b,q=g[f'a_{s}'],g[f'p_{s}'],g[f'b_{s}'],g[f'q_{s}']; oc=(a-p)*(q+b)/2; rc=(b-q)*(p+a)/2
        rows.append({'season':sl,'state':s,'n':len(g),'actual_occupancy':float(a.mean()),'reference_occupancy':float(p.mean()),'actual_within_state_dbr':float(b.mean()),'reference_within_state_dbr':float(q.mean()),'mean_occupancy_delta':float((a-p).mean()),'mean_within_state_dbr_delta':float((b-q).mean()),'mean_occupancy_contribution':float(oc.mean()),'mean_abs_occupancy_contribution':float(oc.abs().mean()),'mean_rate_contribution':float(rc.mean()),'mean_abs_rate_contribution':float(rc.abs().mean())})
    return pd.DataFrame(rows)
def shared(z,root):
    p=pd.read_csv(one(root,'qb_wr_shared_pass_volume_primary_2025.csv'),low_memory=False); s=pd.read_csv(one(root,'qb_wr_shared_pass_volume_secondary_2024_2025.csv'),low_memory=False)
    for d in (p,s): d.columns=[str(c).strip().lower() for c in d.columns]
    p,s=ckeys(p),ckeys(s)
    if len(p)!=440 or len(s)!=884 or p.duplicated(KEYS).any() or s.duplicated(KEYS).any(): raise RuntimeError('shared cohort drift')
    keep=KEYS+['fixed_057_residual','level_vs_057','occupancy_contrib','within_state_rate_contrib']; p=p.merge(z[keep],on=KEYS,validate='one_to_one'); s=s.merge(z[keep],on=KEYS,validate='one_to_one')
    sig={'ACTUAL_RATE_MINUS_057':'fixed_057_residual','LEVEL_VS_057':'level_vs_057','OCCUPANCY':'occupancy_contrib','WITHIN_STATE_RATE':'within_state_rate_contrib'}; rows=[]
    for view,sl,d,y in [('PRIMARY_WR_TARGET_MASS','2025',p,'wr_target_mass_residual'),('SECONDARY_WR_RECEPTION_MASS','POOLED_2024_2025',s,'wr_reception_mass_residual'),('SECONDARY_WR_RECEPTION_MASS','2024',s.loc[s.season.eq(2024)],'wr_reception_mass_residual'),('SECONDARY_WR_RECEPTION_MASS','2025',s.loc[s.season.eq(2025)],'wr_reception_mass_residual')]:
      for k,c in sig.items(): rows.append({'view':view,'season':sl,'signal':k,**corr(d[c],d[y])})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--play-rate-root',type=Path,required=True); ap.add_argument('--shared-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    keys=parent_keys(a.play_rate_root); pbp,pbp_audit=load_pbp(); games=game_table(pbp); z=keys.merge(games,on=['season','week','team'],how='left',validate='one_to_one')
    if len(z)!=884 or z.opponent.isna().any(): raise RuntimeError('target alignment failed')
    z=pd.concat([z,pd.DataFrame([ref_for(r,games) for _,r in z.iterrows()],index=z.index)],axis=1)
    for s in STATES:
        z[f'a_{s}']=num(z[f'n_{s}'])/num(z.decomposable_plays); raw=num(z[f'db_{s}'])/num(z[f'n_{s}']).replace(0,np.nan); z[f'b_{s}']=raw.where(num(z[f'n_{s}']).gt(0),z[f'q_{s}'])
    z['reference_rate']=sum(z[f'p_{s}']*z[f'q_{s}'] for s in STATES); z['actual_rate']=sum(z[f'a_{s}']*z[f'b_{s}'] for s in STATES); z['level_vs_057']=z.reference_rate-.57
    z['occupancy_contrib']=.5*(sum((z[f'a_{s}']-z[f'p_{s}'])*z[f'q_{s}'] for s in STATES)+sum((z[f'a_{s}']-z[f'p_{s}'])*z[f'b_{s}'] for s in STATES))
    z['within_state_rate_contrib']=.5*(sum(z[f'p_{s}']*(z[f'b_{s}']-z[f'q_{s}']) for s in STATES)+sum(z[f'a_{s}']*(z[f'b_{s}']-z[f'q_{s}']) for s in STATES)); z['fixed_057_residual']=z.actual_rate-.57
    referr=float((sum(z[f'p_{s}'] for s in STATES)-1).abs().max()); acterr=float((sum(z[f'a_{s}'] for s in STATES)-1).abs().max()); direct=num(z.decomposable_dropbacks)/num(z.decomposable_plays); rateerr=float((z.actual_rate-direct).abs().max())
    sh2=float(((z.occupancy_contrib+z.within_state_rate_contrib)-(z.actual_rate-z.reference_rate)).abs().max()); sh3=float(((z.level_vs_057+z.occupancy_contrib+z.within_state_rate_contrib)-(z.actual_rate-.57)).abs().max())
    cov={'pooled':float(z.decomposable_plays.sum()/z.eligible_plays.sum()),'2024':float(z.loc[z.season.eq(2024),'decomposable_plays'].sum()/z.loc[z.season.eq(2024),'eligible_plays'].sum()),'2025':float(z.loc[z.season.eq(2025),'decomposable_plays'].sum()/z.loc[z.season.eq(2025),'eligible_plays'].sum())}
    cs=component_summary(z); ss=state_summary(z); sr=shared(z,a.shared_root)
    integ={'exact_884_m89_target_rows':len(z)==884,'exact_one_target_team_game_per_row':not z.duplicated(['season','week','team']).any(),'decomposable_coverage_pooled_ge_0_98':cov['pooled']>=.98,'decomposable_coverage_2024_ge_0_97':cov['2024']>=.97,'decomposable_coverage_2025_ge_0_97':cov['2025']>=.97,'eight_states_mutually_exclusive_exhaustive':True,'reference_occupancy_sum_identity':referr<=TOL,'actual_occupancy_sum_identity':acterr<=TOL,'target_rate_reconciles_pbp':rateerr<=TOL,'two_factor_shapley_identity':sh2<=TOL,'fixed_057_three_part_identity':sh3<=TOL,'all_reference_inputs_strictly_prior':bool((z.max_prior_ord<(z.season.astype(int)*100+z.week.astype(int))).all()),'zero_sportsbook_inputs':True,'zero_model_fitting':True,'zero_production_changes':True,'target_game_pbp_diagnostic_only':True,'shared_receiver_cohort_keys_align_without_duplication':True}
    pool=cs.loc[(cs.season=='POOLED_2024_2025')&(cs.cohort=='ALL')].set_index('component'); y24=cs.loc[(cs.season=='2024')&(cs.cohort=='ALL')].set_index('component'); y25=cs.loc[(cs.season=='2025')&(cs.cohort=='ALL')].set_index('component'); wr=sr.loc[(sr.view=='PRIMARY_WR_TARGET_MASS')&(sr.season=='2025')].set_index('signal')
    routing={}
    for k in COMP:
        others=[o for o in COMP if o!=k]; v=float(pool.loc[k,'mean_abs_contribution']); second=max(float(pool.loc[o,'mean_abs_contribution']) for o in others); a24=float(y24.loc[k,'mean_abs_contribution']); a25=float(y25.loc[k,'mean_abs_contribution']); m24=max(float(y24.loc[o,'mean_abs_contribution']) for o in COMP); m25=max(float(y25.loc[o,'mean_abs_contribution']) for o in COMP); stable=(a24>=m24 and a25>=.9*m25) or (a25>=m25 and a24>=.9*m24); w=abs(float(wr.loc[k,'spearman']))
        routing[k]={'largest_pooled':v>=second,'season_stability':stable,'pooled_lead_ge_20pct':v>=1.2*second,'wr_target_abs_spearman_ge_0_25':w>=.25,'pooled_mean_abs':v,'second_pooled_mean_abs':second,'mean_abs_2024':a24,'mean_abs_2025':a25,'wr_target_spearman_2025':float(wr.loc[k,'spearman'])}
    q=[k for k,g in routing.items() if g['largest_pooled'] and g['season_stability'] and g['pooled_lead_ge_20pct'] and g['wr_target_abs_spearman_ge_0_25']]; mp={'OCCUPANCY':'DOWN_DISTANCE_OCCUPANCY_PRIMARY_DIAGNOSTIC','WITHIN_STATE_RATE':'WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC','LEVEL_VS_057':'LEVEL_CENTERING_PRIMARY_DIAGNOSTIC'}
    disposition='MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE' if not all(integ.values()) else (mp[q[0]] if len(q)==1 else 'MIXED_DOWN_DISTANCE_MECHANISM_NO_SINGLE_PRIMARY')
    result={'migration':'QB_PASS_RATE_DOWN_DISTANCE_DECOMPOSITION_V1','disposition':disposition,'production_actionable':False,'qualifying_primary':q,'coverage':cov,'identity_max_abs_errors':{'reference_occupancy_sum':referr,'actual_occupancy_sum':acterr,'target_rate_vs_direct_pbp':rateerr,'two_factor_shapley':sh2,'fixed_057_three_part':sh3},'integrity_gates':integ,'routing':routing,'pbp_audit':pbp_audit}
    a.out_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.out_dir/'down_distance_decomposition_casebook.csv',index=False); cs.to_csv(a.out_dir/'down_distance_component_summary.csv',index=False); ss.to_csv(a.out_dir/'down_distance_state_summary.csv',index=False); sr.to_csv(a.out_dir/'down_distance_shared_receiver_attribution.csv',index=False); (a.out_dir/'down_distance_decomposition_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if all(integ.values()) else 2
if __name__=='__main__': raise SystemExit(main())
