#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

SOURCE_SEASONS=[2020,2021,2022,2023,2024,2025]
TARGET_SEASONS=[2021,2022,2023,2024,2025]
FIELDS=['avg_separation','avg_cushion','avg_intended_air_yards','percent_share_of_intended_air_yards','avg_yac','avg_expected_yac','avg_yac_above_expectation','catch_percentage']
PLAYER_FIELDS=['player_display_name','player_name','player_short_name']
TEAM_FIELDS=['team_abbr','team','club']

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f'expected one {name}, got {len(h)}')
    return h[0]
def key(v)->str: return ''.join(ch.lower() for ch in str(v or '') if ch.isalnum())
def first(cols,choices): return next((c for c in choices if c in cols),None)
def ordinal(season,week): return int(season)*100+int(week)
def regmask(df):
    s=pd.to_numeric(df.season,errors='coerce'); w=pd.to_numeric(df.week,errors='coerce'); mx=s.map(lambda y:17 if y==2020 else 18)
    return s.isin(SOURCE_SEASONS)&w.ge(1)&w.le(mx)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--wr-r1-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    import nflreadpy as nfl
    ref=pd.read_csv(one(a.wr_r1_root,'wr_r1_paired_wr_casebook.csv'),low_memory=False); ref.columns=[str(c).strip().lower() for c in ref.columns]
    need={'season','week','team','player_clean_key'}
    if need-set(ref.columns): raise RuntimeError(f'missing ref {sorted(need-set(ref.columns))}')
    ref['season']=pd.to_numeric(ref.season,errors='coerce'); ref['week']=pd.to_numeric(ref.week,errors='coerce'); ref=ref.loc[ref.season.isin(TARGET_SEASONS)].copy(); ref['team']=ref.team.map(canon_team); ref['player_key']=ref.player_clean_key.map(key); ref['target_ordinal']=[ordinal(s,w) for s,w in zip(ref.season,ref.week)]
    ref=ref.drop_duplicates(['season','week','team','player_key']).reset_index(drop=True)
    raw=nfl.load_nextgen_stats(seasons=SOURCE_SEASONS,stat_type='receiving'); ngs=raw.to_pandas() if hasattr(raw,'to_pandas') else pd.DataFrame(raw); ngs.columns=[str(c).strip().lower() for c in ngs.columns]
    pc=first(list(ngs.columns),PLAYER_FIELDS); tc=first(list(ngs.columns),TEAM_FIELDS)
    if pc is None or tc is None or 'season' not in ngs or 'week' not in ngs: raise RuntimeError('required NGS identity/time fields missing')
    ngs['season']=pd.to_numeric(ngs.season,errors='coerce'); ngs['week']=pd.to_numeric(ngs.week,errors='coerce')
    if 'season_type' in ngs: ngs=ngs.loc[ngs.season_type.fillna('').astype(str).str.upper().eq('REG')].copy()
    ngs=ngs.loc[regmask(ngs)].copy(); ngs['team']=ngs[tc].map(canon_team); ngs['player_key']=ngs[pc].map(key); ngs=ngs.loc[ngs.player_key.ne('')&ngs.team.notna()].copy(); ngs['obs_ordinal']=[ordinal(s,w) for s,w in zip(ngs.season,ngs.week)]
    k=['season','week','team','player_key']; dup=float(ngs.duplicated(k,keep=False).mean()) if len(ngs) else 1.0; ngs=ngs.sort_values(k).drop_duplicates(k,keep='first').copy()
    for f in FIELDS:
        if f not in ngs: ngs[f]=np.nan
        ngs[f]=pd.to_numeric(ngs[f],errors='coerce')
    # Build player histories. Same-player observations may cross teams/seasons; only chronological order matters.
    history={p:g.sort_values(['obs_ordinal','team']).reset_index(drop=True) for p,g in ngs.groupby('player_key')}
    out=[]; leakage=0
    for r in ref.itertuples(index=False):
        g=history.get(r.player_key)
        rec={'season':int(r.season),'week':int(r.week),'team':r.team,'player_key':r.player_key,'target_ordinal':int(r.target_ordinal),'prior_obs_count':0}
        if g is None:
            for f in FIELDS: rec[f'{f}_prior1']=np.nan; rec[f'{f}_prior3_mean']=np.nan
            out.append(rec); continue
        pg=g.loc[g.obs_ordinal.lt(r.target_ordinal)].copy(); leakage += int((pg.obs_ordinal>=r.target_ordinal).sum())
        rec['prior_obs_count']=int(len(pg))
        for f in FIELDS:
            vals=pg[f].dropna()
            rec[f'{f}_prior1']=float(vals.iloc[-1]) if len(vals)>=1 else np.nan
            rec[f'{f}_prior3_mean']=float(vals.iloc[-3:].mean()) if len(vals)>=3 else np.nan
        out.append(rec)
    feat=pd.DataFrame(out); feat['has_prior1']=feat.prior_obs_count.ge(1); feat['has_prior3']=feat.prior_obs_count.ge(3)
    summary=[]
    for season,g in [('POOLED',feat)]+[(str(y),feat.loc[feat.season.eq(y)]) for y in TARGET_SEASONS]:
        row={'season':season,'target_rows':int(len(g)),'prior1_rate':float(g.has_prior1.mean()) if len(g) else 0.0,'prior3_rate':float(g.has_prior3.mean()) if len(g) else 0.0,'median_prior_obs_count':float(g.prior_obs_count.median()) if len(g) else 0.0}
        for f in FIELDS:
            row[f'{f}_prior1_avail']=float(g[f'{f}_prior1'].notna().mean()) if len(g) else 0.0; row[f'{f}_prior3_avail']=float(g[f'{f}_prior3_mean'].notna().mean()) if len(g) else 0.0
        summary.append(row)
    sm=pd.DataFrame(summary); pool=sm.loc[sm.season.eq('POOLED')].iloc[0]; seasons=sm.loc[sm.season.ne('POOLED')]
    n1=sum(float(pool[f'{f}_prior1_avail'])>=.70 for f in FIELDS); n3=sum(float(pool[f'{f}_prior3_avail'])>=.60 for f in FIELDS)
    gates={'all_target_seasons_present':sorted(feat.season.unique().tolist())==TARGET_SEASONS,'duplicate_rate_le_0_01':dup<=.01,'zero_same_or_future_used':leakage==0,'pooled_prior1_ge_0_70':float(pool.prior1_rate)>=.70,'every_season_prior1_ge_0_60':bool((seasons.prior1_rate>=.60).all()),'pooled_prior3_ge_0_60':float(pool.prior3_rate)>=.60,'four_prior1_fields_ge_0_70':n1>=4,'four_prior3_fields_ge_0_60':n3>=4}
    disp='STRICT_PRIOR_NGS_FEATURES_ELIGIBLE' if all(gates.values()) else 'STRICT_PRIOR_NGS_FEATURES_INELIGIBLE'
    result={'migration':'WR_R10_STRICT_PRIOR_NGS_AVAILABILITY','source_seasons':SOURCE_SEASONS,'target_seasons':TARGET_SEASONS,'target_rows':len(feat),'ngs_unique_rows':len(ngs),'duplicate_rate':dup,'same_or_future_observations_used':leakage,'fields_prior1_ge70':int(n1),'fields_prior3_ge60':int(n3),'gates':gates,'sportsbook_inputs_used':False,'model_fitting_used':False,'production_changed':False,'disposition':disp}
    a.out_dir.mkdir(parents=True,exist_ok=True); feat.to_csv(a.out_dir/'wr_r10_strict_prior_feature_casebook.csv',index=False); sm.to_csv(a.out_dir/'wr_r10_availability_summary.csv',index=False); (a.out_dir/'wr_r10_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True)); print(sm.to_string(index=False)); return 0
if __name__=='__main__': raise SystemExit(main())
