#!/usr/bin/env python3
"""Build strict-prior Phase-J QB distribution-state team context for an active slate.

Production runtime is self-contained: the PBP aggregation helper is embedded here
rather than importing from scripts/backtest/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_season, resolve_week
from scripts.utils.pbp import get_pbp

TEAM_MAP=Path('data/team_week_map.csv')
OUT=Path('data/qb_distribution_state_context.csv')
AUDIT=Path('data/qb_distribution_state_context_audit.csv')


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default,index=df.index,dtype=float)
    return pd.to_numeric(df[col],errors='coerce')


def _regular(x: pd.DataFrame) -> pd.DataFrame:
    if 'season_type' in x.columns:
        q=x.loc[x['season_type'].astype(str).str.upper().eq('REG')].copy()
        if not q.empty: return q
    if 'game_type' in x.columns:
        q=x.loc[x['game_type'].astype(str).str.upper().eq('REG')].copy()
        if not q.empty: return q
    return x


def aggregate_season(season: int) -> pd.DataFrame:
    x=_regular(_lower(get_pbp(int(season),min_rows=1)))
    x['season']=pd.to_numeric(x.get('season'),errors='coerce')
    x['week']=pd.to_numeric(x.get('week'),errors='coerce')
    x=x.loc[x['season'].eq(int(season)) & x['week'].between(1,18)].copy()
    x['season']=int(season); x['week']=x['week'].astype(int)
    x['team']=x.get('posteam',pd.Series('',index=x.index)).map(canon_team)
    x['opponent']=x.get('defteam',pd.Series('',index=x.index)).map(canon_team)
    x=x.loc[x['team'].ne('') & x['opponent'].ne('')].copy()

    sack=_num(x,'sack',0).fillna(0).eq(1)
    pass_attempt=_num(x,'pass_attempt',0).fillna(0).eq(1) & ~sack
    qb_scramble=_num(x,'qb_scramble',0).fillna(0).eq(1)
    qb_kneel=_num(x,'qb_kneel',0).fillna(0).eq(1)
    rush_attempt=_num(x,'rush_attempt',0).fillna(0).eq(1) & ~qb_scramble & ~qb_kneel
    off_play=_num(x,'qb_dropback',0).fillna(0).eq(1) | _num(x,'rush_attempt',0).fillna(0).eq(1)

    py=_num(x,'passing_yards',np.nan)
    if py.notna().sum()==0: py=_num(x,'yards_gained',0)
    py=py.fillna(0)
    ry=_num(x,'rushing_yards',np.nan)
    if ry.notna().sum()==0: ry=_num(x,'yards_gained',0)
    ry=ry.fillna(0)
    epa=_num(x,'epa'); success=_num(x,'success')
    complete=pass_attempt & _num(x,'complete_pass',0).fillna(0).eq(1)
    yac=_num(x,'yards_after_catch')

    x['_off_play']=off_play.astype(float); x['_pass']=pass_attempt.astype(float); x['_rush']=rush_attempt.astype(float)
    x['_pass_yards']=np.where(pass_attempt,py,0.0); x['_rush_yards']=np.where(rush_attempt,ry,0.0)
    x['_pass_epa']=np.where(pass_attempt,epa,np.nan); x['_rush_epa']=np.where(rush_attempt,epa,np.nan)
    x['_pass_success']=np.where(pass_attempt,success,np.nan); x['_rush_success']=np.where(rush_attempt,success,np.nan)
    x['_pass_expl20']=np.where(pass_attempt,py.ge(20).astype(float),np.nan); x['_pass_expl40']=np.where(pass_attempt,py.ge(40).astype(float),np.nan)
    x['_rush_expl10']=np.where(rush_attempt,ry.ge(10).astype(float),np.nan); x['_rush_expl20']=np.where(rush_attempt,ry.ge(20).astype(float),np.nan)
    x['_yac_comp']=np.where(complete,yac,np.nan)

    g=(x.groupby(['season','week','team','opponent'],as_index=False).agg(
        off_plays=('_off_play','sum'),pass_att=('_pass','sum'),rush_att=('_rush','sum'),
        pass_yards=('_pass_yards','sum'),rush_yards=('_rush_yards','sum'),
        pass_epa=('_pass_epa','mean'),rush_epa=('_rush_epa','mean'),
        pass_success=('_pass_success','mean'),rush_success=('_rush_success','mean'),
        pass_expl20=('_pass_expl20','mean'),pass_expl40=('_pass_expl40','mean'),
        rush_expl10=('_rush_expl10','mean'),rush_expl20=('_rush_expl20','mean'),
        yac_per_completion=('_yac_comp','mean')))
    den=g['pass_att']+g['rush_att']
    g['pass_rate']=np.where(den>0,g['pass_att']/den,np.nan); g['rush_rate']=np.where(den>0,g['rush_att']/den,np.nan)
    g['pass_ypa']=np.where(g['pass_att']>0,g['pass_yards']/g['pass_att'],np.nan)
    g['rush_ypc']=np.where(g['rush_att']>0,g['rush_yards']/g['rush_att'],np.nan)
    print(f'[qb_state_context] aggregated PBP {season}: {len(g)} team-games')
    return g


def mean(x,c):
    s=pd.to_numeric(x[c],errors='coerce') if c in x.columns else pd.Series(dtype=float)
    return float(s.mean()) if s.notna().any() else np.nan


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--season',type=int,default=None);ap.add_argument('--week',type=int,default=None);ap.add_argument('--out',default=str(OUT));a=ap.parse_args()
    season=int(a.season if a.season is not None else resolve_season());week=int(a.week if a.week is not None else resolve_week())
    if not TEAM_MAP.exists(): raise RuntimeError(f'missing {TEAM_MAP}')
    tm=pd.read_csv(TEAM_MAP,low_memory=False);tm.columns=[str(c).lower() for c in tm.columns]
    tm['team']=tm['team'].map(canon_team);tm['opponent']=tm['opponent'].map(canon_team)
    slate=tm[(pd.to_numeric(tm['season'],errors='coerce')==season)&(pd.to_numeric(tm['week'],errors='coerce')==week)].copy()
    if 'bye' in slate.columns: slate=slate[~slate['bye'].fillna(False).astype(bool)]
    slate=slate[slate.team.ne('')&slate.opponent.ne('')].drop_duplicates('team')
    if len(slate)!=32: raise RuntimeError(f'expected 32 active team rows, got {len(slate)}')

    seasons=list(range(2019,season))
    hist=pd.concat([aggregate_season(s) for s in seasons],ignore_index=True,sort=False)
    if week>1:
        try:
            cur=aggregate_season(season);cur=cur[cur.week<week]
            if not cur.empty: hist=pd.concat([hist,cur],ignore_index=True,sort=False)
        except Exception as exc:
            print(f'[qb_state_context] current PBP unavailable/unused: {type(exc).__name__}: {exc}')
    hist['team']=hist.team.map(canon_team);hist['opponent']=hist.opponent.map(canon_team)
    hist=hist.sort_values(['season','week'])

    rows=[]
    for _,r in slate.iterrows():
        team=canon_team(r.team);opp=canon_team(r.opponent)
        off=hist[hist.team.eq(team)].tail(5)
        deff=hist[hist.opponent.eq(opp)].tail(5)
        if len(off)<1 or len(deff)<1: raise RuntimeError(f'missing strict-prior history team={team} opponent={opp}')
        rec={'season':season,'week':week,'team':team,'opponent':opp,'off_history_n':len(off),'def_history_n':len(deff),
             'off_prior_off_plays':mean(off,'off_plays'),'off_prior_pass_att':mean(off,'pass_att'),'off_prior_rush_att':mean(off,'rush_att'),'off_prior_pass_rate':mean(off,'pass_rate'),'off_prior_rush_rate':mean(off,'rush_rate'),
             'def_prior_pass_att_faced':mean(deff,'pass_att'),'def_prior_rush_att_faced':mean(deff,'rush_att'),'def_prior_pass_ypa_allowed':mean(deff,'pass_ypa'),'def_prior_rush_ypc_allowed':mean(deff,'rush_ypc'),'def_prior_pass_epa_allowed':mean(deff,'pass_epa'),'def_prior_rush_epa_allowed':mean(deff,'rush_epa'),'def_prior_pass_success_allowed':mean(deff,'pass_success'),'def_prior_rush_success_allowed':mean(deff,'rush_success'),'def_prior_pass_expl20_allowed':mean(deff,'pass_expl20'),'def_prior_pass_expl40_allowed':mean(deff,'pass_expl40'),'def_prior_rush_expl10_allowed':mean(deff,'rush_expl10'),'def_prior_rush_expl20_allowed':mean(deff,'rush_expl20'),'def_prior_yac_per_completion_allowed':mean(deff,'yac_per_completion')}
        rows.append(rec)
    t=pd.DataFrame(rows)
    zcols=['off_prior_off_plays','off_prior_pass_att','off_prior_rush_att','off_prior_pass_rate','off_prior_rush_rate','def_prior_pass_att_faced','def_prior_rush_att_faced','def_prior_pass_ypa_allowed','def_prior_rush_ypc_allowed','def_prior_pass_epa_allowed','def_prior_rush_epa_allowed','def_prior_pass_success_allowed','def_prior_rush_success_allowed','def_prior_pass_expl20_allowed','def_prior_pass_expl40_allowed','def_prior_rush_expl10_allowed','def_prior_rush_expl20_allowed','def_prior_yac_per_completion_allowed']
    for c in zcols:
        s=pd.to_numeric(t[c],errors='coerce');sd=float(s.std());t['z_'+c]=(s-float(s.mean()))/(sd if np.isfinite(sd) and sd>0 else np.nan)
    t['pass_opportunity_spot']=t[['z_off_prior_pass_att','z_off_prior_pass_rate','z_off_prior_off_plays','z_def_prior_pass_att_faced']].mean(axis=1,skipna=True)
    t['pass_efficiency_spot']=t[['z_def_prior_pass_ypa_allowed','z_def_prior_pass_epa_allowed','z_def_prior_pass_success_allowed','z_def_prior_pass_expl20_allowed','z_def_prior_yac_per_completion_allowed']].mean(axis=1,skipna=True)
    t['rush_opportunity_spot']=t[['z_off_prior_rush_att','z_off_prior_rush_rate','z_off_prior_off_plays','z_def_prior_rush_att_faced']].mean(axis=1,skipna=True)
    t['rush_efficiency_spot']=t[['z_def_prior_rush_ypc_allowed','z_def_prior_rush_epa_allowed','z_def_prior_rush_success_allowed','z_def_prior_rush_expl10_allowed','z_def_prior_rush_expl20_allowed']].mean(axis=1,skipna=True)
    required=['pass_opportunity_spot','pass_efficiency_spot','rush_opportunity_spot','rush_efficiency_spot']
    if t[required].isna().any().any(): raise RuntimeError('non-finite Phase-J spot context')
    t['qb_distribution_state_context_version']='PHASE_J_SPOT_CONTEXT_V1';t['sportsbook_inputs_used']=0
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);t.to_csv(p,index=False)
    t[['season','week','team','opponent','off_history_n','def_history_n']+required+['sportsbook_inputs_used']].to_csv(AUDIT,index=False)
    print(f'[qb_state_context] wrote {len(t)} teams season={season} week={week} -> {p}')
    return 0


if __name__=='__main__': raise SystemExit(main())
