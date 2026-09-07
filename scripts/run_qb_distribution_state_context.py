#!/usr/bin/env python3
"""Build strict-prior Phase-J QB distribution-state team context for an active slate."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scripts._opponent_map import canon_team
from scripts.backtest.cross_position_catastrophic_game_spot_v1 import aggregate_season
from scripts.runtime_context import resolve_season, resolve_week

TEAM_MAP=Path('data/team_week_map.csv')
OUT=Path('data/qb_distribution_state_context.csv')
AUDIT=Path('data/qb_distribution_state_context_audit.csv')

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
        deff=hist[hist.opponent.eq(opp)].tail(5)  # offenses faced by this opponent defense
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
