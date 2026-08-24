#!/usr/bin/env python3
"""Migration 50: audit leakage-safe pregame game-script / market signals for QB passing.
Diagnostic only; no production football logic changes.

Uses nflverse schedule market fields that are known before kickoff (spread/total and,
when present, moneyline). These are joined only to stable-primary QB games from the
historical backtest and are never derived from target-game outcomes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def num(x): return pd.to_numeric(x, errors='coerce')
def read(p):
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f'missing {p}')
    return pd.read_csv(p)
def corr(a,b):
    z=pd.DataFrame({'a':num(a),'b':num(b)}).dropna()
    return float(z.a.corr(z.b)) if len(z)>2 and z.a.nunique()>1 and z.b.nunique()>1 else np.nan

def load_schedule(season:int)->pd.DataFrame:
    import nflreadpy as nfl
    x=nfl.load_schedules(int(season))
    if hasattr(x,'to_pandas'): x=x.to_pandas()
    x=pd.DataFrame(x); x.columns=[str(c).strip().lower() for c in x.columns]
    if 'game_type' in x: x=x[x.game_type.astype(str).str.upper().eq('REG')].copy()
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--season',type=int,default=2025);p.add_argument('--participation-trace',type=Path,default=Path('data/backtests/qb_participation_role/qb_participation_player_trace.csv'));p.add_argument('--out-dir',type=Path,default=Path('data/backtests/qb_gamescript_market'));a=p.parse_args()
    z=read(a.participation_trace)
    z=z[(num(z.get('is_actual_primary')).eq(1)) & (num(z.get('actual_qb_attempt_share')).ge(.80))].copy()
    if z.empty: raise RuntimeError('no stable-primary QB rows')
    s=load_schedule(a.season)
    required={'week','home_team','away_team'}
    if not required.issubset(s.columns): raise RuntimeError(f'schedule missing {sorted(required-set(s.columns))}')
    market_cols=[c for c in ['spread_line','total_line','home_moneyline','away_moneyline','home_rest','away_rest','roof','surface'] if c in s.columns]
    s=s[['week','home_team','away_team']+market_cols].copy(); s['week']=num(s.week).astype('Int64')
    rows=[]
    for _,r in s.iterrows():
        spread=num(pd.Series([r.get('spread_line')])).iloc[0] if 'spread_line' in s else np.nan
        total=num(pd.Series([r.get('total_line')])).iloc[0] if 'total_line' in s else np.nan
        for side in ['home','away']:
            team=str(r[f'{side}_team']); opp=str(r['away_team' if side=='home' else 'home_team'])
            team_spread=spread if side=='home' else (-spread if pd.notna(spread) else np.nan)
            implied=(total-team_spread)/2 if pd.notna(total) and pd.notna(team_spread) else np.nan
            opp_implied=(total+team_spread)/2 if pd.notna(total) and pd.notna(team_spread) else np.nan
            rec={'week':int(r.week),'team':team,'opponent':opp,'market_home_away':side,'market_spread':team_spread,'market_total':total,'market_team_implied':implied,'market_opp_implied':opp_implied,'market_is_underdog':int(team_spread>0) if pd.notna(team_spread) else np.nan,'market_abs_spread':abs(team_spread) if pd.notna(team_spread) else np.nan}
            mlcol=f'{side}_moneyline'
            if mlcol in s: rec['market_moneyline']=num(pd.Series([r.get(mlcol)])).iloc[0]
            rows.append(rec)
    m=pd.DataFrame(rows)
    x=z.merge(m,on=['week','team','opponent'],how='left',validate='many_to_one')
    x['pass_error']=num(x.mc_proj)-num(x.actual_pass_yards_raw);x['abs_pass_error']=x.pass_error.abs();x['attempt_error']=num(x.pred_attempts)-num(x.actual_pass_att);x['abs_attempt_error']=x.attempt_error.abs();x['ypa_error']=num(x.pred_ypa)-num(x.actual_ypa);x['abs_ypa_error']=x.ypa_error.abs()
    x['catastrophic_100plus']=(x.abs_pass_error>=100).astype(int);x['big_underprojection_100plus']=(x.pass_error<=-100).astype(int)
    signals=[c for c in ['market_spread','market_abs_spread','market_total','market_team_implied','market_opp_implied','market_is_underdog','market_moneyline'] if c in x.columns]
    rank=[]
    for c in signals:
        rank.append({'signal':c,'n':num(x[c]).notna().sum(),'corr_actual_attempts':corr(x[c],x.actual_pass_att),'corr_attempt_error':corr(x[c],x.attempt_error),'corr_actual_pass_yards':corr(x[c],x.actual_pass_yards_raw),'corr_signed_pass_error':corr(x[c],x.pass_error),'corr_abs_pass_error':corr(x[c],x.abs_pass_error),'corr_catastrophic_100plus':corr(x[c],x.catastrophic_100plus),'corr_big_underprojection_100plus':corr(x[c],x.big_underprojection_100plus)})
    buckets=[]
    for c in signals:
        d=pd.DataFrame({'v':num(x[c]),'att':num(x.actual_pass_att),'yards':num(x.actual_pass_yards_raw),'err':num(x.pass_error),'abs_err':num(x.abs_pass_error),'cat':x.catastrophic_100plus}).dropna(subset=['v'])
        if len(d)<20 or d.v.nunique()<4: continue
        try:d['q']=pd.qcut(d.v.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
        except Exception:continue
        for q,g in d.groupby('q',observed=True): buckets.append({'signal':c,'bucket':str(q),'n':len(g),'signal_mean':g.v.mean(),'actual_attempts':g.att.mean(),'actual_pass_yards':g.yards.mean(),'signed_pass_error':g.err.mean(),'mae':g.abs_err.mean(),'catastrophic_rate':g.cat.mean()})
    groups=[]
    for label,g in [('lt50',x[x.abs_pass_error<50]),('100plus',x[x.abs_pass_error>=100]),('under100plus',x[x.pass_error<=-100]),('over100plus',x[x.pass_error>=100])]:
        rec={'group':label,'n':len(g),'mean_actual_attempts':num(g.actual_pass_att).mean(),'mean_pred_attempts':num(g.pred_attempts).mean(),'mean_actual_ypa':num(g.actual_ypa).mean(),'mean_pred_ypa':num(g.pred_ypa).mean()}
        for c in signals: rec[c]=num(g[c]).mean()
        groups.append(rec)
    a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/'qb_gamescript_market_trace.csv',index=False);pd.DataFrame(rank).to_csv(a.out_dir/'qb_gamescript_signal_ranking.csv',index=False);pd.DataFrame(buckets).to_csv(a.out_dir/'qb_gamescript_quartiles.csv',index=False);pd.DataFrame(groups).to_csv(a.out_dir/'qb_gamescript_error_groups.csv',index=False)
    print('=== SIGNAL RANKING ===');print(pd.DataFrame(rank).to_string(index=False));print('\n=== ERROR GROUPS ===');print(pd.DataFrame(groups).to_string(index=False));print('\n=== QUARTILES ===');print(pd.DataFrame(buckets).to_string(index=False));return 0
if __name__=='__main__': raise SystemExit(main())
