"""Shadow C2 QB-distribution simulator for the Phase-J production candidate.

This module deliberately does not replace production simulation_v2. It replays
that simulator byte-for-byte while capturing its shared team states, then applies
the frozen C2 receiving process. Candidate pricing may expose only the resulting
primary-QB pass_yards array; receiver/rushing production arrays remain canonical.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np, pandas as pd
from scripts.config import MC
from scripts.simulation_v2 import (_num,_clip_prob,_team_inputs,_allocate_counts,_top_n_shares,
    _sharpen_wr_target_shares,_player_key,WR_POSITIONS,MARKET_MAP)

PASS_CATCHER_POSITIONS={"WR","LWR","RWR","SWR","TE","RB","FB"}
C2_RESIDUAL_CATCH_RATE=.64; C2_RESIDUAL_YPT=7.5; C2_YPR_MIN=3.; C2_YPR_MAX=35.

@dataclass
class StateSimulationResult:
    values: Dict[tuple[str,str,str],np.ndarray]
    iterations:int
    team_states:Dict[tuple[str,str,str],np.ndarray]

def simulate_with_states(metrics:pd.DataFrame,*,iterations:int|None=None,seed:int|None=None)->StateSimulationResult:
    iterations=int(iterations or MC.get('iterations',25000));seed=int(MC.get('seed',42) if seed is None else seed);rng=np.random.default_rng(seed);values={};states={}
    if metrics.empty:return StateSimulationResult(values,iterations,states)
    frame=metrics.copy();frame['player_clean_key']=frame.apply(_player_key,axis=1)
    game_key='event_id' if 'event_id' in frame.columns and frame['event_id'].notna().any() else None
    if game_key is None:
        frame['_game_key']=frame.apply(lambda r:'|'.join(sorted([str(r.get('team','')),str(r.get('opponent',''))])),axis=1);game_key='_game_key'
    players=frame.sort_values([game_key,'team','player_clean_key']).drop_duplicates([game_key,'team','player_clean_key'],keep='last')
    for game,gdf in players.groupby(game_key,dropna=False):
        game_pace_shock=rng.normal(0.,2.,iterations)
        for team,tdf in gdf.groupby('team',dropna=False):
            if pd.isna(team) or not str(team).strip():continue
            plays_mean,pass_rate_mean=_team_inputs(tdf);plays=np.rint(np.clip(rng.normal(plays_mean,3.5,iterations)+game_pace_shock,45,85)).astype(int);pass_rate=np.clip(rng.normal(pass_rate_mean,.035,iterations),.25,.82);pass_att=rng.binomial(plays,pass_rate);rush_att=plays-pass_att
            pass_eff=np.clip(rng.normal(1.,.09,iterations),.65,1.35);rush_eff=np.clip(rng.normal(1.,.10,iterations),.60,1.40)
            gs=str(game);ts=str(team);states[(gs,ts,'plays')]=plays.copy();states[(gs,ts,'pass_rate')]=pass_rate.copy();states[(gs,ts,'pass_att')]=pass_att.copy();states[(gs,ts,'rush_att')]=rush_att.copy();states[(gs,ts,'pass_eff_shock')]=pass_eff.copy();states[(gs,ts,'rush_eff_shock')]=rush_eff.copy()
            raw_t=np.array([_num(r,'rules_tgt_share','bayes_tgt_share','target_share','tgt_share',default=0.) for _,r in tdf.iterrows()]);tshares=_sharpen_wr_target_shares(tdf,raw_t);raw_r=np.array([_num(r,'rules_rush_share','bayes_rush_share','rush_share',default=0.) for _,r in tdf.iterrows()]);rshares=_top_n_shares(raw_r,5)
            targets=_allocate_counts(rng,pass_att,tshares);carries=_allocate_counts(rng,rush_att,rshares)
            for j,(_,row) in enumerate(tdf.iterrows()):
                pkey=_player_key(row)
                if not pkey:continue
                role=str(row.get('model_role',row.get('role','')) or '').upper();pos=str(row.get('position','') or '').upper();catch=_clip_prob(_num(row,'rules_catch_rate','bayes_receptions_per_target','receptions_per_target','catch_rate',default=.64),.64);recs=rng.binomial(targets[:,j],catch);vol=float(np.clip(_num(row,'rules_volatility_mult',default=1.),.75,1.50))
                ypt=_num(row,'rules_ypt','bayes_ypt','ypt');ypt=7.5 if not np.isfinite(ypt) or ypt<=0 else ypt;rec_mu=targets[:,j]*ypt*pass_eff;rec_sd=np.maximum(6.,np.sqrt(np.maximum(targets[:,j],1))*ypt*.55)*vol;rec_y=np.clip(rng.normal(rec_mu,rec_sd),0.,None)
                ypc=_num(row,'rules_ypc','bayes_ypc','ypc');ypc=4.2 if not np.isfinite(ypc) or ypc<=0 else ypc;rush_mu=carries[:,j]*ypc*rush_eff;rush_sd=np.maximum(3.,np.sqrt(np.maximum(carries[:,j],1))*ypc*.65)*vol;rush_y=np.clip(rng.normal(rush_mu,rush_sd),0.,None)
                values[(gs,pkey,'receptions')]=recs.astype(float);values[(gs,pkey,'rec_yards')]=rec_y;values[(gs,pkey,'rush_att')]=carries[:,j].astype(float);values[(gs,pkey,'rush_yards')]=rush_y;values[(gs,pkey,'rush_rec_yards')]=rush_y+rec_y
                if pos=='QB' or role.startswith('QB'):
                    ypa=_num(row,'rules_ypa','bayes_ypa','ypa','ypa_prior');ypa=7. if not np.isfinite(ypa) or ypa<=0 else ypa;noise=np.clip(rng.normal(1.,.07*vol,iterations),.72,1.28);values[(gs,pkey,'pass_yards')]=np.clip(pass_att*ypa*pass_eff*noise,0.,None)
                td=_num(row,'offensive_td_rate')
                if np.isfinite(td) and td>=0:
                    rz=_num(row,'rz_share',default=np.nan);rzm=float(np.clip(.75+rz,.75,1.35)) if np.isfinite(rz) else 1.;wp=_num(row,'team_wp');sm=1.+(.08*(wp-.5) if np.isfinite(wp) else 0.);lam=max(0.,td*rzm*sm);shock=np.clip(rng.normal(1.,.12,iterations),.65,1.35);p=np.clip(1.-np.exp(-lam*shock),.001,.98);values[(gs,pkey,'anytime_td')]=rng.binomial(1,p).astype(float)
    return StateSimulationResult(values,iterations,states)

def _primary_qb_row(tdf:pd.DataFrame):
    c=[]
    for idx,row in tdf.iterrows():
        pos=str(row.get('position','') or '').upper().strip();role=str(row.get('model_role',row.get('role','')) or '').upper().strip()
        if pos!='QB' and not role.startswith('QB'):continue
        c.append((_num(row,'qb_projection_eligible',default=0.),_num(row,'qb_role_score',default=0.),str(_player_key(row)),idx,row))
    if not c:return None
    c.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True);return c[0][4]

def apply_c2(base:StateSimulationResult,metrics:pd.DataFrame,*,anchor_map:dict[tuple[str,str],float],seed:int=5601)->StateSimulationResult:
    rng=np.random.default_rng(int(seed));values={k:np.asarray(v).copy() for k,v in base.values.items()};frame=metrics.copy();frame['player_clean_key']=frame.apply(_player_key,axis=1)
    game_key='event_id' if 'event_id' in frame.columns and frame['event_id'].notna().any() else None
    if game_key is None:frame['_game_key']=frame.apply(lambda r:'|'.join(sorted([str(r.get('team','')),str(r.get('opponent',''))])),axis=1);game_key='_game_key'
    players=frame.sort_values([game_key,'team','player_clean_key']).drop_duplicates([game_key,'team','player_clean_key'],keep='last')
    for game,gdf in players.groupby(game_key,dropna=False):
        for team,tdf0 in gdf.groupby('team',dropna=False):
            if pd.isna(team) or not str(team).strip():continue
            gs=str(game);ts=str(team);tdf=tdf0.reset_index(drop=True);pass_att=np.asarray(base.team_states[(gs,ts,'pass_att')],int);pass_eff=np.asarray(base.team_states[(gs,ts,'pass_eff_shock')],float)
            raw=np.array([_num(r,'rules_tgt_share','bayes_tgt_share','target_share','tgt_share',default=0.) for _,r in tdf.iterrows()],float);shares=_sharpen_wr_target_shares(tdf,raw);positions=tdf.get('position',pd.Series('',index=tdf.index)).fillna('').astype(str).str.upper().str.strip().to_numpy();mask=np.isin(positions,list(PASS_CATCHER_POSITIONS));shares=np.where(mask,shares,0.);targets=_allocate_counts(rng,pass_att,shares);res_t=np.maximum(0,pass_att-targets.sum(1));yards={}
            for j,(_,row) in enumerate(tdf.iterrows()):
                if not mask[j]:continue
                pk=_player_key(row)
                if not pk:continue
                catch=_clip_prob(_num(row,'rules_catch_rate','bayes_receptions_per_target','receptions_per_target','catch_rate',default=C2_RESIDUAL_CATCH_RATE),C2_RESIDUAL_CATCH_RATE);recs=rng.binomial(targets[:,j],catch);ypt=_num(row,'rules_ypt','bayes_ypt','ypt');ypt=C2_RESIDUAL_YPT if not np.isfinite(ypt) or ypt<=0 else float(ypt);ypr=float(np.clip(ypt/catch,C2_YPR_MIN,C2_YPR_MAX));vol=float(np.clip(_num(row,'rules_volatility_mult',default=1.),.75,1.5));mu=recs.astype(float)*ypr*pass_eff;sd=np.maximum(3.,np.sqrt(np.maximum(recs,1))*ypr*.55)*vol;y=np.clip(rng.normal(mu,sd),0.,None);yards[pk]=np.where(recs>0,y,0.)
            rr=rng.binomial(res_t,C2_RESIDUAL_CATCH_RATE);rypr=C2_RESIDUAL_YPT/C2_RESIDUAL_CATCH_RATE;rmu=rr.astype(float)*rypr*pass_eff;rsd=np.maximum(3.,np.sqrt(np.maximum(rr,1))*rypr*.55);res_y=np.where(rr>0,np.clip(rng.normal(rmu,rsd),0.,None),0.)
            raw_total=(np.sum(np.vstack(list(yards.values())),axis=0) if yards else np.zeros(base.iterations))+res_y;anchor=float(anchor_map.get((gs,ts),np.nan))
            if not np.isfinite(anchor) or anchor<=0 or not np.isfinite(raw_total.mean()) or raw_total.mean()<=0:raise RuntimeError(f'invalid C2 anchor/raw mean {gs} {ts} anchor={anchor}')
            scale=anchor/float(raw_total.mean());qb_y=raw_total*scale;qr=_primary_qb_row(tdf)
            if qr is not None:
                qk=_player_key(qr)
                if qk:values[(gs,qk,'pass_yards')]=qb_y
    return StateSimulationResult(values,base.iterations,base.team_states)

def lookup(result:StateSimulationResult,row:pd.Series,market:str):
    game=row.get('event_id')
    if pd.isna(game) or not str(game).strip():game='|'.join(sorted([str(row.get('team','')),str(row.get('opponent',''))]))
    return result.values.get((str(game),_player_key(row),MARKET_MAP.get(str(market).lower(),str(market).lower())))
