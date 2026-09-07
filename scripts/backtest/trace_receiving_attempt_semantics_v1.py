#!/usr/bin/env python3
"""Frozen receiving attempt-semantics V1 trace: B0 vs C4 official-attempt target allocation."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts import simulation_v2

PASS_POS = {"WR", "LWR", "RWR", "SWR", "TE", "RB", "FB"}
WR_POS = {"WR", "LWR", "RWR", "SWR"}
TEAM_ALIAS = {"JAC": "JAX", "JAX": "JAX", "LA": "LAR", "LAR": "LAR"}


def read(p: Path) -> pd.DataFrame:
    if not p.exists() or not p.stat().st_size: raise RuntimeError(f"missing {p}")
    return pd.read_csv(p, low_memory=False)

def opt(p: Path) -> pd.DataFrame: return pd.read_csv(p, low_memory=False) if p.exists() and p.stat().st_size else pd.DataFrame()
def num(s): return pd.to_numeric(s, errors="coerce")
def team(v):
    r=str(v or "").strip().upper(); return TEAM_ALIAS.get(r,r)
def finite(v,d=0.0):
    try:
        x=float(v); return x if np.isfinite(x) else float(d)
    except Exception: return float(d)
def pgroup(pos):
    p=str(pos or "").upper().strip()
    if p in WR_POS:return "WR"
    if p=="TE":return "TE"
    if p=="RB":return "RB"
    if p=="FB":return "FB"
    return "OTHER"
def mgroup(pos):
    g=pgroup(pos); return "RB_FB" if g in {"RB","FB"} else g

def prepared(bundle):
    m=cp.build_market_frame(bundle)
    m=apply_bayesian_to_metrics(m,build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(bundle.teams,bundle.players)):
        m=simulation_rules.apply_rules_to_metrics(m)
    m=cp._attach_historical_passing_volume(m,bundle)
    m["player_clean_key"]=m["player_clean_key"].fillna("").astype(str)
    return m

def unique_players(m):
    k=["event_id","team","player_clean_key"]
    return m.sort_values(k).drop_duplicates(k,keep="last").reset_index(drop=True)
def probs(team_df):
    raw=np.array([simulation_v2._num(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0) for _,r in team_df.iterrows()],float)
    x=simulation_v2._sharpen_wr_target_shares(team_df,raw)
    x=np.clip(np.nan_to_num(np.asarray(x,float),nan=0.,posinf=0.,neginf=0.),0.,.95)
    if float(x.sum())>.95:x*=.95/float(x.sum())
    return x

def arr(sim,game,key,market):
    a=sim.values.get((str(game),str(key),str(market))); return a if a is not None and len(a) else None

def c4_sim(players: pd.DataFrame, *, iterations:int, seed:int):
    rng=np.random.default_rng(seed); vals={}; team_rows=[]
    for game,gdf in players.groupby("event_id",dropna=False,sort=False):
        pace_shock=rng.normal(0.,2.,iterations)
        for tm,tdf in gdf.groupby("team",dropna=False,sort=False):
            tdf=tdf.reset_index(drop=True); plays_mean,drop_rate=simulation_v2._team_inputs(tdf)
            rate_series=num(tdf.get("mc_pass_attempts_per_dropback",pd.Series(np.nan,index=tdf.index)))
            source_series=tdf.get("mc_pass_attempt_rate_source",pd.Series("",index=tdf.index)).fillna("").astype(str)
            rate=float(rate_series.dropna().iloc[0]) if rate_series.notna().any() else 1.0
            valid=bool(0.50<=rate<=1.00); rate=float(np.clip(rate,0.50,1.00)) if valid else 1.0
            source="historical_pregame_pbp" if valid and source_series.eq("historical_pregame_pbp").any() else "fallback_1.0"
            plays=np.rint(np.clip(rng.normal(plays_mean,3.5,iterations)+pace_shock,45,85)).astype(int)
            dropbacks=rng.binomial(plays,np.clip(rng.normal(drop_rate,.035,iterations),.25,.82))
            official=rng.binomial(dropbacks,rate)
            pass_eff=np.clip(rng.normal(1.,.09,iterations),.65,1.35)
            sh=probs(tdf); targets=simulation_v2._allocate_counts(rng,official,sh)
            team_rows.append({"event_id":str(game),"team":str(tm),"attempt_rate":rate,"attempt_rate_source":source,"plays_mean":plays_mean,"dropback_rate_mean":drop_rate,"b0_dropback_opportunity_mean":float(plays_mean*drop_rate),"c4_official_attempt_opportunity_mean":float(plays_mean*drop_rate*rate),"mc_dropbacks_mean":float(dropbacks.mean()),"mc_official_attempts_mean":float(official.mean()),"modeled_probability_mass":float(sh[[str(p or '').upper().strip() in PASS_POS for p in tdf.position]].sum())})
            for j,(_,r) in enumerate(tdf.iterrows()):
                pos=str(r.get("position","") or "").upper().strip(); key=str(r.get("player_clean_key","") or "")
                if pos not in PASS_POS: continue
                catch=simulation_v2._clip_prob(simulation_v2._num(r,"rules_catch_rate","bayes_receptions_per_target","receptions_per_target","catch_rate",default=.64),.64)
                rec=rng.binomial(targets[:,j],catch)
                vol=float(np.clip(simulation_v2._num(r,"rules_volatility_mult",default=1.),.75,1.50))
                ypt=simulation_v2._num(r,"rules_ypt","bayes_ypt","ypt"); ypt=7.5 if not np.isfinite(ypt) or ypt<=0 else ypt
                mu=targets[:,j]*ypt*pass_eff; sd=np.maximum(6.,np.sqrt(np.maximum(targets[:,j],1))*ypt*.55)*vol
                yards=np.clip(rng.normal(mu,sd),0.,None)
                vals[(str(game),str(tm),key,"targets")]=targets[:,j].astype(float)
                vals[(str(game),str(tm),key,"receptions")]=rec.astype(float)
                vals[(str(game),str(tm),key,"rec_yards")]=yards
    return vals,pd.DataFrame(team_rows)

def actual_usage(logs,season,weeks):
    x=logs.copy();x.columns=[str(c).strip().lower() for c in x.columns];x["season"]=num(x.season);x["week"]=num(x.week);x=x.loc[x.season.eq(season)&x.week.isin(sorted(weeks))].copy()
    for c in ["targets","receptions","rec_yards","pass_att"]:x[c]=num(x[c]).fillna(0.) if c in x.columns else 0.
    for c in ["team","player","player_clean_key","player_identity_key","position"]:
        if c not in x.columns:x[c]=""
        x[c]=x[c].fillna("").astype(str)
    x["team"]=x.team.map(team);x["position_group"]=x.position.map(pgroup);x["mass_group"]=x.position.map(mgroup);x["join_key"]=np.where(x.player_identity_key.str.strip().ne(""),"id:"+x.player_identity_key.str.strip(),"name:"+x.player_clean_key.str.strip())
    return x[["season","week","team","player","player_clean_key","player_identity_key","join_key","position","position_group","mass_group","targets","receptions","rec_yards","pass_att"]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--season",type=int,required=True);ap.add_argument("--prior-season",type=int,required=True);ap.add_argument("--weeks",required=True);ap.add_argument("--iterations",type=int,default=2000);ap.add_argument("--player-logs",type=Path,required=True);ap.add_argument("--team-weekly",type=Path,required=True);ap.add_argument("--schedule",type=Path,required=True);ap.add_argument("--universe-dir",type=Path,required=True);ap.add_argument("--injuries",type=Path,required=True);ap.add_argument("--weather",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args()
    logs=read(a.player_logs);tw=read(a.team_weekly);sched=read(a.schedule);inj=opt(a.injuries);weather=opt(a.weather);weeks=_parse_weeks(a.weeks);pr=[];tr=[]
    for w in weeks:
        u=read(a.universe_dir/f"{a.season}_week_{w:02d}.csv");b=build_historical_context_bundle(player_logs=logs,team_weekly=tw,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,w),weather=_exact_week(weather,a.season,w));m=prepared(b);players=unique_players(m);b0=simulation_v2.simulate(m,iterations=a.iterations,seed=42+w);c4,teams=c4_sim(players,iterations=a.iterations,seed=400000+42+w);teams["season"]=a.season;teams["week"]=w;tr.append(teams)
        for (game,tm),tdf in players.groupby(["event_id","team"],dropna=False,sort=False):
            tdf=tdf.reset_index(drop=True);sh=probs(tdf);plays,drop_rate=simulation_v2._team_inputs(tdf);rate=num(tdf.get("mc_pass_attempts_per_dropback",pd.Series(np.nan,index=tdf.index))).dropna();attempt_rate=float(rate.iloc[0]) if len(rate) else 1.;attempt_rate=attempt_rate if .5<=attempt_rate<=1 else 1.
            for j,(_,r) in enumerate(tdf.iterrows()):
                pos=str(r.get("position","") or "").upper().strip()
                if pos not in PASS_POS:continue
                key=str(r.get("player_clean_key","") or "");identity=str(r.get("player_identity_key","") or "").strip();jk=f"id:{identity}" if identity else f"name:{key}";br=arr(b0,game,key,"receptions");by=arr(b0,game,key,"rec_yards");ct=c4.get((str(game),str(tm),key,"targets"));cr=c4.get((str(game),str(tm),key,"receptions"));cy=c4.get((str(game),str(tm),key,"rec_yards"))
                pr.append({"season":a.season,"week":w,"event_id":str(game),"team":str(tm),"player":r.get("player",""),"player_clean_key":key,"player_identity_key":identity,"join_key":jk,"position":pos,"position_group":pgroup(pos),"mass_group":mgroup(pos),"b0_target_probability":float(sh[j]),"b0_expected_targets":float(plays*drop_rate*sh[j]),"c4_expected_targets":float(plays*drop_rate*attempt_rate*sh[j]),"c4_mc_targets":float(np.mean(ct)) if ct is not None else np.nan,"b0_receptions":float(np.mean(br)) if br is not None else np.nan,"c4_receptions":float(np.mean(cr)) if cr is not None else np.nan,"b0_rec_yards":float(np.mean(by)) if by is not None else np.nan,"c4_rec_yards":float(np.mean(cy)) if cy is not None else np.nan})
        print(f"[attempt-v1] season={a.season} week={w:02d} players={len(players)}")
    a.out_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(pr).to_csv(a.out_dir/"attempt_v1_player_trace.csv",index=False);pd.concat(tr,ignore_index=True).to_csv(a.out_dir/"attempt_v1_team_trace.csv",index=False);actual_usage(logs,a.season,set(weeks)).to_csv(a.out_dir/"attempt_v1_actual_usage.csv",index=False);return 0
if __name__=="__main__":raise SystemExit(main())
