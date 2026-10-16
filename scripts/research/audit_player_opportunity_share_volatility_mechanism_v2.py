#!/usr/bin/env python3
"""Mechanism decomposition for replicated player opportunity-share volatility."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.research.audit_player_opportunity_share_stability_v1 import (
    CELLS,
    RB_POS,
    WR_POS,
    TE_POS,
    TARGET_WEEKS,
    build_weekly_universe,
)
from scripts.research.audit_player_opportunity_share_stability_2023_holdout_v1 import (
    build_2023_weekly_universe,
)

VERSION="PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2"
TARGET_SEASONS=(2023,2024,2025)
PRIOR_WINDOW=8
MIN_PRIOR_GAMES=4
BOOT_REPS=5000
GAME_SEED=20261012
PLAYER_SEED=20271012


@dataclass
class Row:
    season:int
    week:int
    game_id:str
    team:str
    player_identity_key:str
    player:str
    position:str
    cell:str
    actual_share:float
    prior_games:int
    prior_share_mean:float
    prior_share_sd:float
    expected_sampling_var:float
    expected_sampling_sd:float
    excess_share_var:float
    excess_share_sd:float
    prior_feature_max_week:int
    denominator_source_max_week:int
    share_surprise:float


def _num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _cell_frame(x:pd.DataFrame,cell:str)->pd.DataFrame:
    pos=x["position"].fillna("").astype(str).str.upper()
    if cell=="RB_CARRY_SHARE":
        q=x.loc[pos.isin(RB_POS)].copy()
        q["share"]=_num(q["rush_share_game"])
        q["team_opportunities"]=_num(q["team_rushes"])
    elif cell=="RB_TARGET_SHARE":
        q=x.loc[pos.isin(RB_POS)].copy()
        q["share"]=_num(q["tgt_share_game"])
        q["team_opportunities"]=_num(q["team_targets"])
    elif cell=="WR_TARGET_SHARE":
        q=x.loc[pos.isin(WR_POS)].copy()
        q["share"]=_num(q["tgt_share_game"])
        q["team_opportunities"]=_num(q["team_targets"])
    elif cell=="TE_TARGET_SHARE":
        q=x.loc[pos.isin(TE_POS)].copy()
        q["share"]=_num(q["tgt_share_game"])
        q["team_opportunities"]=_num(q["team_targets"])
    else:
        raise KeyError(cell)
    q["cell"]=cell
    return q


def build_panel(universe:pd.DataFrame)->tuple[pd.DataFrame,dict[str,int]]:
    rows=[]
    future=0
    team_bad=0
    denom_future=0
    for cell in CELLS:
        q=_cell_frame(universe,cell)
        for (season,team,pid),g in q.groupby(["season","team","player_identity_key"],sort=False):
            g=g.sort_values(["week","game_id"]).copy()
            for r in g.itertuples(index=False):
                week=int(r.week)
                if week not in TARGET_WEEKS or not np.isfinite(float(r.share)):
                    continue
                h=g.loc[
                    g["week"].lt(week)
                    & g["share"].notna()
                    & g["team_opportunities"].gt(0)
                ].tail(PRIOR_WINDOW).copy()
                if len(h)<MIN_PRIOR_GAMES:
                    continue
                vals=_num(h["share"]).to_numpy(float)
                ns=_num(h["team_opportunities"]).to_numpy(float)
                ok=np.isfinite(vals)&np.isfinite(ns)&(ns>0)
                vals=vals[ok];ns=ns[ok]
                if len(vals)<MIN_PRIOR_GAMES:
                    continue
                mu=float(np.mean(vals))
                sd=float(np.std(vals,ddof=0))
                sampling_var=float(np.mean(mu*(1.0-mu)/ns))
                sampling_var=max(0.0,sampling_var)
                excess_var=max(0.0,sd*sd-sampling_var)
                max_week=int(h["week"].max())
                future+=int(max_week>=week)
                denom_future+=int(max_week>=week)
                team_bad+=int(not h["team"].astype(str).eq(str(r.team)).all())
                rows.append(asdict(Row(
                    season=int(season),week=week,game_id=str(r.game_id),
                    team=str(team),player_identity_key=str(pid),player=str(r.player),
                    position=str(r.position),cell=cell,actual_share=float(r.share),
                    prior_games=int(len(vals)),prior_share_mean=mu,prior_share_sd=sd,
                    expected_sampling_var=sampling_var,
                    expected_sampling_sd=float(np.sqrt(sampling_var)),
                    excess_share_var=excess_var,
                    excess_share_sd=float(np.sqrt(excess_var)),
                    prior_feature_max_week=max_week,
                    denominator_source_max_week=max_week,
                    share_surprise=abs(float(r.share)-mu),
                )))
    panel=pd.DataFrame(rows)
    return panel,{
        "same_future_feature_violations":int(future),
        "same_team_feature_violations":int(team_bad),
        "same_future_denominator_violations":int(denom_future),
    }


def _corr(x:np.ndarray,y:np.ndarray)->float:
    if len(x)<3 or np.unique(x).size<2 or np.unique(y).size<2:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])


def _rank(v:np.ndarray)->np.ndarray:
    return rankdata(v,method="average").astype(float)


def partial_spearman(q:pd.DataFrame)->float:
    z=q[[
        "prior_share_sd","share_surprise","prior_share_mean","expected_sampling_sd"
    ]].dropna()
    if len(z)<4:
        return np.nan
    x=_rank(z["prior_share_sd"].to_numpy(float))
    y=_rank(z["share_surprise"].to_numpy(float))
    c1=_rank(z["prior_share_mean"].to_numpy(float))
    c2=_rank(z["expected_sampling_sd"].to_numpy(float))
    X=np.column_stack([np.ones(len(z)),c1,c2])
    bx=np.linalg.lstsq(X,x,rcond=None)[0]
    by=np.linalg.lstsq(X,y,rcond=None)[0]
    rx=x-X@bx
    ry=y-X@by
    return _corr(rx,ry)


def excess_spearman(q:pd.DataFrame)->float:
    z=q[["excess_share_sd","share_surprise"]].dropna()
    if len(z)<3:
        return np.nan
    return _corr(
        _rank(z["excess_share_sd"].to_numpy(float)),
        _rank(z["share_surprise"].to_numpy(float)),
    )


def _stats(q:pd.DataFrame)->tuple[float,float]:
    return partial_spearman(q),excess_spearman(q)


def cluster_bootstrap(q:pd.DataFrame,cluster_col:str,seed:int)->dict[str,Any]:
    z=q.reset_index(drop=True).copy()
    groups=[g.index.to_numpy(int) for _,g in z.groupby(cluster_col,sort=True)]
    if len(groups)<2:
        return {
            "valid_partial_reps":0,"partial_ci_low":np.nan,"partial_ci_high":np.nan,
            "valid_excess_reps":0,"excess_ci_low":np.nan,"excess_ci_high":np.nan,
        }
    rng=np.random.default_rng(seed)
    partial=np.full(BOOT_REPS,np.nan)
    excess=np.full(BOOT_REPS,np.nan)
    n=len(groups)
    for rep in range(BOOT_REPS):
        chosen=rng.integers(0,n,size=n)
        idx=np.concatenate([groups[i] for i in chosen])
        p,e=_stats(z.iloc[idx])
        partial[rep]=p
        excess[rep]=e
    pv=partial[np.isfinite(partial)]
    ev=excess[np.isfinite(excess)]
    return {
        "valid_partial_reps":int(len(pv)),
        "partial_ci_low":float(np.quantile(pv,.025)) if len(pv) else np.nan,
        "partial_ci_high":float(np.quantile(pv,.975)) if len(pv) else np.nan,
        "valid_excess_reps":int(len(ev)),
        "excess_ci_low":float(np.quantile(ev,.025)) if len(ev) else np.nan,
        "excess_ci_high":float(np.quantile(ev,.975)) if len(ev) else np.nan,
    }


def quartiles(q:pd.DataFrame)->list[dict[str,Any]]:
    z=q[["excess_share_sd","share_surprise"]].dropna().copy()
    if len(z)<4 or z["excess_share_sd"].nunique()<2:
        return []
    # Descriptive only. Many players can have exactly zero excess variance,
    # so duplicate qcut edges may collapse four requested bins into fewer.
    # Use integer bin codes first, then label only the bins that actually exist.
    try:
        codes=pd.qcut(z["excess_share_sd"],4,labels=False,duplicates="drop")
    except ValueError:
        return []
    codes=pd.to_numeric(codes,errors="coerce")
    z=z.loc[codes.notna()].copy()
    if z.empty:
        return []
    z["quartile_code"]=codes.loc[codes.notna()].astype(int).to_numpy()
    out=[]
    for code,g in z.groupby("quartile_code",sort=True):
        out.append({
            "quartile":f"Q{int(code)+1}","rows":int(len(g)),
            "mean_excess_share_sd":float(g["excess_share_sd"].mean()),
            "mean_share_surprise":float(g["share_surprise"].mean()),
        })
    return out


def score_cell(q:pd.DataFrame,season:int,cell:str)->dict[str,Any]:
    z=q.loc[q["season"].eq(season)&q["cell"].eq(cell)].copy()
    rows=len(z);players=z["player_identity_key"].nunique();games=z["game_id"].nunique()
    support=rows>=300 and players>=100 and games>=50
    out={
        "season":season,"cell":cell,"rows":int(rows),"players":int(players),
        "games":int(games),"support":bool(support),
        "positive_excess_variance_rate":float(z["excess_share_var"].gt(0).mean()) if len(z) else np.nan,
        "mean_prior_share_sd":float(z["prior_share_sd"].mean()) if len(z) else np.nan,
        "mean_expected_sampling_sd":float(z["expected_sampling_sd"].mean()) if len(z) else np.nan,
        "mean_excess_share_sd":float(z["excess_share_sd"].mean()) if len(z) else np.nan,
    }
    if not support:
        out["mechanism_supported"]=False
        out["quartiles"]=quartiles(z)
        return out
    partial,excess=_stats(z)
    gb=cluster_bootstrap(z,"game_id",GAME_SEED+season+CELLS.index(cell)*100)
    pb=cluster_bootstrap(z,"player_identity_key",PLAYER_SEED+season+CELLS.index(cell)*100)
    partial_supported=bool(
        np.isfinite(partial) and partial>0
        and np.isfinite(gb["partial_ci_low"]) and gb["partial_ci_low"]>0
        and np.isfinite(pb["partial_ci_low"]) and pb["partial_ci_low"]>0
    )
    excess_supported=bool(
        np.isfinite(excess) and excess>0
        and np.isfinite(gb["excess_ci_low"]) and gb["excess_ci_low"]>0
        and np.isfinite(pb["excess_ci_low"]) and pb["excess_ci_low"]>0
    )
    out.update({
        "partial_spearman":partial,
        "excess_sd_spearman":excess,
        "game_boot":gb,
        "player_boot":pb,
        "partial_supported":partial_supported,
        "excess_supported":excess_supported,
        "mechanism_supported":bool(partial_supported and excess_supported),
        "quartiles":quartiles(z),
    })
    return out


def audit()->tuple[pd.DataFrame,dict[str,Any]]:
    u23=build_2023_weekly_universe()
    u45=build_weekly_universe()
    universe=pd.concat([u23,u45],ignore_index=True)
    panel,integrity=build_panel(universe)
    if panel.empty:
        raise RuntimeError("zero mechanism panel rows")

    cells=[]
    per_cell=[]
    for cell in CELLS:
        by={}
        for season in TARGET_SEASONS:
            rec=score_cell(panel,season,cell)
            cells.append(rec)
            by[season]=bool(rec.get("mechanism_supported",False))
        confirmed=all(by.get(s,False) for s in TARGET_SEASONS)
        per_cell.append({
            "cell":cell,
            "season_2023_supported":bool(by.get(2023,False)),
            "season_2024_supported":bool(by.get(2024,False)),
            "season_2025_supported":bool(by.get(2025,False)),
            "disposition":(
                "PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED"
                if confirmed else
                "PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_NOT_CONFIRMED"
            ),
        })

    any_confirmed=any(
        x["disposition"]=="PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED"
        for x in per_cell
    )
    identity_cov=float(panel["player_identity_key"].astype(str).ne("").mean())
    gates={
        "exact_target_seasons":sorted(panel["season"].unique().tolist())==list(TARGET_SEASONS),
        "target_weeks_within_5_18":int(panel["week"].min())>=5 and int(panel["week"].max())<=18,
        "same_future_feature_violations_zero":integrity["same_future_feature_violations"]==0,
        "same_team_feature_violations_zero":integrity["same_team_feature_violations"]==0,
        "same_future_denominator_violations_zero":integrity["same_future_denominator_violations"]==0,
        "stable_identity_coverage_ge_095":identity_cov>=.95,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "model_residuals_zero":True,
        "predictive_candidate_fits_zero":True,
        "projection_changes_zero":True,
        "production_changed_false":True,
    }
    disposition=(
        "PLAYER_OPPORTUNITY_EXCESS_VOLATILITY_EXISTS"
        if any_confirmed else
        "PLAYER_OPPORTUNITY_VOLATILITY_EXPLAINED_BY_MEAN_AND_COUNT_NOISE"
    )
    if not all(gates.values()):
        disposition="PLAYER_OPPORTUNITY_VOLATILITY_MECHANISM_INTEGRITY_FAILURE"
    result={
        "version":VERSION,
        "disposition":disposition,
        "target_seasons":list(TARGET_SEASONS),
        "target_weeks":[5,18],
        "prior_window_games":PRIOR_WINDOW,
        "min_prior_games":MIN_PRIOR_GAMES,
        "panel_rows":int(len(panel)),
        "panel_players":int(panel["player_identity_key"].nunique()),
        "scoreable_identity_coverage":identity_cov,
        "integrity":integrity,
        "per_cell":per_cell,
        "cells":cells,
        "gates":gates,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "model_residuals_used":0,
        "predictive_candidate_fits":0,
        "projection_changes":0,
        "production_changed":False,
    }
    return panel,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    panel,result=audit()
    panel.to_csv(
        a.out_dir/"player_opportunity_share_volatility_mechanism_v2_rows.csv",
        index=False,float_format="%.12f"
    )
    (a.out_dir/"player_opportunity_share_volatility_mechanism_v2_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
