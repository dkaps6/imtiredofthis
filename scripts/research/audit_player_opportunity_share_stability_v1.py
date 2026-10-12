#!/usr/bin/env python3
"""Player Opportunity Share Stability V1.

Frozen in docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_PLAN.md.

Diagnostic only. Tests whether strictly-prior realized opportunity-share
volatility predicts absolute next-game share surprise for RB carry share,
RB target share, WR target share and TE target share.

No sportsbook data, model residuals, projection changes or production changes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly

VERSION="PLAYER_OPPORTUNITY_SHARE_STABILITY_V1"
TARGET_SEASONS=(2024,2025)
TARGET_WEEKS=tuple(range(5,19))
PRIOR_WINDOW=8
MIN_PRIOR_GAMES=4
BOOT_REPS=5000
GAME_SEED=20261012
PLAYER_SEED=20271012
RB_POS={"RB","HB","FB"}
WR_POS={"WR","LWR","RWR","SWR"}
TE_POS={"TE"}
CELLS=("RB_CARRY_SHARE","RB_TARGET_SHARE","WR_TARGET_SHARE","TE_TARGET_SHARE")


@dataclass
class PanelRow:
    season:int
    week:int
    game_id:str
    team:str
    opponent:str
    player_identity_key:str
    player:str
    position:str
    cell:str
    actual_share:float
    prior_games:int
    prior_share_mean:float
    prior_share_sd:float
    prior_share_min:float
    prior_share_max:float
    prior_feature_max_week:int
    share_surprise:float


def _to_pandas(obj:Any)->pd.DataFrame:
    if isinstance(obj,pd.DataFrame):
        return obj.copy()
    if hasattr(obj,"to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    if hasattr(obj,"to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _lower(x:pd.DataFrame)->pd.DataFrame:
    y=x.copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y


def _num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _first(df:pd.DataFrame,names:list[str],default:Any="")->pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default,index=df.index)


def _clean(v:Any)->str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s


def _pos(v:Any)->str:
    return str(v or "").strip().upper()


def _load_schedule()->pd.DataFrame:
    import nflreadpy as nfl
    rows=[]
    for season in TARGET_SEASONS:
        try:
            raw=nfl.load_schedules(seasons=[season])
        except TypeError:
            raw=nfl.load_schedules(season)
        x=_lower(_to_pandas(raw))
        if "game_type" in x.columns:
            q=x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
            if not q.empty: x=q
        elif "season_type" in x.columns:
            q=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not q.empty: x=q
        hc="home_team" if "home_team" in x.columns else "home"
        ac="away_team" if "away_team" in x.columns else "away"
        gidc="game_id" if "game_id" in x.columns else None
        for r in x.itertuples(index=False):
            week=int(getattr(r,"week"))
            if week>18:
                continue
            home=canon_team(getattr(r,hc));away=canon_team(getattr(r,ac))
            gid=str(getattr(r,gidc)) if gidc else f"{season}_{week:02d}_{away}_{home}"
            if not home or not away: continue
            rows.append({"season":season,"week":week,"team":home,"opponent":away,"game_id":gid})
            rows.append({"season":season,"week":week,"team":away,"opponent":home,"game_id":gid})
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero schedule rows")
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate schedule team-week")
    return out.sort_values(["season","week","team"]).reset_index(drop=True)


def _load_weekly_stats(season:int)->pd.DataFrame:
    import nflreadpy as nfl
    raw=nfl.load_player_stats(seasons=[season],summary_level="week")
    return _normalize_weekly(_to_pandas(raw),season)


def _load_roster_universe(season:int,schedule:pd.DataFrame)->pd.DataFrame:
    import nflreadpy as nfl
    x=_lower(_to_pandas(nfl.load_rosters_weekly(season)))
    team_col=next((c for c in ("team","team_abbr","club_code") if c in x.columns),None)
    pos_col=next((c for c in ("position","pos","depth_chart_position","ngs_position") if c in x.columns),None)
    name_col=next((c for c in ("full_name","football_name","player_name","player","name") if c in x.columns),None)
    id_col=next((c for c in ("gsis_id","player_id","player_gsis_id") if c in x.columns),None)
    if not all((team_col,pos_col,name_col,id_col)) or "week" not in x.columns:
        raise RuntimeError(f"weekly roster schema incomplete season={season}")
    x["season"]=season
    x["week"]=_num(x["week"]).astype("Int64")
    x=x.loc[x["week"].between(1,18)].copy()
    x["team"]=x[team_col].map(canon_team)
    x["position"]=x[pos_col].map(_pos)
    x["player"]=x[name_col].astype("string").fillna("").str.strip()
    x["player_identity_key"]=x[id_col].map(_clean)

    # Preserve the broad active-roster semantics; reserve/non-roster statuses are
    # excluded when status is available. ACT/INA are both legitimate pregame
    # roster states and zero-opportunity games remain observable.
    if "status" in x.columns:
        st=x["status"].astype(str).str.upper().str.strip()
        eligible=st.isin({"ACT","INA"})
        if eligible.any():
            x=x.loc[eligible].copy()

    x=x.loc[
        x["position"].isin(RB_POS|WR_POS|TE_POS)
        & x["player"].ne("")
        & x["player_identity_key"].ne("")
    ].copy()
    s=schedule.loc[schedule["season"].eq(season)]
    x=x.merge(s,on=["season","week","team"],how="inner",validate="many_to_one")
    x=x.drop_duplicates(["season","week","team","player_identity_key"],keep="last")
    return x[[
        "season","week","game_id","team","opponent","player_identity_key","player","position"
    ]].copy()


def build_weekly_universe()->pd.DataFrame:
    schedule=_load_schedule()
    parts=[]
    for season in TARGET_SEASONS:
        roster=_load_roster_universe(season,schedule)
        stats=_load_weekly_stats(season)
        stats=stats.rename(columns={"player_identity_key":"stats_identity"})
        agg=(stats.groupby(["season","week","team","stats_identity"],as_index=False)
             .agg(targets=("targets","sum"),rushes=("rushes","sum")))
        den=(stats.groupby(["season","week","team"],as_index=False)
             .agg(team_targets=("targets","sum"),team_rushes=("rushes","sum")))
        x=roster.merge(
            agg,
            left_on=["season","week","team","player_identity_key"],
            right_on=["season","week","team","stats_identity"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["stats_identity"],errors="ignore")
        x=x.merge(den,on=["season","week","team"],how="left",validate="many_to_one")
        x["targets"]=num_or_zero(x["targets"])
        x["rushes"]=num_or_zero(x["rushes"])
        x["team_targets"]=num_or_zero(x["team_targets"])
        x["team_rushes"]=num_or_zero(x["team_rushes"])
        x["tgt_share_game"]=np.where(x["team_targets"].gt(0),x["targets"]/x["team_targets"],np.nan)
        x["rush_share_game"]=np.where(x["team_rushes"].gt(0),x["rushes"]/x["team_rushes"],np.nan)
        parts.append(x)
    out=pd.concat(parts,ignore_index=True)
    if out.empty:
        raise RuntimeError("zero weekly roster/player rows")
    return out.sort_values(["season","week","team","player_identity_key"]).reset_index(drop=True)


def num_or_zero(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce").fillna(0.0)


def _cell_rows(x:pd.DataFrame,cell:str)->pd.DataFrame:
    pos=x["position"].fillna("").astype(str).str.upper()
    if cell=="RB_CARRY_SHARE":
        q=x.loc[pos.isin(RB_POS)].copy();q["share"]=pd.to_numeric(q["rush_share_game"],errors="coerce")
    elif cell=="RB_TARGET_SHARE":
        q=x.loc[pos.isin(RB_POS)].copy();q["share"]=pd.to_numeric(q["tgt_share_game"],errors="coerce")
    elif cell=="WR_TARGET_SHARE":
        q=x.loc[pos.isin(WR_POS)].copy();q["share"]=pd.to_numeric(q["tgt_share_game"],errors="coerce")
    elif cell=="TE_TARGET_SHARE":
        q=x.loc[pos.isin(TE_POS)].copy();q["share"]=pd.to_numeric(q["tgt_share_game"],errors="coerce")
    else:
        raise KeyError(cell)
    q["cell"]=cell
    return q


def build_panel(universe:pd.DataFrame)->tuple[pd.DataFrame,dict[str,int]]:
    rows=[]
    future_violations=0
    team_violations=0
    for cell in CELLS:
        q=_cell_rows(universe,cell)
        for (season,team,pid),g in q.groupby(["season","team","player_identity_key"],sort=False):
            g=g.sort_values(["week","game_id"]).copy()
            for r in g.itertuples(index=False):
                week=int(r.week)
                if week not in TARGET_WEEKS or not np.isfinite(float(r.share)):
                    continue
                h=g.loc[g["week"].lt(week)&g["share"].notna()].tail(PRIOR_WINDOW).copy()
                if len(h)<MIN_PRIOR_GAMES:
                    continue
                vals=pd.to_numeric(h["share"],errors="coerce").dropna().to_numpy(float)
                if len(vals)<MIN_PRIOR_GAMES:
                    continue
                max_week=int(h["week"].max())
                future_violations+=int(max_week>=week)
                team_violations+=int(not h["team"].astype(str).eq(str(r.team)).all())
                mean=float(np.mean(vals))
                sd=float(np.std(vals,ddof=0))
                actual=float(r.share)
                rows.append(asdict(PanelRow(
                    season=int(season),week=week,game_id=str(r.game_id),team=str(team),
                    opponent=str(r.opponent),player_identity_key=str(pid),player=str(r.player),
                    position=str(r.position),cell=cell,actual_share=actual,prior_games=int(len(vals)),
                    prior_share_mean=mean,prior_share_sd=sd,prior_share_min=float(np.min(vals)),
                    prior_share_max=float(np.max(vals)),prior_feature_max_week=max_week,
                    share_surprise=abs(actual-mean),
                )))
    panel=pd.DataFrame(rows)
    return panel,{
        "same_future_feature_violations":int(future_violations),
        "same_team_feature_violations":int(team_violations),
    }


def _rho(x:np.ndarray,y:np.ndarray)->float:
    if len(x)<3 or np.unique(x).size<2 or np.unique(y).size<2:
        return np.nan
    return float(spearmanr(x,y).statistic)


def cluster_bootstrap(q:pd.DataFrame,cluster_col:str,seed:int)->dict[str,Any]:
    z=q[["prior_share_sd","share_surprise",cluster_col]].dropna().copy()
    groups=[
        (
            pd.to_numeric(g["prior_share_sd"],errors="coerce").to_numpy(float),
            pd.to_numeric(g["share_surprise"],errors="coerce").to_numpy(float),
        )
        for _,g in z.groupby(cluster_col,sort=True)
    ]
    if len(groups)<2:
        return {"valid_reps":0,"ci_low":np.nan,"ci_high":np.nan,"p_positive":np.nan}
    rng=np.random.default_rng(seed)
    vals=np.full(BOOT_REPS,np.nan)
    n=len(groups)
    for rep in range(BOOT_REPS):
        ids=rng.integers(0,n,size=n)
        x=np.concatenate([groups[i][0] for i in ids])
        y=np.concatenate([groups[i][1] for i in ids])
        vals[rep]=_rho(x,y)
    v=vals[np.isfinite(vals)]
    return {
        "valid_reps":int(len(v)),
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
        "p_positive":float((v>0).mean()) if len(v) else np.nan,
    }


def quartiles(q:pd.DataFrame)->list[dict[str,Any]]:
    z=q[["prior_share_sd","share_surprise"]].dropna().copy()
    if len(z)<4 or z["prior_share_sd"].nunique()<4:
        return []
    z["q"]=pd.qcut(z["prior_share_sd"],4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
    rows=[]
    for label,g in z.groupby("q",observed=True):
        rows.append({
            "quartile":str(label),"rows":int(len(g)),
            "mean_prior_share_sd":float(g["prior_share_sd"].mean()),
            "mean_share_surprise":float(g["share_surprise"].mean()),
        })
    return rows


def score_cell(q:pd.DataFrame,season:int,cell:str)->dict[str,Any]:
    z=q.loc[q["season"].eq(season)&q["cell"].eq(cell)].copy()
    rows=len(z);players=z["player_identity_key"].nunique();games=z["game_id"].nunique()
    support=rows>=300 and players>=100 and games>=50
    out={
        "season":int(season),"cell":cell,"rows":int(rows),"players":int(players),
        "games":int(games),"support":bool(support),
    }
    if not support:
        out["signal_supported"]=False
        out["quartiles"]=quartiles(z)
        return out
    rho=_rho(
        pd.to_numeric(z["prior_share_sd"],errors="coerce").to_numpy(float),
        pd.to_numeric(z["share_surprise"],errors="coerce").to_numpy(float),
    )
    gb=cluster_bootstrap(z,"game_id",GAME_SEED+season+CELLS.index(cell)*100)
    pb=cluster_bootstrap(z,"player_identity_key",PLAYER_SEED+season+CELLS.index(cell)*100)
    supported=bool(
        np.isfinite(rho) and rho>0
        and np.isfinite(gb["ci_low"]) and gb["ci_low"]>0
        and np.isfinite(pb["ci_low"]) and pb["ci_low"]>0
    )
    out.update({
        "spearman_rho":rho,
        "game_boot":gb,
        "player_boot":pb,
        "signal_supported":supported,
        "mean_prior_share_sd":float(z["prior_share_sd"].mean()),
        "mean_share_surprise":float(z["share_surprise"].mean()),
        "quartiles":quartiles(z),
    })
    return out


def audit()->tuple[pd.DataFrame,dict[str,Any]]:
    universe=build_weekly_universe()
    panel,integrity=build_panel(universe)
    if panel.empty:
        raise RuntimeError("zero scoreable share-stability rows")

    cells=[]
    per_cell=[]
    for cell in CELLS:
        by={}
        for season in TARGET_SEASONS:
            rec=score_cell(panel,season,cell)
            cells.append(rec)
            by[season]=bool(rec.get("signal_supported",False))
        replicated=by.get(2024,False) and by.get(2025,False)
        per_cell.append({
            "cell":cell,
            "season_2024_supported":bool(by.get(2024,False)),
            "season_2025_supported":bool(by.get(2025,False)),
            "disposition":(
                "PLAYER_SHARE_VOLATILITY_REPLICATED"
                if replicated else
                "PLAYER_SHARE_VOLATILITY_NOT_REPLICATED"
            ),
        })

    any_rep=any(x["disposition"]=="PLAYER_SHARE_VOLATILITY_REPLICATED" for x in per_cell)
    scoreable_identity_cov=float(panel["player_identity_key"].astype(str).ne("").mean())
    gates={
        "exact_target_seasons":sorted(panel["season"].unique().tolist())==list(TARGET_SEASONS),
        "target_weeks_within_5_18":int(panel["week"].min())>=5 and int(panel["week"].max())<=18,
        "same_future_feature_violations_zero":integrity["same_future_feature_violations"]==0,
        "same_team_feature_violations_zero":integrity["same_team_feature_violations"]==0,
        "stable_identity_coverage_ge_095":scoreable_identity_cov>=.95,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "model_residuals_used_zero":True,
        "fitted_predictive_models_zero":True,
        "candidate_projection_changes_zero":True,
        "production_changed_false":True,
    }
    result={
        "version":VERSION,
        "disposition":(
            "PLAYER_OPPORTUNITY_SHARE_STABILITY_SIGNAL_CONFIRMED"
            if any_rep else
            "PLAYER_OPPORTUNITY_SHARE_STABILITY_NULL"
        ),
        "target_seasons":list(TARGET_SEASONS),
        "target_weeks":[5,18],
        "prior_window_games":PRIOR_WINDOW,
        "min_prior_games":MIN_PRIOR_GAMES,
        "panel_rows":int(len(panel)),
        "panel_players":int(panel["player_identity_key"].nunique()),
        "scoreable_identity_coverage":scoreable_identity_cov,
        "integrity":integrity,
        "per_cell":per_cell,
        "cells":cells,
        "gates":gates,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "model_residuals_used":0,
        "fitted_predictive_models":0,
        "candidate_projection_changes":0,
        "production_changed":False,
    }
    if not all(bool(v) for v in gates.values()):
        result["disposition"]="PLAYER_OPPORTUNITY_SHARE_STABILITY_INTEGRITY_FAILURE"
    return panel,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    panel,result=audit()
    panel.to_csv(a.out_dir/"player_opportunity_share_stability_v1_rows.csv",index=False,float_format="%.12f")
    (a.out_dir/"player_opportunity_share_stability_v1_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
