#!/usr/bin/env python3
"""Prospective future-week lock for replicated player opportunity-share volatility."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research.audit_player_opportunity_share_stability_v1 import (
    _load_roster_universe,
    _load_weekly_stats,
    num_or_zero,
)
from scripts.research.audit_player_state_future_week_coverage_v1 import _load_schedule_all
from scripts.utils.player_identity_v3 import historical_identity_key

VERSION="PLAYER_OPPORTUNITY_SHARE_VOLATILITY_FUTURE_WEEK_LOCK_V1"
PRIOR_WINDOW=8
MIN_PRIOR_GAMES=4
RB_POS={"RB","HB","FB"}
WR_POS={"WR","LWR","RWR","SWR"}
TE_POS={"TE"}


def _num(s:pd.Series)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def _clean(v:Any)->str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s


def _schedule_team_rows(season:int)->pd.DataFrame:
    raw=_load_schedule_all(season)
    rows=[]
    for r in raw.itertuples(index=False):
        gid=f"{season}_{int(r.week):02d}_{r.away}_{r.home}"
        rows.append({"season":season,"week":int(r.week),"team":str(r.home),"opponent":str(r.away),"game_id":gid})
        rows.append({"season":season,"week":int(r.week),"team":str(r.away),"opponent":str(r.home),"game_id":gid})
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero schedule team rows")
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate schedule team-week")
    return out


def build_prior_weekly_universe(season:int,target_week:int)->pd.DataFrame:
    schedule=_schedule_team_rows(season)
    roster=_load_roster_universe(season,schedule)
    roster=roster.loc[_num(roster["week"]).lt(target_week)].copy()

    stats=_load_weekly_stats(season)
    bad=sorted(_num(stats["week"]).dropna().astype(int).loc[lambda s:s.ge(target_week)].unique().tolist())
    if bad:
        raise RuntimeError(f"target/future stats already present: {bad}")
    stats=stats.loc[_num(stats["week"]).lt(target_week)].copy()
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
    x["tgt_share_game"]=(x["targets"]/x["team_targets"]).where(x["team_targets"].gt(0))
    x["rush_share_game"]=(x["rushes"]/x["team_rushes"]).where(x["team_rushes"].gt(0))
    if not x.empty and int(_num(x["week"]).max())>=target_week:
        raise RuntimeError("chronology violation in prior weekly universe")
    return x.sort_values(["season","week","team","player_identity_key"]).reset_index(drop=True)


def _feature(vals:pd.Series,weeks:pd.Series)->dict[str,Any]:
    q=pd.DataFrame({"share":_num(vals),"week":_num(weeks)}).dropna().sort_values("week").tail(PRIOR_WINDOW)
    if len(q)<MIN_PRIOR_GAMES:
        return {
            "supported":False,"prior_games":int(len(q)),
            "mean":np.nan,"sd":np.nan,"min":np.nan,"max":np.nan,
            "source_max_week":int(q["week"].max()) if len(q) else np.nan,
        }
    arr=q["share"].to_numpy(float)
    return {
        "supported":True,"prior_games":int(len(q)),
        "mean":float(np.mean(arr)),"sd":float(np.std(arr,ddof=0)),
        "min":float(np.min(arr)),"max":float(np.max(arr)),
        "source_max_week":int(q["week"].max()),
    }


def lock_from_frames(
    *,
    parent_rows:pd.DataFrame,
    parent_result:dict[str,Any],
    weekly:pd.DataFrame,
    season:int,
    target_week:int,
)->tuple[pd.DataFrame,dict[str,Any]]:
    if parent_result.get("version")!="PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1":
        raise RuntimeError("wrong parent player-state version")
    if int(parent_result.get("season",-1))!=season or int(parent_result.get("target_week",-1))!=target_week:
        raise RuntimeError("parent target mismatch")
    if parent_result.get("disposition")!="PLAYER_STATE_FUTURE_WEEK_READY":
        raise RuntimeError(f"parent not READY: {parent_result.get('disposition')}")
    if not bool(parent_result.get("target_week_roster_source_ready",False)):
        raise RuntimeError("parent lacks exact target-week roster")
    if not bool((parent_result.get("prior_week_source_completion") or {}).get("complete",False)):
        raise RuntimeError("parent prior-week source incomplete")
    if int(parent_result.get("chronology_violations",-1))!=0:
        raise RuntimeError("parent chronology violation")

    p=parent_rows.copy()
    p.columns=[str(c).strip().lower() for c in p.columns]
    required={"team","player","position_group","gsis_id","name_key","roster_source_week"}
    miss=required-set(p.columns)
    if miss:
        raise RuntimeError(f"parent rows missing {sorted(miss)}")
    if not _num(p["roster_source_week"]).eq(target_week).all():
        raise RuntimeError("parent rows are not exact target-week roster")

    p["team"]=p["team"].map(canon_team)
    p["gsis_id"]=p["gsis_id"].map(_clean)
    p["player_identity_key"]=[
        historical_identity_key(pid,name,team)
        for pid,name,team in zip(p["gsis_id"],p["player"],p["team"])
    ]

    w=weekly.copy()
    w.columns=[str(c).strip().lower() for c in w.columns]
    if not w.empty and int(_num(w["week"]).max())>=target_week:
        raise RuntimeError("weekly source contains target/future week")

    rows=[]
    chronology=0
    for r in p.itertuples(index=False):
        pos=str(r.position_group).upper().strip()
        if pos not in {"RB","WR","TE"}:
            continue
        hist=w.loc[
            w["player_identity_key"].astype(str).eq(str(r.player_identity_key))
            & w["team"].map(canon_team).eq(str(r.team))
            & _num(w["week"]).lt(target_week)
        ].copy()
        target_f=_feature(hist["tgt_share_game"] if len(hist) else pd.Series(dtype=float),
                          hist["week"] if len(hist) else pd.Series(dtype=float))
        rush_f=_feature(hist["rush_share_game"] if len(hist) else pd.Series(dtype=float),
                        hist["week"] if len(hist) else pd.Series(dtype=float)) if pos=="RB" else {
            "supported":False,"prior_games":0,"mean":np.nan,"sd":np.nan,"min":np.nan,"max":np.nan,"source_max_week":np.nan
        }
        for f in (target_f,rush_f):
            if f["supported"] and int(f["source_max_week"])>=target_week:
                chronology+=1
        rows.append({
            "season":season,"target_week":target_week,
            "team":str(r.team),"opponent":str(getattr(r,"opponent","")),
            "player":str(r.player),"position":pos,
            "gsis_id":str(r.gsis_id),"player_identity_key":str(r.player_identity_key),
            "target_share_supported":bool(target_f["supported"]),
            "target_share_prior_games":int(target_f["prior_games"]),
            "prior_target_share_mean":target_f["mean"],
            "prior_target_share_sd":target_f["sd"],
            "prior_target_share_min":target_f["min"],
            "prior_target_share_max":target_f["max"],
            "target_share_source_max_week":target_f["source_max_week"],
            "rush_share_supported":bool(rush_f["supported"]) if pos=="RB" else False,
            "rush_share_prior_games":int(rush_f["prior_games"]) if pos=="RB" else 0,
            "prior_rush_share_mean":rush_f["mean"] if pos=="RB" else np.nan,
            "prior_rush_share_sd":rush_f["sd"] if pos=="RB" else np.nan,
            "prior_rush_share_min":rush_f["min"] if pos=="RB" else np.nan,
            "prior_rush_share_max":rush_f["max"] if pos=="RB" else np.nan,
            "rush_share_source_max_week":rush_f["source_max_week"] if pos=="RB" else np.nan,
        })

    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero RB/WR/TE target-week rows")

    rb=out["position"].eq("RB")
    wr=out["position"].eq("WR")
    te=out["position"].eq("TE")
    result={
        "version":VERSION,
        "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "season":season,"target_week":target_week,
        "status":f"WEEK{target_week}_OPPORTUNITY_SHARE_VOLATILITY_LOCK_FROZEN",
        "players":int(len(out)),
        "rb_players":int(rb.sum()),"wr_players":int(wr.sum()),"te_players":int(te.sum()),
        "rb_carry_supported":int((rb&out["rush_share_supported"]).sum()),
        "rb_target_supported":int((rb&out["target_share_supported"]).sum()),
        "wr_target_supported":int((wr&out["target_share_supported"]).sum()),
        "te_target_supported":int((te&out["target_share_supported"]).sum()),
        "chronology_violations":int(chronology),
        "unsupported_rows_imputed":0,
        "thresholds_applied":0,
        "fitted_coefficients":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "target_week_outcomes_read":0,
        "projection_changes":0,
        "production_changed":False,
    }
    if chronology!=0:
        raise RuntimeError("share-volatility lock chronology violation")
    return out,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--target-week",type=int,required=True)
    ap.add_argument("--parent-rows",type=Path,required=True)
    ap.add_argument("--parent-result",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    parent_rows=pd.read_csv(a.parent_rows,low_memory=False)
    parent_result=json.loads(a.parent_result.read_text())
    weekly=build_prior_weekly_universe(a.season,a.target_week)
    rows,result=lock_from_frames(
        parent_rows=parent_rows,parent_result=parent_result,weekly=weekly,
        season=a.season,target_week=a.target_week,
    )
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rows.to_csv(a.out_dir/f"player_opportunity_share_volatility_week{a.target_week}_lock.csv",index=False,float_format="%.12f")
    (a.out_dir/f"player_opportunity_share_volatility_week{a.target_week}_lock.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
