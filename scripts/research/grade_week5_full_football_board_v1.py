#!/usr/bin/env python3
"""Week-5 full underlying football-board postmortem grader.

Grades one unique football projection per player-market from the immutable
preserved Week-5 board. Multiple sportsbook offers do not count as independent
football forecasts. No model fitting or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.operations.grade_market_track_record_gsis_v1 import (
    _snap_participation_sets,
    _status_map,
    build_alias_index,
    load_actual_stats_unfiltered,
    load_roster_identity,
    load_snap_participation,
    resolve_gsis,
)

PRIMARY_MARKETS=("pass_yards","rush_yards","rec_yards","receptions","rush_rec_yards")
TOL=1e-8


def num(s):
    return pd.to_numeric(s,errors="coerce")


def _uniq_numeric(values):
    x=num(pd.Series(values)).dropna().to_numpy(float)
    if len(x)==0:
        return []
    return sorted(set(np.round(x,12).tolist()))


def _median_unique(values):
    x=np.asarray(_uniq_numeric(values),dtype=float)
    return float(np.median(x)) if len(x) else np.nan


def _position_map(actual:pd.DataFrame, roster:pd.DataFrame)->dict[str,str]:
    out={}
    for src in (roster,actual):
        if "position" not in src.columns:
            continue
        for gid,pos in zip(src["gsis_id"].astype(str),src["position"].astype(str)):
            p=str(pos).strip().upper()
            if gid and p and p!="NAN":
                out[gid]=p
    return out


def build_unique_board(priced:pd.DataFrame)->pd.DataFrame:
    x=priced.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    x=x.loc[x["market"].isin(PRIMARY_MARKETS)].copy()
    for c in ("season","week"):
        x[c]=num(x[c]).astype(int)
    x["team"]=x["team"].map(canon_team)
    x["opponent"]=x["opponent"].map(canon_team)
    group=["season","week","event_id","player_clean_key","player","team","opponent","market"]
    rows=[]
    for key,g in x.groupby(group,dropna=False,sort=True):
        mp=_uniq_numeric(g["model_proj"])
        ep=_uniq_numeric(g["ensemble_proj"]) if "ensemble_proj" in g.columns else mp
        if len(mp)!=1:
            raise RuntimeError(f"model projection drift within offers {key}: {mp[:8]}")
        if len(ep)!=1:
            raise RuntimeError(f"ensemble projection drift within offers {key}: {ep[:8]}")
        if abs(mp[0]-ep[0])>1e-6:
            # Preserve both, but model_proj is the historical priced-board mean.
            pass
        line=_median_unique(g["vegas_line"])
        row=dict(zip(group,key))
        row.update({
            "model_proj":float(mp[0]),
            "ensemble_proj":float(ep[0]),
            "median_book_line":line,
            "min_book_line":float(num(g["vegas_line"]).min()),
            "max_book_line":float(num(g["vegas_line"]).max()),
            "book_line_count":int(num(g["vegas_line"]).dropna().nunique()),
            "offer_rows":int(len(g)),
            "max_positive_edge_pct":float(max(0.0,num(g["edge_pct"]).max())) if "edge_pct" in g.columns else np.nan,
            "max_fair_prob":float(num(g["fair_prob"]).max()) if "fair_prob" in g.columns else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def attach_actuals(board:pd.DataFrame, season:int, week:int)->pd.DataFrame:
    actual=load_actual_stats_unfiltered(season,[week])
    roster=load_roster_identity(season,[week])
    snaps=load_snap_participation(season,[week])
    idx=build_alias_index(actual,roster)
    status_map=_status_map(roster)
    snap_exact,snap_base=_snap_participation_sets(snaps)
    rostered=set(zip(roster["season"],roster["week"],roster["team"],roster["gsis_id"]))
    pos_map=_position_map(actual,roster)

    resolved=[]
    for r in board.itertuples(index=False):
        gsis,status=resolve_gsis(str(r.player_clean_key),str(r.team),idx)
        resolved.append((gsis,status))
    out=board.copy()
    out["gsis_id"]=[x[0] for x in resolved]
    out["identity_status"]=[x[1] for x in resolved]
    out["roster_confirmed"]=[
        (int(s),int(w),t,g) in rostered
        for s,w,t,g in zip(out["season"],out["week"],out["team"],out["gsis_id"])
    ]
    out["roster_status"]=[
        status_map.get((int(s),int(w),t,g),"")
        for s,w,t,g in zip(out["season"],out["week"],out["team"],out["gsis_id"])
    ]
    from scripts.operations.grade_market_track_record_gsis_v1 import _suffix_strip
    out["snap_participated"]=[
        (
            (int(s),int(w),t,k) in snap_exact
            or (int(s),int(w),t,_suffix_strip(k)) in snap_base
        )
        for s,w,t,k in zip(out["season"],out["week"],out["team"],out["player_clean_key"])
    ]
    out["position"]=out["gsis_id"].map(pos_map).fillna("UNKNOWN")

    actual_map={}
    for r in actual.itertuples(index=False):
        actual_map[(int(r.season),int(r.week),str(r.gsis_id))]=r

    vals=[]; statuses=[]; sources=[]
    inactive={"INA","INACTIVE","DNP"}
    for r in out.itertuples(index=False):
        key=(int(r.season),int(r.week),str(r.gsis_id))
        a=actual_map.get(key)
        value=np.nan
        if a is not None:
            if r.market=="pass_yards": value=float(a.pass_yards)
            elif r.market=="rush_yards": value=float(a.rush_yards)
            elif r.market=="rec_yards": value=float(a.rec_yards)
            elif r.market=="receptions": value=float(a.receptions)
            elif r.market=="rush_rec_yards": value=float(a.rush_yards)+float(a.rec_yards)
            statuses.append("SETTLED")
            sources.append("stats_table")
        elif r.identity_status=="RESOLVED_GSIS" and bool(r.snap_participated):
            value=0.0
            statuses.append("SETTLED")
            sources.append("snap_confirmed_verified_zero")
        elif (
            r.identity_status=="RESOLVED_GSIS"
            and bool(r.roster_confirmed)
            and str(r.roster_status).strip().upper() in inactive
            and not bool(r.snap_participated)
        ):
            statuses.append("DNP_INACTIVE")
            sources.append("roster_inactive")
        else:
            statuses.append("UNRESOLVED")
            sources.append("unresolved")
        vals.append(value)
    out["actual"]=vals
    out["football_settlement_status"]=statuses
    out["actual_source"]=sources
    return out


def score_rows(x:pd.DataFrame)->dict[str,Any]:
    z=x.loc[x["football_settlement_status"].eq("SETTLED")].copy()
    if z.empty:
        return {"rows":0}
    z["model_error"]=num(z["model_proj"])-num(z["actual"])
    z["line_error"]=num(z["median_book_line"])-num(z["actual"])
    z["abs_model_error"]=z["model_error"].abs()
    z["abs_line_error"]=z["line_error"].abs()
    z["model_line_gap"]=(num(z["model_proj"])-num(z["median_book_line"]))
    z["abs_model_line_gap"]=z["model_line_gap"].abs()
    valid_line=num(z["median_book_line"]).notna()
    side=np.where(z["model_line_gap"]>0,"OVER",np.where(z["model_line_gap"]<0,"UNDER","PUSH"))
    actual_side=np.where(num(z["actual"])>num(z["median_book_line"]),"OVER",
                 np.where(num(z["actual"])<num(z["median_book_line"]),"UNDER","PUSH"))
    decided=valid_line & (side!="PUSH") & (actual_side!="PUSH")
    direction_hit=float(np.mean(np.asarray(side)[decided]==np.asarray(actual_side)[decided])) if decided.any() else np.nan
    corr=np.nan
    if valid_line.sum()>=5 and z.loc[valid_line,"abs_model_line_gap"].nunique()>1:
        corr=float(z.loc[valid_line,["abs_model_line_gap","abs_model_error"]].corr(method="spearman").iloc[0,1])
    return {
        "rows":int(len(z)),
        "mae":float(z["abs_model_error"].mean()),
        "rmse":float(np.sqrt(np.mean(z["model_error"]**2))),
        "bias_model_minus_actual":float(z["model_error"].mean()),
        "median_line_mae":float(z.loc[valid_line,"abs_line_error"].mean()) if valid_line.any() else np.nan,
        "median_line_bias":float(z.loc[valid_line,"line_error"].mean()) if valid_line.any() else np.nan,
        "model_closer_than_line_rate":float((z.loc[valid_line,"abs_model_error"] < z.loc[valid_line,"abs_line_error"]).mean()) if valid_line.any() else np.nan,
        "direction_hit_rate":direction_hit,
        "median_abs_model_line_gap":float(z.loc[valid_line,"abs_model_line_gap"].median()) if valid_line.any() else np.nan,
        "spearman_abs_gap_vs_abs_error":corr,
    }


def bucket_summary(detail:pd.DataFrame)->list[dict[str,Any]]:
    z=detail.loc[detail["football_settlement_status"].eq("SETTLED")].copy()
    z["abs_model_line_gap"]=(num(z["model_proj"])-num(z["median_book_line"])).abs()
    z["abs_model_error"]=(num(z["model_proj"])-num(z["actual"])).abs()
    z=z.loc[z["abs_model_line_gap"].notna()].copy()
    if len(z)<10:
        return []
    z["gap_quintile"]=pd.qcut(z["abs_model_line_gap"].rank(method="first"),5,labels=["Q1","Q2","Q3","Q4","Q5"])
    rows=[]
    for q,g in z.groupby("gap_quintile",observed=True):
        side=np.where(num(g["model_proj"])>num(g["median_book_line"]),"OVER","UNDER")
        actual_side=np.where(num(g["actual"])>num(g["median_book_line"]),"OVER",
                     np.where(num(g["actual"])<num(g["median_book_line"]),"UNDER","PUSH"))
        decided=actual_side!="PUSH"
        rows.append({
            "quintile":str(q),
            "rows":int(len(g)),
            "mean_abs_model_line_gap":float(g["abs_model_line_gap"].mean()),
            "mae":float(g["abs_model_error"].mean()),
            "direction_hit_rate":float(np.mean(side[decided]==actual_side[decided])) if np.any(decided) else np.nan,
        })
    return rows


def build_summary(detail:pd.DataFrame)->dict[str,Any]:
    settled=detail.loc[detail["football_settlement_status"].eq("SETTLED")].copy()
    groups=[]
    def add(scope,frame,position="",market=""):
        rec={"scope":scope,"position":position,"market":market}
        rec.update(score_rows(frame))
        rec["dnp_inactive_rows"]=int((frame["football_settlement_status"]=="DNP_INACTIVE").sum())
        rec["unresolved_rows"]=int((frame["football_settlement_status"]=="UNRESOLVED").sum())
        groups.append(rec)
    add("OVERALL",detail)
    for m,g in detail.groupby("market"): add("MARKET",g,market=str(m))
    for p,g in detail.groupby("position"): add("POSITION",g,position=str(p))
    for (p,m),g in detail.groupby(["position","market"]): add("POSITION_MARKET",g,position=str(p),market=str(m))

    s=settled.copy()
    s["abs_error"]=(num(s["model_proj"])-num(s["actual"])).abs()
    s["signed_error"]=num(s["model_proj"])-num(s["actual"])
    s["abs_gap"]=(num(s["model_proj"])-num(s["median_book_line"])).abs()
    worst=s.sort_values("abs_error",ascending=False).head(40)
    cols=["player","team","opponent","position","market","model_proj","median_book_line","actual","signed_error","abs_error","abs_gap"]
    return {
        "version":"WEEK5_FULL_FOOTBALL_BOARD_POSTMORTEM_V1",
        "season":2026,
        "week":5,
        "unique_player_market_rows":int(len(detail)),
        "settled_rows":int((detail["football_settlement_status"]=="SETTLED").sum()),
        "dnp_inactive_rows":int((detail["football_settlement_status"]=="DNP_INACTIVE").sum()),
        "unresolved_rows":int((detail["football_settlement_status"]=="UNRESOLVED").sum()),
        "scorecard":groups,
        "gap_quintiles":bucket_summary(detail),
        "worst_misses":worst[cols].to_dict("records"),
        "production_changed":False,
        "model_fitting":False,
        "sportsbook_refetch":False,
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    priced=pd.read_csv(args.priced,low_memory=False)
    board=build_unique_board(priced)
    detail=attach_actuals(board,2026,5)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(args.out_dir/"week5_full_board_graded_detail.csv",index=False)
    summary=build_summary(detail)
    (args.out_dir/"week5_full_board_summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    pd.DataFrame(summary["scorecard"]).to_csv(args.out_dir/"week5_full_board_scorecard.csv",index=False)
    pd.DataFrame(summary["gap_quintiles"]).to_csv(args.out_dir/"week5_full_board_gap_quintiles.csv",index=False)
    pd.DataFrame(summary["worst_misses"]).to_csv(args.out_dir/"week5_full_board_worst_misses.csv",index=False)
    print(json.dumps(summary,indent=2,sort_keys=True,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
