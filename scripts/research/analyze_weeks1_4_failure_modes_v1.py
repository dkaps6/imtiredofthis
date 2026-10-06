#!/usr/bin/env python3
"""Weeks 1-4 production failure-mode atlas.

Diagnostic only. Uses the frozen settled production ledger to distinguish:
- systematic center/mean error from skewed catastrophic tails;
- uncertainty-scale failures relative to each row's own model_sd;
- probability ranking from probability calibration;
- MC -> ensemble -> final movement;
- rush+receiving error decomposition into rushing and receiving components.

No parameter fitting, no candidate rule, no sportsbook refetch and no
production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def num(s):
    return pd.to_numeric(s, errors="coerce")


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    mask=np.isfinite(score)
    y=np.asarray(y)[mask]
    score=np.asarray(score)[mask]
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1==0 or n0==0:
        return float("nan")
    ranks=pd.Series(score).rank(method="average").to_numpy(float)
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def _proper_scores(y: pd.Series, p: pd.Series) -> dict:
    yy=num(y).to_numpy(float); pp=num(p).to_numpy(float)
    mask=np.isfinite(yy)&np.isfinite(pp)
    yy=yy[mask]; pp=np.clip(pp[mask],1e-6,1-1e-6)
    if not len(yy):
        return {"n":0,"mean_probability":None,"brier":None,"log_loss":None,"auc":None}
    return {
        "n":int(len(yy)),
        "mean_probability":float(pp.mean()),
        "brier":float(np.mean((pp-yy)**2)),
        "log_loss":float(-np.mean(yy*np.log(pp)+(1-yy)*np.log(1-pp))),
        "auc":_auc(yy,pp),
    }


def _summary(q: pd.DataFrame) -> dict:
    if q.empty:
        return {"n":0}
    return {
        "n":int(len(q)),
        "win_rate":float(q["win"].mean()),
        "units":float(q["unit_result"].sum()),
        "model_mae":float(q["abs_model_error"].mean()),
        "line_mae":float(q["abs_line_error"].mean()),
        "model_mean_error":float(q["model_error"].mean()),
        "model_median_error":float(q["model_error"].median()),
        "model_error_q10":float(q["model_error"].quantile(.10)),
        "model_error_q90":float(q["model_error"].quantile(.90)),
        "line_mean_error":float(q["line_error"].mean()),
        "line_median_error":float(q["line_error"].median()),
        "model_closer_rate":float((q["abs_model_error"]<q["abs_line_error"]).mean()),
        "actual_above_model_rate":float((q["actual"]>q["model_proj"]).mean()),
        "abs_z_gt_2_rate":float((q["abs_z"]>2).mean()),
        "abs_z_gt_3_rate":float((q["abs_z"]>3).mean()),
        "mean_abs_z":float(q["abs_z"].mean()),
    }


def _paired(q: pd.DataFrame, left: str, right: str) -> dict:
    z=q.loc[num(q[left]).notna() & num(q[right]).notna() & num(q["actual"]).notna()].copy()
    if z.empty:
        return {"n":0}
    a=(num(z[left])-num(z["actual"])).abs()
    b=(num(z[right])-num(z["actual"])).abs()
    return {
        "n":int(len(z)),
        f"{left}_mae":float(a.mean()),
        f"{right}_mae":float(b.mean()),
        "paired_abs_error_improvement":float((a-b).mean()),
        "right_better_rate":float((b<a).mean()),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--graded",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    d=pd.read_csv(a.graded,low_memory=False)
    d=d.loc[d["bet_result"].isin(["WIN","LOSS"])].copy()
    if d.empty:
        raise RuntimeError("no decided rows")

    needed=["model_proj","mc_proj","ensemble_proj","model_sd","actual","vegas_line",
            "unit_result","fair_prob","market_prob","week","market","position",
            "bet_result","side","event_id","player","player_clean_key","team"]
    missing=sorted(set(needed)-set(d.columns))
    if missing:
        raise RuntimeError(f"graded ledger missing columns: {missing}")

    for c in ["model_proj","mc_proj","ensemble_proj","model_sd","actual","vegas_line",
              "unit_result","fair_prob","market_prob","edge_abs","edge_pct"]:
        if c in d.columns:
            d[c]=num(d[c])

    if (d["model_sd"]<=0).any() or d["model_sd"].isna().any():
        raise RuntimeError("model_sd must be finite and >0 for every decided row")

    d["win"]=d["bet_result"].eq("WIN").astype(float)
    d["model_error"]=d["model_proj"]-d["actual"]
    d["line_error"]=d["vegas_line"]-d["actual"]
    d["abs_model_error"]=d["model_error"].abs()
    d["abs_line_error"]=d["line_error"].abs()
    d["z"]=(d["actual"]-d["model_proj"])/d["model_sd"]
    d["abs_z"]=d["z"].abs()
    d["model_line_gap"]=d["model_proj"]-d["vegas_line"]
    d["normalized_gap"]=d["model_line_gap"].abs()/d["model_sd"]

    overall={
        "all":_summary(d),
        "week4":_summary(d.loc[d.week.eq(4)]),
    }

    group_rows=[]
    for group_type,col in [("week","week"),("position","position"),("market","market")]:
        for key,q in d.groupby(col,dropna=False):
            rec={"group_type":group_type,"group_value":str(key),**_summary(q)}
            group_rows.append(rec)
    groups=pd.DataFrame(group_rows)

    tail_rows=[]
    for threshold in (2,3):
        q=d.loc[d["abs_z"].gt(threshold)].copy()
        tail_rows.append({
            "threshold_sd":threshold,
            "rows":int(len(q)),
            "share_of_decided":float(len(q)/len(d)),
            "loss_rate":float(q["bet_result"].eq("LOSS").mean()) if len(q) else None,
            "selected_under_rate":float(q["side"].astype(str).str.upper().eq("UNDER").mean()) if len(q) else None,
            "actual_above_model_rate":float((q["actual"]>q["model_proj"]).mean()) if len(q) else None,
            "units":float(q["unit_result"].sum()) if len(q) else 0.0,
        })
    tails=pd.DataFrame(tail_rows)

    p_bins=[0,.55,.60,.65,.70,.80,.90,1.0000001]
    d["fair_prob_band"]=pd.cut(d["fair_prob"],p_bins,right=False)
    conf=[]
    for band,q in d.groupby("fair_prob_band",observed=True):
        conf.append({
            "band":str(band),
            "rows":int(len(q)),
            "mean_stated_probability":float(q["fair_prob"].mean()),
            "realized_win_rate":float(q["win"].mean()),
            "calibration_gap":float(q["win"].mean()-q["fair_prob"].mean()),
            "abs_z_gt_2_rate":float((q["abs_z"]>2).mean()),
            "model_closer_rate":float((q["abs_model_error"]<q["abs_line_error"]).mean()),
            "model_mae":float(q["abs_model_error"].mean()),
            "units":float(q["unit_result"].sum()),
        })
    confidence=pd.DataFrame(conf)

    probability={
        "fair_prob":_proper_scores(d["win"],d["fair_prob"]),
        "market_prob":_proper_scores(d["win"],d["market_prob"]),
        "constant_0_5":_proper_scores(d["win"],pd.Series(0.5,index=d.index)),
        "week4_fair_prob":_proper_scores(d.loc[d.week.eq(4),"win"],d.loc[d.week.eq(4),"fair_prob"]),
        "week4_market_prob":_proper_scores(d.loc[d.week.eq(4),"win"],d.loc[d.week.eq(4),"market_prob"]),
    }

    d["normalized_gap_quintile"]=pd.qcut(d["normalized_gap"],5,duplicates="drop")
    rank_rows=[]
    for band,q in d.groupby("normalized_gap_quintile",observed=True):
        rank_rows.append({
            "quintile":str(band),
            "rows":int(len(q)),
            "mean_normalized_gap":float(q["normalized_gap"].mean()),
            "mean_fair_prob":float(q["fair_prob"].mean()),
            "realized_win_rate":float(q["win"].mean()),
            "units":float(q["unit_result"].sum()),
            "abs_z_gt_2_rate":float((q["abs_z"]>2).mean()),
            "model_mae":float(q["abs_model_error"].mean()),
            "model_closer_rate":float((q["abs_model_error"]<q["abs_line_error"]).mean()),
        })
    ranking=pd.DataFrame(rank_rows)

    attribution_rows=[]
    for label,q in [("ALL",d),("WEEK4",d.loc[d.week.eq(4)])]:
        for market,z in [("ALL",q)]+[(str(k),v) for k,v in q.groupby("market")]:
            attribution_rows.append({
                "scope":label,"market":market,"comparison":"MC_TO_ENSEMBLE",
                **_paired(z,"mc_proj","ensemble_proj")
            })
            attribution_rows.append({
                "scope":label,"market":market,"comparison":"MC_TO_FINAL",
                **_paired(z,"mc_proj","model_proj")
            })
            attribution_rows.append({
                "scope":label,"market":market,"comparison":"ENSEMBLE_TO_FINAL",
                **_paired(z,"ensemble_proj","model_proj")
            })
    attribution=pd.DataFrame(attribution_rows)

    # Decompose rush+receiving error into the selected player's standalone rush
    # and receiving components wherever both are present in the same frozen ledger.
    unique=d.groupby(
        ["season","week","event_id","player_clean_key","market"],as_index=False
    ).agg(
        player=("player","first"),team=("team","first"),position=("position","first"),
        model_proj=("model_proj","first"),actual=("actual","first")
    )
    rr=unique.loc[unique.market.eq("rush_rec_yards")].copy()
    rush=unique.loc[unique.market.eq("rush_yards"),
        ["season","week","event_id","player_clean_key","model_proj","actual"]
    ].rename(columns={"model_proj":"rush_model","actual":"rush_actual"})
    rec=unique.loc[unique.market.eq("rec_yards"),
        ["season","week","event_id","player_clean_key","model_proj","actual"]
    ].rename(columns={"model_proj":"rec_model","actual":"rec_actual"})
    decomp=rr.merge(rush,on=["season","week","event_id","player_clean_key"],how="left").merge(
        rec,on=["season","week","event_id","player_clean_key"],how="left"
    )
    decomp["combo_error"]=decomp["model_proj"]-decomp["actual"]
    decomp["rush_error"]=decomp["rush_model"]-decomp["rush_actual"]
    decomp["rec_error"]=decomp["rec_model"]-decomp["rec_actual"]
    decomp["component_error_sum"]=decomp["rush_error"]+decomp["rec_error"]

    component_summary={}
    for label,q in [("ALL",decomp),("RB",decomp.loc[decomp.position.eq("RB")])]:
        component_summary[label]={
            "combo_rows":int(len(q)),
            "complete_component_rows":int(q[["rush_error","rec_error"]].dropna().shape[0]),
            "combo_mean_error":float(q["combo_error"].mean()),
            "combo_median_error":float(q["combo_error"].median()),
            "rush_component_mean_error":float(q["rush_error"].mean()),
            "rush_component_median_error":float(q["rush_error"].median()),
            "receiving_component_mean_error":float(q["rec_error"].mean()),
            "receiving_component_median_error":float(q["rec_error"].median()),
        }

    extreme=d.sort_values("abs_z",ascending=False).head(100).copy()
    extreme_cols=[c for c in [
        "week","event_id","player","team","position","market","side","vegas_line",
        "model_proj","mc_proj","ensemble_proj","model_sd","actual","model_error",
        "z","fair_prob","market_prob","bet_result","unit_result"
    ] if c in extreme.columns]

    a.out_dir.mkdir(parents=True,exist_ok=True)
    groups.to_csv(a.out_dir/"failure_mode_groups.csv",index=False)
    tails.to_csv(a.out_dir/"failure_mode_tail_concentration.csv",index=False)
    confidence.to_csv(a.out_dir/"failure_mode_confidence_bands.csv",index=False)
    ranking.to_csv(a.out_dir/"failure_mode_normalized_gap_quintiles.csv",index=False)
    attribution.to_csv(a.out_dir/"failure_mode_projection_attribution.csv",index=False)
    decomp.to_csv(a.out_dir/"failure_mode_rush_rec_component_decomposition.csv",index=False)
    extreme[extreme_cols].to_csv(a.out_dir/"failure_mode_top100_standardized_misses.csv",index=False)

    payload={
        "status":"DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE",
        "decided_rows":int(len(d)),
        "overall":overall,
        "tail_concentration":tail_rows,
        "probability_scoring":probability,
        "rush_rec_component_summary":component_summary,
        "notes":{
            "z_definition":"(actual-model_proj)/model_sd; used only as within-model uncertainty scale, not as an assumption of Normality",
            "no_candidate_fit":True,
            "no_threshold_search":True,
            "no_sportsbook_refetch":True,
        },
    }
    (a.out_dir/"failure_mode_atlas_summary.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("=== WEEKS 1-4 FAILURE MODE ATLAS V1 ===")
    print(json.dumps(payload,indent=2,sort_keys=True))
    print("\nBY MARKET/POSITION/WEEK")
    print(groups.to_string(index=False))
    print("\nCONFIDENCE BANDS")
    print(confidence.to_string(index=False))
    print("\nNORMALIZED MODEL-MARKET GAP QUINTILES")
    print(ranking.to_string(index=False))
    print("\nPROJECTION ATTRIBUTION")
    print(attribution.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
