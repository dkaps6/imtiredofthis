#!/usr/bin/env python3
"""Full-stack historical A/B for the frozen Rush Pool Evidence Guard V1.

No fitting and no candidate search occur here. The script:
- rebuilds timestamp-safe historical football inputs;
- applies the existing fold-safe TE-R5P / WR-R15 production-order authority;
- runs the exact same explicit-entitlement MC baseline and V1 candidate;
- applies the current frozen ensemble weights;
- applies RB Rush+Receiving Conservation V2 to eligible RB combo rows;
- checks non-rushing bit invariance and the frozen integration gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions, _attach_component_projection
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.state_v2 import build_state_predictions
from scripts.modeling.rb_rush_rec_conservation_v2 import build_candidate_map
from scripts.modeling.rush_pool_evidence_guard_v1 import ENV_VAR, VERSION
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps as _load_participation_snaps
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    apply_te_fold,
    apply_wr_fold,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import lookup
from scripts.utils.canonical_names import canon_team

KEYS = ["season", "week", "team", "player_clean_key", "market"]
NON_RUSHING_MARKETS = {"receptions", "rec_yards", "pass_yards", "anytime_td"}
RB_POS = {"RB", "HB", "FB", "TB"}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path, low_memory=False)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def _canon_frame(x: pd.DataFrame) -> pd.DataFrame:
    out=x.copy()
    out.columns=[str(c).strip().lower() for c in out.columns]
    if "team" in out.columns: out["team"]=out["team"].map(canon_team)
    if "market" in out.columns: out["market"]=out["market"].astype(str).str.lower()
    if "season" in out.columns: out["season"]=pd.to_numeric(out["season"],errors="raise").astype(int)
    if "week" in out.columns: out["week"]=pd.to_numeric(out["week"],errors="raise").astype(int)
    return out


def _pos_family(v) -> str:
    p="" if v is None or pd.isna(v) else str(v).upper().strip()
    if p in RB_POS or p.startswith("RB") or p.startswith("FB"):
        return "RB_FAMILY"
    if p.startswith("QB"):
        return "QB"
    return "OTHER"


def _set_guard(enabled: bool):
    old=os.environ.get(ENV_VAR)
    if enabled: os.environ[ENV_VAR]="1"
    else: os.environ.pop(ENV_VAR,None)
    return old


def _restore_guard(old):
    if old is None: os.environ.pop(ENV_VAR,None)
    else: os.environ[ENV_VAR]=old


def _run_sim(metrics: pd.DataFrame, *, iterations: int, seed: int, enabled: bool):
    trace=[]
    old=_set_guard(enabled)
    try:
        result=explicit_simulate(metrics,iterations=int(iterations),seed=int(seed),allocation_trace=trace)
    finally:
        _restore_guard(old)
    return result,pd.DataFrame(trace)


def _compare_arrays(base, cand, *, week: int) -> dict:
    if set(base.values)!=set(cand.values):
        raise RuntimeError(f"W{week:02d} simulation key universe changed")
    max_nonrush_gap=0.0
    changed_nonrush=0
    changed_all=0
    for key in base.values:
        a=np.asarray(base.values[key],dtype=float); b=np.asarray(cand.values[key],dtype=float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"W{week:02d} invalid array key={key}")
        gap=float(np.max(np.abs(a-b))) if len(a) else 0.0
        changed_all += int(gap>0)
        if key[2] in NON_RUSHING_MARKETS:
            changed_nonrush += int(gap>0)
            max_nonrush_gap=max(max_nonrush_gap,gap)
    if int(week)==1 and changed_all:
        raise RuntimeError(f"Week-1 V1 no-op failed: changed_arrays={changed_all}")
    if int(week)>1 and changed_nonrush:
        raise RuntimeError(f"W{week:02d} V1 changed non-rushing arrays: {changed_nonrush} max={max_nonrush_gap}")
    return {
        "week1_changed_arrays": int(changed_all) if int(week)==1 else 0,
        "changed_nonrushing_arrays": int(changed_nonrush),
        "max_nonrushing_element_gap": float(max_nonrush_gap),
    }


def _trace_team(trace: pd.DataFrame, *, season: int, week: int, label: str) -> pd.DataFrame:
    need={"event_id","team","team_rush_total_sha256","target_allocation_sha256"}
    missing=need-set(trace.columns)
    if missing: raise RuntimeError(f"{label} trace missing {sorted(missing)}")
    t=trace.copy()
    t["team"]=t["team"].map(canon_team)
    cols=["event_id","team","team_rush_total_sha256","target_allocation_sha256"]
    out=t[cols].drop_duplicates(["event_id","team"])
    if out.duplicated(["event_id","team"]).any():
        raise RuntimeError(f"{label} duplicate team trace")
    out["season"]=int(season); out["week"]=int(week)
    return out


def _score_summary(detail: pd.DataFrame, *, market: str, group: str) -> dict:
    g=detail.loc[detail["market"].eq(market)].copy()
    if group!="ALL": g=g.loc[g["position_family"].eq(group)].copy()
    if g.empty:
        raise RuntimeError(f"empty score group market={market} group={group}")
    actual=g["actual"].to_numpy(float); b=g["baseline_proj"].to_numpy(float); c=g["candidate_proj"].to_numpy(float)
    be=b-actual; ce=c-actual; ba=np.abs(be); ca=np.abs(ce)
    changed=np.abs(b-c)>1e-12
    cand_closer=changed & (ca<ba-1e-12); base_closer=changed & (ba<ca-1e-12)
    decided=cand_closer|base_closer
    threshold=10.0 if market=="rush_att" else 30.0
    return {
        "market":market,"group":group,"n":int(len(g)),
        "baseline_mae":float(np.mean(ba)),"candidate_mae":float(np.mean(ca)),
        "baseline_rmse":float(np.sqrt(np.mean(be*be))),"candidate_rmse":float(np.sqrt(np.mean(ce*ce))),
        "baseline_bias":float(np.mean(be)),"candidate_bias":float(np.mean(ce)),
        "baseline_p90":float(np.quantile(ba,0.90)),"candidate_p90":float(np.quantile(ca,0.90)),
        "baseline_tail_misses":int(np.sum(ba>=threshold)),"candidate_tail_misses":int(np.sum(ca>=threshold)),
        "tail_threshold":threshold,
        "changed_rows":int(changed.sum()),
        "candidate_closer":int(cand_closer.sum()),"baseline_closer":int(base_closer.sum()),
        "candidate_closer_rate":float(cand_closer.sum()/decided.sum()) if int(decided.sum()) else None,
    }


def _ensemble_projection(records: pd.DataFrame, *, weights: pd.DataFrame, mc_col: str) -> np.ndarray:
    x=records[["market","ml_proj","state_proj"]].copy()
    x["mc_proj"]=pd.to_numeric(records[mc_col],errors="coerce")
    out=apply_ensemble(x[["market","mc_proj","ml_proj","state_proj"]],weights=weights)
    return pd.to_numeric(out["ensemble_proj"],errors="coerce").to_numpy(float)


def evaluate_season(*, season:int, prior_season:int, weeks:list[int], player_logs:pd.DataFrame,
                    team_weekly:pd.DataFrame, schedule:pd.DataFrame, universe_dir:Path,
                    injuries:pd.DataFrame, weather:pd.DataFrame,
                    te_params:dict, wr_params:dict|None, snaps:pd.DataFrame,
                    weights:pd.DataFrame, iterations:int) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    details=[]; audits=[]; max_v2_gap=0.0; sportsbook_inputs=0
    for week in weeks:
        u=_read(universe_dir/f"{season}_week_{week:02d}.csv",f"{season} W{week} universe")
        bundle=build_historical_context_bundle(
            player_logs=player_logs,team_weekly=team_weekly,pregame_universe=u,schedule=schedule,
            season=int(season),week=int(week),prior_season=int(prior_season),
            injuries=_exact_week(injuries,int(season),int(week)),
            weather=_exact_week(weather,int(season),int(week)),
        )
        seed=42+int(week)
        # Use the canonical historical builder to carry every current context field.
        metrics=build_mc_predictions(bundle,iterations=20,seed=seed)
        players=metrics.sort_values(["event_id","team","player_clean_key"]).drop_duplicates(
            ["event_id","team","player_clean_key"],keep="last"
        ).copy()
        explicit_base,_=materialize_target_entitlement(players)
        te_final,_,te_audit=apply_te_fold(explicit_base,snaps=snaps,params=te_params)
        if int(season)==2024:
            final,_,wr_audit=apply_wr_fold(te_final,snaps=snaps,params=wr_params)
        else:
            final=te_final
            wr_audit={"same_future_participation":0}
        raw_before=pd.to_numeric(final["rules_rush_share"],errors="coerce").to_numpy(float).copy()

        base,bt=_run_sim(final,iterations=iterations,seed=seed,enabled=False)
        cand,ct=_run_sim(final,iterations=iterations,seed=seed,enabled=True)
        raw_after=pd.to_numeric(final["rules_rush_share"],errors="coerce").to_numpy(float)
        if not np.array_equal(raw_before,raw_after,equal_nan=True):
            raise RuntimeError(f"{season} W{week:02d} simulation mutated raw rules_rush_share")
        arr_audit=_compare_arrays(base,cand,week=week)
        bteam=_trace_team(bt,season=season,week=week,label="baseline")
        cteam=_trace_team(ct,season=season,week=week,label="candidate")
        tj=bteam.merge(cteam,on=["season","week","event_id","team"],suffixes=("_base","_cand"),validate="one_to_one")
        rush_mismatch=int((tj["team_rush_total_sha256_base"]!=tj["team_rush_total_sha256_cand"]).sum())
        target_mismatch=int((tj["target_allocation_sha256_base"]!=tj["target_allocation_sha256_cand"]).sum())
        if rush_mismatch or target_mismatch:
            raise RuntimeError(f"{season} W{week:02d} shared team state drift rush={rush_mismatch} target={target_mismatch}")

        if int(week)==1:
            audits.append({
                "season":season,"week":week,**arr_audit,"team_rush_digest_mismatches":rush_mismatch,
                "target_digest_mismatches":target_mismatch,"te_pool_gap":float(te_audit["team_te_pool_max_abs_gap"]),
                "wr_future_violations":int(wr_audit.get("same_future_participation",0)),
            })
            continue

        _,ml_pred=build_ml(player_logs,bundle.player_consensus,int(season),int(week))
        _,state_pred=build_state_predictions(player_logs,bundle.player_consensus,int(season),int(week))
        mcols=["event_id","team","player","player_clean_key","position","market","season","week"]
        market_rows=metrics[mcols].copy()
        market_rows["team"]=market_rows["team"].map(canon_team); market_rows["market"]=market_rows["market"].astype(str).str.lower()
        market_rows=_attach_component_projection(market_rows,ml_pred,"ml")
        market_rows=_attach_component_projection(market_rows,state_pred,"state")
        actual=build_actual_rows(player_logs,int(season),int(week))
        actual["team"]=actual["team"].map(canon_team); actual["market"]=actual["market"].astype(str).str.lower()
        all_joined=market_rows.merge(
            actual[["team","player_clean_key","market","actual"]],
            on=["team","player_clean_key","market"],how="inner",validate="one_to_one"
        )
        if all_joined.empty: raise RuntimeError(f"{season} W{week:02d} no scored market rows")
        all_joined["position_family"]=all_joined["position"].map(_pos_family)
        joined=all_joined.loc[all_joined["market"].isin(["rush_att","rush_yards","rush_rec_yards"])].copy()
        brec=[]; crec=[]
        for _,row in joined.iterrows():
            ba=lookup(base,row,row["market"]); ca=lookup(cand,row,row["market"])
            if ba is None or ca is None:
                raise RuntimeError(f"{season} W{week:02d} missing array {row['player_clean_key']} {row['market']}")
            brec.append(float(np.mean(np.asarray(ba,float)))); crec.append(float(np.mean(np.asarray(ca,float))))
        joined["baseline_mc_proj"]=brec; joined["candidate_mc_proj"]=crec
        joined["baseline_proj"]=_ensemble_projection(joined,weights=weights,mc_col="baseline_mc_proj")
        joined["candidate_proj"]=_ensemble_projection(joined,weights=weights,mc_col="candidate_mc_proj")

        # Apply current non-Week-1 RB Rush+Receiving Conservation V2 final mean authority.
        scoring_metrics=all_joined.copy()
        bmap,bpayload=build_candidate_map(scoring_metrics,base,weights)
        cmap,cpayload=build_candidate_map(scoring_metrics,cand,weights)
        sportsbook_inputs=max(sportsbook_inputs,int(bpayload.get("sportsbook_inputs_used",0)),int(cpayload.get("sportsbook_inputs_used",0)))
        max_v2_gap=max(max_v2_gap,float(bpayload.get("max_pathwise_identity_gap",0.0)),float(cpayload.get("max_pathwise_identity_gap",0.0)))
        for idx,row in joined.loc[joined["market"].eq("rush_rec_yards") & joined["position_family"].eq("RB_FAMILY")].iterrows():
            key=(str(row["event_id"]),str(row["player_clean_key"]))
            if key in bmap: joined.at[idx,"baseline_proj"]=float(bmap[key]["target_mean"])
            if key in cmap: joined.at[idx,"candidate_proj"]=float(cmap[key]["target_mean"])

        if joined[["actual","baseline_proj","candidate_proj"]].apply(pd.to_numeric,errors="coerce").isna().any().any():
            raise RuntimeError(f"{season} W{week:02d} non-finite scored projection")
        details.append(joined)
        audits.append({
            "season":season,"week":week,**arr_audit,"team_rush_digest_mismatches":rush_mismatch,
            "target_digest_mismatches":target_mismatch,"te_pool_gap":float(te_audit["team_te_pool_max_abs_gap"]),
            "wr_future_violations":int(wr_audit.get("same_future_participation",0)),
            "rb_v2_max_pathwise_gap":max(float(bpayload.get("max_pathwise_identity_gap",0.0)),float(cpayload.get("max_pathwise_identity_gap",0.0))),
        })

    detail=pd.concat(details,ignore_index=True)
    audit=pd.DataFrame(audits)
    summaries={}
    for market in ("rush_att","rush_yards"):
        for group in ("ALL","RB_FAMILY","QB","OTHER"):
            summaries[f"{market}:{group}"]=_score_summary(detail,market=market,group=group)
    summaries["rush_rec_yards:RB_FAMILY"]=_score_summary(detail,market="rush_rec_yards",group="RB_FAMILY")
    scope={
        "week1_changed_arrays":int(audit["week1_changed_arrays"].sum()),
        "changed_nonrushing_arrays":int(audit["changed_nonrushing_arrays"].sum()),
        "max_nonrushing_element_gap":float(audit["max_nonrushing_element_gap"].max()),
        "team_rush_digest_mismatches":int(audit["team_rush_digest_mismatches"].sum()),
        "target_digest_mismatches":int(audit["target_digest_mismatches"].sum()),
        "raw_rules_rush_share_mutations":0,
        "sportsbook_inputs_used":int(sportsbook_inputs),
        "rb_v2_max_pathwise_identity_gap":float(max_v2_gap),
        "wr_same_future_participation":int(audit["wr_future_violations"].sum()),
    }
    return detail,audit,{"season":season,"prior_season":prior_season,"scope":scope,"metrics":summaries}


def _m(s:dict,key:str,metric:str)->float:
    v=s["metrics"][key][metric]
    if v is None: raise RuntimeError(f"missing metric {key}.{metric}")
    return float(v)


def apply_gates(s24:dict,s25:dict)->dict:
    scope24=s24["scope"]; scope25=s25["scope"]
    gates={
        "week1_bit_exact":scope24["week1_changed_arrays"]==0 and scope25["week1_changed_arrays"]==0,
        "nonrushing_bit_exact":scope24["changed_nonrushing_arrays"]==0 and scope25["changed_nonrushing_arrays"]==0,
        "team_rush_arrays_unchanged":scope24["team_rush_digest_mismatches"]==0 and scope25["team_rush_digest_mismatches"]==0,
        "target_allocations_unchanged":scope24["target_digest_mismatches"]==0 and scope25["target_digest_mismatches"]==0,
        "raw_rush_shares_unchanged":scope24["raw_rules_rush_share_mutations"]==0 and scope25["raw_rules_rush_share_mutations"]==0,
        "zero_sportsbook_inputs":scope24["sportsbook_inputs_used"]==0 and scope25["sportsbook_inputs_used"]==0,
        "rb_v2_identity":scope24["rb_v2_max_pathwise_identity_gap"]<=1e-10 and scope25["rb_v2_max_pathwise_identity_gap"]<=1e-10,
    }
    for y,s in ((2024,s24),(2025,s25)):
        gates[f"all_rush_att_mae_improves_{y}"]=_m(s,"rush_att:ALL","candidate_mae") < _m(s,"rush_att:ALL","baseline_mae")
        gates[f"rb_rush_att_mae_improves_{y}"]=_m(s,"rush_att:RB_FAMILY","candidate_mae") < _m(s,"rush_att:RB_FAMILY","baseline_mae")
        gates[f"qb_rush_att_mae_nonworse_{y}"]=_m(s,"rush_att:QB","candidate_mae") <= _m(s,"rush_att:QB","baseline_mae")+1e-12
        gates[f"all_rush_att_p90_nonworse_{y}"]=_m(s,"rush_att:ALL","candidate_p90") <= _m(s,"rush_att:ALL","baseline_p90")+1e-12
        gates[f"rb_rush_att_p90_nonworse_{y}"]=_m(s,"rush_att:RB_FAMILY","candidate_p90") <= _m(s,"rush_att:RB_FAMILY","baseline_p90")+1e-12
        gates[f"other_rush_att_p90_nonworse_{y}"]=_m(s,"rush_att:OTHER","candidate_p90") <= _m(s,"rush_att:OTHER","baseline_p90")+1e-12
        gates[f"other_rush_att_tail_nonworse_{y}"]=_m(s,"rush_att:OTHER","candidate_tail_misses") <= _m(s,"rush_att:OTHER","baseline_tail_misses")
        gates[f"all_rush_yards_mae_nonworse_{y}"]=_m(s,"rush_yards:ALL","candidate_mae") <= _m(s,"rush_yards:ALL","baseline_mae")+1e-12
        gates[f"rb_rush_yards_mae_improves_{y}"]=_m(s,"rush_yards:RB_FAMILY","candidate_mae") < _m(s,"rush_yards:RB_FAMILY","baseline_mae")
        gates[f"rb_rush_yards_p90_nonworse_{y}"]=_m(s,"rush_yards:RB_FAMILY","candidate_p90") <= _m(s,"rush_yards:RB_FAMILY","baseline_p90")+1e-12
        gates[f"rb_rush_yards_tail_nonworse_{y}"]=_m(s,"rush_yards:RB_FAMILY","candidate_tail_misses") <= _m(s,"rush_yards:RB_FAMILY","baseline_tail_misses")
        gates[f"qb_rush_yards_mae_nonworse_{y}"]=_m(s,"rush_yards:QB","candidate_mae") <= _m(s,"rush_yards:QB","baseline_mae")+1e-12
        gates[f"other_rush_yards_p90_nonworse_{y}"]=_m(s,"rush_yards:OTHER","candidate_p90") <= _m(s,"rush_yards:OTHER","baseline_p90")+1e-12
        gates[f"other_rush_yards_tail_nonworse_{y}"]=_m(s,"rush_yards:OTHER","candidate_tail_misses") <= _m(s,"rush_yards:OTHER","baseline_tail_misses")
        gates[f"rb_combo_mae_nonworse_{y}"]=_m(s,"rush_rec_yards:RB_FAMILY","candidate_mae") <= _m(s,"rush_rec_yards:RB_FAMILY","baseline_mae")+1e-12
        gates[f"rb_combo_p90_nonworse_{y}"]=_m(s,"rush_rec_yards:RB_FAMILY","candidate_p90") <= _m(s,"rush_rec_yards:RB_FAMILY","baseline_p90")+1e-12
    return gates


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-2024",type=Path,required=True)
    ap.add_argument("--universe-2025",type=Path,required=True)
    ap.add_argument("--injuries",type=Path,required=True)
    ap.add_argument("--weather",type=Path,required=True)
    ap.add_argument("--te-coefficients",type=Path,required=True)
    ap.add_argument("--wr-coefficients",type=Path,required=True)
    ap.add_argument("--weights",type=Path,default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--weeks",default="1-18")
    ap.add_argument("--iterations",type=int,default=2000)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    logs=_read(a.player_logs,"player logs")
    team=_read(a.team_weekly,"team weekly")
    sched=_read(a.schedule,"schedule")
    injuries=_optional(a.injuries); weather=_optional(a.weather)
    weights=load_weights(a.weights)
    te24=_load_fold_params(a.te_coefficients,test_season=2024,features=TE_FEATURES,label="TE-R5P")
    te25=_load_fold_params(a.te_coefficients,test_season=2025,features=TE_FEATURES,label="TE-R5P")
    wr24=_load_fold_params(a.wr_coefficients,test_season=2024,features=WR_FEATURES,label="WR-R15")
    snaps,dup,_=_load_participation_snaps()
    if dup>0.01: raise RuntimeError(f"participation snap duplicate rate too high: {dup}")
    weeks=_parse_weeks(a.weeks)

    d24,a24,s24=evaluate_season(season=2024,prior_season=2023,weeks=weeks,player_logs=logs,team_weekly=team,
        schedule=sched,universe_dir=a.universe_2024,injuries=injuries,weather=weather,
        te_params=te24,wr_params=wr24,snaps=snaps,weights=weights,iterations=a.iterations)
    d25,a25,s25=evaluate_season(season=2025,prior_season=2024,weeks=weeks,player_logs=logs,team_weekly=team,
        schedule=sched,universe_dir=a.universe_2025,injuries=injuries,weather=weather,
        te_params=te25,wr_params=None,snaps=snaps,weights=weights,iterations=a.iterations)

    gates=apply_gates(s24,s25)
    qualified=all(gates.values())
    disposition=f"{VERSION}_PRODUCTION_INTEGRATION_{'QUALIFIED' if qualified else 'FAILED_CLOSED'}"
    result={"version":VERSION,"disposition":disposition,"qualified":qualified,"production_changed":False,
            "parameters_fit":0,"candidate_variants_scored":1,"season_2024":s24,"season_2025":s25,"gates":gates}

    a.out_dir.mkdir(parents=True,exist_ok=True)
    d24.to_csv(a.out_dir/"detail_2024.csv",index=False); d25.to_csv(a.out_dir/"detail_2025.csv",index=False)
    a24.to_csv(a.out_dir/"audit_2024.csv",index=False); a25.to_csv(a.out_dir/"audit_2025.csv",index=False)
    (a.out_dir/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    rows=[]
    for y,s in ((2024,s24),(2025,s25)):
        for k,v in s["metrics"].items(): rows.append({"season":y,**v})
    pd.DataFrame(rows).to_csv(a.out_dir/"metrics.csv",index=False)
    lines=["# Rush Pool Evidence Guard V1 — Full-Stack Integration Result","",f"Disposition: **{disposition}**",""]
    for y,s in ((2024,s24),(2025,s25)):
        lines += [f"## {y}",""]
        for key in ("rush_att:ALL","rush_att:RB_FAMILY","rush_att:QB","rush_att:OTHER","rush_yards:ALL","rush_yards:RB_FAMILY","rush_yards:QB","rush_yards:OTHER","rush_rec_yards:RB_FAMILY"):
            m=s["metrics"][key]
            lines.append(f"- {key}: MAE `{m['baseline_mae']:.6f} -> {m['candidate_mae']:.6f}`; p90 `{m['baseline_p90']:.6f} -> {m['candidate_p90']:.6f}`; tail `{m['baseline_tail_misses']} -> {m['candidate_tail_misses']}`")
        lines.append("")
    lines += ["## Frozen gates",""]+[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in gates.items()]
    (a.out_dir/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
