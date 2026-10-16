#!/usr/bin/env python3
"""Freeze prospective Week-6 WR trust labels from exact opportunity difficulty.

Research only. Uses completed 2026 Weeks 1-5 and leaves WR-R15 unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import lookup

VERSION="WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1"
QUALIFIED_PARENT="WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED"
PACK_STATUS="WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_PACK_FROZEN"
WR_VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"
SEASON=2026
WEEK=6
EXPECTED_COHORT=152
EXPECTED_PARITY_ROWS=3224
EXPECTED_PARITY_PLAYERS=211
MIN_PRIOR=4
PRIOR_WINDOW=8
TOL=1e-9
WR_POS={"WR","LWR","RWR","SWR"}


def read_csv(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def num(s): return pd.to_numeric(s,errors="coerce")


def semantic_parity(parent:pd.DataFrame,authority:pd.DataFrame)->dict:
    needp={
      "season","week","team","player_clean_key","current_opportunity_error"
    }
    needa={
      "variant","season","week","team","player_clean_key",
      "pred_targets","mc_rec_yards","actual_targets"
    }
    if needp-set(parent.columns):
        raise RuntimeError(f"WR parent parity columns missing {sorted(needp-set(parent.columns))}")
    if needa-set(authority.columns):
        raise RuntimeError(f"WR authority parity columns missing {sorted(needa-set(authority.columns))}")
    p=parent.loc[num(parent["season"]).isin([2023,2024])].copy()
    a=authority.loc[authority["variant"].astype(str).eq(WR_VARIANT)].copy()
    keys=["season","week","team","player_clean_key"]
    for label,x in [("parent",p),("authority",a)]:
        if x.duplicated(keys).any():
            raise RuntimeError(f"{label} duplicate parity identity")
    z=p.merge(
      a[keys+["pred_targets","mc_rec_yards","actual_targets"]],
      on=keys,how="inner",validate="one_to_one"
    )
    pred_targets=num(z["pred_targets"])
    if pred_targets.le(0).any() or pred_targets.isna().any():
        raise RuntimeError("invalid historical WR predicted targets")
    pred_ypt=num(z["mc_rec_yards"])/pred_targets
    recon=(pred_targets-num(z["actual_targets"]))*pred_ypt
    gap=(recon-num(z["current_opportunity_error"])).abs()
    out={
      "matched_rows":int(len(z)),
      "players":int(z["player_clean_key"].nunique()),
      "max_abs_opportunity_error_gap":float(gap.max()) if len(z) else np.inf,
      "formula":"(pred_targets-actual_targets)*(mc_rec_yards/pred_targets)",
      "tolerance":TOL,
    }
    if (
      out["matched_rows"]!=EXPECTED_PARITY_ROWS
      or out["players"]!=EXPECTED_PARITY_PLAYERS
      or out["max_abs_opportunity_error_gap"]>TOL
    ):
        raise RuntimeError(f"WR opportunity semantic parity failed: {out}")
    return out


def current_cohort(path:Path)->pd.DataFrame:
    x=read_csv(path,"Week6 WR trajectory parent")
    need={
      "season","week","event_id","team","opponent","player","player_clean_key",
      "position","baseline_entitlement_tgt_share","wr_r15_anchor"
    }
    if need-set(x.columns):
        raise RuntimeError(f"Week6 WR cohort missing {sorted(need-set(x.columns))}")
    x=x.loc[
      num(x["season"]).eq(SEASON)
      &num(x["week"]).eq(WEEK)
      &x["position"].astype(str).str.upper().eq("WR")
    ].copy()
    x["team"]=x["team"].map(canon_team)
    x["player_clean_key"]=x["player_clean_key"].fillna("").astype(str).str.strip()
    if len(x)!=EXPECTED_COHORT:
        raise RuntimeError(f"Week6 WR cohort drift {len(x)} != {EXPECTED_COHORT}")
    if x["player_clean_key"].eq("").any() or x["player_clean_key"].duplicated().any():
        raise RuntimeError("Week6 WR cohort blank/duplicate player identity")
    share=num(x["baseline_entitlement_tgt_share"])
    if share.isna().any() or share.lt(0).any():
        raise RuntimeError("Week6 WR baseline entitlement invalid")
    return x.sort_values(["team","player_clean_key"]).reset_index(drop=True)


def actual_target_map(logs:pd.DataFrame,week:int)->dict:
    x=logs.loc[num(logs["season"]).eq(SEASON)&num(logs["week"]).eq(week)].copy()
    if x.empty:
        raise RuntimeError(f"missing completed 2026 Week{week} player logs")
    x["team"]=x["team"].map(canon_team)
    if "player_clean_key" not in x.columns:
        raise RuntimeError("historical player logs missing player_clean_key")
    x["player_clean_key"]=x["player_clean_key"].fillna("").astype(str)
    x["targets"]=num(x.get("targets")).fillna(0.0)
    g=x.groupby(["team","player_clean_key"],as_index=False)["targets"].sum()
    return {(str(r.team),str(r.player_clean_key)):float(r.targets) for r in g.itertuples(index=False)}


def reconstruct_2026_history(
    *,
    cohort:pd.DataFrame,
    player_logs:pd.DataFrame,
    team_weekly:pd.DataFrame,
    schedule:pd.DataFrame,
    universe_dir:Path,
)->pd.DataFrame:
    logs=player_logs.copy()
    logs.columns=[str(c).strip().lower() for c in logs.columns]
    if "player_clean_key" not in logs.columns:
        raise RuntimeError("player logs do not expose canonical player_clean_key")
    bad=sorted(
      num(logs.loc[num(logs["season"]).eq(SEASON)&num(logs["week"]).ge(WEEK),"week"])
      .dropna().astype(int).unique().tolist()
    )
    if bad:
        raise RuntimeError(f"Week6-or-later outcomes present; refuse WR lock: {bad}")

    wanted=set(cohort["player_clean_key"].astype(str))
    rows=[]
    for week in range(1,WEEK):
        universe=read_csv(universe_dir/f"{SEASON}_week_{week:02d}.csv",f"2026 W{week} universe")
        bundle=build_historical_context_bundle(
          player_logs=player_logs,
          team_weekly=team_weekly,
          pregame_universe=universe,
          schedule=schedule,
          season=SEASON,
          week=week,
          prior_season=2025,
          injuries=None,
          weather=None,
        )
        metrics=build_mc_predictions(bundle,iterations=2000,seed=42+week)
        players=(
          metrics.sort_values(["event_id","team","player_clean_key"])
          .drop_duplicates(["event_id","team","player_clean_key"],keep="last")
          .copy()
        )
        baseline,_=materialize_target_entitlement(players)
        te_final,_,_=apply_te_r5p_entitlement(baseline)
        final,_,wr_audit=apply_wr_r15_entitlement(te_final)
        if str(wr_audit.get("model_version"))!="WR_R15_PRODUCTION_MODEL_V1":
            raise RuntimeError(f"WR-R15 model version drift W{week}")
        sims=explicit_simulate(final,iterations=2000,seed=42+week)
        actuals=actual_target_map(logs,week)

        wr=final.loc[
          final["position"].astype(str).str.upper().isin(WR_POS)
          &final["player_clean_key"].astype(str).isin(wanted)
        ].copy()
        if wr.duplicated(["team","player_clean_key"]).any():
            raise RuntimeError(f"duplicate reconstructed WR identity W{week}")
        for _,r in wr.iterrows():
            team=canon_team(r.get("team"))
            pk=str(r.get("player_clean_key"))
            plays=float(pd.to_numeric(pd.Series([r.get("rules_plays_est")]),errors="coerce").iloc[0])
            rate=float(pd.to_numeric(pd.Series([r.get("rules_pass_rate")]),errors="coerce").iloc[0])
            share=float(pd.to_numeric(pd.Series([r.get("entitlement_tgt_share")]),errors="coerce").iloc[0])
            if not (np.isfinite(plays) and np.isfinite(rate) and np.isfinite(share) and share>=0):
                raise RuntimeError(f"invalid WR pregame state W{week} {team} {pk}")
            pred_team_targets=float(np.clip(plays,50.0,80.0)*np.clip(rate,.35,.75))
            pred_targets=pred_team_targets*share
            if pred_targets<=0:
                continue
            arr=lookup(sims,r,"rec_yards")
            if arr is None or len(arr)==0 or not np.isfinite(np.asarray(arr,float)).all():
                raise RuntimeError(f"missing WR rec-yard simulation W{week} {team} {pk}")
            pred_rec_yards=float(np.mean(np.asarray(arr,float)))
            pred_ypt=pred_rec_yards/pred_targets
            actual_targets=float(actuals.get((team,pk),0.0))
            opp_error=(pred_targets-actual_targets)*pred_ypt
            rows.append({
              "season":SEASON,"week":week,"event_id":str(r.get("event_id")),
              "team":team,"player":str(r.get("player")),"player_clean_key":pk,
              "pred_team_targets":pred_team_targets,"pred_targets":pred_targets,
              "pred_rec_yards":pred_rec_yards,"pred_ypt":pred_ypt,
              "actual_targets":actual_targets,"opportunity_error":opp_error,
              "source_max_week":week-1,
              "history_source":"RECONSTRUCTED_2026_STRICT_PRIOR_WR_R15",
            })
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero reconstructed 2026 WR opportunity-error rows")
    if out.duplicated(["season","week","player_clean_key"]).any():
        sample=out.loc[
          out.duplicated(["season","week","player_clean_key"],keep=False),
          ["season","week","team","player","player_clean_key"]
        ].head(20).to_dict("records")
        raise RuntimeError(f"duplicate reconstructed WR player-week: {sample}")
    if int(num(out["week"]).max())>=WEEK:
        raise RuntimeError("Week6 WR history leakage")
    return out.sort_values(["week","team","player_clean_key"]).reset_index(drop=True)


def lock(cohort:pd.DataFrame,history:pd.DataFrame,parity:dict,parent:dict,pack:dict)->tuple[pd.DataFrame,dict]:
    if parent.get("disposition")!=QUALIFIED_PARENT:
        raise RuntimeError("WR trust parent is not qualified")
    if pack.get("status")!=PACK_STATUS:
        raise RuntimeError("Week6 opportunity pack is not frozen")
    if int(pack.get("target_week_outcomes_read",1))!=0:
        raise RuntimeError("Week6 opportunity pack read target outcomes")
    state=(pack.get("components") or {}).get("player_state") or {}
    if state.get("disposition")!="PLAYER_STATE_FUTURE_WEEK_READY" or int(state.get("roster_source_week",-1))!=WEEK:
        raise RuntimeError(f"Week6 player-state boundary drift: {state}")

    rows=[]
    for r in cohort.itertuples(index=False):
        pk=str(r.player_clean_key)
        q=history.loc[history["player_clean_key"].astype(str).eq(pk)].sort_values(["season","week"]).tail(PRIOR_WINDOW)
        supported=len(q)>=MIN_PRIOR
        mae=float(np.mean(np.abs(q["opportunity_error"].to_numpy(float)))) if supported else np.nan
        rows.append({
          "season":SEASON,"week":WEEK,"event_id":str(r.event_id),
          "team":str(r.team),"opponent":str(r.opponent),"player":str(r.player),
          "player_clean_key":pk,"wr_r15_anchor":bool(r.wr_r15_anchor),
          "baseline_entitlement_tgt_share":float(r.baseline_entitlement_tgt_share),
          "prior_games":int(len(q)),"player_mae":mae,"supported":bool(supported),
          "history_min_week":int(q.iloc[0]["week"]) if len(q) else np.nan,
          "history_max_week":int(q.iloc[-1]["week"]) if len(q) else np.nan,
        })
    out=pd.DataFrame(rows)
    sup=out.loc[out["supported"]].copy()
    if sup.empty:
        raise RuntimeError("zero supported Week6 WR trust rows")
    pool=float(sup["player_mae"].median())
    if not np.isfinite(pool) or pool<=0:
        raise RuntimeError("invalid Week6 WR pool MAE")
    sup["week_pool_mae"]=pool
    sup["difficulty_ratio_vs_pool"]=sup["player_mae"]/pool
    sup["trust_fragility_score"]=sup["player_mae"].rank(method="average",pct=True)
    sup["trust_bucket"]="MIDDLE"
    sup.loc[sup["trust_fragility_score"].le(.25),"trust_bucket"]="HIGH_TRUST"
    sup.loc[sup["trust_fragility_score"].ge(.75),"trust_bucket"]="LOW_TRUST"
    out=out.merge(
      sup[["player_clean_key","week_pool_mae","difficulty_ratio_vs_pool","trust_fragility_score","trust_bucket"]],
      on="player_clean_key",how="left",validate="one_to_one"
    )
    out["trust_bucket"]=out["trust_bucket"].fillna("UNSUPPORTED")
    out=out.sort_values(["team","player_clean_key"]).reset_index(drop=True)
    payload=out.to_csv(index=False,float_format="%.12f",lineterminator="\n").encode()
    digest="sha256:"+hashlib.sha256(payload).hexdigest()
    buckets={str(k):int(v) for k,v in out["trust_bucket"].value_counts().to_dict().items()}
    result={
      "version":VERSION,"status":"WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN",
      "parent_disposition":parent.get("disposition"),
      "season":SEASON,"week":WEEK,"current_wr_cohort":int(len(out)),
      "supported_wr":int(out["supported"].sum()),"unsupported_wr":int((~out["supported"]).sum()),
      "trust_bucket_counts":buckets,"week_pool_mae":pool,
      "reconstructed_2026_rows":int(len(history)),
      "history_source_max_week_2026":int(num(history["week"]).max()),
      "historical_semantic_parity":parity,
      "row_csv_sha256":digest,"prior_window":PRIOR_WINDOW,"min_prior_games":MIN_PRIOR,
      "sportsbook_inputs_used":0,"oddsapi_calls":0,"week6_outcomes_read":0,
      "models_fit":0,"projection_distribution_values_changed":0,"production_changed":False,
    }
    return out,result


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--week6-cohort",type=Path,required=True)
    ap.add_argument("--week6-pack-result",type=Path,required=True)
    ap.add_argument("--parent-trust-result",type=Path,required=True)
    ap.add_argument("--mechanism-parent",type=Path,required=True)
    ap.add_argument("--wr-authority",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    mechanism=read_csv(a.mechanism_parent,"WR mechanism parent")
    authority=read_csv(a.wr_authority,"WR recovered authority")
    parity=semantic_parity(mechanism,authority)
    cohort=current_cohort(a.week6_cohort)
    logs=read_csv(a.player_logs,"historical player logs")
    history=reconstruct_2026_history(
      cohort=cohort,player_logs=logs,
      team_weekly=read_csv(a.team_weekly,"team weekly"),
      schedule=read_csv(a.schedule,"schedule"),
      universe_dir=a.universe_dir,
    )
    rows,result=lock(
      cohort,history,parity,
      json.loads(a.parent_trust_result.read_text()),
      json.loads(a.week6_pack_result.read_text()),
    )
    rows.to_csv(a.out_dir/"wr_week6_trust_lock.csv",index=False,float_format="%.12f")
    history.to_csv(a.out_dir/"wr_week6_reconstructed_2026_opportunity_errors.csv",index=False,float_format="%.12f")
    (a.out_dir/"wr_week6_trust_lock.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
