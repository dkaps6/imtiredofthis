#!/usr/bin/env python3
"""Retrospective 2026 Weeks 1-4 impact replay for RB receiving-room share V1.

This answers whether the frozen no-fit RB receiving-room redistribution would
have moved individual RB receiving projections closer to actual results. It is
retrospective mechanism impact, not out-of-sample promotion evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import (
    assert_no_future_rows,
    build_historical_context_bundle,
)
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.rb_pricing_adapter_v1 import load_rb_context
from scripts.modeling.rb_receiving_identity_runtime_v1 import identity_atlas
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.modeling.rb_receiving_room_share_shadow_v1 import (
    apply_rb_receiving_room_share_shadow,
)
from scripts.research.run_all_player_all_position_replay_v1 import (
    SEASON,
    PRIOR_SEASON,
    _read,
    _attach_ml_state,
    _recompute_mc_from_specialist,
    _canonical_market,
    _required_market,
    _pos,
    _apply_rb_authorities,
    _attach_actuals,
)
from scripts.research.run_player_opportunity_allocation_audit_v1 import (
    _trace_explicit_simulation,
)
from scripts.simulation_v2 import SimulationResult

WEEKS=(1,2,3,4)
TOL=1e-10
RB_MARKETS={"rush_yards","rec_yards","receptions","rush_rec_yards"}


def _build_entitlement_state(metrics:pd.DataFrame)->pd.DataFrame:
    cols=["event_id","team","player_clean_key"]
    players=metrics.sort_values(cols).drop_duplicates(cols,keep="last").copy()
    if players.duplicated(cols).any():
        raise RuntimeError("impact replay player universe is not unique")
    base,_=materialize_target_entitlement(players)
    te,_,_=apply_te_r5p_entitlement(base)
    final,_,_=apply_wr_r15_entitlement(te)
    return final


def _hybrid_receiving_only(
    baseline:SimulationResult,
    candidate:SimulationResult,
    rb_keys:set[tuple[str,str]],
)->SimulationResult:
    """Use candidate receiving arrays for RB/FB but baseline rushing everywhere."""
    values={k:np.asarray(v,dtype=float).copy() for k,v in baseline.values.items()}
    for game,pkey in rb_keys:
        for market in ("receptions","rec_yards"):
            key=(str(game),str(pkey),market)
            if key not in candidate.values or key not in baseline.values:
                raise RuntimeError(f"missing receiving array for hybrid key={key}")
            values[key]=np.asarray(candidate.values[key],dtype=float).copy()
        rush_key=(str(game),str(pkey),"rush_yards")
        rec_key=(str(game),str(pkey),"rec_yards")
        combo_key=(str(game),str(pkey),"rush_rec_yards")
        if rush_key not in baseline.values:
            raise RuntimeError(f"missing baseline rush array for hybrid key={rush_key}")
        values[rush_key]=np.asarray(baseline.values[rush_key],dtype=float).copy()
        if (str(game),str(pkey),"rush_att") in baseline.values:
            values[(str(game),str(pkey),"rush_att")]=np.asarray(
                baseline.values[(str(game),str(pkey),"rush_att")],dtype=float
            ).copy()
        values[combo_key]=values[rush_key]+values[rec_key]
    return SimulationResult(values=values,iterations=baseline.iterations)


def _target_rows(trace:pd.DataFrame,week:int,actual_point:pd.DataFrame)->pd.DataFrame:
    t=trace.loc[
        trace["opportunity_type"].eq("targets")
        & trace["position"].map(_pos).isin({"RB","FB"})
    ].copy()
    if t.empty:
        raise RuntimeError(f"W{week}: zero RB target trace rows")
    t["week"]=int(week)
    t["position_family"]=t["position"].map(_pos)
    actual=actual_point.loc[
        actual_point["market"].eq("rec_yards"),
        ["event_id","team","player_clean_key","actual_opportunities"],
    ].drop_duplicates(["event_id","team","player_clean_key"])
    t=t.merge(
        actual,
        on=["event_id","team","player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    if t["actual_opportunities"].isna().any():
        raise RuntimeError(f"W{week}: unresolved RB actual targets in trace")
    return t


def _prepare_point(
    metrics:pd.DataFrame,
    sims:SimulationResult,
    *,
    weights:pd.DataFrame,
    rb_context:pd.DataFrame,
    player_logs:pd.DataFrame,
    universe:pd.DataFrame,
    week:int,
)->pd.DataFrame:
    x=_recompute_mc_from_specialist(metrics,sims)
    x["market"]=x["market"].map(_canonical_market)
    x["position_family"]=x["position"].map(_pos)
    x=x.loc[
        x["position_family"].isin({"RB","FB"})
        & x["market"].isin(RB_MARKETS)
    ].copy()
    if x.empty:
        raise RuntimeError(f"W{week}: zero RB point rows")
    x=apply_ensemble(x,weights=weights)
    x["season"]=SEASON
    x["week"]=int(week)
    x["projection_mean"]=pd.to_numeric(x["ensemble_proj"],errors="coerce")
    x=_apply_rb_authorities(x,sims=sims,weights=weights,rb_context=rb_context)
    x=_attach_actuals(x,player_logs=player_logs,pregame_universe=universe,week=week)
    if x["projection_mean"].isna().any() or x["actual"].isna().any():
        raise RuntimeError(f"W{week}: unresolved RB point projection/actual")
    return x


def _compare_point(base:pd.DataFrame,cand:pd.DataFrame)->pd.DataFrame:
    key=["season","week","event_id","team","player_clean_key","market"]
    keep=key+["player","position_family","projection_mean","actual","actual_opportunities"]
    b=base[keep].rename(columns={"projection_mean":"baseline_projection"})
    c=cand[key+["projection_mean"]].rename(columns={"projection_mean":"candidate_projection"})
    out=b.merge(c,on=key,how="inner",validate="one_to_one")
    if len(out)!=len(b) or len(out)!=len(c):
        raise RuntimeError("baseline/candidate point identity drift")
    out["baseline_error"]=out["baseline_projection"]-out["actual"]
    out["candidate_error"]=out["candidate_projection"]-out["actual"]
    out["baseline_abs_error"]=out["baseline_error"].abs()
    out["candidate_abs_error"]=out["candidate_error"].abs()
    out["abs_error_improvement"]=out["baseline_abs_error"]-out["candidate_abs_error"]
    out["winner"]=np.select(
        [
            out["candidate_abs_error"]<out["baseline_abs_error"]-TOL,
            out["baseline_abs_error"]<out["candidate_abs_error"]-TOL,
        ],
        ["CANDIDATE","BASELINE"],
        default="TIE",
    )
    return out


def _compare_targets(base:pd.DataFrame,cand:pd.DataFrame)->pd.DataFrame:
    key=["week","event_id","team","player_clean_key"]
    b=base[key+["player","position_family","predicted_opportunities","actual_opportunities"]].rename(
        columns={"predicted_opportunities":"baseline_predicted_targets"}
    )
    c=cand[key+["predicted_opportunities"]].rename(
        columns={"predicted_opportunities":"candidate_predicted_targets"}
    )
    out=b.merge(c,on=key,how="inner",validate="one_to_one")
    if len(out)!=len(b) or len(out)!=len(c):
        raise RuntimeError("baseline/candidate target identity drift")
    out["baseline_error"]=out["baseline_predicted_targets"]-out["actual_opportunities"]
    out["candidate_error"]=out["candidate_predicted_targets"]-out["actual_opportunities"]
    out["baseline_abs_error"]=out["baseline_error"].abs()
    out["candidate_abs_error"]=out["candidate_error"].abs()
    out["abs_error_improvement"]=out["baseline_abs_error"]-out["candidate_abs_error"]
    out["winner"]=np.select(
        [
            out["candidate_abs_error"]<out["baseline_abs_error"]-TOL,
            out["baseline_abs_error"]<out["candidate_abs_error"]-TOL,
        ],
        ["CANDIDATE","BASELINE"],
        default="TIE",
    )
    return out


def _metrics(g:pd.DataFrame,base_col:str,cand_col:str,actual_col:str)->dict:
    b=pd.to_numeric(g[base_col],errors="coerce")
    c=pd.to_numeric(g[cand_col],errors="coerce")
    y=pd.to_numeric(g[actual_col],errors="coerce")
    be=b-y
    ce=c-y
    return {
        "rows":int(len(g)),
        "baseline_mae":float(be.abs().mean()),
        "candidate_mae":float(ce.abs().mean()),
        "mae_improvement":float(be.abs().mean()-ce.abs().mean()),
        "relative_mae_improvement":float(
            (be.abs().mean()-ce.abs().mean())/be.abs().mean()
        ) if float(be.abs().mean())>0 else np.nan,
        "baseline_median_ae":float(be.abs().median()),
        "candidate_median_ae":float(ce.abs().median()),
        "baseline_bias":float(be.mean()),
        "candidate_bias":float(ce.mean()),
        "baseline_rmse":float(np.sqrt(np.mean(np.square(be)))),
        "candidate_rmse":float(np.sqrt(np.mean(np.square(ce)))),
        "candidate_closer":int((ce.abs()<be.abs()-TOL).sum()),
        "baseline_closer":int((be.abs()<ce.abs()-TOL).sum()),
        "ties":int((np.abs(be.abs()-ce.abs())<=TOL).sum()),
    }


def run(
    *,
    player_logs_path:Path,
    team_weekly_path:Path,
    schedule_path:Path,
    universe_dir:Path,
    rb_week1_context_path:Path,
    out_dir:Path,
    iterations:int,
)->dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    player_logs=_read(player_logs_path,"player logs")
    team_weekly=_read(team_weekly_path,"team weekly history")
    schedule=_read(schedule_path,"schedule history")
    weights=load_weights()
    if weights.empty:
        raise RuntimeError("missing frozen ensemble weights")
    rb_context=load_rb_context(rb_week1_context_path)

    states,prev=identity_atlas(2013,SEASON)

    point_parts=[]
    target_parts=[]
    audit_parts=[]
    shadow_summaries={}

    for week in WEEKS:
        universe=_read(universe_dir/f"{SEASON}_week_{week:02d}.csv",f"ACT-only universe W{week}")
        bundle=build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=SEASON,
            week=week,
            prior_season=PRIOR_SEASON,
        )
        assert_no_future_rows(bundle.player_history,SEASON,week,f"W{week} player_history")
        assert_no_future_rows(bundle.team_history,SEASON,week,f"W{week} team_history")

        seed=42+week
        metrics=build_mc_predictions(bundle,iterations=int(iterations),seed=seed)
        metrics=_attach_ml_state(metrics,bundle,player_logs,week=week)
        final=_build_entitlement_state(metrics)

        baseline_sims,baseline_trace=_trace_explicit_simulation(
            final,iterations=int(iterations),seed=seed
        )
        candidate_final,room_audit,shadow_summary=apply_rb_receiving_room_share_shadow(
            final,season=SEASON,week=week,states=states,prev=prev
        )
        candidate_raw,candidate_trace=_trace_explicit_simulation(
            candidate_final,iterations=int(iterations),seed=seed
        )

        rb_rows=final.loc[final["position"].map(_pos).isin({"RB","FB"})]
        rb_keys=set(zip(rb_rows["event_id"].astype(str),rb_rows["player_clean_key"].astype(str)))
        hybrid=_hybrid_receiving_only(baseline_sims,candidate_raw,rb_keys)

        baseline_point=_prepare_point(
            metrics,baseline_sims,weights=weights,rb_context=rb_context,
            player_logs=player_logs,universe=universe,week=week
        )
        candidate_point=_prepare_point(
            metrics,hybrid,weights=weights,rb_context=rb_context,
            player_logs=player_logs,universe=universe,week=week
        )

        point_cmp=_compare_point(baseline_point,candidate_point)
        # Candidate is receiving-only: rushing point projections must be exact.
        rush=point_cmp.loc[point_cmp["market"].eq("rush_yards")]
        if not np.allclose(
            rush["baseline_projection"],rush["candidate_projection"],
            rtol=0,atol=TOL,equal_nan=True
        ):
            raise RuntimeError(
                f"W{week}: receiving-room shadow changed rush_yards point projection"
            )

        base_targets=_target_rows(baseline_trace,week,baseline_point)
        cand_targets=_target_rows(candidate_trace,week,baseline_point)
        target_cmp=_compare_targets(base_targets,cand_targets)

        point_parts.append(point_cmp)
        target_parts.append(target_cmp)
        if not room_audit.empty:
            audit_parts.append(room_audit)
        shadow_summaries[str(week)]=shadow_summary

        print(
            f"[rb-room-impact] W{week} point_rows={len(point_cmp)} "
            f"target_rows={len(target_cmp)} applied_rooms={shadow_summary['applied_rooms']}"
        )

    points=pd.concat(point_parts,ignore_index=True,sort=False)
    targets=pd.concat(target_parts,ignore_index=True,sort=False)
    audits=pd.concat(audit_parts,ignore_index=True,sort=False) if audit_parts else pd.DataFrame()

    point_summary=[]
    for market,g in points.groupby("market",dropna=False):
        rec={"market":str(market),"scope":"ALL"}
        rec.update(_metrics(g,"baseline_projection","candidate_projection","actual"))
        point_summary.append(rec)
        for week,w in g.groupby("week"):
            rr={"market":str(market),"scope":f"W{int(week)}"}
            rr.update(_metrics(w,"baseline_projection","candidate_projection","actual"))
            point_summary.append(rr)
    point_summary_df=pd.DataFrame(point_summary)

    target_summary=[]
    rec={"scope":"ALL"}
    rec.update(_metrics(
        targets,"baseline_predicted_targets","candidate_predicted_targets","actual_opportunities"
    ))
    target_summary.append(rec)
    for week,w in targets.groupby("week"):
        rr={"scope":f"W{int(week)}"}
        rr.update(_metrics(
            w,"baseline_predicted_targets","candidate_predicted_targets","actual_opportunities"
        ))
        target_summary.append(rr)
    target_summary_df=pd.DataFrame(target_summary)

    # Descriptive high-workload subsets only; no gating or rule selection.
    high_targets=targets.loc[pd.to_numeric(targets["actual_opportunities"],errors="coerce").ge(6)]
    high_point=points.loc[
        points["market"].isin({"rec_yards","receptions","rush_rec_yards"})
        & pd.to_numeric(points["actual_opportunities"],errors="coerce").ge(6)
    ]
    high_summary={
        "target_rows_actual_6plus":int(len(high_targets)),
        "target_metrics":_metrics(
            high_targets,"baseline_predicted_targets","candidate_predicted_targets","actual_opportunities"
        ) if len(high_targets) else {},
        "point_by_market":{
            str(m):_metrics(g,"baseline_projection","candidate_projection","actual")
            for m,g in high_point.groupby("market")
        },
    }

    points.to_csv(out_dir/"rb_receiving_room_share_point_impact.csv",index=False)
    targets.to_csv(out_dir/"rb_receiving_room_share_target_impact.csv",index=False)
    point_summary_df.to_csv(out_dir/"rb_receiving_room_share_point_summary.csv",index=False)
    target_summary_df.to_csv(out_dir/"rb_receiving_room_share_target_summary.csv",index=False)
    audits.to_csv(out_dir/"rb_receiving_room_share_transform_audit.csv",index=False)

    summary={
        "version":"RB_RECEIVING_ROOM_SHARE_IMPACT_V1",
        "season":SEASON,
        "weeks":list(WEEKS),
        "iterations":int(iterations),
        "interpretation_boundary":"RETROSPECTIVE_MECHANISM_IMPACT__NOT_OOS_PROMOTION_EVIDENCE",
        "point_summary":point_summary_df.loc[point_summary_df.scope.eq("ALL")].to_dict("records"),
        "target_summary":target_summary_df.loc[target_summary_df.scope.eq("ALL")].to_dict("records"),
        "high_workload_summary":high_summary,
        "shadow_summaries":shadow_summaries,
        "transform_applied_rooms":int(audits["shadow_applied"].sum()) if len(audits) else 0,
        "transform_rooms":int(len(audits)),
        "rush_yards_projection_max_abs_gap":float(
            (
                points.loc[points.market.eq("rush_yards"),"candidate_projection"]
                - points.loc[points.market.eq("rush_yards"),"baseline_projection"]
            ).abs().max()
        ),
        "parameters_fit":0,
        "automatic_promotion":False,
        "sportsbook_inputs_used_upstream":False,
        "paid_odds_api_used":False,
    }
    (out_dir/"rb_receiving_room_share_impact_summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True,default=str)+"\n"
    )
    print(json.dumps(summary,indent=2,sort_keys=True,default=str))
    return summary


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--player-logs",type=Path,required=True)
    p.add_argument("--team-weekly",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--universe-dir",type=Path,required=True)
    p.add_argument("--rb-week1-context",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=5000)
    a=p.parse_args()
    run(
        player_logs_path=a.player_logs,
        team_weekly_path=a.team_weekly,
        schedule_path=a.schedule,
        universe_dir=a.universe_dir,
        rb_week1_context_path=a.rb_week1_context,
        out_dir=a.out_dir,
        iterations=a.iterations,
    )
    return 0

if __name__=="__main__":
    raise SystemExit(main())
