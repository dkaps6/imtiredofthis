#!/usr/bin/env python3
"""Player Landscape Transmission Audit V1.

Architecture + dynamic pregame audit.  The static matrix asks whether a concrete
football input actually reaches individual opportunity, efficiency, mean, or
distribution.  The dynamic Week-5 trace follows named players through the
certified football stack without target-week outcomes or sportsbook inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import (
    _attach_component_projection,
    build_mc_predictions,
)
from scripts.backtest.historical_context import (
    assert_no_future_rows,
    build_historical_context_bundle,
)
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.state_v2 import build_state_predictions
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import lookup
from scripts.research.run_all_player_all_position_replay_v1 import (
    _apply_qb_synthesis,
    _apply_rb_authorities,
)

SEASON=2026
WEEK=5
PRIOR_SEASON=2025
TOL=1e-10

MARKETS={
    "QB":["pass_yards"],
    "RB":["rush_yards","rec_yards","receptions","rush_rec_yards"],
    "WR":["rec_yards","receptions"],
    "TE":["rec_yards","receptions"],
}
LAYERS=[
    "IDENTITY_AVAILABILITY",
    "INDIVIDUAL_RECENT_USAGE",
    "ROOM_HIERARCHY_COMPETITION",
    "TEAM_OPPORTUNITY_ENVIRONMENT",
    "INDIVIDUAL_EFFICIENCY",
    "OPPONENT_DEFENSE",
    "INDIVIDUAL_MATCHUP_INTERACTION",
    "INJURIES_VACANCIES",
    "DISTRIBUTION_UNCERTAINTY",
]
VALID_STATUS={
    "CONSUMED_DIRECT_PLAYER",
    "CONSUMED_PLAYER_VIA_SPECIALIST",
    "CONSUMED_TEAM_CONTEXT",
    "CONSUMED_OPPONENT_CONTEXT",
    "CONSUMED_INDIRECT",
    "AVAILABLE_BUT_DROPPED",
    "AVAILABLE_BUT_NOT_CONSUMED",
    "SOURCE_PARITY_BLOCKED",
    "PROSPECTIVE_ONLY_FROZEN",
    "TESTED_AND_CLOSED",
    "NOT_AVAILABLE",
}


def _spec(
    feature,layer,scope,status,source,consumer,
    positions,markets="*",opportunity=False,efficiency=False,mean=False,distribution=False,
    historical="",current="",reopen="",notes="",
):
    return dict(
        feature=feature,landscape_layer=layer,feature_scope=scope,
        production_consumption_status=status,source_artifact_module=source,
        consuming_module_function=consumer,positions=",".join(positions),
        markets=markets if isinstance(markets,str) else ",".join(markets),
        strict_prior_live_availability=True,current_season_freshness="LIVE_OR_STRICT_PRIOR",
        affects_opportunity=bool(opportunity),affects_efficiency=bool(efficiency),
        affects_mean=bool(mean),affects_distribution=bool(distribution),
        historical_validation_status=historical,current_season_validation_status=current,
        closed_reopen_rule=reopen,notes=notes,
    )


def feature_specs()->list[dict]:
    s=[]
    # Identity / availability.
    s += [
      _spec("player_team_opponent_identity","IDENTITY_AVAILABILITY","PLAYER","CONSUMED_DIRECT_PLAYER",
            "historical_context.py + context_bridge.py","build_player_contexts / simulation keys",
            ["QB","RB","WR","TE"],opportunity=True,mean=True,
            historical="CORE_IDENTITY_CONTRACT",current="CURRENT_W5_TRACE",reopen="",
            notes="Every simulation row is keyed to named player/team/opponent."),
      _spec("pregame_roster_status","IDENTITY_AVAILABILITY","PLAYER","CONSUMED_INDIRECT",
            "historical_inputs.py","pregame universe membership",
            ["QB","RB","WR","TE"],opportunity=True,mean=True,
            historical="LEAKAGE_SAFE_ROSTER_UNIVERSE",current="CURRENT_W5_TRACE"),
      _spec("depth_role_string","IDENTITY_AVAILABILITY","PLAYER","CONSUMED_INDIRECT",
            "historical_inputs.py + simulation_rules.py","QB opportunity / WR role labeling",
            ["QB","WR"],opportunity=True,mean=True,
            historical="ROLE_ROUTING_ACTIVE",current="CURRENT_W5_TRACE"),
      _spec("depth_role_string_rb_te","IDENTITY_AVAILABILITY","PLAYER","AVAILABLE_BUT_NOT_CONSUMED",
            "historical_inputs.py","generic RB/TE rules",
            ["RB","TE"],opportunity=False,mean=False,
            notes="RB/TE current role is expressed primarily through shares/specialists, not the raw role label."),
    ]
    # Player recent usage / history.
    s += [
      _spec("target_share","INDIVIDUAL_RECENT_USAGE","PLAYER","CONSUMED_DIRECT_PLAYER",
            "player_form_consensus -> bayesian_v2 -> simulation_rules","rules_tgt_share",
            ["RB","WR","TE"],["rec_yards","receptions"],opportunity=True,mean=True,
            historical="BAYESIAN_CURRENT_STATE_TRANSMISSION_MISMATCH_KNOWN",current="W1_4_ALLOCATION_COMPRESSION_CONFIRMED"),
      _spec("rush_share","INDIVIDUAL_RECENT_USAGE","PLAYER","CONSUMED_DIRECT_PLAYER",
            "player_form_consensus -> bayesian_v2 -> simulation_rules","rules_rush_share",
            ["RB"],["rush_yards","rush_rec_yards"],opportunity=True,mean=True,
            historical="BAYESIAN_CURRENT_STATE_TRANSMISSION_MISMATCH_KNOWN",current="W1_4_ALLOCATION_COMPRESSION_CONFIRMED"),
      _spec("qb_pass_attempt_share","INDIVIDUAL_RECENT_USAGE","PLAYER","CONSUMED_PLAYER_VIA_SPECIALIST",
            "qb_opportunity.py + component_predictions.py","mc_expected_pass_attempts + M89/M90",
            ["QB"],["pass_yards"],opportunity=True,mean=True,
            historical="QB_PLAYER_ROLE_AUTHORITY",current="W1_4_REPLAY_COMPLETE"),
      _spec("lagged_player_game_history_ml","INDIVIDUAL_RECENT_USAGE","PLAYER","CONSUMED_INDIRECT",
            "ml_v2.py","market-specific HistGradientBoosting -> ensemble",
            ["QB","RB","WR","TE"],opportunity=False,mean=True,
            historical="OOS_ENSEMBLE_COMPONENT",current="CURRENT_W5_TRACE",
            notes="Prev/mean3/mean5 player box-score and share history; no opponent features."),
      _spec("lagged_player_outcome_state","INDIVIDUAL_RECENT_USAGE","PLAYER","CONSUMED_INDIRECT",
            "state_v2.py","Markov state prediction -> ensemble",
            ["QB","RB","WR","TE"],opportunity=False,mean=True,
            historical="OOS_ENSEMBLE_COMPONENT",current="CURRENT_W5_TRACE",
            notes="Player-specific recent outcome regime; no opponent features."),
      _spec("route_rate","INDIVIDUAL_RECENT_USAGE","PLAYER","SOURCE_PARITY_BLOCKED",
            "PlayerForm route schema / public route-volume source frontier","generic rules/MC",
            ["RB","WR","TE"],["rec_yards","receptions"],
            historical="HISTORICAL_WEEKLY_ROUTE_PARITY_NOT_CLEARED",
            reopen="Clear a reproducible weekly routes-run source with historical/live semantic parity first.",
            notes="The schema supports route rate, but canonical Week-5 parity reconstruction has no populated route values; do not treat an empty column as available player state."),
      _spec("yprr","INDIVIDUAL_RECENT_USAGE","PLAYER","SOURCE_PARITY_BLOCKED",
            "PlayerForm route schema / public route-volume source frontier","generic rules/MC",
            ["WR","TE"],["rec_yards"],
            historical="HISTORICAL_WEEKLY_ROUTE_PARITY_NOT_CLEARED",
            reopen="Clear the same routes-run source gate before any YPRR transmission candidate.",
            notes="YPRR requires real routes run. Target/dropback counts may not be relabeled as routes."),
      _spec("rb_prior_receiving_room_share","INDIVIDUAL_RECENT_USAGE","PLAYER","PROSPECTIVE_ONLY_FROZEN",
            "rb_receiving_identity_runtime_v1.py","RB_RECEIVING_ROOM_SHARE_SHADOW_V1",
            ["RB"],["rec_yards","receptions","rush_rec_yards"],opportunity=True,mean=True,
            historical="W1_4_RETROSPECTIVE_IMPACT_CONFIRMED",current="W5_PREGAME_LOCK_FROZEN",
            reopen="Grade exact W5+ frozen rule before promotion."),
      _spec("wr_te_target_share_trajectory","INDIVIDUAL_RECENT_USAGE","PLAYER","PROSPECTIVE_ONLY_FROZEN",
            "PLAYER_TARGET_SHARE_TRAJECTORY_V1","Week5+ trajectory shadow",
            ["WR","TE"],["rec_yards","receptions"],opportunity=True,mean=True,
            historical="HISTORICALLY_CONFIRMED",current="W1_4_INELIGIBLE_W5_LOCK_FROZEN",
            reopen="Do not weaken four-prior-same-season-game eligibility."),
    ]
    # Room hierarchy.
    s += [
      _spec("m38_wr_hierarchy","ROOM_HIERARCHY_COMPETITION","ROOM","CONSUMED_PLAYER_VIA_SPECIALIST",
            "target_entitlement_v1.py","materialize_target_entitlement",
            ["WR"],["rec_yards","receptions"],opportunity=True,mean=True,
            historical="PROMOTED_PRODUCTION",current="CURRENT_W5_TRACE"),
      _spec("te_r5p_room_redistribution","ROOM_HIERARCHY_COMPETITION","ROOM","CONSUMED_PLAYER_VIA_SPECIALIST",
            "te_r5p_entitlement_adapter_v1.py","apply_te_r5p_entitlement",
            ["TE"],["rec_yards","receptions"],opportunity=True,mean=True,
            historical="PROMOTED_PRODUCTION",current="CURRENT_W5_TRACE"),
      _spec("wr_r15_secondary_room_redistribution","ROOM_HIERARCHY_COMPETITION","ROOM","CONSUMED_PLAYER_VIA_SPECIALIST",
            "wr_r15_entitlement_adapter_v1.py","apply_wr_r15_entitlement",
            ["WR"],["rec_yards","receptions"],opportunity=True,mean=True,
            historical="PROMOTED_PRODUCTION",current="CURRENT_W5_TRACE"),
      _spec("alpha_receiver_injury_vacancy","ROOM_HIERARCHY_COMPETITION","ROOM","CONSUMED_DIRECT_PLAYER",
            "simulation_rules.py","_injury_target_overrides",
            ["RB","WR","TE"],["rec_yards","receptions"],opportunity=True,mean=True),
      _spec("rb_carry_snap_room_state","ROOM_HIERARCHY_COMPETITION","ROOM","PROSPECTIVE_ONLY_FROZEN",
            "RB_PLAYER_STATE_ALLOCATION_SHADOW_V1","Week5+ RB allocation shadow",
            ["RB"],["rush_yards","rush_rec_yards"],opportunity=True,mean=True,
            historical="RETROSPECTIVE_M96_STOP_PRESERVED",current="W5_PREGAME_LOCK_FROZEN"),
    ]
    # Team environment.
    s += [
      _spec("plays_est_and_pace","TEAM_OPPORTUNITY_ENVIRONMENT","TEAM","CONSUMED_TEAM_CONTEXT",
            "TeamContext","rules_v2.estimate_plays",
            ["QB","RB","WR","TE"],opportunity=True,mean=True,historical="PRODUCTION",current="CURRENT_W5_TRACE"),
      _spec("generic_pass_rush_split_0_57","TEAM_OPPORTUNITY_ENVIRONMENT","TEAM","CONSUMED_TEAM_CONTEXT",
            "rules_v2.py","project_game_script",
            ["QB","RB","WR","TE"],opportunity=True,mean=True,
            historical="ARCHITECTURE_AUDITED",current="CURRENT_W5_TRACE",
            notes="Generic skill stack hardcodes pass_share=0.57 before specialist paths."),
      _spec("offensive_proe","TEAM_OPPORTUNITY_ENVIRONMENT","TEAM","AVAILABLE_BUT_NOT_CONSUMED",
            "TeamContext.proe","generic simulation rules_pass_rate shadows fallback",
            ["RB","WR","TE"],opportunity=True,mean=True,
            historical="FMT_SIGNAL_REPLICATED_BUT_SIMPLE_INTEGRATION_FAILED",current="",
            reopen="FMT-INT-WR-TRUE-PROE-V1 is tested and closed; new mechanism must differ."),
      _spec("game_script_lead_neutral_trail","TEAM_OPPORTUNITY_ENVIRONMENT","TEAM","AVAILABLE_BUT_NOT_CONSUMED",
            "rules_v2.project_game_script","simulation_v2",
            ["RB","WR","TE"],opportunity=True,
            historical="PHASE_A_ARCHITECTURE_GAP",current="",
            notes="Probabilities are calculated but generic simulator does not alter pass/rush split from them."),
      _spec("qb_team_attempt_environment","TEAM_OPPORTUNITY_ENVIRONMENT","TEAM","CONSUMED_PLAYER_VIA_SPECIALIST",
            "component_predictions + qb_pass_synthesis_v1","M89/M90",
            ["QB"],["pass_yards"],opportunity=True,mean=True,
            historical="PROTECTED_QB_SPECIALIST",current="W1_4_TEAM_VOLUME_DOMINANT"),
    ]
    # Individual efficiency.
    s += [
      _spec("player_ypt","INDIVIDUAL_EFFICIENCY","PLAYER","CONSUMED_DIRECT_PLAYER",
            "PlayerForm -> bayesian_v2","rules_ypt -> simulation_v2",
            ["RB","WR","TE"],["rec_yards"],efficiency=True,mean=True),
      _spec("player_catch_rate","INDIVIDUAL_EFFICIENCY","PLAYER","CONSUMED_DIRECT_PLAYER",
            "PlayerForm -> bayesian_v2","rules_catch_rate -> simulation_v2",
            ["RB","WR","TE"],["receptions"],efficiency=True,mean=True),
      _spec("player_ypc","INDIVIDUAL_EFFICIENCY","PLAYER","CONSUMED_DIRECT_PLAYER",
            "PlayerForm -> bayesian_v2","rules_ypc -> simulation_v2",
            ["RB"],["rush_yards","rush_rec_yards"],efficiency=True,mean=True),
      _spec("qb_ypa_efficiency","INDIVIDUAL_EFFICIENCY","PLAYER","CONSUMED_PLAYER_VIA_SPECIALIST",
            "PlayerForm/Bayes + QB synthesis","rules_ypa + M89/M90",
            ["QB"],["pass_yards"],efficiency=True,mean=True),
      _spec("player_target_depth_dispersion","INDIVIDUAL_EFFICIENCY","PLAYER","TESTED_AND_CLOSED",
            "PLAYER_TARGET_DEPTH_DISTRIBUTION_SHADOW_V1","distribution shadow only",
            ["WR","TE"],["rec_yards"],distribution=True,
            historical="FEATURE_CONFIRMED_TRANSFORM_NOT_CONFIRMED",current="W1_4_CRPS_SLIGHTLY_WORSE",
            reopen="Do not post-hoc threshold the same W1-4 outcomes."),
    ]
    # Opponent defense.
    s += [
      _spec("opponent_pressure","OPPONENT_DEFENSE","OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "TeamContext.pressure_rate_generated","rules_v2.matchup_multipliers",
            ["QB","RB","WR","TE"],opportunity=True,efficiency=True,mean=True,distribution=True,
            historical="PRESSURE_CALIBRATION_PROMOTED",current="CURRENT_W5_TRACE"),
      _spec("coverage_man_zone_middle","OPPONENT_DEFENSE","OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "TeamContext coverage fields","rules_v2.matchup_multipliers",
            ["RB","WR","TE"],["rec_yards","receptions"],opportunity=True,mean=True,
            historical="GENERIC_ROLE_MATCHUP_RULES",current="CURRENT_W5_TRACE"),
      _spec("light_heavy_box_rates","OPPONENT_DEFENSE","OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "TeamContext box rates","rules_v2.matchup_multipliers",
            ["RB"],["rush_yards","rush_rec_yards"],efficiency=True,mean=True),
      _spec("def_rush_epa","OPPONENT_DEFENSE","OPPONENT","AVAILABLE_BUT_NOT_CONSUMED",
            "TeamContext.def_rush_epa","generic rules/MC",
            ["RB"],["rush_yards","rush_rec_yards"],efficiency=True,mean=True,
            historical="M95A_M95B_GENERIC_INTERACTION_CLOSED",current="",
            reopen="Do not add arbitrary weak-run-defense multiplier."),
      _spec("def_pass_epa","OPPONENT_DEFENSE","OPPONENT","AVAILABLE_BUT_NOT_CONSUMED",
            "TeamContext.def_pass_epa","generic RB/WR/TE rules",
            ["RB","WR","TE"],["rec_yards","receptions"],efficiency=True,mean=True),
      _spec("explosive_play_rate_allowed","OPPONENT_DEFENSE","OPPONENT","AVAILABLE_BUT_NOT_CONSUMED",
            "TeamContext.explosive_play_rate_allowed","generic rules/MC",
            ["RB","WR","TE"],["rush_yards","rec_yards","rush_rec_yards"],distribution=True),
      _spec("position_specific_ypt_allowed","OPPONENT_DEFENSE","OPPONENT","AVAILABLE_BUT_DROPPED",
            "make_team_form.py: wr/te/rb_ypt_allowed","context_bridge.TeamContext",
            ["RB","WR","TE"],["rec_yards"],efficiency=True,mean=True,
            historical="TE_POSITION_YPT_DIAGNOSTIC_REPLICATED_SOURCE_PARITY_BLOCKED"),
      _spec("outside_slot_ypt_allowed","OPPONENT_DEFENSE","OPPONENT","AVAILABLE_BUT_DROPPED",
            "make_team_form.py","context_bridge.TeamContext",
            ["WR"],["rec_yards"],efficiency=True,mean=True,
            historical="SOURCE_PARITY_BLOCKED"),
      _spec("dl_ybc_stuff_rate","OPPONENT_DEFENSE","OPPONENT","AVAILABLE_BUT_DROPPED",
            "make_team_form.py","context_bridge.TeamContext",
            ["RB"],["rush_yards"],efficiency=True,mean=True,
            historical="SOURCE_PARITY_BLOCKED"),
    ]
    # Interaction layer.
    s += [
      _spec("wr_role_x_coverage","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "simulation_rules + rules_v2","WR1/WR1_5/SLOT target multipliers",
            ["WR"],["rec_yards","receptions"],opportunity=True,mean=True),
      _spec("te_role_x_zone_middle","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "rules_v2","TE target multiplier",
            ["TE"],["rec_yards","receptions"],opportunity=True,mean=True),
      _spec("rb_receiving_role_x_zone_pressure","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "rules_v2","RB receiving target multiplier",
            ["RB"],["rec_yards","receptions"],opportunity=True,mean=True),
      _spec("rb_role_x_run_defense_quality","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+OPPONENT","TESTED_AND_CLOSED",
            "M95A/M95B","generic role x defensive rushing vulnerability",
            ["RB"],["rush_yards","rush_rec_yards"],opportunity=True,efficiency=True,mean=True,
            historical="DESCRIPTIVE_TRUTH_BUT_PROSPECTIVE_MODELS_MIXED_FAILED",
            reopen="Do not repackage M95A/M95B with new weights."),
      _spec("wr_cb_assignment","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+OPPONENT","NOT_AVAILABLE",
            "retired coverage_penalty","none",
            ["WR"],["rec_yards","receptions"],opportunity=True,efficiency=True,
            historical="PAID_SOURCE_NOT_AVAILABLE_HEURISTIC_RETIRED",
            reopen="Requires free reproducible full-slate assignment source."),
      _spec("fmt_rb_pass_rate_faced_candidate","INDIVIDUAL_MATCHUP_INTERACTION","TEAM+OPPONENT","TESTED_AND_CLOSED",
            "FMT-INT-RB-DEF-PASS-RATE-FACED-V1","none",
            ["RB"],["rush_yards","rush_rec_yards"],opportunity=True,mean=True,
            historical="HISTORICAL_INTEGRATION_FAIL_CLOSED"),
      _spec("fmt_wr_true_proe_candidate","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+TEAM","TESTED_AND_CLOSED",
            "FMT-INT-WR-TRUE-PROE-V1","none",
            ["WR"],["rec_yards"],opportunity=True,mean=True,
            historical="HISTORICAL_INTEGRATION_FAIL_CLOSED"),
      _spec("fmt_te_pass_success_candidate","INDIVIDUAL_MATCHUP_INTERACTION","PLAYER+OPPONENT","TESTED_AND_CLOSED",
            "FMT-INT-TE-DEF-PASS-SUCCESS-V1","none",
            ["TE"],["rec_yards"],efficiency=True,mean=True,
            historical="HISTORICAL_INTEGRATION_FAIL_CLOSED"),
    ]
    # Injuries.
    s += [
      _spec("player_own_injury_status","INJURIES_VACANCIES","PLAYER","CONSUMED_DIRECT_PLAYER",
            "context_bridge injuries","simulation_rules._injury_limited",
            ["RB","WR","TE"],"*",opportunity=True,mean=True),
      _spec("alpha_receiver_vacancy_redistribution","INJURIES_VACANCIES","ROOM","CONSUMED_DIRECT_PLAYER",
            "simulation_rules","_injury_target_overrides",
            ["RB","WR","TE"],["rec_yards","receptions"],opportunity=True,mean=True),
      _spec("opponent_defender_injury","INJURIES_VACANCIES","OPPONENT","SOURCE_PARITY_BLOCKED",
            "nflverse injury source audit","none",
            ["QB","RB","WR","TE"],"*",
            historical="SOURCE_PARITY_NOT_CLEARED",
            reopen="Need qualified pregame defender-status source; blanks cannot mean healthy."),
    ]
    # Distribution.
    s += [
      _spec("generic_mc_efficiency_shocks","DISTRIBUTION_UNCERTAINTY","PLAYER+TEAM","CONSUMED_INDIRECT",
            "simulation_v2","pass/rush efficiency shocks + yardage noise",
            ["QB","RB","WR","TE"],"*",distribution=True),
      _spec("rules_volatility_mult","DISTRIBUTION_UNCERTAINTY","PLAYER+OPPONENT","CONSUMED_OPPONENT_CONTEXT",
            "rules_v2 + simulation_rules","simulation_v2 rec/rush/pass SD",
            ["QB","RB","WR","TE"],"*",distribution=True),
      _spec("right_tail_asymmetry","DISTRIBUTION_UNCERTAINTY","MARKET","PROSPECTIVE_ONLY_FROZEN",
            "DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_V1","none in production",
            ["RB","WR","TE"],["rush_yards","rec_yards","receptions","rush_rec_yards"],distribution=True,
            historical="REPLICATED_2024_2025",current="",
            reopen="Separate mean-neutral asymmetric candidate only."),
    ]
    return s


def _expand_matrix(specs:list[dict])->pd.DataFrame:
    rows=[]
    for spec in specs:
        if spec["production_consumption_status"] not in VALID_STATUS:
            raise RuntimeError(f"invalid status {spec['production_consumption_status']}")
        positions=spec["positions"].split(",")
        market_token=spec["markets"]
        for pos in positions:
            applicable=MARKETS[pos] if market_token=="*" else [
                m for m in market_token.split(",") if m in MARKETS[pos]
            ]
            for market in applicable:
                row={k:v for k,v in spec.items() if k not in {"positions","markets"}}
                row["position_family"]=pos
                row["market"]=market
                rows.append(row)
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("landscape matrix empty")
    return out.sort_values(["position_family","market","landscape_layer","feature"]).reset_index(drop=True)


def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _position(v)->str:
    p=str(v or "").upper().strip()
    if p in {"HB","FB","TB"} or p.startswith("RB"):
        return "RB"
    if p in {"LWR","RWR","SWR"} or p.startswith("WR"):
        return "WR"
    return p


def _roof_map()->dict[tuple[int,int,str],float]:
    try:
        import nflreadpy as nfl
        raw=nfl.load_schedules(seasons=[SEASON])
        s=raw.to_pandas() if hasattr(raw,"to_pandas") else pd.DataFrame(raw)
    except Exception:
        return {}
    s.columns=[str(c).strip().lower() for c in s.columns]
    q=s.loc[pd.to_numeric(s.get("week"),errors="coerce").eq(WEEK)].copy()
    out={}
    for _,r in q.iterrows():
        roof=str(r.get("roof","") or "").lower()
        val=float(int(any(x in roof for x in ("dome","closed","indoor")))) if roof else np.nan
        for c in ("home_team","away_team"):
            t=canon_team(r.get(c))
            if t: out[(SEASON,WEEK,t)]=val
    return out


def _market_mc_from_sims(metrics:pd.DataFrame,sims)->pd.DataFrame:
    out=metrics.copy()
    vals=[]
    for _,row in out.iterrows():
        arr=lookup(sims,row,str(row["market"]))
        if arr is None or len(arr)==0:
            vals.append(np.nan)
            continue
        x=np.asarray(arr,dtype=float)
        if str(row["market"])=="pass_yards":
            ar=pd.to_numeric(pd.Series([row.get("mc_pass_attempts_per_dropback")]),errors="coerce").iloc[0]
            sh=pd.to_numeric(pd.Series([row.get("qb_pass_att_share")]),errors="coerce").iloc[0]
            if pd.notna(ar): x=x*float(np.clip(ar,0.50,1.00))
            if pd.notna(sh): x=x*float(np.clip(sh,0.0,1.0))
        vals.append(float(np.mean(x)))
    out["mc_proj"]=vals
    return out


def _build_dynamic_trace(
    *,
    player_logs:pd.DataFrame,
    team_weekly:pd.DataFrame,
    schedule:pd.DataFrame,
    universe:pd.DataFrame,
    iterations:int,
)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    assert_no_future_rows(player_logs,SEASON,WEEK,"landscape_player_logs")
    assert_no_future_rows(team_weekly,SEASON,WEEK,"landscape_team_weekly")

    bundle=build_historical_context_bundle(
        player_logs=player_logs,team_weekly=team_weekly,pregame_universe=universe,
        schedule=schedule,season=SEASON,week=WEEK,prior_season=PRIOR_SEASON,
    )
    raw=build_mc_predictions(bundle,iterations=int(iterations),seed=20261007)

    keys=["event_id","team","player_clean_key"]
    players=raw.sort_values(keys).drop_duplicates(keys,keep="last").copy()
    base,base_trace=materialize_target_entitlement(players)
    te,te_trace,te_audit=apply_te_r5p_entitlement(base)
    final,wr_trace,wr_audit=apply_wr_r15_entitlement(te)

    alloc_trace=[]
    sims=explicit_simulate(final,iterations=int(iterations),seed=20261007,allocation_trace=alloc_trace)
    market=_market_mc_from_sims(raw,sims)

    _,ml=build_ml(player_logs,bundle.player_consensus,SEASON,WEEK)
    _,state=build_state_predictions(player_logs,bundle.player_consensus,SEASON,WEEK)
    market=_attach_component_projection(market,ml,"ml")
    market=_attach_component_projection(market,state,"state")
    weights=load_weights()
    market=apply_ensemble(market,weights=weights)
    market["position_family"]=market["position"].map(_position)
    market["week"]=WEEK
    market["season"]=SEASON

    # Existing point authorities. Week 5 has no P3 scope, but V2 rush+rec remains
    # available. QB M89/M90 is applied with current strict-prior history.
    market=_apply_qb_synthesis(
        market,player_logs=player_logs,team_weekly=team_weekly,
        controlled_map=_roof_map(),
    )
    empty_rb_context=pd.DataFrame(columns=[
        "season","week","player","team","opponent","rb_synthesis_proj",
        "rb_synthesis_route","rb_synthesis_version","rb_synthesis_applied",
        "football_only_no_odds","sportsbook_inputs_used","player_base_key",
    ])
    market=_apply_rb_authorities(
        market,sims=sims,weights=weights,rb_context=empty_rb_context,
    )

    # Specialist entitlement state -> expected target opportunity.
    pcols=[
        "event_id","team","player_clean_key","player","position",
        "rules_plays_est","rules_pass_rate","rules_tgt_share","rules_rush_share",
        "rules_ypt","rules_ypc","rules_catch_rate","rules_pass_eff_mult",
        "rules_rush_eff_mult","rules_volatility_mult",
        "entitlement_tgt_share","entitlement_residual_share",
    ]
    for c in pcols:
        if c not in final.columns: final[c]=np.nan
    p=final[pcols].copy()
    p["position_family"]=p["position"].map(_position)
    p["team_pass_opportunities_mean"]=(
        pd.to_numeric(p["rules_plays_est"],errors="coerce")
        * pd.to_numeric(p["rules_pass_rate"],errors="coerce")
    )
    p["expected_targets"]=(
        p["team_pass_opportunities_mean"]
        * pd.to_numeric(p["entitlement_tgt_share"],errors="coerce")
    )

    rush=pd.DataFrame(alloc_trace)
    if not rush.empty:
        rush=rush[["event_id","team","player_clean_key","expected_carries_from_final_probability"]]
        rush=rush.rename(columns={"expected_carries_from_final_probability":"expected_carries"})
        p=p.merge(rush,on=["event_id","team","player_clean_key"],how="left",validate="one_to_one")
    else:
        p["expected_carries"]=np.nan
    p["expected_carries"]=pd.to_numeric(p["expected_carries"],errors="coerce").fillna(0.0)
    p["expected_targets"]=pd.to_numeric(p["expected_targets"],errors="coerce").fillna(0.0)

    # Add opponent/team context values from the exact bundle.
    ctx=[]
    for pc in bundle.players:
        off,defn=pc.offense,pc.defense
        f=pc.features or {}
        ctx.append({
            "team":pc.team,"player_clean_key":next(
                (r.player_clean_key for r in bundle.player_form.itertuples(index=False)
                 if str(r.team)==str(pc.team) and str(r.player)==str(pc.player)),""
            ),
            "opponent":pc.opponent,"role":pc.role,
            "player_tgt_share":f.get("tgt_share"),"player_rush_share":f.get("rush_share"),
            "player_route_rate":f.get("route_rate"),"player_yprr":f.get("yprr"),
            "player_ypt":f.get("ypt"),"player_ypc":f.get("ypc"),
            "player_catch_rate":f.get("catch_rate"),
            "injury_status":f.get("injury_status"),"injury_designation":f.get("injury_designation"),
            "off_plays_est":getattr(off,"plays_est",np.nan),
            "off_proe":getattr(off,"proe",np.nan),
            "off_success_rate":getattr(off,"success_rate_off",np.nan),
            "opp_success_rate_def":getattr(defn,"success_rate_def",np.nan),
            "opp_pressure_generated":getattr(defn,"pressure_rate_generated",np.nan),
            "opp_def_pass_epa":getattr(defn,"def_pass_epa",np.nan),
            "opp_def_rush_epa":getattr(defn,"def_rush_epa",np.nan),
            "opp_explosive_allowed":getattr(defn,"explosive_play_rate_allowed",np.nan),
            "opp_man_rate":getattr(defn,"coverage_man_rate",np.nan),
            "opp_zone_rate":getattr(defn,"coverage_zone_rate",np.nan),
            "opp_middle_open_rate":getattr(defn,"middle_open_rate",np.nan),
            "opp_light_box_rate":getattr(defn,"light_box_rate",np.nan),
            "opp_heavy_box_rate":getattr(defn,"heavy_box_rate",np.nan),
        })
    ctx=pd.DataFrame(ctx)
    if not ctx.empty:
        ctx=ctx.drop_duplicates(["team","player_clean_key"])
        p=p.merge(ctx,on=["team","player_clean_key"],how="left",validate="one_to_one")

    # Workload selection is player-level and deterministic, never outcome-based.
    p["workload_score"]=np.select(
        [
            p["position_family"].eq("QB"),
            p["position_family"].eq("RB"),
            p["position_family"].isin(["WR","TE"]),
        ],
        [
            0.0, # overwritten from pass-attempt trace below
            p["expected_carries"]+p["expected_targets"],
            p["expected_targets"],
        ],
        default=0.0,
    )
    qbatt=market.loc[
        market["position_family"].eq("QB") & market["market"].eq("pass_yards"),
        ["team","player_clean_key","mc_expected_pass_attempts"],
    ].drop_duplicates(["team","player_clean_key"])
    if not qbatt.empty:
        p=p.merge(qbatt,on=["team","player_clean_key"],how="left",validate="one_to_one")
        p.loc[p["position_family"].eq("QB"),"workload_score"]=pd.to_numeric(
            p.loc[p["position_family"].eq("QB"),"mc_expected_pass_attempts"],errors="coerce"
        ).fillna(0.0)

    selected=[]
    for pos in ("QB","RB","WR","TE"):
        # Select only from players who actually own at least one required
        # projected market for this position. This matters most for QB, where
        # the pregame roster can contain backups but only the frozen
        # projection-eligible QB receives pass_yards.
        elig = market.loc[
            market["position_family"].eq(pos)
            & market["market"].isin(MARKETS[pos]),
            ["team","player_clean_key"],
        ].drop_duplicates()
        q=p.loc[p["position_family"].eq(pos)].merge(
            elig,on=["team","player_clean_key"],how="inner",validate="one_to_one"
        )
        if len(q) < 3:
            raise RuntimeError(
                f"Week5 dynamic trace has fewer than 3 projected {pos} players: {len(q)}"
            )
        q=q.sort_values(["workload_score","team","player_clean_key"],kind="mergesort").reset_index(drop=True)
        picks=[
            ("LOW",0),
            ("MEDIAN",(len(q)-1)//2),
            ("HIGH",len(q)-1),
        ]
        for tier,idx in picks:
            r=q.iloc[int(idx)]
            selected.append((pos,tier,str(r["team"]),str(r["player_clean_key"])))
    sel=pd.DataFrame(selected,columns=["position_family","workload_tier","team","player_clean_key"])
    sp=p.merge(sel,on=["position_family","team","player_clean_key"],how="inner",validate="one_to_one")

    trace=market.merge(
        sp,
        on=["position_family","team","player_clean_key"],
        how="inner",
        suffixes=("","_player"),
    )
    trace=trace.loc[
        [
            m in MARKETS.get(pos,[])
            for m,pos in zip(trace["market"],trace["position_family"])
        ]
    ].copy()
    trace["trace_chain"]="identity>availability>player_state>room_state>team_environment>opponent_environment>opportunity>efficiency>final_mean>distribution"
    trace["sportsbook_inputs_used"]=False
    trace["week5_outcomes_used"]=False
    trace["final_mean_authority"]=np.where(
        trace["position_family"].eq("QB") & trace["market"].eq("pass_yards"),
        "QB_M89_M90_SYNTHESIS",
        np.where(
            trace["position_family"].eq("RB") & trace["market"].eq("rush_rec_yards"),
            "RB_RUSH_REC_V2_OR_ENSEMBLE",
            "ENSEMBLE_AFTER_PROMOTED_RECEIVING_ENTITLEMENT",
        ),
    )

    selected_identity_tiers=trace[
        ["position_family","workload_tier","team","player_clean_key"]
    ].drop_duplicates()
    if len(selected_identity_tiers) != 12:
        raise RuntimeError(
            f"dynamic trace lost selected position/workload tiers: {len(selected_identity_tiers)} != 12"
        )

    dynamic_fields=[
        "player_tgt_share","player_rush_share","player_route_rate","player_yprr",
        "player_ypt","player_ypc","player_catch_rate","off_plays_est","off_proe",
        "off_success_rate","opp_success_rate_def","opp_pressure_generated",
        "opp_def_pass_epa","opp_def_rush_epa","opp_explosive_allowed",
        "opp_man_rate","opp_zone_rate","opp_middle_open_rate",
        "opp_light_box_rate","opp_heavy_box_rate",
    ]
    dynamic_coverage={}
    for c in dynamic_fields:
        if c in p.columns:
            dynamic_coverage[c]={
                "nonnull_rows":int(p[c].notna().sum()),
                "rows":int(len(p)),
                "nonnull_rate":float(p[c].notna().mean()),
            }

    payload={
        "season":SEASON,"week":WEEK,
        "pregame_players":int(len(p)),
        "context_provenance":"HISTORICAL_AVAILABILITY_PARITY_WEEK5_RECONSTRUCTION",
        "live_supplemental_team_sources_included":False,
        "dynamic_feature_coverage":dynamic_coverage,
        "trace_players":int(trace[["team","player_clean_key"]].drop_duplicates().shape[0]),
        "trace_position_tiers":int(len(selected_identity_tiers)),
        "trace_rows":int(len(trace)),
        "positions":sorted(trace["position_family"].unique().tolist()),
        "te_r5p_audit_disposition":str(te_audit.get("disposition","")),
        "wr_r15_audit_disposition":str(wr_audit.get("disposition","")),
        "parameters_fit_by_audit":0,
        "sportsbook_inputs_used":False,
        "week5_outcomes_used":False,
    }
    return p,trace,payload


def _position_market_summary(matrix:pd.DataFrame)->pd.DataFrame:
    rows=[]
    consumed={
        "CONSUMED_DIRECT_PLAYER","CONSUMED_PLAYER_VIA_SPECIALIST",
        "CONSUMED_TEAM_CONTEXT","CONSUMED_OPPONENT_CONTEXT","CONSUMED_INDIRECT",
    }
    for (pos,market),g in matrix.groupby(["position_family","market"],sort=True):
        row={"position_family":pos,"market":market,"feature_rows":int(len(g))}
        for layer in LAYERS:
            q=g.loc[g["landscape_layer"].eq(layer)]
            row[f"{layer.lower()}_rows"]=int(len(q))
            row[f"{layer.lower()}_consumed_rows"]=int(q["production_consumption_status"].isin(consumed).sum())
            row[f"{layer.lower()}_gap_rows"]=int((~q["production_consumption_status"].isin(consumed)).sum())
        row["consumed_feature_rows"]=int(g["production_consumption_status"].isin(consumed).sum())
        row["gap_feature_rows"]=int((~g["production_consumption_status"].isin(consumed)).sum())
        row["individualized_usage"]=bool(
            g.loc[g["landscape_layer"].isin(["INDIVIDUAL_RECENT_USAGE","ROOM_HIERARCHY_COMPETITION"]),
                  "production_consumption_status"].isin(consumed).any()
        )
        row["individualized_efficiency"]=bool(
            g.loc[g["landscape_layer"].eq("INDIVIDUAL_EFFICIENCY"),
                  "production_consumption_status"].isin(consumed).any()
        )
        row["opponent_materially_consumed"]=bool(
            g.loc[g["landscape_layer"].isin(["OPPONENT_DEFENSE","INDIVIDUAL_MATCHUP_INTERACTION"]),
                  "production_consumption_status"].isin(consumed).any()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run(args)->dict:
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    specs=feature_specs()
    inventory=pd.DataFrame(specs)
    matrix=_expand_matrix(specs)
    summary=_position_market_summary(matrix)

    logs=_read(Path(args.player_logs),"player logs")
    team=_read(Path(args.team_weekly),"team weekly")
    sched=_read(Path(args.schedule),"schedule")
    universe=_read(Path(args.universe),"Week5 universe")
    # Explicit no-outcome guard.
    cur=logs.loc[pd.to_numeric(logs["season"],errors="coerce").eq(SEASON)]
    if not cur.empty and pd.to_numeric(cur["week"],errors="coerce").max()>=WEEK:
        raise RuntimeError("player landscape audit saw Week5/future outcome rows")

    players,trace,dynamic=_build_dynamic_trace(
        player_logs=logs,team_weekly=team,schedule=sched,universe=universe,
        iterations=int(args.iterations),
    )

    inventory.to_csv(out/"player_landscape_feature_inventory.csv",index=False)
    matrix.to_csv(out/"player_landscape_transmission_matrix.csv",index=False)
    summary.to_csv(out/"player_landscape_position_market_summary.csv",index=False)
    trace.to_csv(out/"player_landscape_trace_examples.csv",index=False)
    players.to_csv(out/"player_landscape_week5_player_state.csv",index=False)

    consumed_status=[
        "CONSUMED_DIRECT_PLAYER","CONSUMED_PLAYER_VIA_SPECIALIST",
        "CONSUMED_TEAM_CONTEXT","CONSUMED_OPPONENT_CONTEXT","CONSUMED_INDIRECT",
    ]
    payload={
        "version":"PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1",
        "season":SEASON,"week":WEEK,
        "feature_inventory_rows":int(len(inventory)),
        "transmission_matrix_rows":int(len(matrix)),
        "position_market_rows":int(len(summary)),
        "trace_rows":int(len(trace)),
        "trace_players":int(trace[["team","player_clean_key"]].drop_duplicates().shape[0]),
        "consumed_matrix_rows":int(matrix["production_consumption_status"].isin(consumed_status).sum()),
        "gap_matrix_rows":int((~matrix["production_consumption_status"].isin(consumed_status)).sum()),
        "status_counts":matrix["production_consumption_status"].value_counts().to_dict(),
        "layer_gap_counts":{
            layer:int(
                (~matrix.loc[matrix["landscape_layer"].eq(layer),"production_consumption_status"].isin(consumed_status)).sum()
            ) for layer in LAYERS
        },
        "dynamic_trace":dynamic,
        "parent_authorities":{
            "all_player_replay_run":37683439543,
            "rb_receiving_impact_run":37703522415,
            "rb_receiving_week5_lock_run":37705464974,
            "matchup_phase_a_run":37504432928,
            "matchup_phase_bc_run":37514137803,
            "matchup_integration_run":37519246784,
        },
        "parameters_fit":0,
        "threshold_searches":0,
        "sportsbook_inputs_used":False,
        "paid_odds_api_used":False,
        "week5_outcomes_used":False,
        "production_changed":False,
    }
    (out/"player_landscape_audit_summary.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n"
    )
    print(json.dumps(payload,indent=2,sort_keys=True,default=str))
    return payload


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--player-logs",required=True)
    p.add_argument("--team-weekly",required=True)
    p.add_argument("--schedule",required=True)
    p.add_argument("--universe",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--iterations",type=int,default=1200)
    args=p.parse_args()
    run(args)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
