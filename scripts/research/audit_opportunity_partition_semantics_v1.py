#!/usr/bin/env python3
"""Read-only audit of team opportunity-partition semantics.

No target-game outcomes and no sportsbook information are used. This script
changes no production arrays and scores no historical outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
from scripts._opponent_map import canon_team
from scripts.modeling.qb_c2_production_adapter_v1 import annotate_primary_qbs
from scripts.research.audit_hierarchical_receiver_reconciliation_v1 import _build_current_pricing_metrics
from scripts.simulation_c2_qb_candidate import _target_shares, simulate_with_states
from scripts.utils.pbp import get_pbp

OUT=Path("data/research/opportunity_partition_semantics_v1")
SUMMARY=OUT/"summary.json"
TEAM_CSV=OUT/"team_semantics_diagnostic.csv"
SOURCE_JSON=OUT/"source_semantics_proof.json"
ITERATIONS=5000
SEED=42
PASS_CATCHER_POSITIONS={"WR","LWR","RWR","SWR","TE","RB","FB"}
TOL=1e-10

def _read_text(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size<=0:
        raise RuntimeError(f"source file missing for semantics proof: {p}")
    return p.read_text(encoding="utf-8")

def _source_semantics_proof():
    component=_read_text("scripts/backtest/component_predictions.py")
    sim=_read_text("scripts/simulation_v2.py")
    pricing=_read_text("scripts/run_pricing_v2.py")
    hist=_read_text("scripts/backtest/historical_inputs.py")
    required={
        "historical_dropback_definition":
            ("Scrambles are already excluded from" in hist and "remain dropbacks" in hist and "pass_attempts_per_dropback" in hist),
        "rules_pass_rate_is_dropback_rate":
            ("rules_pass_rate is derived from qb_dropback share in the historical PBP." in component
             and 'metrics["mc_dropback_rate"] = pd.to_numeric(metrics.get("rules_pass_rate")' in component),
        "historical_mc_converts_dropbacks_to_official_attempts":
            ('metrics["mc_pass_rate"] = metrics["mc_dropback_rate"] * metrics["mc_pass_attempts_per_dropback"]' in component),
        "canonical_receiver_targets_use_unconverted_pass_state":
            ("targets=_allocate_counts(rng,pass_att,target_shares)" in sim),
        "canonical_rush_pool_is_plays_minus_pass_state":
            ("rush_att=plays-pass_att" in sim),
        "production_qb_pricing_applies_attempt_conversion":
            ("qb_attempt_rate = attempt_conversion(row, qb_team_context)" in pricing
             and "base_outcomes = base_outcomes * qb_attempt_rate * qb_share" in pricing),
    }
    if not all(required.values()):
        raise RuntimeError(f"source semantics proof failed: {required}")
    return {"status":"SOURCE_SEMANTICS_CONFIRMED",**required}

def _identity_id(row):
    for col in ("player_id","gsis_id","player_gsis_id"):
        val=str(row.get(col,"") or "").strip()
        if val.startswith("00-"):
            return val
    ident=str(row.get("player_identity_key","") or "").strip()
    if ident.lower().startswith("gsis:"):
        return ident.split(":",1)[1].strip()
    return ""

def _prior_primary_qb_scramble_rates(primary,season,week):
    pbp=get_pbp(int(season),min_rows=1)
    if hasattr(pbp,"to_pandas"):
        pbp=pbp.to_pandas()
    pbp=pd.DataFrame(pbp).copy()
    pbp.columns=[str(c).strip().lower() for c in pbp.columns]
    if "season_type" in pbp.columns:
        reg=pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            pbp=reg
    pbp["week"]=pd.to_numeric(pbp["week"],errors="coerce")
    pbp=pbp.loc[pbp["week"].lt(int(week))].copy()
    if pbp.empty:
        raise RuntimeError("no strict-prior PBP rows")
    for col in ("qb_dropback","qb_scramble"):
        pbp[col]=pd.to_numeric(pbp.get(col),errors="coerce").fillna(0).astype(int)
    if "passer_player_id" not in pbp.columns or "rusher_player_id" not in pbp.columns:
        raise RuntimeError("PBP missing passer/rusher ids")
    rows=[]
    for _,r in primary.iterrows():
        pid=_identity_id(r)
        team=canon_team(r["team"])
        if not pid:
            raise RuntimeError(f"primary QB missing GSIS id team={team} player={r.get('player')}")
        drop=pbp.loc[pbp["passer_player_id"].astype(str).eq(pid)&pbp["qb_dropback"].eq(1)]
        scr=pbp.loc[pbp["rusher_player_id"].astype(str).eq(pid)&pbp["qb_scramble"].eq(1)]
        if len(drop)<=0:
            raise RuntimeError(f"primary QB zero strict-prior dropbacks team={team} player={r.get('player')}")
        rows.append({
            "team":team,
            "primary_qb":str(r.get("player")),
            "primary_qb_id":pid,
            "prior_dropbacks":int(len(drop)),
            "prior_scrambles":int(len(scr)),
            "prior_scramble_rate_per_dropback":float(len(scr)/len(drop)),
        })
    out=pd.DataFrame(rows)
    if out["team"].duplicated().any():
        raise RuntimeError("duplicate primary-QB scramble team")
    return out

def _team_context():
    p=Path("data/team_context_v3.csv")
    if not p.exists() or p.stat().st_size<=0:
        raise RuntimeError("team_context_v3 missing")
    x=pd.read_csv(p,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    x["team"]=x["team"].map(canon_team)
    x["pass_attempts_per_dropback"]=pd.to_numeric(x["pass_attempts_per_dropback"],errors="coerce")
    if x["pass_attempts_per_dropback"].isna().any() or not x["pass_attempts_per_dropback"].between(.50,1.0).all():
        raise RuntimeError("invalid pass_attempts_per_dropback")
    return x.drop_duplicates("team",keep="last")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    proof=_source_semantics_proof()

    synthetic=_build_current_pricing_metrics()
    base._identity_frame=v2._canonical_identity_frame
    universe,_,universe_audit=v3._build_with_promoted_entitlement_specialists(synthetic)
    seasons=sorted(pd.to_numeric(universe["season"],errors="coerce").dropna().astype(int).unique())
    weeks=sorted(pd.to_numeric(universe["week"],errors="coerce").dropna().astype(int).unique())
    if len(seasons)!=1 or len(weeks)!=1:
        raise RuntimeError(f"expected one season/week got {seasons}/{weeks}")
    season,week=int(seasons[0]),int(weeks[0])

    state=simulate_with_states(universe,iterations=ITERATIONS,seed=SEED)
    annotated,_=annotate_primary_qbs(universe,season=season,week=week)
    primary=annotated.loc[pd.to_numeric(annotated["qb_projection_eligible"],errors="coerce").eq(1)].copy()
    teams=sorted(universe["team"].map(canon_team).dropna().astype(str).unique())
    if set(primary["team"].map(canon_team))!=set(teams):
        raise RuntimeError("primary-QB coverage != current universe")

    scramble=_prior_primary_qb_scramble_rates(primary,season,week)
    context=_team_context().set_index("team",drop=False)
    rows=[]

    for team in teams:
        tdf=universe.loc[universe["team"].map(canon_team).eq(team)].copy()
        events=tdf["event_id"].dropna().astype(str).unique().tolist()
        if len(events)!=1:
            raise RuntimeError(f"expected one event team={team} got={events}")
        game=str(events[0])

        plays=np.asarray(state.team_states[(game,team,"plays")],float)
        dropbacks=np.asarray(state.team_states[(game,team,"pass_att")],float)
        rush_pool=np.asarray(state.team_states[(game,team,"rush_att")],float)
        if float(np.max(np.abs((plays-dropbacks)-rush_pool)))>TOL:
            raise RuntimeError(f"rush complement identity failed team={team}")

        conv=float(context.loc[team,"pass_attempts_per_dropback"])
        official=dropbacks*conv
        nonattempt=dropbacks-official

        sr=scramble.loc[scramble["team"].eq(team)]
        if len(sr)!=1:
            raise RuntimeError(f"scramble authority missing/duplicate team={team}")
        sr=sr.iloc[0]
        scramble_rate=float(sr["prior_scramble_rate_per_dropback"])
        implied_scrambles=dropbacks*scramble_rate
        non_scramble_nonattempt=nonattempt-implied_scrambles

        shares=_target_shares(tdf)
        positions=tdf["position"].fillna("").astype(str).str.upper().str.strip().to_numpy()
        mask=np.isin(positions,list(PASS_CATCHER_POSITIONS))
        named_share=float(np.where(mask,shares,0.0).sum())
        if named_share<0 or named_share>0.950000000001:
            raise RuntimeError(f"invalid named target share team={team} value={named_share}")

        prow=primary.loc[primary["team"].map(canon_team).eq(team)].iloc[0]
        qpk=str(prow["player_clean_key"])
        qb_rush=np.asarray(state.values[(game,qpk,"rush_att")],float)
        qb_pass_y=np.asarray(state.values[(game,qpk,"pass_yards")],float)

        rec_means=[]
        for _,pr in tdf.loc[mask].iterrows():
            arr=state.values.get((game,str(pr["player_clean_key"]),"rec_yards"))
            if arr is not None:
                rec_means.append(float(np.mean(np.asarray(arr,float))))
        named_rec_yards_sum=float(np.sum(rec_means))
        raw_qb_pass_yards_mean=float(qb_pass_y.mean())

        db=float(dropbacks.mean())
        off=float(official.mean())
        excess=float(nonattempt.mean())
        current_rush=float(rush_pool.mean())
        implied_scr=float(implied_scrambles.mean())

        rows.append({
            "season":season,"week":week,"event_id":game,"team":team,
            "primary_qb":str(prow["player"]),
            "mean_plays":float(plays.mean()),
            "mean_dropbacks_current_pass_state":db,
            "pass_attempts_per_dropback":conv,
            "implied_official_pass_attempts":off,
            "implied_nonattempt_dropbacks":excess,
            "receiver_target_pool_current":db,
            "receiver_target_pool_excess":excess,
            "receiver_target_pool_inflation_pct_vs_official_attempts":float(excess/off) if off>0 else np.nan,
            "named_target_share":named_share,
            "expected_named_target_mass_from_nonattempt_dropbacks":float(excess*named_share),
            "current_non_dropback_rush_pool":current_rush,
            "prior_primary_qb_dropbacks":int(sr["prior_dropbacks"]),
            "prior_primary_qb_scrambles":int(sr["prior_scrambles"]),
            "prior_primary_qb_scramble_rate":scramble_rate,
            "implied_scramble_rush_attempts":implied_scr,
            "raw_nonattempt_minus_implied_scramble":float(non_scramble_nonattempt.mean()),
            "implied_rush_pool_with_scrambles_restored":float((rush_pool+implied_scrambles).mean()),
            "restored_scramble_pct_of_current_rush_pool":float(implied_scr/current_rush) if current_rush>0 else np.nan,
            "current_primary_qb_sim_rush_att":float(qb_rush.mean()),
            "current_primary_qb_sim_rush_minus_implied_scramble":float(qb_rush.mean()-implied_scr),
            "canonical_named_receiver_yards_sum":named_rec_yards_sum,
            "canonical_raw_qb_pass_yards_mean":raw_qb_pass_yards_mean,
            "named_receiver_minus_raw_qb_yards":float(named_rec_yards_sum-raw_qb_pass_yards_mean),
        })

    team_df=pd.DataFrame(rows).sort_values("team").reset_index(drop=True)

    def corr(a,b):
        z=team_df[[a,b]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(z)<3 or z[a].nunique()<2 or z[b].nunique()<2:
            return None
        v=z[a].corr(z[b])
        return float(v) if pd.notna(v) else None

    payload={
        "study":"OPPORTUNITY_PARTITION_SEMANTICS_V1",
        "status":"READ_ONLY_AUDIT_COMPLETE",
        "season":season,"week":week,"iterations":ITERATIONS,"teams":int(len(team_df)),
        "sportsbook_inputs_used":False,"target_game_outcomes_used":False,"production_changed":False,
        "source_semantics_proof":proof,
        "pass_attempt_conversion_min":float(team_df["pass_attempts_per_dropback"].min()),
        "pass_attempt_conversion_median":float(team_df["pass_attempts_per_dropback"].median()),
        "pass_attempt_conversion_max":float(team_df["pass_attempts_per_dropback"].max()),
        "mean_receiver_target_pool_excess":float(team_df["receiver_target_pool_excess"].mean()),
        "median_receiver_target_pool_excess":float(team_df["receiver_target_pool_excess"].median()),
        "median_receiver_target_pool_inflation_pct":float(team_df["receiver_target_pool_inflation_pct_vs_official_attempts"].median()),
        "min_receiver_target_pool_inflation_pct":float(team_df["receiver_target_pool_inflation_pct_vs_official_attempts"].min()),
        "max_receiver_target_pool_inflation_pct":float(team_df["receiver_target_pool_inflation_pct_vs_official_attempts"].max()),
        "mean_expected_named_target_mass_from_nonattempt_dropbacks":float(team_df["expected_named_target_mass_from_nonattempt_dropbacks"].mean()),
        "median_expected_named_target_mass_from_nonattempt_dropbacks":float(team_df["expected_named_target_mass_from_nonattempt_dropbacks"].median()),
        "mean_implied_scramble_rush_attempts":float(team_df["implied_scramble_rush_attempts"].mean()),
        "median_implied_scramble_rush_attempts":float(team_df["implied_scramble_rush_attempts"].median()),
        "median_restored_scramble_pct_of_current_rush_pool":float(team_df["restored_scramble_pct_of_current_rush_pool"].median()),
        "mean_raw_nonattempt_minus_implied_scramble":float(team_df["raw_nonattempt_minus_implied_scramble"].mean()),
        "teams_negative_nonattempt_residual":int(team_df["raw_nonattempt_minus_implied_scramble"].lt(-TOL).sum()),
        "corr_target_inflation_vs_named_receiver_over_raw_qb_yards":corr("receiver_target_pool_inflation_pct_vs_official_attempts","named_receiver_minus_raw_qb_yards"),
        "corr_target_excess_vs_named_receiver_over_raw_qb_yards":corr("receiver_target_pool_excess","named_receiver_minus_raw_qb_yards"),
        "median_current_qb_sim_rush_att":float(team_df["current_primary_qb_sim_rush_att"].median()),
        "median_current_qb_sim_rush_minus_implied_scramble":float(team_df["current_primary_qb_sim_rush_minus_implied_scramble"].median()),
        "universe_audit":{
            "football_player_rows":int(universe_audit["football_player_rows"]),
            "football_teams":int(universe_audit["football_teams"]),
            "canonical_games":int(universe_audit["canonical_games"]),
            "sportsbook_rows_used_to_define_player_universe":int(universe_audit["sportsbook_rows_used_to_define_player_universe"]),
        },
    }
    team_df.to_csv(TEAM_CSV,index=False)
    SOURCE_JSON.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    SUMMARY.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
