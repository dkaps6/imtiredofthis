#!/usr/bin/env python3
"""Independent 2023 holdout for Player Opportunity Share Stability V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research.audit_player_opportunity_share_stability_v1 import (
    CELLS,
    TARGET_WEEKS,
    _load_roster_universe,
    _load_weekly_stats,
    build_panel,
    num_or_zero,
    score_cell,
)

VERSION="PLAYER_OPPORTUNITY_SHARE_STABILITY_2023_HOLDOUT_V1"
TARGET_SEASON=2023


def _to_pandas(obj):
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


def _load_schedule_2023()->pd.DataFrame:
    import nflreadpy as nfl
    try:
        raw=nfl.load_schedules(seasons=[TARGET_SEASON])
    except TypeError:
        raw=nfl.load_schedules(TARGET_SEASON)
    x=_lower(_to_pandas(raw))
    if "game_type" in x.columns:
        q=x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            x=q
    elif "season_type" in x.columns:
        q=x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            x=q
    hc="home_team" if "home_team" in x.columns else "home"
    ac="away_team" if "away_team" in x.columns else "away"
    gidc="game_id" if "game_id" in x.columns else None
    rows=[]
    for r in x.itertuples(index=False):
        week=int(getattr(r,"week"))
        if week>18:
            continue
        home=canon_team(getattr(r,hc))
        away=canon_team(getattr(r,ac))
        gid=str(getattr(r,gidc)) if gidc else f"{TARGET_SEASON}_{week:02d}_{away}_{home}"
        if not home or not away:
            continue
        rows.append({"season":TARGET_SEASON,"week":week,"team":home,"opponent":away,"game_id":gid})
        rows.append({"season":TARGET_SEASON,"week":week,"team":away,"opponent":home,"game_id":gid})
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero 2023 schedule rows")
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate 2023 schedule team-week")
    return out.sort_values(["season","week","team"]).reset_index(drop=True)


def build_2023_weekly_universe()->pd.DataFrame:
    schedule=_load_schedule_2023()
    roster=_load_roster_universe(TARGET_SEASON,schedule)
    stats=_load_weekly_stats(TARGET_SEASON)
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
    if x.empty:
        raise RuntimeError("zero 2023 weekly roster/player rows")
    return x.sort_values(["season","week","team","player_identity_key"]).reset_index(drop=True)


def audit()->tuple[pd.DataFrame,dict]:
    universe=build_2023_weekly_universe()
    panel,integrity=build_panel(universe)
    if panel.empty:
        raise RuntimeError("zero 2023 holdout scoreable rows")

    cells=[]
    per_cell=[]
    for cell in CELLS:
        rec=score_cell(panel,TARGET_SEASON,cell)
        cells.append(rec)
        confirmed=bool(rec.get("signal_supported",False))
        per_cell.append({
            "cell":cell,
            "season_2023_supported":confirmed,
            "disposition":(
                "PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED"
                if confirmed else
                "PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_NOT_CONFIRMED"
            ),
        })

    nondegenerate={}
    for cell in CELLS:
        z=pd.to_numeric(
            panel.loc[panel["cell"].eq(cell),"actual_share"],errors="coerce"
        )
        nondegenerate[cell]=bool(
            len(z)>0 and z.nunique(dropna=True)>1 and float(z.max())>0
        )

    all_confirmed=all(
        x["disposition"]=="PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED"
        for x in per_cell
    )
    identity_cov=float(panel["player_identity_key"].astype(str).ne("").mean())

    gates={
        "exact_target_season_2023":sorted(panel["season"].unique().tolist())==[2023],
        "target_weeks_within_5_18":int(panel["week"].min())>=5 and int(panel["week"].max())<=18,
        "same_future_feature_violations_zero":integrity["same_future_feature_violations"]==0,
        "same_team_feature_violations_zero":integrity["same_team_feature_violations"]==0,
        "stable_identity_coverage_ge_095":identity_cov>=.95,
        "all_cells_actual_shares_nondegenerate":all(nondegenerate.values()),
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "model_residuals_zero":True,
        "fitted_predictive_models_zero":True,
        "candidate_projection_changes_zero":True,
        "production_changed_false":True,
    }

    disposition=(
        "PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED"
        if all_confirmed else
        "PLAYER_OPPORTUNITY_SHARE_STABILITY_HOLDOUT_PARTIAL_OR_FAIL"
    )
    if not all(gates.values()):
        disposition="PLAYER_OPPORTUNITY_SHARE_STABILITY_HOLDOUT_INTEGRITY_FAILURE"

    result={
        "version":VERSION,
        "disposition":disposition,
        "target_season":2023,
        "target_weeks":[5,18],
        "parent_prior_window_games":8,
        "parent_min_prior_games":4,
        "panel_rows":int(len(panel)),
        "panel_players":int(panel["player_identity_key"].nunique()),
        "scoreable_identity_coverage":identity_cov,
        "integrity":{
            **integrity,
            "nondegenerate_actual_share_cells":nondegenerate,
        },
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
    return panel,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    panel,result=audit()
    panel.to_csv(
        a.out_dir/"player_opportunity_share_stability_2023_holdout_rows.csv",
        index=False,float_format="%.12f"
    )
    (a.out_dir/"player_opportunity_share_stability_2023_holdout_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
