#!/usr/bin/env python3
"""Certify a future-week individual opportunity lock pack.

Mechanical manifest builder only. It validates component metadata, chronology /
conservation boundaries, and file digests. It does not score football outcomes,
fit models, read sportsbook data, or change production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION="INDIVIDUAL_OPPORTUNITY_FUTURE_WEEK_LOCK_PACK_V1"


def _read_json(path:Path,label:str)->dict[str,Any]:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    obj=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj,dict):
        raise RuntimeError(f"{label} must be JSON object")
    return obj


def _sha256(path:Path)->str:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing artifact file: {path}")
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return "sha256:"+h.hexdigest()


def _zero(meta:dict[str,Any], keys:tuple[str,...], label:str)->None:
    for key in keys:
        if key not in meta:
            continue
        try:
            value=int(meta[key])
        except Exception as exc:
            raise RuntimeError(f"{label} invalid {key}: {meta[key]!r}") from exc
        if value!=0:
            raise RuntimeError(f"{label} contamination {key}={value}")


def _false(meta:dict[str,Any], keys:tuple[str,...], label:str)->None:
    for key in keys:
        if key in meta and bool(meta[key]):
            raise RuntimeError(f"{label} contamination {key}={meta[key]!r}")


def certify(
    *,
    season:int,
    week:int,
    player_state:dict[str,Any],
    rb_carry:dict[str,Any],
    rb_receiving:dict[str,Any],
    wrte_trajectory:dict[str,Any],
)->dict[str,Any]:
    if int(player_state.get("season",-1))!=season or int(player_state.get("target_week",-1))!=week:
        raise RuntimeError("player-state target mismatch")
    if player_state.get("disposition")!="PLAYER_STATE_FUTURE_WEEK_READY":
        raise RuntimeError(f"player-state not READY: {player_state.get('disposition')}")
    if not bool(player_state.get("target_week_roster_source_ready",False)):
        raise RuntimeError("player-state lacks exact target-week roster authority")
    completion=player_state.get("prior_week_source_completion") or {}
    if not bool(completion.get("complete",False)):
        raise RuntimeError("player-state prior-week source incomplete")
    if int(player_state.get("chronology_violations",-1))!=0:
        raise RuntimeError("player-state chronology violations")

    for label,meta in (
        ("rb_carry",rb_carry),
        ("rb_receiving",rb_receiving),
        ("wrte_trajectory",wrte_trajectory),
    ):
        if int(meta.get("season",-1))!=season or int(meta.get("week",-1))!=week:
            raise RuntimeError(f"{label} target mismatch")
        if meta.get("status")!=f"WEEK{week}_PREGAME_LOCK_FROZEN":
            raise RuntimeError(f"{label} not frozen: {meta.get('status')}")
        if bool(meta.get("scientific_formula_changed_from_week5",False)):
            raise RuntimeError(f"{label} formula changed from Week 5")
        _zero(meta,("target_week_outcomes_read","week5_outcomes_read","sportsbook_inputs_used","parameters_fit"),label)
        _false(meta,("production_changed",),label)

    _zero(
        player_state,
        ("target_week_outcomes_read","sportsbook_inputs_used","candidate_models_fit"),
        "player_state",
    )
    _false(player_state,("production_changed",),"player_state")

    if float(rb_carry.get("max_team_control_sum_gap",1.0))>1e-10:
        raise RuntimeError("RB carry control conservation failed")
    if float(rb_carry.get("max_team_shadow_sum_gap",1.0))>1e-10:
        raise RuntimeError("RB carry shadow conservation failed")

    if float(rb_receiving.get("max_candidate_conservation_gap",1.0))>1e-10:
        raise RuntimeError("RB receiving conservation failed")

    if int(wrte_trajectory.get("same_or_future_feature_violations",-1))!=0:
        raise RuntimeError("WR/TE trajectory chronology violation")
    if float(wrte_trajectory.get("max_room_pool_gap",1.0))>1e-10:
        raise RuntimeError("WR/TE room conservation failed")
    if float(wrte_trajectory.get("max_team_modeled_target_mass_gap",1.0))>1e-10:
        raise RuntimeError("WR/TE team target mass conservation failed")

    components={
        "player_state":{
            "version":player_state.get("version",""),
            "disposition":player_state.get("disposition",""),
            "skill_players":int(player_state.get("skill_players",0)),
            "scheduled_teams":int(player_state.get("scheduled_teams",0)),
            "roster_source_week":player_state.get("roster_source_week"),
            "prior_week_source_completion":completion,
        },
        "rb_carry_snap":{
            "version":rb_carry.get("version",""),
            "status":rb_carry.get("status",""),
            "locked_teams":int(rb_carry.get("locked_teams",0)),
            "locked_players":int(rb_carry.get("locked_players",0)),
            "row_csv_digest":rb_carry.get("row_csv_digest",""),
        },
        "rb_receiving":{
            "version":rb_receiving.get("version",""),
            "status":rb_receiving.get("status",""),
            "teams":int(rb_receiving.get("teams",0)),
            "players":int(rb_receiving.get("players",0)),
            "row_csv_sha256":rb_receiving.get("row_csv_sha256",""),
        },
        "wrte_trajectory":{
            "version":wrte_trajectory.get("version",""),
            "status":wrte_trajectory.get("status",""),
            "locked_players":int(wrte_trajectory.get("locked_players",0)),
            "locked_wr":int(wrte_trajectory.get("locked_wr",0)),
            "locked_te":int(wrte_trajectory.get("locked_te",0)),
            "trajectory_available_players":int(wrte_trajectory.get("trajectory_available_players",0)),
            "row_csv_digest":wrte_trajectory.get("row_csv_digest",""),
        },
    }

    if components["rb_carry_snap"]["locked_players"]<=0:
        raise RuntimeError("empty RB carry/snap lock")
    if components["rb_receiving"]["players"]<=0:
        raise RuntimeError("empty RB receiving lock")
    if components["wrte_trajectory"]["locked_players"]<=0:
        raise RuntimeError("empty WR/TE trajectory lock")

    return {
        "version":VERSION,
        "status":f"WEEK{week}_INDIVIDUAL_OPPORTUNITY_LOCK_PACK_FROZEN",
        "season":season,
        "week":week,
        "components":components,
        "target_week_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "parameters_fit":0,
        "production_changed":False,
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--player-state-result",type=Path,required=True)
    ap.add_argument("--rb-carry-result",type=Path,required=True)
    ap.add_argument("--rb-carry-rows",type=Path,required=True)
    ap.add_argument("--rb-receiving-result",type=Path,required=True)
    ap.add_argument("--rb-receiving-rows",type=Path,required=True)
    ap.add_argument("--wrte-result",type=Path,required=True)
    ap.add_argument("--wrte-rows",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    player_state=_read_json(a.player_state_result,"player-state result")
    rb_carry=_read_json(a.rb_carry_result,"RB carry result")
    rb_receiving=_read_json(a.rb_receiving_result,"RB receiving result")
    wrte=_read_json(a.wrte_result,"WR/TE result")

    result=certify(
        season=a.season,week=a.week,
        player_state=player_state,
        rb_carry=rb_carry,
        rb_receiving=rb_receiving,
        wrte_trajectory=wrte,
    )

    result["artifact_sha256"]={
        "rb_carry_rows":_sha256(a.rb_carry_rows),
        "rb_receiving_rows":_sha256(a.rb_receiving_rows),
        "wrte_rows":_sha256(a.wrte_rows),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
