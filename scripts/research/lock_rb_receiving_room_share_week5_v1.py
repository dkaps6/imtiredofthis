#!/usr/bin/env python3
"""Freeze Week-5 pregame RB receiving-room share shadow V1.

Uses the already-frozen Week-5 RB player-state lock as the identity authority.
No Week-5 outcomes or sportsbook inputs are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import (
    assert_no_future_rows,
    build_historical_context_bundle,
)
from scripts.modeling.rb_receiving_identity_runtime_v1 import identity_atlas
from scripts.modeling.rb_receiving_room_share_shadow_v1 import (
    apply_rb_receiving_room_share_shadow,
)

SEASON=2026
WEEK=5
PRIOR_SEASON=2025
TOL=1e-10


def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _key(v)->str:
    return re.sub(r"[^a-z0-9]","",str(v or "").lower())


def _pos(v)->str:
    p=str(v or "").upper().strip()
    if p in {"HB","TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    return p


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def run(
    *,
    player_logs_path:Path,
    team_weekly_path:Path,
    schedule_path:Path,
    universe_path:Path,
    parent_rb_lock_path:Path,
    out_dir:Path,
    iterations:int,
)->dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    player_logs=_read(player_logs_path,"player logs")
    team_weekly=_read(team_weekly_path,"team weekly history")
    schedule=_read(schedule_path,"schedule")
    universe=_read(universe_path,"Week-5 pregame universe")
    parent=_read(parent_rb_lock_path,"frozen Week-5 RB player-state lock")

    if len(parent)!=98:
        raise RuntimeError(f"parent Week-5 RB lock row drift: {len(parent)} != 98")
    if int(parent["team"].nunique())!=30:
        raise RuntimeError(f"parent Week-5 RB lock team drift: {parent['team'].nunique()} != 30")
    if not pd.to_numeric(parent["week"],errors="coerce").eq(WEEK).all():
        raise RuntimeError("parent RB lock contains non-Week5 row")

    # Outcome boundary: completed player history must stop before Week 5.
    current=pd.to_numeric(player_logs["season"],errors="coerce").eq(SEASON)
    if current.any():
        max_w=int(pd.to_numeric(player_logs.loc[current,"week"],errors="coerce").max())
        if max_w>=WEEK:
            raise RuntimeError(f"Week-5/future outcome row present in player logs max_week={max_w}")
    assert_no_future_rows(player_logs,SEASON,WEEK,"week5_receiving_room_player_history")
    assert_no_future_rows(team_weekly,SEASON,WEEK,"week5_receiving_room_team_history")

    bundle=build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=SEASON,
        week=WEEK,
        prior_season=PRIOR_SEASON,
    )
    assert_no_future_rows(bundle.player_history,SEASON,WEEK,"week5_bundle_player_history")
    assert_no_future_rows(bundle.team_history,SEASON,WEEK,"week5_bundle_team_history")

    metrics=build_mc_predictions(bundle,iterations=int(iterations),seed=42+WEEK)
    pcols=["event_id","team","opponent","player","player_clean_key","position"]
    optional=[c for c in (
        "rules_tgt_share","bayes_tgt_share","target_share","tgt_share"
    ) if c in metrics.columns]
    players=metrics[pcols+optional].sort_values(
        ["event_id","team","player_clean_key"]
    ).drop_duplicates(["event_id","team","player_clean_key"],keep="last").copy()
    players["team"]=players["team"].map(canon_team)
    players["position_family"]=players["position"].map(_pos)
    players=players.loc[players["position_family"].isin({"RB","FB"})].copy()
    players["name_key"]=players["player"].map(_key)

    parent["team"]=parent["team"].map(canon_team)
    parent["name_key"]=parent["player"].map(_key)
    if parent.duplicated(["team","name_key"]).any():
        raise RuntimeError("parent RB lock duplicate team/name identity")
    if players.duplicated(["team","name_key"]).any():
        d=players.loc[players.duplicated(["team","name_key"],keep=False),["team","player"]]
        raise RuntimeError(f"model RB universe duplicate team/name identity: {d.to_dict('records')[:20]}")

    matched=parent.merge(
        players,
        on=["team","name_key"],
        how="left",
        suffixes=("_parent",""),
        validate="one_to_one",
    )
    if matched["player_clean_key"].isna().any():
        bad=matched.loc[matched["player_clean_key"].isna(),["team","player_parent","gsis_id"]]
        raise RuntimeError(f"parent Week-5 RB identities missing from football universe: {bad.to_dict('records')}")

    share_col=next((c for c in ("rules_tgt_share","bayes_tgt_share","target_share","tgt_share") if c in matched.columns),None)
    if share_col is None:
        raise RuntimeError("Week-5 RB football state has no target-share field")
    matched["canonical_target_share_raw"]=pd.to_numeric(matched[share_col],errors="coerce")
    if matched["canonical_target_share_raw"].isna().any() or matched["canonical_target_share_raw"].lt(0).any():
        raise RuntimeError("invalid canonical RB target-share state")

    # The shadow helper operates on target-entitlement mass. For the prospective
    # lock we use the generic canonical RB target-share mass; downstream
    # M38/TE-R5P/WR-R15 preserve RB ratios and RB mass.
    shadow_input=pd.DataFrame({
        "event_id":matched["event_id"],
        "team":matched["team"],
        "player":matched["player_parent"],
        "player_clean_key":matched["player_clean_key"],
        "position":matched["position"],
        "entitlement_tgt_share":matched["canonical_target_share_raw"],
    })

    states,prev=identity_atlas(2013,SEASON)
    candidate,audit,shadow_summary=apply_rb_receiving_room_share_shadow(
        shadow_input,season=SEASON,week=WEEK,states=states,prev=prev
    )

    join_cols=["event_id","team","player_clean_key"]
    extra=candidate[join_cols+[
        "prior_rb_room_share","prior_games",
        "rb_room_current_share","rb_room_candidate_share",
        "rb_receiving_room_shadow_applied",
        "rb_receiving_room_history_available",
        "rb_receiving_room_fallback_reason",
    ]].copy()
    lock=matched.merge(extra,on=join_cols,how="left",validate="one_to_one")

    out=pd.DataFrame({
        "season":SEASON,
        "week":WEEK,
        "team":lock["team"],
        "opponent":lock["opponent"],
        "state_key":lock["state_key"],
        "player":lock["player_parent"],
        "gsis_id":lock["gsis_id"],
        "pfr_id":lock["pfr_id"],
        "room_size":lock["room_size"],
        "canonical_target_share_source":share_col,
        "canonical_target_share_raw":lock["canonical_target_share_raw"],
        "current_rb_room_share":lock["rb_room_current_share"],
        "prior_rb_room_share":lock["prior_rb_room_share"],
        "prior_games":lock["prior_games"],
        "history_available":lock["rb_receiving_room_history_available"],
        "candidate_rb_receiving_room_share":lock["rb_room_candidate_share"],
        "shadow_applied":lock["rb_receiving_room_shadow_applied"],
        "fallback_reason":lock["rb_receiving_room_fallback_reason"],
        "reference_carry_state_share":lock["shadow_player_state_share"],
    })
    out=out.sort_values(["team","player"]).reset_index(drop=True)

    if len(out)!=98 or out["team"].nunique()!=30:
        raise RuntimeError("Week-5 receiving lock population drift")
    if out["candidate_rb_receiving_room_share"].isna().any():
        raise RuntimeError("candidate Week-5 RB receiving share contains missing values")

    team_sums=out.groupby("team").agg(
        current_sum=("current_rb_room_share","sum"),
        candidate_sum=("candidate_rb_receiving_room_share","sum"),
        players=("player","size"),
        history_players=("history_available","sum"),
    ).reset_index()
    if float((team_sums["current_sum"]-1.0).abs().max())>TOL:
        raise RuntimeError("current RB room shares do not conserve to 1")
    if float((team_sums["candidate_sum"]-1.0).abs().max())>TOL:
        raise RuntimeError("candidate RB room shares do not conserve to 1")

    out["abs_room_share_change"]=(
        out["candidate_rb_receiving_room_share"]-out["current_rb_room_share"]
    ).abs()
    changed=out.groupby("team")["abs_room_share_change"].max()
    team_sums=team_sums.merge(
        changed.rename("max_abs_player_share_change"),
        on="team",how="left",validate="one_to_one"
    )

    csv_path=out_dir/"rb_receiving_room_share_week5_lock.csv"
    team_path=out_dir/"rb_receiving_room_share_week5_team_summary.csv"
    out.to_csv(csv_path,index=False)
    team_sums.to_csv(team_path,index=False)

    payload={
        "version":"RB_RECEIVING_ROOM_SHARE_SHADOW_V1_WEEK5_LOCK",
        "status":"WEEK5_PREGAME_LOCK_FROZEN",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "season":SEASON,
        "week":WEEK,
        "players":int(len(out)),
        "teams":int(out["team"].nunique()),
        "rooms_applied":int(team_sums["max_abs_player_share_change"].gt(TOL).sum()),
        "history_available_players":int(out["history_available"].sum()),
        "history_available_rate":float(out["history_available"].mean()),
        "median_team_max_abs_player_share_change":float(team_sums["max_abs_player_share_change"].median()),
        "max_team_current_conservation_gap":float((team_sums["current_sum"]-1).abs().max()),
        "max_team_candidate_conservation_gap":float((team_sums["candidate_sum"]-1).abs().max()),
        "parent_rb_player_state_run":37560824479,
        "parent_rb_player_state_artifact":11456916566,
        "parent_rb_player_state_digest":"sha256:edf51bbd93920ef0af580a0396422af3288cf511785be59deeea95c117062af4",
        "row_csv_sha256":"sha256:"+_sha256(csv_path),
        "team_csv_sha256":"sha256:"+_sha256(team_path),
        "parameters_fit":0,
        "sportsbook_inputs_used":0,
        "week5_outcomes_read":0,
        "production_changed":False,
        "retrospective_w1_w4_impact_run":37703522415,
        "retrospective_result_used_to_change_rule":False,
        "shadow_summary":shadow_summary,
    }
    (out_dir/"rb_receiving_room_share_week5_lock.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n"
    )
    print(json.dumps(payload,indent=2,sort_keys=True,default=str))
    return payload


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--player-logs",type=Path,required=True)
    p.add_argument("--team-weekly",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--universe",type=Path,required=True)
    p.add_argument("--parent-rb-lock",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=1000)
    a=p.parse_args()
    run(
        player_logs_path=a.player_logs,
        team_weekly_path=a.team_weekly,
        schedule_path=a.schedule,
        universe_path=a.universe,
        parent_rb_lock_path=a.parent_rb_lock,
        out_dir=a.out_dir,
        iterations=a.iterations,
    )
    return 0

if __name__=="__main__":
    raise SystemExit(main())
