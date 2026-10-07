#!/usr/bin/env python3
"""Create immutable Week-5 RB Player State Allocation Shadow V1 lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION="RB_PLAYER_STATE_ALLOCATION_SHADOW_V1"
SEASON=2026
WEEK=5
PARENT_RUN=37560311001
PARENT_ARTIFACT=11456556226
PARENT_DIGEST="sha256:a39b958e492a781e310de0f14d34153e21ca589a76cd226479b3b39e10f9328e"

FORBIDDEN=(
    "actual","result","target_game","week5_rush","week5_target",
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)

def num(x):
    return pd.to_numeric(x,errors="coerce")

def canonical_csv_bytes(df:pd.DataFrame)->bytes:
    cols=[
        "season","week","team","opponent","state_key","player","gsis_id","pfr_id",
        "room_size","raw_last3_room_opportunity_share","raw_last3_room_snap_fraction",
        "control_recent_carry_share","shadow_player_state_share",
    ]
    x=df[cols].copy().sort_values(["team","state_key"],kind="mergesort").reset_index(drop=True)
    return x.to_csv(index=False,float_format="%.12f",lineterminator="\n").encode("utf-8")

def build(rows:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    x=rows.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]

    bad=[c for c in x.columns if any(t in c for t in FORBIDDEN)]
    # Parent source intentionally contains no target outcome columns; broad words
    # like "current_total_rush_yards" are strictly-prior state, so only fail on
    # explicit target/outcome or sportsbook semantics above.
    bad=[c for c in bad if not c.startswith(("current_","prior_","last"))]
    if bad:
        raise RuntimeError(f"forbidden target/sportsbook fields in parent rows: {bad}")

    required={
        "team","opponent","state_key","player","gsis_id","pfr_id","position_group",
        "room_size","last3_room_opportunity_share","last3_room_snap_fraction",
        "chronology_valid",
    }
    miss=required-set(x.columns)
    if miss:
        raise RuntimeError(f"parent rows missing required columns: {sorted(miss)}")

    rb=x.loc[x["position_group"].astype(str).str.upper().eq("RB")].copy()
    rb["carry_raw"]=num(rb["last3_room_opportunity_share"])
    rb["snap_raw"]=num(rb["last3_room_snap_fraction"])
    rb["chronology_valid_b"]=rb["chronology_valid"].astype(str).str.lower().isin({"true","1","yes"})
    rb=rb.loc[rb["carry_raw"].notna()&rb["snap_raw"].notna()&rb["chronology_valid_b"]].copy()

    locked=[]
    teams=[]
    for team,g in rb.groupby("team",sort=True):
        g=g.copy()
        if len(g)<2:
            continue
        carry_sum=float(g["carry_raw"].sum())
        snap_sum=float(g["snap_raw"].sum())
        if not np.isfinite(carry_sum) or not np.isfinite(snap_sum) or carry_sum<=0 or snap_sum<=0:
            continue
        g["control_recent_carry_share"]=g["carry_raw"]/carry_sum
        g["snap_state_norm"]=g["snap_raw"]/snap_sum
        g["shadow_player_state_share"]=0.5*g["control_recent_carry_share"]+0.5*g["snap_state_norm"]
        # defensive renormalization; should already sum to 1 within FP noise.
        ss=float(g["shadow_player_state_share"].sum())
        g["shadow_player_state_share"]=g["shadow_player_state_share"]/ss
        g["season"]=SEASON
        g["week"]=WEEK
        g["raw_last3_room_opportunity_share"]=g["carry_raw"]
        g["raw_last3_room_snap_fraction"]=g["snap_raw"]

        cg=float(g["control_recent_carry_share"].sum())
        sg=float(g["shadow_player_state_share"].sum())
        if abs(cg-1.0)>1e-12 or abs(sg-1.0)>1e-12:
            raise RuntimeError(f"share conservation failure team={team} control={cg} shadow={sg}")
        locked.append(g)
        teams.append({
            "team":team,
            "locked_players":int(len(g)),
            "control_sum":cg,
            "shadow_sum":sg,
            "max_abs_shadow_minus_control":float((g["shadow_player_state_share"]-g["control_recent_carry_share"]).abs().max()),
            "mean_abs_shadow_minus_control":float((g["shadow_player_state_share"]-g["control_recent_carry_share"]).abs().mean()),
        })

    if not locked:
        raise RuntimeError("zero lockable RB teams")

    out=pd.concat(locked,ignore_index=True)
    team_summary=pd.DataFrame(teams)
    payload=canonical_csv_bytes(out)
    digest="sha256:"+hashlib.sha256(payload).hexdigest()

    result={
        "version":VERSION,
        "season":SEASON,
        "week":WEEK,
        "status":"WEEK5_PREGAME_LOCK_FROZEN",
        "parent_run":PARENT_RUN,
        "parent_artifact":PARENT_ARTIFACT,
        "parent_digest":PARENT_DIGEST,
        "locked_teams":int(out["team"].nunique()),
        "locked_players":int(len(out)),
        "locked_player_identities":int(out["state_key"].nunique()),
        "control":"renormalized strictly-prior last3 RB-room carry share",
        "shadow":"0.50 control carry state + 0.50 renormalized strictly-prior last3 RB-room snap fraction",
        "row_csv_digest":digest,
        "max_team_control_sum_gap":float((team_summary["control_sum"]-1.0).abs().max()),
        "max_team_shadow_sum_gap":float((team_summary["shadow_sum"]-1.0).abs().max()),
        "teams_with_nonzero_shadow_change":int(team_summary["max_abs_shadow_minus_control"].gt(1e-12).sum()),
        "median_team_max_abs_shadow_change":float(team_summary["max_abs_shadow_minus_control"].median()),
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "week5_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "parameters_fit":0,
        "production_changed":False,
    }
    return out,team_summary,result

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    if not a.parent_rows.exists():
        raise RuntimeError(f"missing parent rows: {a.parent_rows}")
    rows=pd.read_csv(a.parent_rows,low_memory=False)
    out,teams,result=build(rows)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    payload=canonical_csv_bytes(out)
    (a.out_dir/"rb_player_state_allocation_week5_lock.csv").write_bytes(payload)
    teams.to_csv(a.out_dir/"rb_player_state_allocation_week5_team_summary.csv",index=False,float_format="%.12f")
    (a.out_dir/"rb_player_state_allocation_week5_lock.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    print("\nTEAM SUMMARY")
    print(teams.to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
