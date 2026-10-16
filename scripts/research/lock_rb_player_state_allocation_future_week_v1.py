#!/usr/bin/env python3
"""Build a future-week RB Player State Allocation Shadow V1 lock.

Mechanical continuation only. The scientific formula is unchanged from the
frozen Week-5 authority: 50% current carry-state share + 50% current snap-state
share, each normalized within the current certified RB room.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION="RB_PLAYER_STATE_ALLOCATION_SHADOW_FUTURE_WEEK_V1"
FORBIDDEN=(
    "actual","target_game",
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)

def num(x): return pd.to_numeric(x,errors="coerce")

def canonical_csv_bytes(df:pd.DataFrame)->bytes:
    cols=[
        "season","week","team","opponent","state_key","player","gsis_id","pfr_id",
        "room_size","raw_last3_room_opportunity_share","raw_last3_room_snap_fraction",
        "control_recent_carry_share","shadow_player_state_share",
    ]
    x=df[cols].copy().sort_values(["team","state_key"],kind="mergesort").reset_index(drop=True)
    return x.to_csv(index=False,float_format="%.12f",lineterminator="\n").encode("utf-8")

def certify_parent_result(parent:dict,season:int,week:int)->None:
    if int(parent.get("season",-1))!=season or int(parent.get("target_week",-1))!=week:
        raise RuntimeError(f"parent target mismatch: {parent}")
    if parent.get("disposition")!="PLAYER_STATE_FUTURE_WEEK_READY":
        raise RuntimeError(f"parent not READY: {parent.get('disposition')}")
    if int(parent.get("target_week_outcomes_read",-1))!=0:
        raise RuntimeError("parent target-week outcomes boundary violated")
    if int(parent.get("sportsbook_inputs_used",-1))!=0:
        raise RuntimeError("parent sportsbook boundary violated")
    if int(parent.get("candidate_models_fit",-1))!=0:
        raise RuntimeError("parent fitted candidate model unexpectedly")
    if bool(parent.get("production_changed",True)):
        raise RuntimeError("parent production changed")
    completion=parent.get("prior_week_source_completion") or {}
    if not bool(completion.get("complete",False)):
        raise RuntimeError("parent prior-week source state incomplete")

def build(rows:pd.DataFrame,season:int,week:int,parent_meta:dict)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    x=rows.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]

    bad=[c for c in x.columns if any(t in c for t in FORBIDDEN)]
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

    if x["state_key"].astype(str).duplicated().any():
        raise RuntimeError("future-week parent contains duplicate state_key")

    chrono=x["chronology_valid"].astype(str).str.lower().isin({"true","1","yes"})
    if not chrono.all():
        bad_rows=x.loc[~chrono,["team","player","state_key"]].head(20)
        raise RuntimeError(f"future-week chronology failure: {bad_rows.to_dict('records')}")

    for c in ("last3_source_max_week","snap_source_max_week"):
        if c in x.columns:
            v=num(x[c]).dropna()
            if len(v) and not v.lt(week).all():
                raise RuntimeError(f"same/future source rows in {c}: {sorted(v[v.ge(week)].unique())}")

    rb=x.loc[x["position_group"].astype(str).str.upper().eq("RB")].copy()
    rb["carry_raw"]=num(rb["last3_room_opportunity_share"])
    rb["snap_raw"]=num(rb["last3_room_snap_fraction"])
    rb=rb.loc[rb["carry_raw"].notna()&rb["snap_raw"].notna()].copy()

    locked=[];teams=[]
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
        g["shadow_player_state_share"]=(
            0.50*g["control_recent_carry_share"]+
            0.50*g["snap_state_norm"]
        )
        ss=float(g["shadow_player_state_share"].sum())
        if not np.isfinite(ss) or ss<=0:
            raise RuntimeError(f"invalid shadow mass team={team}")
        g["shadow_player_state_share"]/=ss
        g["season"]=season;g["week"]=week
        g["raw_last3_room_opportunity_share"]=g["carry_raw"]
        g["raw_last3_room_snap_fraction"]=g["snap_raw"]

        cg=float(g["control_recent_carry_share"].sum())
        sg=float(g["shadow_player_state_share"].sum())
        if abs(cg-1.0)>1e-12 or abs(sg-1.0)>1e-12:
            raise RuntimeError(f"share conservation failure team={team} control={cg} shadow={sg}")
        locked.append(g)
        teams.append({
            "team":team,"locked_players":int(len(g)),
            "control_sum":cg,"shadow_sum":sg,
            "max_abs_shadow_minus_control":float(
                (g["shadow_player_state_share"]-g["control_recent_carry_share"]).abs().max()
            ),
            "mean_abs_shadow_minus_control":float(
                (g["shadow_player_state_share"]-g["control_recent_carry_share"]).abs().mean()
            ),
        })

    if not locked:
        raise RuntimeError("zero lockable future-week RB teams")

    out=pd.concat(locked,ignore_index=True)
    team_summary=pd.DataFrame(teams)
    digest="sha256:"+hashlib.sha256(canonical_csv_bytes(out)).hexdigest()
    result={
        "version":VERSION,
        "season":season,"week":week,
        "status":f"WEEK{week}_PREGAME_LOCK_FROZEN",
        "parent_state_version":parent_meta.get("version",""),
        "parent_state_disposition":parent_meta.get("disposition",""),
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
        "target_week_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "parameters_fit":0,
        "production_changed":False,
        "scientific_formula_changed_from_week5":False,
    }
    return out,team_summary,result

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--parent-rows",type=Path,required=True)
    ap.add_argument("--parent-result",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    rows=pd.read_csv(a.parent_rows,low_memory=False)
    parent=json.loads(a.parent_result.read_text(encoding="utf-8"))
    certify_parent_result(parent,a.season,a.week)
    out,teams,result=build(rows,a.season,a.week,parent)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    prefix=f"rb_player_state_allocation_week{a.week}_lock"
    (a.out_dir/f"{prefix}.csv").write_bytes(canonical_csv_bytes(out))
    teams.to_csv(a.out_dir/f"rb_player_state_allocation_week{a.week}_team_summary.csv",index=False,float_format="%.12f")
    (a.out_dir/f"{prefix}.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
