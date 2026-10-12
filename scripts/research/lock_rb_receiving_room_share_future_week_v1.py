#!/usr/bin/env python3
"""Build a future-week RB Receiving Room Share Shadow V1 lock.

Mechanical continuation of the frozen Week-5 rule. No coefficient, threshold,
or fallback is fit. The exact current certified RB room is preserved and each
player's strict-prior receiving-room share is normalized within that room.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.rb_receiving_identity_runtime_v1 import identity_atlas

VERSION="RB_RECEIVING_ROOM_SHARE_SHADOW_FUTURE_WEEK_V1"
TOL=1e-10

def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def _bool_series(s:pd.Series)->pd.Series:
    return s.astype(str).str.lower().isin({"true","1","yes"})

def certify_parents(
    *,
    season:int,
    week:int,
    rb_lock_meta:dict,
    live_state_meta:dict,
)->None:
    if int(rb_lock_meta.get("season",-1))!=season or int(rb_lock_meta.get("week",-1))!=week:
        raise RuntimeError("RB allocation parent target mismatch")
    if rb_lock_meta.get("status")!=f"WEEK{week}_PREGAME_LOCK_FROZEN":
        raise RuntimeError(f"RB allocation parent not frozen: {rb_lock_meta.get('status')}")
    if int(rb_lock_meta.get("target_week_outcomes_read",-1))!=0:
        raise RuntimeError("RB allocation parent outcome boundary violated")
    if int(rb_lock_meta.get("sportsbook_inputs_used",-1))!=0:
        raise RuntimeError("RB allocation parent sportsbook boundary violated")
    if bool(rb_lock_meta.get("production_changed",True)):
        raise RuntimeError("RB allocation parent production changed")

    if int(live_state_meta.get("season",-1))!=season or int(live_state_meta.get("target_week",-1))!=week:
        raise RuntimeError("live-state parent target mismatch")
    if live_state_meta.get("disposition")!="PLAYER_STATE_FUTURE_WEEK_READY":
        raise RuntimeError(f"live-state parent not ready: {live_state_meta.get('disposition')}")
    if int(live_state_meta.get("target_week_outcomes_read",-1))!=0:
        raise RuntimeError("live-state parent outcome boundary violated")
    if int(live_state_meta.get("sportsbook_inputs_used",-1))!=0:
        raise RuntimeError("live-state parent sportsbook boundary violated")
    if bool(live_state_meta.get("production_changed",True)):
        raise RuntimeError("live-state parent production changed")
    if not bool((live_state_meta.get("prior_week_source_completion") or {}).get("complete",False)):
        raise RuntimeError("live-state prior-week source incomplete")

def _snapshot_prior_room_share_by_gsis(
    queries:pd.DataFrame,
    states:pd.DataFrame,
    *,
    season:int,
    week:int,
)->pd.DataFrame:
    q=queries.copy()
    q["gsis_id"]=q["gsis_id"].fillna("").astype(str).str.strip()
    if q["gsis_id"].eq("").any():
        bad=q.loc[q["gsis_id"].eq(""),["team","player","state_key"]]
        raise RuntimeError(f"future-week frozen RB missing GSIS ID: {bad.to_dict('records')}")
    q["time_key"]=season*100+week
    q["_qrow"]=np.arange(len(q))

    st=states.copy()
    if "player_id" not in st.columns:
        raise RuntimeError("RB receiving identity states missing player_id")
    st["gsis_id"]=st["player_id"].fillna("").astype(str).str.strip()
    st=st.loc[st["gsis_id"].ne("")].copy()
    required={"time_key","after_rb_room_share","after_games"}
    missing=required-set(st.columns)
    if missing:
        raise RuntimeError(f"RB receiving identity states missing columns: {sorted(missing)}")
    dup=st.duplicated(["gsis_id","time_key"],keep=False)
    if dup.any():
        bad=st.loc[dup,["gsis_id","time_key","team"]].head(20)
        raise RuntimeError(f"duplicate GSIS/time receiving state: {bad.to_dict('records')}")

    hist=st[["gsis_id","time_key","after_rb_room_share","after_games"]].copy().rename(
        columns={"after_rb_room_share":"prior_rb_room_share","after_games":"prior_games"}
    )
    hist=hist.sort_values(["time_key","gsis_id"])
    qsort=q.sort_values(["time_key","gsis_id"])
    snap=pd.merge_asof(
        qsort,hist,on="time_key",by="gsis_id",
        direction="backward",allow_exact_matches=False,
    )
    return snap.sort_values("_qrow").drop(columns=["_qrow","time_key"]).reset_index(drop=True)

def _attach_prior_room_share(
    lock:pd.DataFrame,
    live:pd.DataFrame,
    *,
    season:int,
    week:int,
)->pd.DataFrame:
    required_lock={
        "season","week","team","opponent","state_key","player","gsis_id","pfr_id",
        "room_size","control_recent_carry_share","shadow_player_state_share",
    }
    miss=required_lock-set(lock.columns)
    if miss:
        raise RuntimeError(f"future-week RB lock missing columns: {sorted(miss)}")
    if not pd.to_numeric(lock["season"],errors="coerce").eq(season).all():
        raise RuntimeError("future-week RB lock contains wrong season")
    if not pd.to_numeric(lock["week"],errors="coerce").eq(week).all():
        raise RuntimeError("future-week RB lock contains wrong week")
    if lock["state_key"].astype(str).duplicated().any():
        raise RuntimeError("future-week RB lock duplicate state_key")

    required_live={
        "state_key","team","player","name_key","position_group",
        "chronology_valid","last3_source_max_week","snap_source_max_week",
        "roster_source_week","current_games","same_team_current_games","last3_targets",
    }
    miss=required_live-set(live.columns)
    if miss:
        raise RuntimeError(f"future-week live parent missing columns: {sorted(miss)}")

    live=live.copy()
    live["team"]=live["team"].map(canon_team)
    if live["state_key"].astype(str).duplicated().any():
        raise RuntimeError("future-week live parent duplicate state_key")

    parent=lock.merge(
        live[[
            "state_key","team","name_key","position_group","chronology_valid",
            "last3_source_max_week","snap_source_max_week","roster_source_week",
            "current_games","same_team_current_games","last3_targets",
        ]],
        on=["state_key","team"],how="left",validate="one_to_one",
    )
    if parent["name_key"].isna().any():
        bad=parent.loc[parent["name_key"].isna(),["team","player","state_key"]]
        raise RuntimeError(f"frozen RB identity absent from live parent: {bad.to_dict('records')}")
    if not parent["position_group"].astype(str).str.upper().eq("RB").all():
        raise RuntimeError("future-week receiving lock includes non-RB parent")
    if not _bool_series(parent["chronology_valid"]).all():
        raise RuntimeError("future-week live parent chronology invalid")
    for c in ("last3_source_max_week","snap_source_max_week"):
        v=pd.to_numeric(parent[c],errors="coerce").dropna()
        if len(v) and not v.lt(week).all():
            raise RuntimeError(f"same/future source week in {c}")

    states,_prev=identity_atlas(2013,season)
    feat=_snapshot_prior_room_share_by_gsis(
        parent[["state_key","team","player","gsis_id"]].copy(),
        states,season=season,week=week,
    )
    out=parent.merge(
        feat[["state_key","gsis_id","prior_rb_room_share","prior_games"]],
        on=["state_key","gsis_id"],how="left",validate="one_to_one",
    )
    out["prior_rb_room_share"]=pd.to_numeric(out["prior_rb_room_share"],errors="coerce")
    out["prior_games"]=pd.to_numeric(out["prior_games"],errors="coerce")
    out["identity_history_route"]="EXACT_GSIS_STRICT_PRIOR"
    missing=out["prior_rb_room_share"].isna()|~np.isfinite(out["prior_rb_room_share"])
    if missing.any():
        bad=out.loc[missing,["team","player","state_key","gsis_id"]]
        raise RuntimeError(
            "future-week frozen RB identities missing strict-prior receiving history: "
            f"{bad.to_dict('records')}"
        )
    if out["prior_rb_room_share"].lt(0).any():
        raise RuntimeError("negative prior_rb_room_share")
    return out

def normalize_receiving_room_share(x:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    x=x.copy()
    x["candidate_rb_receiving_room_share"]=np.nan
    team_rows=[]
    for team,idx in x.groupby("team",sort=True).groups.items():
        g=x.loc[idx].copy()
        if len(g)<2:
            raise RuntimeError(f"future-week frozen RB room has <2 players team={team}")
        state=pd.to_numeric(g["prior_rb_room_share"],errors="coerce")
        total=float(state.sum())
        if not np.isfinite(total) or total<=0:
            raise RuntimeError(f"nonpositive prior RB receiving-room mass team={team}")
        candidate=state/total
        x.loc[idx,"candidate_rb_receiving_room_share"]=candidate.to_numpy(float)
        team_rows.append({
            "team":team,"players":int(len(g)),
            "prior_state_sum":total,"candidate_sum":float(candidate.sum()),
            "candidate_max":float(candidate.max()),"candidate_min":float(candidate.min()),
            "candidate_hhi":float(np.square(candidate.to_numpy(float)).sum()),
            "carry_reference_hhi":float(np.square(
                pd.to_numeric(g["shadow_player_state_share"],errors="coerce")
                .fillna(0.0).to_numpy(float)
            ).sum()),
        })
    teams=pd.DataFrame(team_rows)
    if x["candidate_rb_receiving_room_share"].isna().any():
        raise RuntimeError("future-week candidate receiving share missing")
    if float((teams["candidate_sum"]-1.0).abs().max())>TOL:
        raise RuntimeError("future-week receiving-room shares fail conservation")
    return x,teams

def build_candidate(
    lock:pd.DataFrame,
    live:pd.DataFrame,
    *,
    season:int,
    week:int,
)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    x=_attach_prior_room_share(lock.copy(),live.copy(),season=season,week=week)
    x,teams=normalize_receiving_room_share(x)
    out=pd.DataFrame({
        "season":season,"week":week,
        "team":x["team"],"opponent":x["opponent"],"state_key":x["state_key"],
        "player":x["player"],"name_key":x["name_key"],"gsis_id":x["gsis_id"],
        "pfr_id":x["pfr_id"],"room_size":x["room_size"],"prior_games":x["prior_games"],
        "prior_rb_room_share":x["prior_rb_room_share"],
        "candidate_rb_receiving_room_share":x["candidate_rb_receiving_room_share"],
        "identity_history_route":x["identity_history_route"],
        "reference_control_recent_carry_share":x["control_recent_carry_share"],
        "reference_carry_snap_shadow_share":x["shadow_player_state_share"],
        "current_games":x["current_games"],
        "same_team_current_games":x["same_team_current_games"],
        "last3_targets":x["last3_targets"],
        "last3_source_max_week":x["last3_source_max_week"],
        "snap_source_max_week":x["snap_source_max_week"],
        "chronology_valid":x["chronology_valid"],
        "roster_source_week":x["roster_source_week"],
    }).sort_values(["team","state_key"],kind="mergesort").reset_index(drop=True)

    result={
        "version":VERSION,
        "status":f"WEEK{week}_PREGAME_LOCK_FROZEN",
        "season":season,"week":week,
        "players":int(len(out)),"teams":int(out["team"].nunique()),
        "history_available_players":int(out["prior_rb_room_share"].notna().sum()),
        "history_available_rate":float(out["prior_rb_room_share"].notna().mean()),
        "rooms_locked":int(len(teams)),
        "max_candidate_conservation_gap":float((teams["candidate_sum"]-1.0).abs().max()),
        "median_candidate_hhi":float(teams["candidate_hhi"].median()),
        "parameters_fit":0,"sportsbook_inputs_used":0,
        "target_week_outcomes_read":0,"production_changed":False,
        "scientific_formula_changed_from_week5":False,
        "receiving_state":"prior_rb_room_share",
        "redistribution_rule":"normalize strict-prior prior_rb_room_share within exact frozen RB room",
    }
    return out,teams,result

def run(
    *,
    season:int,
    week:int,
    parent_rb_lock_path:Path,
    parent_rb_lock_result_path:Path,
    parent_live_rows_path:Path,
    parent_live_result_path:Path,
    out_dir:Path,
)->dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    lock=_read(parent_rb_lock_path,"future-week RB player-state lock")
    live=_read(parent_live_rows_path,"future-week live player-state rows")
    rb_meta=json.loads(parent_rb_lock_result_path.read_text(encoding="utf-8"))
    live_meta=json.loads(parent_live_result_path.read_text(encoding="utf-8"))
    certify_parents(season=season,week=week,rb_lock_meta=rb_meta,live_state_meta=live_meta)
    out,teams,result=build_candidate(lock,live,season=season,week=week)

    row_path=out_dir/f"rb_receiving_room_share_week{week}_lock.csv"
    team_path=out_dir/f"rb_receiving_room_share_week{week}_team_summary.csv"
    out.to_csv(row_path,index=False,float_format="%.12f")
    teams.to_csv(team_path,index=False,float_format="%.12f")
    result["row_csv_sha256"]="sha256:"+_sha256(row_path)
    result["team_csv_sha256"]="sha256:"+_sha256(team_path)
    result["generated_at"]=datetime.now(timezone.utc).isoformat()
    (out_dir/f"rb_receiving_room_share_week{week}_lock.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    return result

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--parent-rb-lock",type=Path,required=True)
    ap.add_argument("--parent-rb-lock-result",type=Path,required=True)
    ap.add_argument("--parent-live-rows",type=Path,required=True)
    ap.add_argument("--parent-live-result",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    run(
        season=a.season,week=a.week,
        parent_rb_lock_path=a.parent_rb_lock,
        parent_rb_lock_result_path=a.parent_rb_lock_result,
        parent_live_rows_path=a.parent_live_rows,
        parent_live_result_path=a.parent_live_result,
        out_dir=a.out_dir,
    )
    return 0

if __name__=="__main__":
    raise SystemExit(main())
