#!/usr/bin/env python3
"""Freeze a private-safe pregame GSIS RB successor allocation lock.

This is research-only. It consumes a private GSIS Lineup Detail snapshot plus
the already-frozen RB Vacancy Opportunity V1 no-outcome state. Player-level
locked rows must remain private. The public manifest contains aggregate
integrity/timing facts only.

No target outcomes or sportsbook inputs are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research.gsis_rb_successor_lineup_v1 import (
    build_private_candidate,
    load_snapshot,
    parse_offense_lineups,
    team_key,
)

TOL=1e-10
FORBIDDEN={
    "actual","actual_rushes","actual_rush_yards","target_game_snaps",
    "line","odds","over_odds","under_odds","book","bookmaker",
    "market_prob","edge_pct","fair_prob","result","profit_units",
}

def _iso(v:Any)->datetime:
    s=str(v).strip()
    if not s:
        raise RuntimeError("missing UTC timestamp")
    if s.endswith("Z"):
        s=s[:-1]+"+00:00"
    dt=datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt=dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _no_forbidden(df:pd.DataFrame,label:str)->None:
    bad=[]
    for col in df.columns:
        lc=str(col).strip().lower()
        if lc in FORBIDDEN or lc.startswith("sportsbook_"):
            bad.append(str(col))
    if bad:
        raise RuntimeError(f"{label} contains forbidden target/market fields: {sorted(bad)}")

def _capture_by_team(payload:dict[str,Any])->dict[str,datetime]:
    out={}
    for rec in payload.get("records",[]):
        if rec.get("report")!="Lineup Detail" or str(rec.get("mode",""))!="Offense":
            continue
        team=""
        for f in rec.get("filters",[]):
            if f.get("id")=="select2":
                team=team_key(f.get("value",""))
                break
        if not team:
            raise RuntimeError("Lineup Detail offense record missing team filter")
        ts=_iso(rec.get("capture_timestamp_utc"))
        out[team]=max(out.get(team,ts),ts)
    return out

def build_lock(
    *,
    snapshot_path:Path,
    expected_snapshot_sha256:str,
    vacancy:pd.DataFrame,
    events:pd.DataFrame,
    successor_pool:pd.DataFrame,
)->tuple[pd.DataFrame,dict[str,Any]]:
    _no_forbidden(vacancy,"vacancy state")
    _no_forbidden(events,"event schedule")
    _no_forbidden(successor_pool,"successor pool")
    req_v={
        "target_season","target_week","team","successor_player_clean_key",
        "vacated_rush_share","successor_weight","transfer_rush_share",
        "unavailable_players",
    }
    req_e={"target_season","target_week","team","event_id","kickoff_utc"}
    req_p={"target_season","target_week","team","successor_player_clean_key"}
    if req_v-set(vacancy.columns):
        raise RuntimeError(f"vacancy state missing {sorted(req_v-set(vacancy.columns))}")
    if req_e-set(events.columns):
        raise RuntimeError(f"event schedule missing {sorted(req_e-set(events.columns))}")
    if req_p-set(successor_pool.columns):
        raise RuntimeError(f"successor pool missing {sorted(req_p-set(successor_pool.columns))}")

    v=vacancy.copy()
    e=events.copy()
    p=successor_pool.copy()
    for x in (v,e,p):
        x["team"]=x["team"].map(team_key)
        x["target_season"]=pd.to_numeric(x["target_season"],errors="raise").astype(int)
        x["target_week"]=pd.to_numeric(x["target_week"],errors="raise").astype(int)
    p["successor_player_clean_key"]=p["successor_player_clean_key"].astype(str).str.strip()
    if e.duplicated(["target_season","target_week","team"]).any():
        raise RuntimeError("event schedule has duplicate season/week/team")
    if p.duplicated(["target_season","target_week","team","successor_player_clean_key"]).any():
        raise RuntimeError("successor pool has duplicate identities")
    if v.empty:
        return pd.DataFrame(),{
            "disposition":"NO_QUALIFYING_VACANCY_EVENT",
            "events_seen":0,"events_locked":0,
            "target_outcomes_read":False,"sportsbook_inputs_read":False,
            "player_identifiers_emitted_publicly":False,
        }

    payload,digest=load_snapshot(snapshot_path)
    want=str(expected_snapshot_sha256).strip().lower()
    if not want or digest.lower()!=want:
        raise RuntimeError(f"GSIS snapshot SHA mismatch got={digest} expected={want}")
    snap_season=int(payload.get("season"))
    if not v["target_season"].eq(snap_season).all():
        raise RuntimeError("GSIS snapshot season != target season")
    rows=parse_offense_lineups(payload)
    capture=_capture_by_team(payload)
    gsis,mech=build_private_candidate(rows,v,successor_pool=p)

    # One frozen event row per vacancy team.
    event_base=v[["target_season","target_week","team","vacated_rush_share","unavailable_players"]].copy()
    event_base["vacated_rush_share"]=pd.to_numeric(event_base["vacated_rush_share"],errors="raise")
    event_base=event_base.sort_values(["target_season","target_week","team"]).drop_duplicates(
        ["target_season","target_week","team"],keep="first"
    )
    for ident,g in v.groupby(["target_season","target_week","team"]):
        vals=pd.to_numeric(g["vacated_rush_share"],errors="raise").astype(float)
        if float(vals.max()-vals.min())>1e-12:
            raise RuntimeError(f"vacated share inconsistent within event {ident}")

    out=[]
    event_status=[]
    for _,er in event_base.iterrows():
        season=int(er.target_season); week=int(er.target_week); team=str(er.team)
        ev=e.loc[
            e.target_season.eq(season)&e.target_week.eq(week)&e.team.eq(team)
        ]
        if len(ev)!=1:
            raise RuntimeError(f"expected one target event for {season} W{week} {team}, got {len(ev)}")
        ev=ev.iloc[0]
        kickoff=_iso(ev.kickoff_utc)
        cap=capture.get(team)
        if cap is None:
            event_status.append({"status":"NO_TEAM_LINEUP_CAPTURE","team":team})
            continue
        if not cap < kickoff:
            event_status.append({"status":"SOURCE_TIMING_INVALID","team":team})
            continue

        g=v.loc[v.target_season.eq(season)&v.target_week.eq(week)&v.team.eq(team)].copy()
        pool_event=p.loc[p.target_season.eq(season)&p.target_week.eq(week)&p.team.eq(team)].copy()
        if pool_event.empty:
            raise RuntimeError(f"active successor pool empty for {team}")
        gc=gsis.loc[
            pd.to_numeric(gsis.get("target_season"),errors="coerce").eq(season)
            &pd.to_numeric(gsis.get("target_week"),errors="coerce").eq(week)
            &gsis.get("team",pd.Series(dtype=str)).astype(str).map(team_key).eq(team)
        ].copy() if not gsis.empty else pd.DataFrame()
        if gc.empty:
            event_status.append({"status":"NO_GSIS_SUCCESSOR_EXPOSURE","team":team})
            continue

        gsis_map={
            str(r.successor_player_clean_key):(float(r.gsis_successor_weight),float(r.gsis_transfer_rush_share))
            for _,r in gc.iterrows()
        }
        V=float(er.vacated_rush_share)
        snap_w=pd.to_numeric(g.successor_weight,errors="raise").astype(float)
        snap_t=pd.to_numeric(g.transfer_rush_share,errors="raise").astype(float)
        if (snap_w<0).any() or (snap_t<0).any():
            raise RuntimeError(f"negative frozen Vacancy V1 transfer for {team}")
        if abs(float(snap_w.sum())-1.0)>TOL or abs(float(snap_t.sum())-V)>TOL:
            raise RuntimeError(f"Vacancy V1 conservation failed for {team}")

        snap_map={
            str(r.successor_player_clean_key):(float(r.successor_weight),float(r.transfer_rush_share))
            for _,r in g.iterrows()
        }
        snap_successors=set(snap_map)
        active_successors=set(pool_event.successor_player_clean_key.astype(str))
        if not snap_successors.issubset(active_successors):
            raise RuntimeError(f"Vacancy V1 successor missing from active pool for {team}")
        locked=[]
        for _,r in pool_event.iterrows():
            pk=str(r.successor_player_clean_key)
            sw,st=snap_map.get(pk,(0.0,0.0))
            gw,gt=gsis_map.get(pk,(0.0,0.0))
            locked.append({
                "target_season":season,
                "target_week":week,
                "event_id":str(ev.event_id),
                "team":team,
                "kickoff_utc":kickoff.isoformat(),
                "gsis_team_capture_utc":cap.isoformat(),
                "gsis_snapshot_sha256":digest,
                "successor_player_clean_key":pk,
                "unavailable_players":str(er.unavailable_players),
                "vacated_rush_share":V,
                "snap_successor_weight":sw,
                "snap_transfer_rush_share":st,
                "gsis_successor_weight":gw,
                "gsis_transfer_rush_share":gt,
            })
        ldf=pd.DataFrame(locked)
        if abs(float(ldf.gsis_successor_weight.sum())-1.0)>TOL:
            raise RuntimeError(f"GSIS successor weights do not conserve for {team}")
        if abs(float(ldf.gsis_transfer_rush_share.sum())-V)>TOL:
            raise RuntimeError(f"GSIS transfer does not conserve frozen vacated share for {team}")
        out.extend(locked)
        event_status.append({"status":"LOCKED","team":team,"successors":len(ldf)})

    private=pd.DataFrame(out)
    locked_events=[x for x in event_status if x["status"]=="LOCKED"]
    disposition="GSIS_RB_SUCCESSOR_LINEUP_V1_PREGAME_ALLOCATION_LOCKED" if locked_events else (
        "SOURCE_TIMING_INVALID" if any(x["status"]=="SOURCE_TIMING_INVALID" for x in event_status)
        else "NO_GSIS_SUCCESSOR_EXPOSURE"
    )
    audit={
        "disposition":disposition,
        "snapshot_sha256":digest,
        "snapshot_id":payload.get("snapshot_id"),
        "snapshot_season":snap_season,
        "snapshot_phase":payload.get("phase"),
        "offense_lineup_rows_parsed":len(rows),
        "events_seen":len(event_base),
        "events_locked":len(locked_events),
        "candidate_rows_private":len(private),
        "events_source_timing_invalid":sum(x["status"]=="SOURCE_TIMING_INVALID" for x in event_status),
        "events_no_team_lineup_capture":sum(x["status"]=="NO_TEAM_LINEUP_CAPTURE" for x in event_status),
        "events_no_gsis_successor_exposure":sum(x["status"]=="NO_GSIS_SUCCESSOR_EXPOSURE" for x in event_status),
        "max_snap_conservation_gap":0.0 if private.empty else float(
            private.groupby(["target_season","target_week","team"]).apply(
                lambda q: abs(q.snap_transfer_rush_share.sum()-q.vacated_rush_share.iloc[0])
            ).max()
        ),
        "max_gsis_conservation_gap":0.0 if private.empty else float(
            private.groupby(["target_season","target_week","team"]).apply(
                lambda q: abs(q.gsis_transfer_rush_share.sum()-q.vacated_rush_share.iloc[0])
            ).max()
        ),
        "target_outcomes_read":False,
        "sportsbook_inputs_read":False,
        "raw_lineups_emitted_publicly":False,
        "player_identifiers_emitted_publicly":False,
        "private_rows_must_not_be_committed":True,
        "candidate_mechanics_events_seen":int(mech.get("events_seen",0)),
        "successor_pool_mode":str(mech.get("successor_pool_mode","")),
    }
    return private,audit

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",type=Path,required=True)
    ap.add_argument("--expected-snapshot-sha256",required=True)
    ap.add_argument("--vacancy-state",type=Path,required=True)
    ap.add_argument("--events",type=Path,required=True)
    ap.add_argument("--successor-pool",type=Path,required=True)
    ap.add_argument("--private-lock-out",type=Path,required=True)
    ap.add_argument("--public-manifest-out",type=Path,required=True)
    a=ap.parse_args()
    private,audit=build_lock(
        snapshot_path=a.snapshot,
        expected_snapshot_sha256=a.expected_snapshot_sha256,
        vacancy=pd.read_csv(a.vacancy_state,low_memory=False),
        events=pd.read_csv(a.events,low_memory=False),
        successor_pool=pd.read_csv(a.successor_pool,low_memory=False),
    )
    a.public_manifest_out.parent.mkdir(parents=True,exist_ok=True)
    a.public_manifest_out.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not private.empty:
        a.private_lock_out.parent.mkdir(parents=True,exist_ok=True)
        private.to_csv(a.private_lock_out,index=False)
    elif a.private_lock_out.exists():
        a.private_lock_out.unlink()
    print(json.dumps(audit,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
