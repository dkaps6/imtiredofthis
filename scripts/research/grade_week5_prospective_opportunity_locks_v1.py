#!/usr/bin/env python3
"""Grade immutable Week-5 player-opportunity locks against realized usage.

Diagnostic only. Week 5 is prospective lock #1 for carry/snap and target-share
trajectory. No formula changes or final promotion decisions are made here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._opponent_map import canon_team
from scripts.operations.grade_market_track_record_gsis_v1 import (
    _snap_participation_sets,
    _status_map,
    build_alias_index,
    load_actual_stats_unfiltered,
    load_roster_identity,
    load_snap_participation,
    resolve_gsis,
)
from scripts.player_form_v2 import _normalize_weekly

SEASON=2026
WEEK=5
RB_POS={"RB","HB","FB","TB"}
WR_POS={"WR","LWR","RWR","SWR"}
INACTIVE={"INA","INACTIVE","DNP"}


def _to_pandas(obj:Any)->pd.DataFrame:
    if isinstance(obj,pd.DataFrame):
        return obj.copy()
    if hasattr(obj,"to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    if hasattr(obj,"to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def _num(s): return pd.to_numeric(s,errors="coerce")


def _position(v)->str:
    p=str(v or "").strip().upper()
    if p in {"HB","FB","TB"} or p.startswith("RB"): return "RB"
    if p in WR_POS or p.startswith("WR"): return "WR"
    if p.startswith("TE"): return "TE"
    return p


def _load_usage()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    import nflreadpy as nfl
    raw=_to_pandas(nfl.load_player_stats(seasons=[SEASON],summary_level="week"))
    weekly=_normalize_weekly(raw,SEASON)
    weekly=weekly.loc[_num(weekly["week"]).eq(WEEK)].copy()
    weekly["team"]=weekly["team"].map(canon_team)
    weekly["position_family"]=weekly["position"].map(_position)
    weekly["player_id"]=weekly["player_id"].fillna("").astype(str).str.strip()
    if weekly.duplicated(["team","player_id"]).any():
        # Aggregate only if provider emits split rows for the same target game.
        weekly=(weekly.groupby(["team","player_id"],as_index=False)
            .agg(
                player=("player","last"),
                position=("position","last"),
                position_family=("position_family","last"),
                targets=("targets","sum"),
                rushes=("rushes","sum"),
                rec_yards=("rec_yards","sum"),
                rush_yards=("rush_yards","sum"),
                team_targets=("team_targets","max"),
                team_rushes=("team_rushes","max"),
            ))
    roster=load_roster_identity(SEASON,[WEEK])
    snaps=load_snap_participation(SEASON,[WEEK])
    actual=load_actual_stats_unfiltered(SEASON,[WEEK])
    return weekly,roster,snaps,actual


def _identity_context(weekly,roster,snaps,actual):
    idx=build_alias_index(actual,roster)
    status=_status_map(roster)
    snap_exact,snap_base=_snap_participation_sets(snaps)
    rostered=set(zip(roster["season"],roster["week"],roster["team"],roster["gsis_id"]))
    stat_map={
        (str(r.team),str(r.player_id)):r
        for r in weekly.itertuples(index=False)
        if str(r.player_id).strip()
    }
    return idx,status,snap_exact,snap_base,rostered,stat_map


def _resolve_locked(lock:pd.DataFrame,weekly,roster,snaps,actual,*,id_col:str|None,name_col:str)->pd.DataFrame:
    idx,status,snap_exact,snap_base,rostered,stat_map=_identity_context(weekly,roster,snaps,actual)
    out=lock.copy()
    out["team"]=out["team"].map(canon_team)
    gids=[]; routes=[]; targets=[]; rushes=[]; scoreable=[]
    for r in out.itertuples(index=False):
        team=str(r.team)
        gid=""
        if id_col and hasattr(r,id_col):
            raw=str(getattr(r,id_col) or "").strip()
            if raw.lower() not in {"","nan","none","<na>"}:
                gid=raw
        route="LOCKED_GSIS" if gid else ""
        if not gid:
            clean=str(getattr(r,name_col) or "").strip()
            gid2,st=resolve_gsis(clean,team,idx)
            gid=str(gid2 or "")
            route=st
        rec=stat_map.get((team,gid))
        if rec is not None:
            t=float(rec.targets); ru=float(rec.rushes); ok=True
        else:
            roster_ok=(SEASON,WEEK,team,gid) in rostered if gid else False
            rs=status.get((SEASON,WEEK,team,gid),"") if gid else ""
            # A locked target-week identity with no weekly stat row is a valid
            # zero-usage observation when roster evidence exists. We do not
            # infer a zero for an unresolved identity.
            ok=bool(gid and roster_ok)
            t=0.0 if ok else np.nan
            ru=0.0 if ok else np.nan
            if ok and str(rs).upper() in INACTIVE:
                route=f"{route}|ROSTER_INACTIVE_ZERO"
            elif ok:
                route=f"{route}|ROSTER_ZERO_USAGE"
        gids.append(gid);routes.append(route);targets.append(t);rushes.append(ru);scoreable.append(ok)
    out["resolved_gsis_id"]=gids
    out["actual_identity_route"]=routes
    out["actual_targets"]=targets
    out["actual_rushes"]=rushes
    out["actual_identity_scoreable"]=scoreable
    return out


def _actual_rb_denoms(weekly:pd.DataFrame)->pd.DataFrame:
    rb=weekly.loc[weekly["position_family"].eq("RB")].copy()
    return (rb.groupby("team",as_index=False)
        .agg(
            actual_rb_room_rushes=("rushes","sum"),
            actual_rb_room_targets=("targets","sum"),
        ))


def _team_target_denoms(weekly:pd.DataFrame)->pd.DataFrame:
    return (weekly.groupby("team",as_index=False)
        .agg(actual_team_targets=("team_targets","max")))


def _actual_rb_users(weekly:pd.DataFrame,kind:str)->dict[str,set[str]]:
    rb=weekly.loc[weekly["position_family"].eq("RB")].copy()
    col="rushes" if kind=="rush" else "targets"
    rb=rb.loc[_num(rb[col]).gt(0)]
    out={}
    for team,g in rb.groupby("team"):
        out[str(team)]=set(g["player_id"].astype(str))
    return out


def _safe_spearman(a,b)->float:
    x=np.asarray(a,float);y=np.asarray(b,float)
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<2 or len(np.unique(x))<2 or len(np.unique(y))<2:return np.nan
    return float(spearmanr(x,y).statistic)


def grade_rb_carry(lock,weekly,roster,snaps,actual)->tuple[pd.DataFrame,dict]:
    x=_resolve_locked(lock,weekly,roster,snaps,actual,id_col="gsis_id",name_col="player")
    den=_actual_rb_denoms(weekly)
    x=x.merge(den,on="team",how="left",validate="many_to_one")
    actual_users=_actual_rb_users(weekly,"rush")
    locked={team:set(g["resolved_gsis_id"].astype(str)) for team,g in x.groupby("team")}
    unscored={}
    for team,users in actual_users.items():
        missing=sorted(users-locked.get(team,set()))
        if missing:unscored[team]=missing
    x["team_scoreable"]=~x["team"].isin(unscored)
    x["actual_rb_room_carry_share"]=np.where(
        x["actual_rb_room_rushes"].gt(0),
        x["actual_rushes"]/x["actual_rb_room_rushes"],
        np.nan,
    )
    q=x.loc[x["team_scoreable"]&x["actual_identity_scoreable"]&x["actual_rb_room_carry_share"].notna()].copy()
    for c in ["control_recent_carry_share","shadow_player_state_share"]:
        q[c]=_num(q[c])
    q["control_ae"]=(q["control_recent_carry_share"]-q["actual_rb_room_carry_share"]).abs()
    q["shadow_ae"]=(q["shadow_player_state_share"]-q["actual_rb_room_carry_share"]).abs()
    team_rows=[]
    for team,g in q.groupby("team"):
        actual_top=set(g.loc[g["actual_rb_room_carry_share"].eq(g["actual_rb_room_carry_share"].max()),"resolved_gsis_id"].astype(str))
        control_top=set(g.loc[g["control_recent_carry_share"].eq(g["control_recent_carry_share"].max()),"resolved_gsis_id"].astype(str))
        shadow_top=set(g.loc[g["shadow_player_state_share"].eq(g["shadow_player_state_share"].max()),"resolved_gsis_id"].astype(str))
        team_rows.append({
            "team":team,
            "players":len(g),
            "control_mae":float(g["control_ae"].mean()),
            "shadow_mae":float(g["shadow_ae"].mean()),
            "mae_improvement_control_minus_shadow":float((g["control_ae"]-g["shadow_ae"]).mean()),
            "control_rank_rho":_safe_spearman(g["control_recent_carry_share"],g["actual_rb_room_carry_share"]),
            "shadow_rank_rho":_safe_spearman(g["shadow_player_state_share"],g["actual_rb_room_carry_share"]),
            "control_top_correct":bool(actual_top&control_top),
            "shadow_top_correct":bool(actual_top&shadow_top),
        })
    t=pd.DataFrame(team_rows)
    result={
        "status":"WEEK5_DIAGNOSTIC_ONLY__NO_FINAL_V1_DISPOSITION",
        "scoreable_teams":int(t["team"].nunique()) if len(t) else 0,
        "unscored_new_or_missing_carrier_teams":unscored,
        "scoreable_players":int(len(q)),
        "control_player_share_mae":float(q["control_ae"].mean()) if len(q) else np.nan,
        "shadow_player_share_mae":float(q["shadow_ae"].mean()) if len(q) else np.nan,
        "mean_player_ae_improvement":float((q["control_ae"]-q["shadow_ae"]).mean()) if len(q) else np.nan,
        "control_rmse":float(np.sqrt(np.mean((q["control_recent_carry_share"]-q["actual_rb_room_carry_share"])**2))) if len(q) else np.nan,
        "shadow_rmse":float(np.sqrt(np.mean((q["shadow_player_state_share"]-q["actual_rb_room_carry_share"])**2))) if len(q) else np.nan,
        "control_top_back_accuracy":float(t["control_top_correct"].mean()) if len(t) else np.nan,
        "shadow_top_back_accuracy":float(t["shadow_top_correct"].mean()) if len(t) else np.nan,
        "mean_control_room_rank_rho":float(t["control_rank_rho"].mean()) if len(t) else np.nan,
        "mean_shadow_room_rank_rho":float(t["shadow_rank_rho"].mean()) if len(t) else np.nan,
        "teams_shadow_mae_better":int((t["shadow_mae"]<t["control_mae"]).sum()) if len(t) else 0,
        "teams_control_mae_better":int((t["control_mae"]<t["shadow_mae"]).sum()) if len(t) else 0,
        "ties":int(np.isclose(t["shadow_mae"],t["control_mae"]).sum()) if len(t) else 0,
    }
    return q,t,result


def grade_rb_receiving(lock,weekly,roster,snaps,actual)->tuple[pd.DataFrame,dict]:
    x=_resolve_locked(lock,weekly,roster,snaps,actual,id_col="gsis_id",name_col="player")
    den=_actual_rb_denoms(weekly)
    x=x.merge(den,on="team",how="left",validate="many_to_one")
    actual_users=_actual_rb_users(weekly,"target")
    locked={team:set(g["resolved_gsis_id"].astype(str)) for team,g in x.groupby("team")}
    unscored={}
    for team,users in actual_users.items():
        missing=sorted(users-locked.get(team,set()))
        if missing:unscored[team]=missing
    x["team_scoreable"]=~x["team"].isin(unscored)
    x["actual_rb_room_target_share"]=np.where(
        x["actual_rb_room_targets"].gt(0),
        x["actual_targets"]/x["actual_rb_room_targets"],
        np.nan,
    )
    x["candidate_rb_receiving_room_share"]=_num(x["candidate_rb_receiving_room_share"])
    q=x.loc[x["team_scoreable"]&x["actual_identity_scoreable"]&x["actual_rb_room_target_share"].notna()].copy()
    q["candidate_ae"]=(q["candidate_rb_receiving_room_share"]-q["actual_rb_room_target_share"]).abs()
    return q,{
        "status":"WEEK5_CANDIDATE_ABSOLUTE_DIAGNOSTIC_ONLY",
        "baseline_comparison_available":False,
        "baseline_comparison_reason":"IMMUTABLE_WEEK5_LOCK_OMITTED_REQUIRED_CURRENT_CANONICAL_RB_RECEIVING_ROOM_SHARE",
        "scoreable_players":int(len(q)),
        "scoreable_teams":int(q["team"].nunique()),
        "unscored_new_or_missing_targeter_teams":unscored,
        "candidate_room_share_mae":float(q["candidate_ae"].mean()) if len(q) else np.nan,
        "candidate_room_share_rmse":float(np.sqrt(np.mean((q["candidate_rb_receiving_room_share"]-q["actual_rb_room_target_share"])**2))) if len(q) else np.nan,
        "candidate_room_share_bias":float((q["candidate_rb_receiving_room_share"]-q["actual_rb_room_target_share"]).mean()) if len(q) else np.nan,
        "evidence_defect":"LOCK_CONTRACT_REQUIRED_BASELINE_BUT_LOCK_CONSTRUCTOR_DID_NOT_PERSIST_IT",
    }


def grade_trajectory(lock,weekly,roster,snaps,actual)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    x=_resolve_locked(lock,weekly,roster,snaps,actual,id_col="receiver_id",name_col="player_clean_key")
    den=_team_target_denoms(weekly)
    x=x.merge(den,on="team",how="left",validate="many_to_one")
    x["actual_team_target_share"]=np.where(
        x["actual_team_targets"].gt(0),
        x["actual_targets"]/x["actual_team_targets"],
        np.nan,
    )
    x["baseline_entitlement_tgt_share"]=_num(x["baseline_entitlement_tgt_share"])
    x["shadow_entitlement_tgt_share"]=_num(x["shadow_entitlement_tgt_share"])
    q=x.loc[x["actual_identity_scoreable"]&x["actual_team_target_share"].notna()].copy()
    q["baseline_ae"]=(q["baseline_entitlement_tgt_share"]-q["actual_team_target_share"]).abs()
    q["shadow_ae"]=(q["shadow_entitlement_tgt_share"]-q["actual_team_target_share"]).abs()
    q["position_family"]=q["position_family"].astype(str).str.upper()
    rows=[]
    for label,g in [("ALL",q),("WR",q.loc[q["position_family"].eq("WR")]),("TE",q.loc[q["position_family"].eq("TE")])]:
        rows.append({
            "scope":label,
            "players":int(len(g)),
            "baseline_mae":float(g["baseline_ae"].mean()) if len(g) else np.nan,
            "shadow_mae":float(g["shadow_ae"].mean()) if len(g) else np.nan,
            "mae_improvement_baseline_minus_shadow":float((g["baseline_ae"]-g["shadow_ae"]).mean()) if len(g) else np.nan,
            "baseline_rmse":float(np.sqrt(np.mean((g["baseline_entitlement_tgt_share"]-g["actual_team_target_share"])**2))) if len(g) else np.nan,
            "shadow_rmse":float(np.sqrt(np.mean((g["shadow_entitlement_tgt_share"]-g["actual_team_target_share"])**2))) if len(g) else np.nan,
        })
    scopes=pd.DataFrame(rows)
    room_rows=[]
    for (team,pos),g in q.groupby(["team","position_family"]):
        if pos=="WR":
            rg=g.loc[~g["wr_r15_anchor"].astype(str).str.lower().isin({"true","1","yes"})].copy()
        elif pos=="TE":
            rg=g.copy()
        else:
            continue
        if len(rg)<2:continue
        actual_top=set(rg.loc[rg["actual_team_target_share"].eq(rg["actual_team_target_share"].max()),"resolved_gsis_id"].astype(str))
        base_top=set(rg.loc[rg["baseline_entitlement_tgt_share"].eq(rg["baseline_entitlement_tgt_share"].max()),"resolved_gsis_id"].astype(str))
        shadow_top=set(rg.loc[rg["shadow_entitlement_tgt_share"].eq(rg["shadow_entitlement_tgt_share"].max()),"resolved_gsis_id"].astype(str))
        room_rows.append({
            "team":team,"position_family":pos,"players":len(rg),
            "baseline_rank_rho":_safe_spearman(rg["baseline_entitlement_tgt_share"],rg["actual_team_target_share"]),
            "shadow_rank_rho":_safe_spearman(rg["shadow_entitlement_tgt_share"],rg["actual_team_target_share"]),
            "baseline_top_correct":bool(actual_top&base_top),
            "shadow_top_correct":bool(actual_top&shadow_top),
        })
    rooms=pd.DataFrame(room_rows)
    result={
        "status":"WEEK5_DIAGNOSTIC_ONLY__NO_FINAL_V1_DISPOSITION",
        "locked_rows":int(len(x)),
        "scoreable_rows":int(len(q)),
        "identity_unscoreable_rows":int((~x["actual_identity_scoreable"]).sum()),
        "scopes":scopes.to_dict("records"),
        "rooms":int(len(rooms)),
        "baseline_mean_room_rank_rho":float(rooms["baseline_rank_rho"].mean()) if len(rooms) else np.nan,
        "shadow_mean_room_rank_rho":float(rooms["shadow_rank_rho"].mean()) if len(rooms) else np.nan,
        "baseline_top_accuracy":float(rooms["baseline_top_correct"].mean()) if len(rooms) else np.nan,
        "shadow_top_accuracy":float(rooms["shadow_top_correct"].mean()) if len(rooms) else np.nan,
    }
    return q,rooms,result


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--rb-carry-lock",type=Path,required=True)
    p.add_argument("--rb-receiving-lock",type=Path,required=True)
    p.add_argument("--trajectory-lock",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)

    weekly,roster,snaps,actual=_load_usage()
    carry=_read(a.rb_carry_lock,"RB carry/snap lock")
    recv=_read(a.rb_receiving_lock,"RB receiving lock")
    traj=_read(a.trajectory_lock,"WR/TE trajectory lock")

    cq,ct,cr=grade_rb_carry(carry,weekly,roster,snaps,actual)
    rq,rr=grade_rb_receiving(recv,weekly,roster,snaps,actual)
    tq,tr,ts=grade_trajectory(traj,weekly,roster,snaps,actual)

    cq.to_csv(a.out_dir/"week5_rb_carry_snap_graded_rows.csv",index=False)
    ct.to_csv(a.out_dir/"week5_rb_carry_snap_team_summary.csv",index=False)
    rq.to_csv(a.out_dir/"week5_rb_receiving_candidate_graded_rows.csv",index=False)
    tq.to_csv(a.out_dir/"week5_wrte_trajectory_graded_rows.csv",index=False)
    tr.to_csv(a.out_dir/"week5_wrte_trajectory_room_summary.csv",index=False)
    result={
        "version":"WEEK5_PROSPECTIVE_PLAYER_OPPORTUNITY_LOCK_POSTMORTEM_V1",
        "season":SEASON,"week":WEEK,
        "rb_carry_snap":cr,
        "rb_receiving":rr,
        "wrte_target_share_trajectory":ts,
        "week5_is_lock_number_one":True,
        "final_promotion_dispositions_authorized":False,
        "production_changed":False,
    }
    (a.out_dir/"week5_prospective_opportunity_locks_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
