#!/usr/bin/env python3
"""Prospective lock for historically-passed joint opportunity allocation V1.

Locks RB-carry and WR-target joint Dirichlet room parameters before target-week
outcomes. Research-only; production simulation is untouched.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research.joint_opportunity_allocation_dirichlet_v1 import (
    beta_intraclass_rho,
    room_rho,
)
from scripts.research.score_joint_opportunity_allocation_dirichlet_v1 import (
    build_week_metrics,
    canonical_player_probabilities,
    clean_key,
    read_csv,
)

VERSION="PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW"
CELLS=("RB_CARRY_SHARE","WR_TARGET_SHARE")
MEAN_TOL=1e-12


def _num(s):
    return pd.to_numeric(s,errors="coerce")


def _bool(s:pd.Series)->pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes"})


def validate_authorities(
    *,
    season:int,
    week:int,
    player_state:dict[str,Any],
    volatility_result:dict[str,Any],
)->None:
    if player_state.get("version")!="PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1":
        raise RuntimeError("wrong player-state authority")
    if int(player_state.get("season",-1))!=season or int(player_state.get("target_week",-1))!=week:
        raise RuntimeError("player-state target mismatch")
    if player_state.get("disposition")!="PLAYER_STATE_FUTURE_WEEK_READY":
        raise RuntimeError(f"player state not READY: {player_state.get('disposition')}")
    if not bool(player_state.get("target_week_roster_source_ready",False)):
        raise RuntimeError("exact target-week roster authority unavailable")
    if int(player_state.get("roster_source_week",-1))!=week:
        raise RuntimeError("roster source is not exact target week")
    if not bool((player_state.get("prior_week_source_completion") or {}).get("complete",False)):
        raise RuntimeError("prior-week stats/snaps incomplete")
    if int(player_state.get("chronology_violations",-1))!=0:
        raise RuntimeError("player-state chronology violation")

    if volatility_result.get("version")!="PLAYER_OPPORTUNITY_SHARE_VOLATILITY_FUTURE_WEEK_LOCK_V1":
        raise RuntimeError("wrong volatility lock authority")
    if int(volatility_result.get("season",-1))!=season or int(volatility_result.get("target_week",-1))!=week:
        raise RuntimeError("volatility target mismatch")
    if volatility_result.get("status")!=f"WEEK{week}_OPPORTUNITY_SHARE_VOLATILITY_LOCK_FROZEN":
        raise RuntimeError("volatility lock not frozen")
    if int(volatility_result.get("chronology_violations",-1))!=0:
        raise RuntimeError("volatility chronology violation")
    if int(volatility_result.get("target_week_outcomes_read",-1))!=0:
        raise RuntimeError("volatility lock read target-week outcomes")
    if volatility_result.get("allocation_uncertainty_candidate_cells")!=list(CELLS):
        raise RuntimeError("volatility candidate-cell authority drift")
    if bool(volatility_result.get("rb_target_betabinomial_authorized",True)):
        raise RuntimeError("RB target uncertainty unexpectedly authorized")
    if bool(volatility_result.get("te_target_betabinomial_authorized",True)):
        raise RuntimeError("TE target uncertainty unexpectedly authorized")


def prepare_lock_rows(rows:pd.DataFrame,season:int,week:int)->pd.DataFrame:
    x=rows.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={
        "season","target_week","team","player","position","player_identity_key",
        "prior_target_share_mean","prior_target_excess_share_var",
        "target_share_source_max_week","wr_target_betabinomial_eligible",
        "prior_rush_share_mean","prior_rush_excess_share_var",
        "rush_share_source_max_week","rb_carry_betabinomial_eligible",
    }
    miss=req-set(x.columns)
    if miss:
        raise RuntimeError(f"volatility rows missing {sorted(miss)}")
    x=x.loc[
        _num(x["season"]).eq(season)&_num(x["target_week"]).eq(week)
    ].copy()
    x["team"]=x["team"].map(canon_team)
    x["player_clean_key"]=x["player"].map(clean_key)
    if x.duplicated(["team","player_identity_key"]).any():
        raise RuntimeError("duplicate target-week volatility identity")
    return x


def lock_shadow(
    *,
    season:int,
    week:int,
    volatility_rows:pd.DataFrame,
    volatility_result:dict[str,Any],
    player_state:dict[str,Any],
    team_weekly:pd.DataFrame,
    player_logs:pd.DataFrame,
    schedule:pd.DataFrame,
    universe:pd.DataFrame,
)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
    validate_authorities(
        season=season,week=week,
        player_state=player_state,volatility_result=volatility_result,
    )
    v=prepare_lock_rows(volatility_rows,season,week)

    metrics=build_week_metrics(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=season,
        week=week,
    )

    player_rows=[]
    room_rows=[]
    q_match=[]
    chronology=0
    max_share_gap=0.0
    locked_cells={cell:0 for cell in CELLS}
    locked_players={cell:set() for cell in CELLS}

    for cell in CELLS:
        probs=canonical_player_probabilities(metrics,cell)
        source=v.copy()

        if cell=="RB_CARRY_SHARE":
            source=source.loc[source["position"].astype(str).str.upper().eq("RB")].copy()
            source["eligible_flag"]=_bool(source["rb_carry_betabinomial_eligible"])
            source["prior_mean"]=_num(source["prior_rush_share_mean"])
            source["excess_var"]=_num(source["prior_rush_excess_share_var"])
            source["source_max_week"]=_num(source["rush_share_source_max_week"])
        else:
            source=source.loc[source["position"].astype(str).str.upper().eq("WR")].copy()
            source["eligible_flag"]=_bool(source["wr_target_betabinomial_eligible"])
            source["prior_mean"]=_num(source["prior_target_share_mean"])
            source["excess_var"]=_num(source["prior_target_excess_share_var"])
            source["source_max_week"]=_num(source["target_share_source_max_week"])

        source=source.merge(
            probs[["team","player_clean_key","q"]],
            on=["team","player_clean_key"],
            how="left",
            validate="one_to_one",
        )
        q_match.append(float(source["q"].notna().mean()) if len(source) else 0.0)
        chronology+=int(source["source_max_week"].dropna().ge(week).sum())

        rhos=[]
        for r in source.itertuples(index=False):
            rho=beta_intraclass_rho(float(r.prior_mean),float(r.excess_var)) if (
                pd.notna(r.prior_mean) and pd.notna(r.excess_var)
            ) else None
            rhos.append(rho)
        source["rho_i"]=rhos
        source["joint_eligible"]=(
            source["eligible_flag"]
            & source["rho_i"].notna()
            & _num(source["q"]).gt(0)
            & _num(source["q"]).lt(1)
            & source["source_max_week"].lt(week)
        )

        for team,g in source.groupby("team",sort=True):
            e=g.loc[g["joint_eligible"]].copy()
            if e.empty:
                continue
            q=e["q"].to_numpy(float)
            rho=e["rho_i"].to_numpy(float)
            complement=1.0-float(q.sum())
            if not (0.0<complement<1.0):
                continue
            rr=room_rho(
                np.append(q,complement),
                np.append(np.ones(len(q),dtype=bool),False),
                np.append(rho,np.nan),
            )
            if rr is None:
                continue
            rho_room,kappa,_=rr
            alpha=q*kappa
            alpha_c=complement*kappa
            alpha0=float(alpha.sum()+alpha_c)
            expected_q=alpha/alpha0
            gap=float(np.max(np.abs(expected_q-q))) if len(q) else 0.0
            comp_gap=abs(alpha_c/alpha0-complement)
            max_share_gap=max(max_share_gap,gap,comp_gap)
            if max(gap,comp_gap)>MEAN_TOL:
                raise RuntimeError("joint shadow mean identity violated")

            room_rows.append({
                "season":season,"target_week":week,"team":team,"cell":cell,
                "eligible_players":int(len(e)),
                "eligible_probability":float(q.sum()),
                "complement_probability":float(complement),
                "rho_room":float(rho_room),
                "kappa":float(kappa),
                "alpha0":alpha0,
                "complement_alpha":float(alpha_c),
                "probability_mass_gap":abs(float(q.sum())+complement-1.0),
                "max_expected_share_gap":max(gap,comp_gap),
            })
            locked_cells[cell]+=1
            for j,(_,r) in enumerate(e.iterrows()):
                player_rows.append({
                    "season":season,"target_week":week,"team":team,"cell":cell,
                    "player":str(r["player"]),
                    "player_identity_key":str(r["player_identity_key"]),
                    "player_clean_key":str(r["player_clean_key"]),
                    "canonical_q":float(q[j]),
                    "prior_share_mean":float(r["prior_mean"]),
                    "excess_share_var":float(r["excess_var"]),
                    "rho_i":float(r["rho_i"]),
                    "rho_room":float(rho_room),
                    "kappa":float(kappa),
                    "alpha_i":float(alpha[j]),
                    "source_max_week":int(r["source_max_week"]),
                })
                locked_players[cell].add(str(r["player_identity_key"]))

    rooms=pd.DataFrame(room_rows)
    players=pd.DataFrame(player_rows)
    if rooms.empty or players.empty:
        raise RuntimeError("zero joint prospective shadow rooms")
    if chronology!=0:
        raise RuntimeError("joint prospective shadow chronology violation")

    result={
        "version":VERSION,
        "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        "season":season,"target_week":week,
        "status":f"WEEK{week}_JOINT_ALLOCATION_DIRICHLET_SHADOW_FROZEN",
        "authorized_cells":list(CELLS),
        "rooms":int(len(rooms)),
        "player_rows":int(len(players)),
        "rb_carry_rooms":int(locked_cells["RB_CARRY_SHARE"]),
        "wr_target_rooms":int(locked_cells["WR_TARGET_SHARE"]),
        "rb_carry_distinct_players":int(len(locked_players["RB_CARRY_SHARE"])),
        "wr_target_distinct_players":int(len(locked_players["WR_TARGET_SHARE"])),
        "min_canonical_q_match_rate":min(q_match) if q_match else 0.0,
        "max_expected_share_gap":float(max_share_gap),
        "chronology_violations":0,
        "target_week_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "fitted_coefficients":0,
        "threshold_searches":0,
        "production_changed":False,
    }
    if result["min_canonical_q_match_rate"]<.95:
        raise RuntimeError(f"canonical q identity match too low: {result['min_canonical_q_match_rate']}")
    return rooms,players,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--player-state-result",type=Path,required=True)
    ap.add_argument("--volatility-rows",type=Path,required=True)
    ap.add_argument("--volatility-result",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    state=json.loads(a.player_state_result.read_text())
    vres=json.loads(a.volatility_result.read_text())
    vrows=read_csv(a.volatility_rows,"volatility rows")
    team=read_csv(a.team_weekly,"team weekly")
    logs=read_csv(a.player_logs,"player logs")
    schedule=read_csv(a.schedule,"schedule")
    universe=read_csv(a.universe,"pregame universe")
    for frame in (team,logs,schedule):
        frame["season"]=_num(frame["season"]).astype("Int64")
        frame["week"]=_num(frame["week"]).astype("Int64")
    team["team"]=team["team"].map(canon_team)
    logs["team"]=logs["team"].map(canon_team)
    if "opponent" in logs.columns:
        logs["opponent"]=logs["opponent"].map(canon_team)
    schedule["team"]=schedule["team"].map(canon_team)
    schedule["opponent"]=schedule["opponent"].map(canon_team)

    rooms,players,result=lock_shadow(
        season=a.season,week=a.week,
        volatility_rows=vrows,volatility_result=vres,player_state=state,
        team_weekly=team,player_logs=logs,schedule=schedule,universe=universe,
    )
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rooms.to_csv(a.out_dir/f"joint_allocation_dirichlet_week{a.week}_rooms.csv",index=False,float_format="%.12f")
    players.to_csv(a.out_dir/f"joint_allocation_dirichlet_week{a.week}_players.csv",index=False,float_format="%.12f")
    (a.out_dir/f"joint_allocation_dirichlet_week{a.week}_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
