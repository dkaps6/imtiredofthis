#!/usr/bin/env python3
"""Active-roster receiver room-state V1 diagnostic.

Discovery only: 2022-2023. Uses strict-prior player blend-4 target-share state
to compose WR/TE/RB_FB room shares. Fits no parameter and changes no production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.research.audit_wr1_current_state_anchor_v1 import (
    attach_strict_prior_identity,
    optional,
    read,
    state_for_week,
)

VERSION="ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1"
ROOMS=("WR","TE","RB_FB")
WR_POS={"WR","LWR","RWR","SWR"}


def room(value:object)->str:
    p="" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in WR_POS or p.startswith("WR"): return "WR"
    if p=="TE" or p.startswith("TE"): return "TE"
    if p in {"RB","HB","TB","FB"} or p.startswith("RB") or p.startswith("FB"): return "RB_FB"
    return "OTHER"


def metric(actual,pred):
    z=pd.DataFrame({"actual":pd.to_numeric(actual,errors="coerce"),
                    "pred":pd.to_numeric(pred,errors="coerce")}).dropna()
    if z.empty:return {"n":0}
    e=z.pred.to_numpy(float)-z.actual.to_numpy(float); ae=np.abs(e)
    return {
        "n":int(len(z)),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(e*e))),
        "bias":float(e.mean()),"median_ae":float(np.quantile(ae,.5)),
        "p90_ae":float(np.quantile(ae,.9)),
        "corr":float(np.corrcoef(z.actual,z.pred)[0,1])
            if len(z)>1 and z.actual.std()>0 and z.pred.std()>0 else None,
    }


def paired(actual,baseline,state):
    return {
        "baseline":metric(actual,baseline),
        "state":metric(actual,state),
    }


def actual_target_week(logs:pd.DataFrame,season:int,week:int):
    s=pd.to_numeric(logs.season,errors="coerce")
    w=pd.to_numeric(logs.week,errors="coerce")
    q=logs.loc[s.eq(season)&w.eq(week)].copy()
    if q.duplicated(["player_identity_key"]).any():
        raise RuntimeError(f"{season} W{week} duplicate target-week stable identity")
    q["team"]=q.team.map(canon_team)
    q["room"]=q.position.map(room)
    q["targets"]=pd.to_numeric(q.targets,errors="coerce").fillna(0.0)
    q["team_targets"]=pd.to_numeric(q.team_targets,errors="coerce")
    team=q.groupby("team",as_index=False).agg(
        actual_complete_team_targets=("team_targets","max")
    )
    player=q[["player_identity_key","team","targets","room"]].copy()
    return player,team


def summarize(rooms:pd.DataFrame)->dict:
    out={"samples":{}}
    samples=[("pooled",rooms)]+[(str(s),rooms.loc[rooms.season.eq(s)]) for s in (2022,2023)]
    for label,f in samples:
        block={"rooms":{}}
        for rr in ROOMS:
            q=f.loc[f.room.eq(rr)]
            block["rooms"][rr]={
                "composition":paired(q.actual_room_composition,q.baseline_room_composition,q.state_room_composition),
                "oracle_targets":paired(q.actual_room_targets,q.baseline_oracle_room_targets,q.state_oracle_room_targets),
                "absolute_team_share":paired(q.actual_room_team_share,q.baseline_room_mass,q.state_raw_room_mass),
            }
        for metric_name in ("composition","oracle_targets","absolute_team_share"):
            b=[block["rooms"][rr][metric_name]["baseline"]["mae"] for rr in ROOMS]
            s=[block["rooms"][rr][metric_name]["state"]["mae"] for rr in ROOMS]
            block[f"macro_{metric_name}_baseline_mae"]=float(np.mean(b))
            block[f"macro_{metric_name}_state_mae"]=float(np.mean(s))
        out["samples"][label]=block

    cov={}
    for rr in ROOMS:
        q=rooms.loc[rooms.room.eq(rr)]
        cov[rr]={
            "player_rows":int(q.player_rows.sum()),
            "state_rows":int(q.state_rows.sum()),
            "row_coverage":float(q.state_rows.sum()/q.player_rows.sum()) if q.player_rows.sum()>0 else None,
            "baseline_entitlement_mass":float(q.baseline_entitlement_mass.sum()),
            "state_eligible_entitlement_mass":float(q.state_eligible_entitlement_mass.sum()),
            "entitlement_weighted_coverage":float(q.state_eligible_entitlement_mass.sum()/q.baseline_entitlement_mass.sum())
                if q.baseline_entitlement_mass.sum()>0 else None,
            "team_games_with_fallback":int(q.has_fallback.sum()),
            "team_games_full_state":int((~q.has_fallback).sum()),
        }
    out["coverage"]=cov

    wr=rooms.loc[rooms.room.eq("WR") & rooms.wr1_state_gap.notna()].copy()
    if len(wr)>=3:
        out["wr1_interaction"]={
            "rows":int(len(wr)),
            "corr_state_gap_vs_needed_wr_room_mass":float(
                wr.wr1_state_gap.corr(wr.needed_wr_room_mass_correction,method="spearman")
            ),
            "mean_wr1_state_gap":float(wr.wr1_state_gap.mean()),
            "positive_wr1_state_gap_rate":float(wr.wr1_state_gap.gt(0).mean()),
        }
    else:
        out["wr1_interaction"]={"rows":int(len(wr)),"corr_state_gap_vs_needed_wr_room_mass":None}
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-2022",type=Path,required=True)
    ap.add_argument("--universe-2023",type=Path,required=True)
    ap.add_argument("--injuries",type=Path,required=True)
    ap.add_argument("--weather",type=Path,required=True)
    ap.add_argument("--weeks",default="2-18")
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    logs=read(a.player_logs,"player logs")
    team=read(a.team_weekly,"team weekly")
    sched=read(a.schedule,"schedule")
    inj=optional(a.injuries); weather=optional(a.weather)
    weeks=_parse_weeks(a.weeks)
    room_rows=[]; player_rows=[]; integrity=[]

    for season,prior,udir in [(2022,2021,a.universe_2022),(2023,2022,a.universe_2023)]:
        for week in weeks:
            universe=read(udir/f"{season}_week_{int(week):02d}.csv",f"{season} W{week} universe")
            bundle=build_historical_context_bundle(
                player_logs=logs,team_weekly=team,pregame_universe=universe,schedule=sched,
                season=season,week=int(week),prior_season=prior,
                injuries=_exact_week(inj,season,int(week)),
                weather=_exact_week(weather,season,int(week)))
            metrics=build_mc_predictions(bundle,iterations=20,seed=42+int(week))
            players=(metrics.sort_values(["event_id","team","player_clean_key"])
                     .drop_duplicates(["event_id","team","player_clean_key"],keep="last").copy())
            explicit,trace=materialize_target_entitlement(players)
            explicit["team"]=explicit.team.map(canon_team)
            explicit["room"]=explicit.position.map(room)
            explicit=explicit.loc[explicit.room.isin(ROOMS)].copy()
            explicit=attach_strict_prior_identity(explicit,logs,season,int(week))
            state=state_for_week(logs,season,int(week))
            explicit=explicit.merge(state,on="player_identity_key",how="left",validate="many_to_one")
            actual,team_actual=actual_target_week(logs,season,int(week))
            explicit=explicit.merge(
                actual[["player_identity_key","team","targets"]],
                on=["player_identity_key","team"],how="left",validate="many_to_one")
            explicit["targets"]=pd.to_numeric(explicit.targets,errors="coerce").fillna(0.0)
            explicit["entitlement_tgt_share"]=pd.to_numeric(explicit.entitlement_tgt_share,errors="raise")
            explicit["state_eligible"]=explicit.blend4_tgt_share.notna()
            explicit["state_player_share"]=np.where(
                explicit.state_eligible,
                pd.to_numeric(explicit.blend4_tgt_share,errors="coerce"),
                explicit.entitlement_tgt_share)
            if not np.isfinite(explicit.state_player_share.to_numpy(float)).all() or explicit.state_player_share.lt(0).any():
                raise RuntimeError(f"{season} W{week} invalid state player shares")

            team_actual=team_actual.set_index("team")
            for (game,tm),g0 in explicit.groupby(["event_id","team"],sort=True):
                tm=canon_team(tm); g=g0.copy()
                if tm not in team_actual.index: raise RuntimeError(f"missing actual team total {tm}")
                actual_complete=float(team_actual.loc[tm,"actual_complete_team_targets"])
                if not np.isfinite(actual_complete) or actual_complete<=0: continue
                base_total=float(g.entitlement_tgt_share.sum())
                state_total=float(g.state_player_share.sum())
                actual_modeled=float(g.targets.sum())
                if base_total<=0 or state_total<=0 or actual_modeled<=0: continue

                # M38 WR1 descriptive interaction only.
                wr=g.loc[g.room.eq("WR")].copy()
                wr1_gap=np.nan
                if not wr.empty:
                    ai=wr.entitlement_tgt_share.idxmax()
                    ar=wr.loc[ai]
                    if pd.notna(ar.blend4_tgt_share):
                        wr1_gap=float(ar.blend4_tgt_share-ar.entitlement_tgt_share)

                actual_room_map=g.groupby("room").targets.sum().to_dict()
                base_room_map=g.groupby("room").entitlement_tgt_share.sum().to_dict()
                state_room_map=g.groupby("room").state_player_share.sum().to_dict()

                actual_wr_team_share=float(actual_room_map.get("WR",0.0)/actual_complete)
                needed_wr=float(actual_wr_team_share-base_room_map.get("WR",0.0))

                for rr in ROOMS:
                    rg=g.loc[g.room.eq(rr)]
                    actual_room=float(actual_room_map.get(rr,0.0))
                    base_mass=float(base_room_map.get(rr,0.0))
                    state_mass=float(state_room_map.get(rr,0.0))
                    base_comp=base_mass/base_total
                    state_comp=state_mass/state_total
                    actual_comp=actual_room/actual_modeled
                    room_rows.append({
                        "season":season,"week":int(week),"event_id":str(game),"team":tm,"room":rr,
                        "baseline_room_mass":base_mass,"state_raw_room_mass":state_mass,
                        "baseline_room_composition":base_comp,"state_room_composition":state_comp,
                        "actual_room_targets":actual_room,"actual_modeled_receiver_targets":actual_modeled,
                        "actual_complete_team_targets":actual_complete,
                        "actual_room_composition":actual_comp,
                        "actual_room_team_share":actual_room/actual_complete,
                        "baseline_oracle_room_targets":actual_modeled*base_comp,
                        "state_oracle_room_targets":actual_modeled*state_comp,
                        "player_rows":int(len(rg)),
                        "state_rows":int(rg.state_eligible.sum()),
                        "baseline_entitlement_mass":float(rg.entitlement_tgt_share.sum()),
                        "state_eligible_entitlement_mass":float(rg.loc[rg.state_eligible,"entitlement_tgt_share"].sum()),
                        "has_fallback":bool((~rg.state_eligible).any()),
                        "wr1_state_gap":wr1_gap if rr=="WR" else np.nan,
                        "needed_wr_room_mass_correction":needed_wr if rr=="WR" else np.nan,
                    })
                pc=g[["season","week","event_id","team","player","player_clean_key","player_identity_key",
                      "position","room","entitlement_tgt_share","state_eligible","prior_games","current_games",
                      "prior_tgt_share","current_tgt_share","blend4_tgt_share","state_player_share","targets"]].copy()
                player_rows.append(pc)

            team_trace=trace.drop_duplicates(["event_id","team"])
            mass_gap=float(np.max(np.abs(
                pd.to_numeric(team_trace.modeled_player_sum,errors="raise").to_numpy(float)
                +pd.to_numeric(team_trace.residual_share,errors="raise").to_numpy(float)-1.0
            ))) if len(team_trace) else 0.0
            integrity.append({"season":season,"week":int(week),"entitlement_mass_gap":mass_gap})

    rooms=pd.DataFrame(room_rows)
    players=pd.concat(player_rows,ignore_index=True) if player_rows else pd.DataFrame()
    integ=pd.DataFrame(integrity)
    if rooms.empty: raise RuntimeError("room-state diagnostic produced zero rows")
    if not rooms.season.isin([2022,2023]).all(): raise RuntimeError("forbidden confirmation season entered")
    if float(integ.entitlement_mass_gap.max())>1e-10: raise RuntimeError("entitlement conservation failed")

    summary=summarize(rooms)
    p=summary["samples"]["pooled"]
    s22=summary["samples"]["2022"]; s23=summary["samples"]["2023"]
    comp_base=p["macro_composition_baseline_mae"]; comp_state=p["macro_composition_state_mae"]
    oracle_base=p["macro_oracle_targets_baseline_mae"]; oracle_state=p["macro_oracle_targets_state_mae"]
    gates={
        "pooled_macro_composition_improves":comp_state<comp_base,
        "pooled_wr_composition_improves":
            p["rooms"]["WR"]["composition"]["state"]["mae"]<p["rooms"]["WR"]["composition"]["baseline"]["mae"],
        "pooled_te_composition_nonworse":
            p["rooms"]["TE"]["composition"]["state"]["mae"]<=p["rooms"]["TE"]["composition"]["baseline"]["mae"],
        "pooled_rbfb_composition_nonworse":
            p["rooms"]["RB_FB"]["composition"]["state"]["mae"]<=p["rooms"]["RB_FB"]["composition"]["baseline"]["mae"],
        "wr_composition_improves_2022":
            s22["rooms"]["WR"]["composition"]["state"]["mae"]<s22["rooms"]["WR"]["composition"]["baseline"]["mae"],
        "wr_composition_improves_2023":
            s23["rooms"]["WR"]["composition"]["state"]["mae"]<s23["rooms"]["WR"]["composition"]["baseline"]["mae"],
        "pooled_macro_oracle_targets_improves":oracle_state<oracle_base,
        "pooled_wr_oracle_targets_improves":
            p["rooms"]["WR"]["oracle_targets"]["state"]["mae"]<p["rooms"]["WR"]["oracle_targets"]["baseline"]["mae"],
        "sportsbook_inputs_zero":True,
        "parameters_fit_zero":True,
        "candidate_variants_zero":True,
    }
    supported=all(gates.values())
    disposition=("ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_SUPPORTED"
                 if supported else "ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_NOT_SUPPORTED")
    payload={
        "version":VERSION,"disposition":disposition,"supported":supported,
        "discovery_seasons":[2022,2023],"confirmation_seasons_inspected":[],
        "production_changed":False,"sportsbook_inputs_used":0,"parameters_fit":0,
        "candidate_variants_scored":0,"room_rows":int(len(rooms)),"player_rows":int(len(players)),
        "max_entitlement_mass_gap":float(integ.entitlement_mass_gap.max()),
        "summary":summary,"gates":gates,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rooms.to_csv(a.out_dir/"room_state_detail.csv",index=False)
    players.to_csv(a.out_dir/"player_state_detail.csv",index=False)
    integ.to_csv(a.out_dir/"integrity.csv",index=False)
    (a.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    lines=["# Active-Roster Receiver Room State V1","",f"Disposition: **{disposition}**","",
           "## Pooled discovery","",
           f"- macro composition MAE: {comp_base:.6f} -> {comp_state:.6f}",
           f"- macro oracle room-target MAE: {oracle_base:.6f} -> {oracle_state:.6f}",""]
    for rr in ROOMS:
        x=p["rooms"][rr]
        lines.append(
            f"- {rr} composition MAE: {x['composition']['baseline']['mae']:.6f} -> "
            f"{x['composition']['state']['mae']:.6f}; oracle target MAE "
            f"{x['oracle_targets']['baseline']['mae']:.6f} -> {x['oracle_targets']['state']['mae']:.6f}"
        )
    lines+=["","## Frozen gates",""]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in gates.items()]
    (a.out_dir/"RESULT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
