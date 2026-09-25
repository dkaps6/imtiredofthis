#!/usr/bin/env python3
"""Diagnostic of immutable M38 WR1 anchor versus strict-prior current-season state.

Discovery only: 2022-2023. No candidate is qualified and 2024-2025 are not read.
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
from scripts.research.evaluate_receiver_targetable_dropback_v1_team_calibration import (
    build_team_actual_history,
    strict_prior_rate,
)
from scripts.simulation_c2_qb_candidate import simulate_with_states

VERSION="WR1_CURRENT_STATE_ANCHOR_DIAGNOSTIC_V1"
PSEUDO_PRIOR_GAMES=4.0
WR_POS={"WR","LWR","RWR","SWR"}


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        return pd.DataFrame()
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def score(actual, pred):
    z=pd.DataFrame({"actual":pd.to_numeric(actual,errors="coerce"),"pred":pd.to_numeric(pred,errors="coerce")}).dropna()
    if z.empty:return {"n":0}
    e=z["pred"].to_numpy(float)-z["actual"].to_numpy(float); ae=np.abs(e)
    return {
        "n":int(len(z)),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(e*e))),
        "bias":float(e.mean()),"median_ae":float(np.quantile(ae,.5)),
        "p90_ae":float(np.quantile(ae,.9)),
        "corr":float(np.corrcoef(z["actual"],z["pred"])[0,1])
            if len(z)>1 and z["actual"].std()>0 and z["pred"].std()>0 else None,
    }


def corr(x,y,method="spearman"):
    z=pd.DataFrame({"x":pd.to_numeric(x,errors="coerce"),"y":pd.to_numeric(y,errors="coerce")}).dropna()
    if len(z)<3 or z.x.nunique()<2 or z.y.nunique()<2:return None
    return float(z.x.corr(z.y,method=method))


def aggregate_tgt_state(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=["player_identity_key","games","targets","team_targets","tgt_share"])
    req={"player_identity_key","week","targets","team_targets"}
    miss=req-set(logs.columns)
    if miss: raise RuntimeError(f"state logs missing {sorted(miss)}")
    g=logs.groupby("player_identity_key",dropna=False).agg(
        games=("week","nunique"),targets=("targets","sum"),team_targets=("team_targets","sum")
    ).reset_index()
    g["tgt_share"]=np.where(g["team_targets"]>0,g["targets"]/g["team_targets"],np.nan)
    return g


def state_for_week(logs: pd.DataFrame, season:int, week:int) -> pd.DataFrame:
    s=pd.to_numeric(logs["season"],errors="coerce")
    w=pd.to_numeric(logs["week"],errors="coerce")
    prior=aggregate_tgt_state(logs.loc[s.eq(season-1)].copy()).rename(columns={
        "games":"prior_games","tgt_share":"prior_tgt_share"})
    current=aggregate_tgt_state(logs.loc[s.eq(season)&w.lt(week)].copy()).rename(columns={
        "games":"current_games","tgt_share":"current_tgt_share"})
    x=prior[["player_identity_key","prior_games","prior_tgt_share"]].merge(
        current[["player_identity_key","current_games","current_tgt_share"]],
        on="player_identity_key",how="inner",validate="one_to_one")
    for c in ("prior_games","current_games","prior_tgt_share","current_tgt_share"):
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna()
    x=x.loc[x.prior_games.ge(1)&x.current_games.ge(1)].copy()
    x["w_current"]=x["current_games"]/(x["current_games"]+PSEUDO_PRIOR_GAMES)
    x["blend4_tgt_share"]=(1-x["w_current"])*x["prior_tgt_share"]+x["w_current"]*x["current_tgt_share"]
    return x


def actual_week(logs: pd.DataFrame, season:int, week:int) -> pd.DataFrame:
    s=pd.to_numeric(logs["season"],errors="coerce")
    w=pd.to_numeric(logs["week"],errors="coerce")
    q=logs.loc[s.eq(season)&w.eq(week)].copy()
    if q.duplicated(["player_identity_key"]).any():
        raise RuntimeError(f"{season} W{week} duplicate stable identity")
    q["team"]=q["team"].map(canon_team)
    q["position"]=q["position"].astype(str).str.upper().str.strip()
    q["targets"]=pd.to_numeric(q["targets"],errors="coerce").fillna(0.0)
    q["team_targets"]=pd.to_numeric(q["team_targets"],errors="coerce")
    q["actual_tgt_share"]=np.where(q["team_targets"].gt(0),q["targets"]/q["team_targets"],np.nan)
    wr=q.loc[q["position"].isin(WR_POS)].copy()
    room=wr.groupby("team",as_index=False)["targets"].sum().rename(columns={"targets":"actual_wr_room_targets"})
    return q,room


def sign_agreement(a,b):
    x=pd.to_numeric(a,errors="coerce"); y=pd.to_numeric(b,errors="coerce")
    m=x.notna()&y.notna()&x.ne(0)&y.ne(0)
    return (int(m.sum()), float(np.sign(x[m]).eq(np.sign(y[m])).mean()) if m.any() else None)


def summarize(frame:pd.DataFrame)->dict:
    out={}
    for label,q in [("pooled",frame)]+[(str(s),frame.loc[frame.season.eq(s)]) for s in (2022,2023)]:
        nsg,sg=sign_agreement(q.state_gap,q.needed_correction)
        room_q=q.loc[q.room_state_complete.eq(True)].copy()
        rn,rsg=sign_agreement(room_q.state_room_gap,room_q.needed_room_correction)
        out[label]={
            "rows":int(len(q)),
            "m38_team_share":score(q.actual_tgt_share,q.m38_anchor_share),
            "prior_team_share":score(q.actual_tgt_share,q.prior_tgt_share),
            "current_team_share":score(q.actual_tgt_share,q.current_tgt_share),
            "blend4_team_share":score(q.actual_tgt_share,q.blend4_tgt_share),
            "state_gap_vs_needed_spearman":corr(q.state_gap,q.needed_correction,"spearman"),
            "state_gap_vs_needed_pearson":corr(q.state_gap,q.needed_correction,"pearson"),
            "state_gap_sign_n":nsg,
            "state_gap_sign_agreement":sg,
            "room_complete_rows":int(len(room_q)),
            "m38_room_share":score(room_q.actual_wr1_room_share,room_q.m38_wr1_room_share),
            "state_room_share":score(room_q.actual_wr1_room_share,room_q.state_wr1_room_share),
            "state_room_gap_vs_needed_spearman":corr(room_q.state_room_gap,room_q.needed_room_correction,"spearman"),
            "state_room_gap_sign_n":rn,
            "state_room_gap_sign_agreement":rsg,
        }
    out["targetable_cohorts"]={}
    for label,val in [("helps",True),("hurts",False)]:
        q=frame.loc[frame.targetable_helps.eq(val)]
        nsg,sg=sign_agreement(q.state_gap,q.needed_correction)
        out["targetable_cohorts"][label]={
            "rows":int(len(q)),
            "mean_state_gap":float(q.state_gap.mean()) if len(q) else None,
            "median_state_gap":float(q.state_gap.median()) if len(q) else None,
            "positive_state_gap_rate":float(q.state_gap.gt(0).mean()) if len(q) else None,
            "state_gap_vs_needed_spearman":corr(q.state_gap,q.needed_correction,"spearman"),
            "sign_n":nsg,"sign_agreement":sg,
        }
    out["game_buckets"]={}
    def bucket(n):
        if n<=4:return str(int(n))
        if n<=8:return "5-8"
        return "9+"
    z=frame.copy(); z["current_games_bucket"]=z.current_games.astype(int).map(bucket)
    for b,q in z.groupby("current_games_bucket"):
        out["game_buckets"][str(b)]={
            "rows":int(len(q)),
            "m38_mae":score(q.actual_tgt_share,q.m38_anchor_share)["mae"],
            "blend4_mae":score(q.actual_tgt_share,q.blend4_tgt_share)["mae"],
            "state_gap_vs_needed_spearman":corr(q.state_gap,q.needed_correction,"spearman"),
        }
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
    ap.add_argument("--iterations",type=int,default=2000)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    logs=read(a.player_logs,"player logs")
    team=read(a.team_weekly,"team weekly")
    sched=read(a.schedule,"schedule")
    inj=optional(a.injuries); weather=optional(a.weather)
    rate_history=build_team_actual_history(logs,team)
    weeks=_parse_weeks(a.weeks)
    rows=[]; integrity=[]

    for season,prior,udir in [(2022,2021,a.universe_2022),(2023,2022,a.universe_2023)]:
        for week in weeks:
            upath=udir/f"{season}_week_{int(week):02d}.csv"
            universe=read(upath,f"{season} W{week} universe")
            bundle=build_historical_context_bundle(
                player_logs=logs,team_weekly=team,pregame_universe=universe,schedule=sched,
                season=season,week=int(week),prior_season=prior,
                injuries=_exact_week(inj,season,int(week)),weather=_exact_week(weather,season,int(week)))
            metrics=build_mc_predictions(bundle,iterations=20,seed=42+int(week))
            players=(metrics.sort_values(["event_id","team","player_clean_key"])
                     .drop_duplicates(["event_id","team","player_clean_key"],keep="last").copy())
            explicit,trace=materialize_target_entitlement(players)
            team_trace=trace.drop_duplicates(["event_id","team"]).copy()
            mass_gap=float(np.max(np.abs(
                pd.to_numeric(team_trace["modeled_player_sum"],errors="raise").to_numpy(float)
                + pd.to_numeric(team_trace["residual_share"],errors="raise").to_numpy(float)
                - 1.0
            ))) if len(team_trace) else 0.0
            if mass_gap>1e-10:
                raise RuntimeError(f"explicit entitlement mass drift {mass_gap}")
            explicit["team"]=explicit["team"].map(canon_team)
            explicit["position"]=explicit["position"].astype(str).str.upper().str.strip()
            wr=explicit.loc[explicit.position.isin(WR_POS)].copy()
            if wr.empty: raise RuntimeError(f"{season} W{week} zero WR rows")
            if "player_identity_key" not in wr.columns:
                raise RuntimeError("explicit entitlement missing stable identity")
            state=state_for_week(logs,season,int(week))
            actual,room_actual=actual_week(logs,season,int(week))
            actcols=["player_identity_key","team","targets","actual_tgt_share"]
            wr=wr.merge(state,on="player_identity_key",how="left",validate="one_to_one")
            wr=wr.merge(actual[actcols],on=["player_identity_key","team"],how="left",validate="one_to_one")
            wr=wr.merge(room_actual,on="team",how="left",validate="many_to_one")
            wr["entitlement_tgt_share"]=pd.to_numeric(wr.entitlement_tgt_share,errors="coerce")
            wr["actual_wr_room_targets"]=pd.to_numeric(wr.actual_wr_room_targets,errors="coerce").fillna(0.0)

            sim=simulate_with_states(explicit,iterations=int(a.iterations),seed=42+int(week))
            for (game,tm),g0 in wr.groupby(["event_id","team"],sort=True):
                g=g0.copy()
                ent=g.entitlement_tgt_share
                if ent.isna().any() or ent.lt(0).any(): raise RuntimeError("invalid WR entitlement")
                idx=ent.idxmax(); anchor=g.loc[idx]
                if pd.isna(anchor.get("blend4_tgt_share")) or pd.isna(anchor.get("actual_tgt_share")):
                    continue
                room_mass=float(ent.sum())
                if room_mass<=0: continue
                db=float(np.asarray(sim.team_states[(str(game),canon_team(tm),"pass_att")],float).mean())
                rt_info=strict_prior_rate(rate_history,season,int(week),canon_team(tm),prior)
                rt=float(rt_info["targetable_dropback_rate"])
                base_targets=db*float(anchor.entitlement_tgt_share)
                targ_targets=db*rt*float(anchor.entitlement_tgt_share)
                actual_targets=float(anchor.targets)
                ba=abs(base_targets-actual_targets); ca=abs(targ_targets-actual_targets)

                complete_state=g[["blend4_tgt_share"]].notna().all(axis=1).all()
                state_room=np.nan
                actual_room_share=np.nan
                if complete_state and float(g.blend4_tgt_share.sum())>0 and float(anchor.actual_wr_room_targets)>0:
                    state_room=float(anchor.blend4_tgt_share/g.blend4_tgt_share.sum())
                    actual_room_share=float(actual_targets/float(anchor.actual_wr_room_targets))
                m38_room=float(anchor.entitlement_tgt_share/room_mass)
                rows.append({
                    "season":season,"week":int(week),"event_id":str(game),"team":canon_team(tm),
                    "player":str(anchor.player),"player_clean_key":str(anchor.player_clean_key),
                    "player_identity_key":str(anchor.player_identity_key),
                    "m38_anchor_share":float(anchor.entitlement_tgt_share),"wr_room_mass":room_mass,
                    "m38_wr1_room_share":m38_room,
                    "prior_games":int(anchor.prior_games),"current_games":int(anchor.current_games),
                    "prior_tgt_share":float(anchor.prior_tgt_share),"current_tgt_share":float(anchor.current_tgt_share),
                    "blend4_tgt_share":float(anchor.blend4_tgt_share),"w_current":float(anchor.w_current),
                    "actual_targets":actual_targets,"actual_tgt_share":float(anchor.actual_tgt_share),
                    "actual_wr_room_targets":float(anchor.actual_wr_room_targets),
                    "room_state_complete":bool(complete_state and np.isfinite(state_room) and np.isfinite(actual_room_share)),
                    "state_wr1_room_share":state_room,"actual_wr1_room_share":actual_room_share,
                    "state_gap":float(anchor.blend4_tgt_share-anchor.entitlement_tgt_share),
                    "needed_correction":float(anchor.actual_tgt_share-anchor.entitlement_tgt_share),
                    "state_room_gap":float(state_room-m38_room) if np.isfinite(state_room) else np.nan,
                    "needed_room_correction":float(actual_room_share-m38_room) if np.isfinite(actual_room_share) else np.nan,
                    "targetable_rate":rt,"targetable_prior_history_games":int(rt_info["prior_history_games"]),
                    "baseline_pred_targets":base_targets,"targetable_pred_targets":targ_targets,
                    "baseline_target_abs_error":ba,"targetable_target_abs_error":ca,
                    "targetable_helps":bool(ca<ba-1e-12),
                })
            integrity.append({"season":season,"week":int(week),
                              "entitlement_mass_identity_gap":mass_gap})

    frame=pd.DataFrame(rows)
    if frame.empty: raise RuntimeError("diagnostic produced zero WR1 rows")
    if not frame.season.isin([2022,2023]).all(): raise RuntimeError("forbidden season entered diagnostic")
    summary=summarize(frame)
    pooled=summary["pooled"]; hurt=summary["targetable_cohorts"]["hurts"]
    rule_support=bool(
        pooled["blend4_team_share"]["mae"] < pooled["m38_team_share"]["mae"]
        and (pooled["state_gap_vs_needed_spearman"] or -1)>0
        and (pooled["state_gap_sign_agreement"] or 0)>.50
        and pooled["room_complete_rows"]>0
        and pooled["state_room_share"]["mae"] < pooled["m38_room_share"]["mae"]
        and (hurt["state_gap_vs_needed_spearman"] or -1)>0
    )
    payload={
        "version":VERSION,"disposition":"WR1_CURRENT_STATE_ANCHOR_DIAGNOSTIC_V1_COMPLETE",
        "discovery_seasons":[2022,2023],"confirmation_seasons_inspected":[],
        "production_changed":False,"sportsbook_inputs_used":0,"parameters_fit":0,
        "candidate_variants_scored":0,"player_rows":int(len(frame)),
        "summary":summary,"candidate_justified":rule_support,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(a.out_dir/"wr1_state_detail.csv",index=False)
    pd.DataFrame(integrity).to_csv(a.out_dir/"integrity.csv",index=False)
    (a.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    lines=[
        "# WR1 Current-State Anchor Diagnostic V1","",
        f"Candidate justified by frozen diagnostic rule: **{rule_support}**","",
        "## Pooled discovery 2022-2023","",
        f"- M38 team-share MAE: {pooled['m38_team_share']['mae']:.6f}",
        f"- Blend-4 team-share MAE: {pooled['blend4_team_share']['mae']:.6f}",
        f"- state-gap vs needed Spearman: {pooled['state_gap_vs_needed_spearman']}",
        f"- sign agreement: {pooled['state_gap_sign_agreement']}",
        f"- M38 room-share MAE: {pooled['m38_room_share']['mae']:.6f}",
        f"- state-normalized room-share MAE: {pooled['state_room_share']['mae']:.6f}",
        "",
        "## Targetable-hurts cohort","",
        f"- rows: {hurt['rows']}",
        f"- mean state gap: {hurt['mean_state_gap']}",
        f"- positive state-gap rate: {hurt['positive_state_gap_rate']}",
        f"- state-gap vs needed Spearman: {hurt['state_gap_vs_needed_spearman']}",
    ]
    (a.out_dir/"RESULT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
