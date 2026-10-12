#!/usr/bin/env python3
"""Score frozen individual RB run-location x opponent-location matchup candidate.

Frozen in:
docs/research/RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_PREDICTIVE_V1_FREEZE.md

No fitted coefficient, threshold search, sportsbook input, 2026 outcomes, or
production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.simulation_v2 import lookup, simulate
from scripts.research.score_player_role_matchup_depth_predictive_v1 import (
    read_csv,
    key,
    num,
    check_forbidden,
    bool_col,
    base_components,
    cluster_bootstrap,
    attach_game_id,
    verify_baseline,
)

VERSION="RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_PREDICTIVE_V1"
FREEZE="docs/research/RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_PREDICTIVE_V1_FREEZE.md"
CANDIDATE="RB-INDIVIDUAL-RUN-LOCATION-YPC-V1"
TARGET_SEASONS=(2024,2025)
TARGET_WEEKS=tuple(range(2,19))
ITERATIONS=2000
BOOT_REPS=5000
BOOT_SEED=20261012
MULT_FLOOR=0.50
MULT_CEIL=1.80
LOCATIONS=("left","middle","right")
RB_POS={"RB","HB","FB"}


def prepare_source(source:pd.DataFrame)->pd.DataFrame:
    req={
        "season","week","team","opponent","player","gsis_id",
        "matchup_state_ready","defense_state_ready",
        "player_share_left","player_share_middle","player_share_right",
        "def_rushes_left","def_rushes_middle","def_rushes_right",
        "def_ypc_left","def_ypc_middle","def_ypc_right",
    }
    miss=req-set(source.columns)
    if miss:
        raise RuntimeError(f"source artifact missing columns: {sorted(miss)}")
    x=source.copy()
    x["season"]=num(x["season"]).astype(int)
    x["week"]=num(x["week"]).astype(int)
    x=x.loc[x["season"].isin(TARGET_SEASONS)&x["week"].isin(TARGET_WEEKS)].copy()
    x["team"]=x["team"].map(canon_team)
    x["opponent"]=x["opponent"].map(canon_team)
    x["player_clean_key"]=x["player"].map(key)
    x["matchup_state_ready"]=bool_col(x["matchup_state_ready"])
    x["defense_state_ready"]=bool_col(x["defense_state_ready"])
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        d=x.loc[
            x.duplicated(["season","week","team","player_clean_key"],keep=False),
            ["season","week","team","player","gsis_id"],
        ].head(20)
        raise RuntimeError(f"duplicate source player keys: {d.to_dict('records')}")
    return x


def build_league_location_ypc(source:pd.DataFrame)->pd.DataFrame:
    cols=[]
    for loc in LOCATIONS:
        cols.extend([f"def_rushes_{loc}",f"def_ypc_{loc}"])
    d=source.loc[source["defense_state_ready"],["season","week","opponent",*cols]].copy()
    for c in cols:
        d[c]=num(d[c])
    grouped=[]
    for (season,week,opp),g in d.groupby(["season","week","opponent"],sort=True):
        rec={"season":int(season),"week":int(week),"defense":str(opp)}
        valid=True
        for loc in LOCATIONS:
            n=num(g[f"def_rushes_{loc}"]).dropna().unique()
            y=num(g[f"def_ypc_{loc}"]).dropna().unique()
            if len(n)!=1 or len(y)!=1:
                valid=False
                break
            rec[f"rushes_{loc}"]=float(n[0])
            rec[f"ypc_{loc}"]=float(y[0])
        if valid:
            grouped.append(rec)
    z=pd.DataFrame(grouped)
    if z.empty:
        raise RuntimeError("zero unique defense-week run-location states")

    rows=[]
    for (season,week),g in z.groupby(["season","week"],sort=True):
        rec={"season":int(season),"week":int(week)}
        for loc in LOCATIONS:
            n=num(g[f"rushes_{loc}"])
            y=num(g[f"ypc_{loc}"])
            ok=n.gt(0)&n.notna()&y.notna()
            den=float(n.loc[ok].sum())
            rec[f"league_ypc_{loc}"]=(
                float((n.loc[ok]*y.loc[ok]).sum()/den) if den>0 else np.nan
            )
        rows.append(rec)
    out=pd.DataFrame(rows)
    expected=len(TARGET_SEASONS)*len(TARGET_WEEKS)
    if len(out)!=expected:
        raise RuntimeError(f"league run-location baseline week-cell drift {len(out)} != {expected}")
    return out


def attach_multiplier(source:pd.DataFrame)->pd.DataFrame:
    league=build_league_location_ypc(source)
    x=source.merge(league,on=["season","week"],how="left",validate="many_to_one")
    opp_mix=[];league_mix=[];mult=[]
    for r in x.itertuples(index=False):
        if not bool(r.matchup_state_ready):
            opp_mix.append(np.nan);league_mix.append(np.nan);mult.append(1.0);continue
        shares=np.array([float(getattr(r,f"player_share_{loc}")) for loc in LOCATIONS],dtype=float)
        opp=np.array([float(getattr(r,f"def_ypc_{loc}")) for loc in LOCATIONS],dtype=float)
        lg=np.array([float(getattr(r,f"league_ypc_{loc}")) for loc in LOCATIONS],dtype=float)
        if not (np.isfinite(shares).all() and np.isfinite(opp).all() and np.isfinite(lg).all()):
            opp_mix.append(np.nan);league_mix.append(np.nan);mult.append(1.0);continue
        om=float(np.dot(shares,opp))
        lm=float(np.dot(shares,lg))
        raw=om/lm if lm>0 else np.nan
        m=float(np.clip(raw,MULT_FLOOR,MULT_CEIL)) if np.isfinite(raw) else 1.0
        opp_mix.append(om);league_mix.append(lm);mult.append(m)
    x["opp_mix_ypc"]=opp_mix
    x["league_mix_ypc"]=league_mix
    x["run_location_matchup_mult"]=mult
    x["source_ready_for_candidate"]=(
        x["matchup_state_ready"]
        & num(x["opp_mix_ypc"]).notna()
        & num(x["league_mix_ypc"]).gt(0)
        & num(x["run_location_matchup_mult"]).notna()
    )
    return x


def candidate_mc(
    base_metrics:pd.DataFrame,
    source_week:pd.DataFrame,
    *,
    iterations:int,
    seed:int,
)->tuple[pd.DataFrame,dict]:
    m=base_metrics.copy()
    before=m.copy(deep=True)
    s=source_week[[
        "season","week","team","player_clean_key",
        "source_ready_for_candidate","run_location_matchup_mult",
    ]].drop_duplicates(["season","week","team","player_clean_key"])
    m["team"]=m["team"].map(canon_team)
    m=m.merge(s,on=["season","week","team","player_clean_key"],how="left",validate="many_to_one")
    pos=m["position"].fillna("").astype(str).str.upper().str.strip()
    ready=bool_col(m["source_ready_for_candidate"].fillna(False))
    mult=num(m["run_location_matchup_mult"])
    base_ypc=num(m["rules_ypc"])
    mask=pos.isin(RB_POS)&ready&mult.notna()&base_ypc.notna()
    if not mask.any():
        raise RuntimeError("zero RB run-location candidate-applied rows")
    m["candidate_applied"]=mask
    m["baseline_rules_ypc_for_audit"]=base_ypc
    m.loc[mask,"rules_ypc"]=base_ypc.loc[mask]*mult.loc[mask]

    # Input isolation: the only model input changed by this scorer is rules_ypc.
    shared=[c for c in before.columns if c in m.columns and c!="rules_ypc"]
    numeric_deltas=[]
    for c in shared:
        a=num(before[c]);b=num(m.loc[before.index,c])
        ok=a.notna()&b.notna()
        if ok.any():
            numeric_deltas.append(float((a.loc[ok]-b.loc[ok]).abs().max()))
    input_non_ypc_max_delta=max(numeric_deltas) if numeric_deltas else 0.0

    base_trace=[]
    cand_trace=[]
    base_sims=simulate(before,iterations=int(iterations),seed=int(seed),allocation_trace=base_trace)
    cand_sims=simulate(m,iterations=int(iterations),seed=int(seed),allocation_trace=cand_trace)

    vals=[]
    rush_att_deltas=[]
    rec_yards_deltas=[]
    receptions_deltas=[]
    for idx,row in m.iterrows():
        market=str(row["market"])
        arr=lookup(cand_sims,row,market)
        vals.append(float(np.mean(arr)) if arr is not None and len(arr) else np.nan)

        if bool(mask.loc[idx]):
            rbrow=before.loc[idx]
            for mk,store in (
                ("rush_att",rush_att_deltas),
                ("rec_yards",rec_yards_deltas),
                ("receptions",receptions_deltas),
            ):
                ba=lookup(base_sims,rbrow,mk)
                ca=lookup(cand_sims,row,mk)
                if ba is not None and ca is not None and len(ba) and len(ca):
                    store.append(float(abs(np.mean(ca)-np.mean(ba))))

    out=m[[
        "season","week","team","opponent","player_clean_key","position","market",
        "candidate_applied","source_ready_for_candidate","run_location_matchup_mult",
    ]].copy()
    out["candidate_mc_proj"]=vals
    audit={
        "candidate_applied_metric_rows":int(mask.sum()),
        "candidate_applied_players":int(
            m.loc[mask,["season","week","team","player_clean_key"]].drop_duplicates().shape[0]
        ),
        "mult_min":float(mult.loc[mask].min()),
        "mult_max":float(mult.loc[mask].max()),
        "mult_mean":float(mult.loc[mask].mean()),
        "input_non_ypc_max_delta":float(input_non_ypc_max_delta),
        "max_rush_att_mean_delta":max(rush_att_deltas) if rush_att_deltas else 0.0,
        "max_rb_rec_yards_mean_delta":max(rec_yards_deltas) if rec_yards_deltas else 0.0,
        "max_rb_receptions_mean_delta":max(receptions_deltas) if receptions_deltas else 0.0,
    }
    return out,audit


def metric_cell(q:pd.DataFrame,season:int,population:str)->dict:
    z=q.loc[q["season"].eq(season)].copy()
    z=z.loc[
        num(z["actual"]).notna()
        & num(z["baseline_proj"]).notna()
        & num(z["candidate_proj"]).notna()
    ].copy()
    if population=="PRIMARY_SOURCE_READY":
        z=z.loc[bool_col(z["source_ready_for_candidate"].fillna(False))].copy()
    rows=len(z);games=z["game_id"].nunique();players=z["player_clean_key"].nunique()
    support=rows>=500 and games>=100 and players>=50
    rec={
        "candidate":CANDIDATE,"season":int(season),"population":population,
        "rows":int(rows),"games":int(games),"players":int(players),"support":bool(support),
    }
    if not support:
        return rec
    actual=num(z["actual"]);base=num(z["baseline_proj"]);cand=num(z["candidate_proj"])
    eb=base-actual;ec=cand-actual
    z["ae_improvement"]=eb.abs()-ec.abs()
    game=cluster_bootstrap(z,"game_id",reps=BOOT_REPS,seed=BOOT_SEED+season)
    player=cluster_bootstrap(z,"player_clean_key",reps=BOOT_REPS,seed=BOOT_SEED+10000+season)
    rec.update({
        "baseline_mae":float(eb.abs().mean()),
        "candidate_mae":float(ec.abs().mean()),
        "mae_improvement":float(z["ae_improvement"].mean()),
        "baseline_rmse":float(np.sqrt(np.mean(np.square(eb)))),
        "candidate_rmse":float(np.sqrt(np.mean(np.square(ec)))),
        "baseline_bias":float(eb.mean()),
        "candidate_bias":float(ec.mean()),
        "game_ci_low":game["ci_low"],"game_ci_high":game["ci_high"],
        "player_ci_low":player["ci_low"],"player_ci_high":player["ci_high"],
        "supported_improvement":bool(
            np.isfinite(game["ci_low"]) and game["ci_low"]>0
            and np.isfinite(player["ci_low"]) and player["ci_low"]>0
        ),
        "supported_harm":bool(
            np.isfinite(game["ci_high"]) and game["ci_high"]<0
            and np.isfinite(player["ci_high"]) and player["ci_high"]<0
        ),
        "rmse_nonworse":bool(float(np.sqrt(np.mean(np.square(ec))))<=float(np.sqrt(np.mean(np.square(eb))))),
    })
    return rec


def output_invariants(base:pd.DataFrame,cand:pd.DataFrame)->dict:
    keys=["season","week","team","opponent","player_clean_key","position","market"]
    z=base[keys+["ensemble_proj"]].rename(columns={"ensemble_proj":"base"}).merge(
        cand[keys+["ensemble_proj"]].rename(columns={"ensemble_proj":"cand"}),
        on=keys,how="inner",validate="one_to_one"
    )
    if len(z)!=len(base):
        raise RuntimeError("candidate final identity drift")
    pos=z["position"].fillna("").astype(str).str.upper()
    rb=pos.isin(RB_POS)
    allowed=rb&z["market"].isin({"rush_yards","rush_rec_yards"})
    untreated=~allowed
    rb_rec=rb&z["market"].eq("rec_yards")
    rb_recs=rb&z["market"].eq("receptions")
    rb_att=rb&z["market"].eq("rush_att")
    def gap(mask):
        if not mask.any(): return 0.0
        return float((num(z.loc[mask,"cand"])-num(z.loc[mask,"base"])).abs().max())
    return {
        "max_untreated_projection_delta":gap(untreated),
        "max_rb_rec_yards_projection_delta":gap(rb_rec),
        "max_rb_receptions_projection_delta":gap(rb_recs),
        "max_rb_rush_att_projection_delta":gap(rb_att),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-dir-2024",type=Path,required=True)
    ap.add_argument("--universe-dir-2025",type=Path,required=True)
    ap.add_argument("--right-tail-detail",type=Path,required=True)
    ap.add_argument("--source-rows",type=Path,required=True)
    ap.add_argument("--weights",type=Path,default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--iterations",type=int,default=ITERATIONS)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    team=read_csv(a.team_weekly,"team-week history")
    logs=read_csv(a.player_logs,"player logs")
    schedule=read_csv(a.schedule,"schedule")
    right_tail=read_csv(a.right_tail_detail,"right-tail baseline detail")
    source=attach_multiplier(prepare_source(read_csv(a.source_rows,"qualified RB source rows")))
    weights=read_csv(a.weights,"ensemble weights")
    for label,frame in [("team",team),("logs",logs),("schedule",schedule),("right_tail",right_tail),("source",source)]:
        check_forbidden(label,frame)

    for frame in (team,logs,schedule):
        frame["season"]=num(frame["season"]).astype("Int64")
        frame["week"]=num(frame["week"]).astype("Int64")
    team["team"]=team["team"].map(canon_team)
    logs["team"]=logs["team"].map(canon_team);logs["opponent"]=logs["opponent"].map(canon_team)
    schedule["team"]=schedule["team"].map(canon_team);schedule["opponent"]=schedule["opponent"].map(canon_team)

    baseline_parts=[];candidate_parts=[];application_audit=[]
    for season in TARGET_SEASONS:
        prior=season-1
        udir=a.universe_dir_2024 if season==2024 else a.universe_dir_2025
        for week in TARGET_WEEKS:
            universe=read_csv(udir/f"{season}_week_{week:02d}.csv",f"pregame universe {season} W{week}")
            bundle=build_historical_context_bundle(
                player_logs=logs,team_weekly=team,pregame_universe=universe,
                schedule=schedule,season=season,week=week,prior_season=prior,
            )
            seed=42+week
            base_comp,base_metrics=base_components(
                bundle,logs,season,week,iterations=int(a.iterations),seed=seed
            )
            baseline_parts.append(base_comp)
            sw=source.loc[source["season"].eq(season)&source["week"].eq(week)].copy()
            cmc,audit=candidate_mc(base_metrics,sw,iterations=int(a.iterations),seed=seed)
            audit.update({"season":season,"week":week})
            application_audit.append(audit)
            keys=["season","week","team","opponent","player_clean_key","position","market"]
            c=base_comp.drop(columns=["mc_proj"]).merge(
                cmc[keys+["candidate_mc_proj"]],
                on=keys,how="left",validate="one_to_one"
            )
            if c["candidate_mc_proj"].isna().any():
                raise RuntimeError(f"candidate MC missing rows season={season} week={week}")
            c=c.rename(columns={"candidate_mc_proj":"mc_proj"})
            candidate_parts.append(c)

    baseline_comp=pd.concat(baseline_parts,ignore_index=True)
    baseline_final=apply_ensemble(baseline_comp,weights=weights)
    baseline_final=attach_game_id(baseline_final,schedule)
    authority=verify_baseline(baseline_final,right_tail)

    cand_comp=pd.concat(candidate_parts,ignore_index=True)
    cand_final=apply_ensemble(cand_comp,weights=weights)
    cand_final=attach_game_id(cand_final,schedule)
    invariants=output_invariants(baseline_final,cand_final)

    keys=["season","week","team","opponent","player_clean_key","position","market","game_id"]
    detail=baseline_final[keys+["actual","ensemble_proj"]].rename(
        columns={"ensemble_proj":"baseline_proj"}
    ).merge(
        cand_final[keys+["ensemble_proj"]].rename(columns={"ensemble_proj":"candidate_proj"}),
        on=keys,how="inner",validate="one_to_one"
    )
    source_keys=source[[
        "season","week","team","player_clean_key","source_ready_for_candidate",
        "run_location_matchup_mult",
    ]]
    detail=detail.merge(
        source_keys,on=["season","week","team","player_clean_key"],
        how="left",validate="many_to_one"
    )
    detail["source_ready_for_candidate"]=bool_col(detail["source_ready_for_candidate"].fillna(False))

    rush=detail.loc[
        detail["position"].fillna("").astype(str).str.upper().isin(RB_POS)
        & detail["market"].eq("rush_yards")
    ].copy()
    cells=[]
    for season in TARGET_SEASONS:
        cells.append(metric_cell(rush,season,"PRIMARY_SOURCE_READY"))
        cells.append(metric_cell(rush,season,"BROAD_ALL_RB"))
    cells=pd.DataFrame(cells)

    primary=cells.loc[cells["population"].eq("PRIMARY_SOURCE_READY")].sort_values("season")
    broad=cells.loc[cells["population"].eq("BROAD_ALL_RB")].sort_values("season")
    primary_pass=bool(
        len(primary)==2
        and primary["support"].fillna(False).all()
        and (num(primary["mae_improvement"])>0).all()
        and primary["supported_improvement"].fillna(False).all()
        and primary["rmse_nonworse"].fillna(False).all()
    )
    broad_pass=bool(
        len(broad)==2
        and broad["support"].fillna(False).all()
        and (num(broad["candidate_mae"])<=num(broad["baseline_mae"])+1e-12).all()
        and ~broad["supported_harm"].fillna(False).any()
    )
    app_invariant=all(
        float(x["input_non_ypc_max_delta"])<=1e-10
        and float(x["max_rush_att_mean_delta"])<=1e-10
        and float(x["max_rb_rec_yards_mean_delta"])<=1e-10
        and float(x["max_rb_receptions_mean_delta"])<=1e-10
        for x in application_audit
    )
    output_invariant=bool(
        float(invariants["max_untreated_projection_delta"])<=1e-10
        and float(invariants["max_rb_rec_yards_projection_delta"])<=1e-10
        and float(invariants["max_rb_receptions_projection_delta"])<=1e-10
        and float(invariants["max_rb_rush_att_projection_delta"])<=1e-10
    )
    passed=primary_pass and broad_pass and app_invariant and output_invariant

    summary={
        "candidate":CANDIDATE,
        "primary_gate_pass":primary_pass,
        "broad_protection_pass":broad_pass,
        "input_and_opportunity_invariant_pass":app_invariant,
        "output_isolation_pass":output_invariant,
        **invariants,
        "disposition":(
            "RB_RUN_LOCATION_MATCHUP_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW"
            if passed else
            "RB_RUN_LOCATION_MATCHUP_HISTORICAL_FAIL_CLOSED"
        ),
        "production_change_authorized":False,
    }

    a.out_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.out_dir/"rb_individual_run_location_matchup_predictive_v1_detail.csv",index=False)
    cells.to_csv(a.out_dir/"rb_individual_run_location_matchup_predictive_v1_cells.csv",index=False)
    pd.DataFrame([summary]).to_csv(a.out_dir/"rb_individual_run_location_matchup_predictive_v1_summary.csv",index=False)
    payload={
        "version":VERSION,"freeze":FREEZE,
        "evaluation_seasons":list(TARGET_SEASONS),"evaluation_weeks":[2,18],
        "iterations":int(a.iterations),"bootstrap_reps":BOOT_REPS,"bootstrap_seed":BOOT_SEED,
        "multiplier_bounds":[MULT_FLOOR,MULT_CEIL],
        "baseline_authority_reproduction":authority,
        "source_ready_rows":int(source["source_ready_for_candidate"].sum()),
        "application_audit":application_audit,
        "candidate":summary,
        "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_used":0,
        "candidate_coefficients_fit":0,"threshold_searches":0,
        "target_game_usage_used":False,"production_change_authorized":False,
    }
    (a.out_dir/"rb_individual_run_location_matchup_predictive_v1_result.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,sort_keys=True))
    print(cells.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
