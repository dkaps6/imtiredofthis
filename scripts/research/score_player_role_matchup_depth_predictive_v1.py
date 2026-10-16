#!/usr/bin/env python3
"""Score frozen player-specific target-depth x opponent-depth matchup candidates.

Frozen in:
docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_PREDICTIVE_V1_FREEZE.md

No coefficient fitting, threshold search, sportsbook input, 2026 outcomes, or
production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import (
    _attach_component_projection,
    build_actual_rows,
    build_mc_predictions,
)
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.modeling.ml_v2 import build_and_train as build_ml
from scripts.modeling.state_v2 import build_state_predictions
from scripts.simulation_v2 import lookup, simulate

VERSION="PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_PREDICTIVE_V1"
FREEZE="docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_PREDICTIVE_V1_FREEZE.md"
TARGET_SEASONS=(2024,2025)
TARGET_WEEKS=tuple(range(2,19))
ITERATIONS=2000
BOOT_REPS=5000
BOOT_SEED=20261012
MULT_FLOOR=0.50
MULT_CEIL=1.80
BANDS=("behind","short","intermediate","deep")
CANDIDATES={
    "PLAYER-DEPTH-MATCHUP-YPT-WR-V1":"WR",
    "PLAYER-DEPTH-MATCHUP-YPT-TE-V1":"TE",
}
FORBIDDEN=(
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)


def read_csv(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def key(v)->str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def check_forbidden(label:str,frame:pd.DataFrame)->None:
    bad=[c for c in frame.columns if any(t in str(c).lower() for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sportsbook/odds columns in {label}: {bad}")


def bool_col(s:pd.Series)->pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes"})


def prepare_source(source:pd.DataFrame)->pd.DataFrame:
    req={
        "season","week","team","opponent","position","player","gsis_id",
        "matchup_state_ready",
        "player_share_behind","player_share_short",
        "player_share_intermediate","player_share_deep",
        "def_targets_behind","def_targets_short",
        "def_targets_intermediate","def_targets_deep",
        "def_ypt_behind","def_ypt_short","def_ypt_intermediate","def_ypt_deep",
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
    x["position"]=x["position"].astype(str).str.upper().str.strip()
    x["player_clean_key"]=x["player"].map(key)
    x["matchup_state_ready"]=bool_col(x["matchup_state_ready"])
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        d=x.loc[x.duplicated(["season","week","team","player_clean_key"],keep=False),
                ["season","week","team","player","gsis_id"]].head(20)
        raise RuntimeError(f"duplicate source player keys: {d.to_dict('records')}")
    return x


def build_league_band_ypt(source:pd.DataFrame)->pd.DataFrame:
    # Defense state repeats once per opposing rostered receiver. Collapse to one
    # row per defense-week before the target-weighted cross-defense average.
    ready=source.copy()
    defense_cols=[]
    for b in BANDS:
        defense_cols.extend([f"def_targets_{b}",f"def_ypt_{b}"])
    d=ready[["season","week","opponent",*defense_cols]].copy()
    # A defense-week must be internally identical across repeated player rows.
    for c in defense_cols:
        d[c]=num(d[c])
    grouped=[]
    for (season,week,opp),g in d.groupby(["season","week","opponent"],sort=True):
        rec={"season":int(season),"week":int(week),"defense":str(opp)}
        valid=True
        for b in BANDS:
            n=num(g[f"def_targets_{b}"]).dropna().unique()
            y=num(g[f"def_ypt_{b}"]).dropna().unique()
            if len(n)!=1 or len(y)!=1:
                valid=False
                break
            rec[f"targets_{b}"]=float(n[0])
            rec[f"ypt_{b}"]=float(y[0])
        if valid:
            grouped.append(rec)
    z=pd.DataFrame(grouped)
    if z.empty:
        raise RuntimeError("zero unique defense-week source states")
    rows=[]
    for (season,week),g in z.groupby(["season","week"],sort=True):
        rec={"season":int(season),"week":int(week)}
        for b in BANDS:
            n=num(g[f"targets_{b}"])
            y=num(g[f"ypt_{b}"])
            ok=n.gt(0)&n.notna()&y.notna()
            den=float(n.loc[ok].sum())
            rec[f"league_ypt_{b}"]=(
                float((n.loc[ok]*y.loc[ok]).sum()/den) if den>0 else np.nan
            )
        rows.append(rec)
    out=pd.DataFrame(rows)
    expected=len(TARGET_SEASONS)*len(TARGET_WEEKS)
    if len(out)!=expected:
        raise RuntimeError(f"league depth baseline week-cell drift {len(out)} != {expected}")
    return out


def attach_multiplier(source:pd.DataFrame)->pd.DataFrame:
    league=build_league_band_ypt(source)
    x=source.merge(league,on=["season","week"],how="left",validate="many_to_one")
    mult=[]
    opp_mix=[]
    league_mix=[]
    for r in x.itertuples(index=False):
        if not bool(r.matchup_state_ready):
            mult.append(1.0); opp_mix.append(np.nan); league_mix.append(np.nan); continue
        shares=np.array([float(getattr(r,f"player_share_{b}")) for b in BANDS],dtype=float)
        opp=np.array([float(getattr(r,f"def_ypt_{b}")) for b in BANDS],dtype=float)
        lg=np.array([float(getattr(r,f"league_ypt_{b}")) for b in BANDS],dtype=float)
        if not (np.isfinite(shares).all() and np.isfinite(opp).all() and np.isfinite(lg).all()):
            mult.append(1.0); opp_mix.append(np.nan); league_mix.append(np.nan); continue
        om=float(np.dot(shares,opp))
        lm=float(np.dot(shares,lg))
        raw=om/lm if lm>0 else np.nan
        m=float(np.clip(raw,MULT_FLOOR,MULT_CEIL)) if np.isfinite(raw) else 1.0
        mult.append(m); opp_mix.append(om); league_mix.append(lm)
    x["opp_mix_ypt"]=opp_mix
    x["league_mix_ypt"]=league_mix
    x["depth_matchup_mult"]=mult
    x["source_ready_for_candidate"]=(
        x["matchup_state_ready"]
        & num(x["depth_matchup_mult"]).notna()
        & num(x["opp_mix_ypt"]).notna()
        & num(x["league_mix_ypt"]).gt(0)
    )
    return x


def base_components(bundle,player_logs:pd.DataFrame,season:int,week:int,*,iterations:int,seed:int):
    mc=build_mc_predictions(bundle,iterations=iterations,seed=seed)
    _,ml_pred=build_ml(player_logs,bundle.player_consensus,int(season),int(week))
    _,state_pred=build_state_predictions(player_logs,bundle.player_consensus,int(season),int(week))
    cols=[
        "player","player_clean_key","team","opponent","season","week",
        "position","role","event_id","market","mc_proj",
    ]
    out=mc[cols].copy()
    out=_attach_component_projection(out,ml_pred,"ml")
    out=_attach_component_projection(out,state_pred,"state")
    actual=build_actual_rows(player_logs,int(season),int(week))
    out=out.merge(actual,on=["team","player_clean_key","market"],how="left",validate="one_to_one")
    out=out.loc[num(out["actual"]).notna()].reset_index(drop=True)
    return out,mc


def candidate_mc(
    base_metrics:pd.DataFrame,
    source_week:pd.DataFrame,
    target_position:str,
    *,
    iterations:int,
    seed:int,
)->tuple[pd.DataFrame,dict]:
    m=base_metrics.copy()
    s=source_week[[
        "season","week","team","player_clean_key","position",
        "source_ready_for_candidate","depth_matchup_mult",
    ]].copy()
    s=s.loc[s["position"].eq(target_position)].drop_duplicates(
        ["season","week","team","player_clean_key"]
    )
    m["team"]=m["team"].map(canon_team)
    m=m.merge(
        s.rename(columns={"position":"source_position"}),
        on=["season","week","team","player_clean_key"],
        how="left",validate="many_to_one",
    )
    pos=m["position"].fillna("").astype(str).str.upper().str.strip()
    ready=bool_col(m["source_ready_for_candidate"].fillna(False))
    mult=num(m["depth_matchup_mult"])
    base_ypt=num(m["rules_ypt"])
    mask=pos.eq(target_position)&ready&mult.notna()&base_ypt.notna()
    m["candidate_applied"]=mask
    m["baseline_rules_ypt_for_audit"]=base_ypt
    m.loc[mask,"rules_ypt"]=base_ypt.loc[mask]*mult.loc[mask]

    changed=int(mask.sum())
    if changed<=0:
        raise RuntimeError(f"zero candidate-applied metrics position={target_position}")

    sims=simulate(m,iterations=int(iterations),seed=int(seed))
    vals=[]
    for _,row in m.iterrows():
        arr=lookup(sims,row,str(row["market"]))
        if arr is not None and len(arr) and str(row["market"])=="pass_yards":
            attempt_rate=pd.to_numeric(
                pd.Series([row.get("mc_pass_attempts_per_dropback")]),errors="coerce"
            ).iloc[0]
            share=pd.to_numeric(
                pd.Series([row.get("qb_pass_att_share")]),errors="coerce"
            ).iloc[0]
            if pd.notna(attempt_rate):
                arr=arr*float(np.clip(attempt_rate,0.50,1.00))
            if pd.notna(share):
                arr=arr*float(np.clip(share,0.0,1.0))
        vals.append(float(np.mean(arr)) if arr is not None and len(arr) else np.nan)

    out=m[[
        "season","week","team","opponent","player_clean_key","position","market",
        "candidate_applied","source_ready_for_candidate","depth_matchup_mult",
    ]].copy()
    out["candidate_mc_proj"]=vals
    audit={
        "candidate_applied_metric_rows":changed,
        "candidate_applied_players":int(
            m.loc[mask,["season","week","team","player_clean_key"]].drop_duplicates().shape[0]
        ),
        "mult_min":float(mult.loc[mask].min()),
        "mult_max":float(mult.loc[mask].max()),
        "mult_mean":float(mult.loc[mask].mean()),
    }
    return out,audit


def cluster_bootstrap(q:pd.DataFrame,cluster_col:str,*,reps:int,seed:int)->dict:
    z=q[["ae_improvement",cluster_col]].dropna().copy()
    if z.empty:
        return {"ci_low":np.nan,"ci_high":np.nan,"valid_reps":0}
    g=z.groupby(cluster_col)["ae_improvement"].agg(["sum","count"])
    if len(g)<2:
        return {"ci_low":np.nan,"ci_high":np.nan,"valid_reps":0}
    a=g[["sum","count"]].to_numpy(float)
    rng=np.random.default_rng(seed)
    p=np.full(len(a),1.0/len(a),dtype=float)
    vals=[];done=0
    while done<reps:
        k=min(250,reps-done)
        counts=rng.multinomial(len(a),p,size=k).astype(float)
        sm=counts@a
        vals.append(np.where(sm[:,1]>0,sm[:,0]/sm[:,1],np.nan))
        done+=k
    v=np.concatenate(vals)
    v=v[np.isfinite(v)]
    return {
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
        "valid_reps":int(len(v)),
    }


def metric_cell(q:pd.DataFrame,*,candidate:str,position:str,season:int,population:str)->dict:
    z=q.loc[q["season"].eq(season)].copy()
    z=z.loc[
        num(z["actual"]).notna()
        & num(z["baseline_proj"]).notna()
        & num(z["candidate_proj"]).notna()
    ].copy()
    if population=="PRIMARY_SOURCE_READY":
        z=z.loc[bool_col(z["source_ready_for_candidate"].fillna(False))].copy()
    rows=len(z);games=z["game_id"].nunique();players=z["player_clean_key"].nunique()
    if position=="WR":
        support=rows>=500 and games>=50 and players>=50
    else:
        support=rows>=200 and games>=50 and players>=25
    rec={
        "candidate":candidate,"position":position,"season":int(season),
        "population":population,"rows":int(rows),"games":int(games),
        "players":int(players),"support":bool(support),
    }
    if not support:
        return rec
    actual=num(z["actual"]);base=num(z["baseline_proj"]);cand=num(z["candidate_proj"])
    eb=base-actual;ec=cand-actual
    z["ae_improvement"]=eb.abs()-ec.abs()
    game=cluster_bootstrap(z,"game_id",reps=BOOT_REPS,seed=BOOT_SEED+season+(0 if position=="WR" else 100))
    player=cluster_bootstrap(z,"player_clean_key",reps=BOOT_REPS,seed=BOOT_SEED+10000+season+(0 if position=="WR" else 100))
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


def attach_game_id(frame:pd.DataFrame,schedule:pd.DataFrame)->pd.DataFrame:
    s=schedule[["season","week","team","opponent","game_id"]].copy()
    s["season"]=num(s["season"]).astype(int);s["week"]=num(s["week"]).astype(int)
    s["team"]=s["team"].map(canon_team);s["opponent"]=s["opponent"].map(canon_team)
    s=s.drop_duplicates(["season","week","team"])
    out=frame.drop(columns=["game_id"],errors="ignore").merge(
        s,on=["season","week","team","opponent"],how="left",validate="many_to_one"
    )
    if out["game_id"].isna().any():
        raise RuntimeError("missing authoritative game_id")
    return out


def verify_baseline(baseline:pd.DataFrame,right_tail:pd.DataFrame)->dict:
    keys=["season","week","team","opponent","player_clean_key","market"]
    rt=right_tail.copy()
    rt["season"]=num(rt["season"]).astype(int);rt["week"]=num(rt["week"]).astype(int)
    rt=rt.loc[rt["season"].isin(TARGET_SEASONS)&rt["week"].isin(TARGET_WEEKS)].copy()
    rt["team"]=rt["team"].map(canon_team);rt["opponent"]=rt["opponent"].map(canon_team)
    rt["player_clean_key"]=rt["player_clean_key"].map(key);rt["final_mean"]=num(rt["final_mean"])
    b=baseline[keys+["ensemble_proj"]].copy()
    z=rt[keys+["final_mean"]].merge(b,on=keys,how="left",validate="one_to_one")
    if z["ensemble_proj"].isna().any():
        raise RuntimeError(f"baseline authority missing rows={int(z['ensemble_proj'].isna().sum())}")
    gap=(num(z["ensemble_proj"])-num(z["final_mean"])).abs()
    max_gap=float(gap.max()) if len(gap) else np.nan
    if not np.isfinite(max_gap) or max_gap>1e-8:
        raise RuntimeError(f"baseline authority numerical drift max_gap={max_gap}")
    return {"rows":int(len(z)),"max_abs_gap":max_gap}


def score_candidate(detail:pd.DataFrame,candidate:str,position:str,invariants:dict)->tuple[pd.DataFrame,dict]:
    q=detail.loc[
        detail["position"].fillna("").astype(str).str.upper().eq(position)
        & detail["market"].eq("rec_yards")
    ].copy()
    rows=[]
    for season in TARGET_SEASONS:
        rows.append(metric_cell(q,candidate=candidate,position=position,season=season,population="PRIMARY_SOURCE_READY"))
        rows.append(metric_cell(q,candidate=candidate,position=position,season=season,population="BROAD_ALL_POSITION"))
    cells=pd.DataFrame(rows)
    primary=cells.loc[cells["population"].eq("PRIMARY_SOURCE_READY")].sort_values("season")
    broad=cells.loc[cells["population"].eq("BROAD_ALL_POSITION")].sort_values("season")
    primary_pass=bool(
        len(primary)==2
        and primary["support"].fillna(False).all()
        and (pd.to_numeric(primary["mae_improvement"],errors="coerce")>0).all()
        and primary["supported_improvement"].fillna(False).all()
        and primary["rmse_nonworse"].fillna(False).all()
    )
    broad_pass=bool(
        len(broad)==2
        and broad["support"].fillna(False).all()
        and (pd.to_numeric(broad["candidate_mae"],errors="coerce")
             <=pd.to_numeric(broad["baseline_mae"],errors="coerce")+1e-12).all()
        and ~broad["supported_harm"].fillna(False).any()
    )
    invariant_pass=bool(
        float(invariants.get("max_untreated_projection_delta",np.inf))<=1e-10
        and float(invariants.get("max_receptions_projection_delta",np.inf))<=1e-10
    )
    passed=primary_pass and broad_pass and invariant_pass
    return cells,{
        "candidate":candidate,"position":position,
        "primary_gate_pass":primary_pass,
        "broad_protection_pass":broad_pass,
        "opportunity_invariant_pass":invariant_pass,
        **invariants,
        "disposition":(
            "PLAYER_DEPTH_MATCHUP_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW"
            if passed else
            "PLAYER_DEPTH_MATCHUP_HISTORICAL_FAIL_CLOSED"
        ),
        "production_change_authorized":False,
    }


def candidate_invariants(base_final:pd.DataFrame,cand_final:pd.DataFrame,position:str)->dict:
    keys=["season","week","team","opponent","player_clean_key","position","market"]
    z=base_final[keys+["ensemble_proj"]].rename(columns={"ensemble_proj":"base"}).merge(
        cand_final[keys+["ensemble_proj"]].rename(columns={"ensemble_proj":"cand"}),
        on=keys,how="inner",validate="one_to_one"
    )
    if len(z)!=len(base_final):
        raise RuntimeError("candidate final identity drift")
    pos=z["position"].fillna("").astype(str).str.upper()
    treated=pos.eq(position)&z["market"].eq("rec_yards")
    untreated=~treated
    rec=pos.eq(position)&z["market"].eq("receptions")
    return {
        "max_untreated_projection_delta":float((num(z.loc[untreated,"cand"])-num(z.loc[untreated,"base"])).abs().max()) if untreated.any() else 0.0,
        "max_receptions_projection_delta":float((num(z.loc[rec,"cand"])-num(z.loc[rec,"base"])).abs().max()) if rec.any() else 0.0,
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
    right_tail=read_csv(a.right_tail_detail,"right-tail detail")
    source=attach_multiplier(prepare_source(read_csv(a.source_rows,"qualified source rows")))
    weights=read_csv(a.weights,"ensemble weights")
    for label,frame in [("team",team),("logs",logs),("schedule",schedule),("right_tail",right_tail),("source",source)]:
        check_forbidden(label,frame)

    for frame in (team,logs,schedule):
        frame["season"]=num(frame["season"]).astype("Int64")
        frame["week"]=num(frame["week"]).astype("Int64")
    team["team"]=team["team"].map(canon_team)
    logs["team"]=logs["team"].map(canon_team);logs["opponent"]=logs["opponent"].map(canon_team)
    schedule["team"]=schedule["team"].map(canon_team);schedule["opponent"]=schedule["opponent"].map(canon_team)

    baseline_parts=[]
    candidate_parts={c:[] for c in CANDIDATES}
    application_audit={c:[] for c in CANDIDATES}

    for season in TARGET_SEASONS:
        prior=season-1
        udir=a.universe_dir_2024 if season==2024 else a.universe_dir_2025
        for week in TARGET_WEEKS:
            upath=udir/f"{season}_week_{week:02d}.csv"
            universe=read_csv(upath,f"pregame universe {season} W{week}")
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
            for cand,pos in CANDIDATES.items():
                cmc,audit=candidate_mc(
                    base_metrics,sw,pos,iterations=int(a.iterations),seed=seed
                )
                audit.update({"season":season,"week":week})
                application_audit[cand].append(audit)
                keys=["season","week","team","opponent","player_clean_key","position","market"]
                c=base_comp.copy()
                c=c.drop(columns=["mc_proj"]).merge(
                    cmc[keys+["candidate_mc_proj"]],
                    on=keys,how="left",validate="one_to_one"
                )
                if c["candidate_mc_proj"].isna().any():
                    raise RuntimeError(f"candidate MC missing rows {cand} {season} W{week}")
                c=c.rename(columns={"candidate_mc_proj":"mc_proj"})
                candidate_parts[cand].append(c)

    baseline_comp=pd.concat(baseline_parts,ignore_index=True)
    baseline_final=apply_ensemble(baseline_comp,weights=weights)
    baseline_final=attach_game_id(baseline_final,schedule)
    authority=verify_baseline(baseline_final,right_tail)

    source_keys=source[[
        "season","week","team","player_clean_key","position",
        "source_ready_for_candidate","depth_matchup_mult",
    ]].copy()

    all_cells=[];summaries=[]
    a.out_dir.mkdir(parents=True,exist_ok=True)

    for cand,pos in CANDIDATES.items():
        comp=pd.concat(candidate_parts[cand],ignore_index=True)
        final=apply_ensemble(comp,weights=weights)
        final=attach_game_id(final,schedule)
        invariants=candidate_invariants(baseline_final,final,pos)
        keys=["season","week","team","opponent","player_clean_key","position","market","game_id"]
        detail=baseline_final[keys+["actual","ensemble_proj"]].rename(
            columns={"ensemble_proj":"baseline_proj"}
        ).merge(
            final[keys+["ensemble_proj"]].rename(columns={"ensemble_proj":"candidate_proj"}),
            on=keys,how="inner",validate="one_to_one"
        )
        detail=detail.merge(
            source_keys.rename(columns={"position":"source_position"}),
            on=["season","week","team","player_clean_key"],
            how="left",validate="many_to_one"
        )
        detail["source_ready_for_candidate"]=bool_col(
            detail["source_ready_for_candidate"].fillna(False)
        )
        cells,summary=score_candidate(detail,cand,pos,invariants)
        all_cells.append(cells);summaries.append(summary)
        safe=cand.lower().replace("-","_")
        detail.to_csv(a.out_dir/f"{safe}_detail.csv",index=False)

    cells=pd.concat(all_cells,ignore_index=True)
    cells.to_csv(a.out_dir/"player_depth_matchup_predictive_v1_cells.csv",index=False)
    pd.DataFrame(summaries).to_csv(a.out_dir/"player_depth_matchup_predictive_v1_summary.csv",index=False)
    payload={
        "version":VERSION,"freeze":FREEZE,
        "evaluation_seasons":list(TARGET_SEASONS),"evaluation_weeks":[2,18],
        "iterations":int(a.iterations),"bootstrap_reps":BOOT_REPS,"bootstrap_seed":BOOT_SEED,
        "multiplier_bounds":[MULT_FLOOR,MULT_CEIL],
        "baseline_authority_reproduction":authority,
        "source_ready_rows":int(source["source_ready_for_candidate"].sum()),
        "application_audit":application_audit,
        "candidates":summaries,
        "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_used":0,
        "candidate_coefficients_fit":0,"threshold_searches":0,
        "target_game_usage_used":False,"production_change_authorized":False,
    }
    (a.out_dir/"player_depth_matchup_predictive_v1_result.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,sort_keys=True))
    print(cells.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
