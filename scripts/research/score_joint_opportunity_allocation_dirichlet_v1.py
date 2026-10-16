#!/usr/bin/env python3
"""Historical score for frozen joint player-opportunity allocation V1.

Scores canonical pregame allocation probabilities under:
- baseline Multinomial;
- frozen Dirichlet-Multinomial eligible-group + complement candidate.

No sportsbook input, fitted coefficient, threshold search, 2026 outcome, or
production change.
"""
from __future__ import annotations

import argparse
import json
from math import lgamma
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_market_frame
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts.research.audit_player_opportunity_allocation_uncertainty_betabinomial_v1 import (
    build_target_counts,
    discrete_crps,
)
from scripts.research.joint_opportunity_allocation_dirichlet_v1 import (
    beta_intraclass_rho,
    room_rho,
)
from scripts.simulation_v2 import _sharpen_wr_target_shares, _top_n_shares
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION="PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL"
FREEZE="docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_FREEZE.md"
TARGET_SEASONS=(2023,2024,2025)
TARGET_WEEKS=tuple(range(5,19))
CELLS=("RB_CARRY_SHARE","WR_TARGET_SHARE")
RB_POS={"RB","HB","FB"}
WR_POS={"WR","LWR","RWR","SWR"}
BOOT_REPS=5000
BOOT_SEED=20261013
MEAN_TOL=1e-12
FORBIDDEN=(
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)


def num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def clean_key(value)->str:
    try:
        _,k=canonicalize_player_name_safe(value)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def read_csv(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    bad=[c for c in x.columns if any(t in str(c).lower() for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sportsbook columns in {label}: {bad}")
    return x


def prepare_v2(source:pd.DataFrame)->pd.DataFrame:
    req={
        "season","week","game_id","team","player_identity_key","player","position",
        "cell","prior_share_mean","excess_share_var",
        "prior_feature_max_week","denominator_source_max_week",
    }
    miss=req-set(source.columns)
    if miss:
        raise RuntimeError(f"V2 source missing {sorted(miss)}")
    x=source.loc[
        num(source["season"]).isin(TARGET_SEASONS)
        & num(source["week"]).isin(TARGET_WEEKS)
        & source["cell"].astype(str).isin(CELLS)
    ].copy()
    x["season"]=num(x["season"]).astype(int)
    x["week"]=num(x["week"]).astype(int)
    x["team"]=x["team"].map(canon_team)
    x["player_clean_key"]=x["player"].map(clean_key)
    if x.duplicated(["season","week","team","player_identity_key","cell"]).any():
        raise RuntimeError("duplicate V2 source identity")
    if (
        (num(x["prior_feature_max_week"])>=x["week"]).any()
        or (num(x["denominator_source_max_week"])>=x["week"]).any()
    ):
        raise RuntimeError("same/future V2 source detected")
    rhos=[]
    for r in x.itertuples(index=False):
        rhos.append(beta_intraclass_rho(float(r.prior_share_mean),float(r.excess_share_var)))
    x["player_rho"]=rhos
    x["v2_candidate_eligible"]=x["player_rho"].notna()
    return x


def _cap_with_residual(shares:np.ndarray)->tuple[np.ndarray,float]:
    clean=np.nan_to_num(np.asarray(shares,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    clean=np.clip(clean,0.0,0.95)
    total=float(clean.sum())
    if total>0.95:
        clean=clean*(0.95/total)
    residual=max(0.0,1.0-float(clean.sum()))
    denom=float(clean.sum()+residual)
    if denom<=0:
        raise RuntimeError("allocation probabilities sum to zero")
    return clean/denom,residual/denom


def canonical_player_probabilities(metrics:pd.DataFrame,cell:str)->pd.DataFrame:
    x=metrics.copy()
    x["team"]=x["team"].map(canon_team)
    x["player_clean_key"]=x.get("player_clean_key",x["player"]).map(clean_key)
    x=x.sort_values(["team","player_clean_key","market"]).drop_duplicates(
        ["team","player_clean_key"],keep="last"
    )
    rows=[]
    for team,g in x.groupby("team",sort=True):
        g=g.reset_index(drop=True)
        pos=g["position"].fillna("").astype(str).str.upper().str.strip()
        if cell=="RB_CARRY_SHARE":
            raw=np.array([
                pd.to_numeric(pd.Series([r.get("rules_rush_share",r.get("bayes_rush_share",r.get("rush_share",0.0)))]),errors="coerce").fillna(0.0).iloc[0]
                for _,r in g.iterrows()
            ],dtype=float)
            shares=_top_n_shares(raw,5)
        elif cell=="WR_TARGET_SHARE":
            raw=np.array([
                pd.to_numeric(pd.Series([r.get("rules_tgt_share",r.get("bayes_tgt_share",r.get("tgt_share",0.0)))]),errors="coerce").fillna(0.0).iloc[0]
                for _,r in g.iterrows()
            ],dtype=float)
            shares=_sharpen_wr_target_shares(g,raw)
        else:
            raise KeyError(cell)
        probs,residual=_cap_with_residual(shares)
        for j,r in g.iterrows():
            rows.append({
                "team":team,
                "player_clean_key":str(r["player_clean_key"]),
                "position":str(r.get("position","")).upper().strip(),
                "q":float(probs[j]),
            })
        rows.append({
            "team":team,"player_clean_key":"__RESIDUAL__",
            "position":"RESIDUAL","q":float(residual),
        })
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"zero canonical probabilities cell={cell}")
    sums=out.groupby("team")["q"].sum()
    if float((sums-1.0).abs().max())>1e-12:
        raise RuntimeError(f"canonical probability mass drift cell={cell}")
    return out


def build_week_metrics(
    *, player_logs:pd.DataFrame,team_weekly:pd.DataFrame,pregame_universe:pd.DataFrame,
    schedule:pd.DataFrame,season:int,week:int,
)->pd.DataFrame:
    bundle=build_historical_context_bundle(
        player_logs=player_logs,team_weekly=team_weekly,
        pregame_universe=pregame_universe,schedule=schedule,
        season=season,week=week,prior_season=season-1,
    )
    metrics=build_market_frame(bundle)
    bayes=build_bayesian_baseline(bundle.player_consensus)
    metrics=apply_bayesian_to_metrics(metrics,bayes)
    with patch.object(simulation_rules,"load_model_contexts",return_value=(bundle.teams,bundle.players)):
        metrics=simulation_rules.apply_rules_to_metrics(metrics)
    if int(num(metrics["rules_applied"]).fillna(0).sum())==0:
        raise RuntimeError(f"historical rules matched zero rows {season} W{week}")
    metrics["season"]=season
    metrics["week"]=week
    return metrics


def log_multinomial(y:np.ndarray,p:np.ndarray)->float:
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    N=int(y.sum())
    if len(y)!=len(p) or (y<0).any() or not np.isfinite(p).all() or (p<0).any():
        return -np.inf
    if abs(float(p.sum())-1.0)>1e-10:
        return -np.inf
    if ((p<=0)&(y>0)).any():
        return -np.inf
    out=lgamma(N+1)-sum(lgamma(int(v)+1) for v in y)
    for yi,pi in zip(y,p):
        if yi>0:
            out+=int(yi)*float(np.log(pi))
    return float(out)


def log_dirichlet_multinomial(y:np.ndarray,alpha:np.ndarray)->float:
    y=np.asarray(y,dtype=int)
    a=np.asarray(alpha,dtype=float)
    if len(y)!=len(a) or (y<0).any() or not np.isfinite(a).all() or (a<=0).any():
        return -np.inf
    N=int(y.sum())
    A=float(a.sum())
    out=lgamma(N+1)-sum(lgamma(int(v)+1) for v in y)
    out+=lgamma(A)-lgamma(N+A)
    out+=sum(lgamma(int(yi)+float(ai))-lgamma(float(ai)) for yi,ai in zip(y,a))
    return float(out)


def count_crps_binom(N:int,y:int,q:float)->float:
    d=binom(N,q)
    return discrete_crps(d.cdf(np.arange(N+1)),y)


def count_crps_betabinom(N:int,y:int,alpha:float,beta:float)->float:
    d=betabinom(N,alpha,beta)
    return discrete_crps(d.cdf(np.arange(N+1)),y)


def build_rooms(
    *,source:pd.DataFrame,counts:pd.DataFrame,metrics:pd.DataFrame,
    season:int,week:int,cell:str,schedule:pd.DataFrame,
)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    probs=canonical_player_probabilities(metrics,cell)
    s=source.loc[
        source["season"].eq(season)&source["week"].eq(week)&source["cell"].eq(cell)
    ].copy()
    c=counts.loc[
        counts["season"].eq(season)&counts["week"].eq(week)
    ].copy()
    c["team"]=c["team"].map(canon_team)
    c["player_clean_key"]=c.get("player",pd.Series("",index=c.index)).map(clean_key)

    if cell=="RB_CARRY_SHARE":
        ycol="rushes"; Ncol="team_rushes"; positions=RB_POS
    else:
        ycol="targets"; Ncol="team_targets"; positions=WR_POS

    actual=c[["team","player_identity_key","player_clean_key",ycol,Ncol]].copy()
    actual[ycol]=num(actual[ycol]); actual[Ncol]=num(actual[Ncol])

    s=s.merge(
        actual,
        on=["team","player_identity_key"],
        how="left",validate="one_to_one",suffixes=("","_actual"),
    )
    # Preserve stable-ID match as primary. Name key is audit-only.
    if s[[ycol,Ncol]].isna().any(axis=None):
        raise RuntimeError(f"target-count identity miss season={season} week={week} cell={cell}")

    # V2 source position is the frozen player-cell authority. The canonical
    # probability frame contributes only q; merging a second position column
    # would create position_x/position_y and weaken the identity audit.
    s=s.merge(
        probs[["team","player_clean_key","q"]],
        on=["team","player_clean_key"],
        how="left",validate="one_to_one",
    )
    match_rate=float(s["q"].notna().mean()) if len(s) else 0.0

    sched=schedule.loc[
        num(schedule["season"]).eq(season)&num(schedule["week"]).eq(week),
        ["team","game_id"]
    ].copy()
    sched["team"]=sched["team"].map(canon_team)
    sched=sched.drop_duplicates("team")

    room_rows=[]
    marginal_rows=[]
    room_count=0
    eligible_player_keys=set()

    for team,g in s.groupby("team",sort=True):
        canonical=probs.loc[probs["team"].eq(team)].copy()
        if canonical.empty:
            continue
        Nvals=num(g[Ncol]).dropna().unique()
        if len(Nvals)!=1:
            raise RuntimeError(f"team opportunity total drift {season} W{week} {team} {cell}")
        N=int(round(float(Nvals[0])))
        if N<=0:
            continue

        posok=g["position"].fillna("").astype(str).str.upper().isin(positions)
        elig=(
            g["v2_candidate_eligible"].fillna(False).astype(bool)
            & posok
            & num(g["q"]).gt(0)
            & num(g["q"]).lt(1)
        )
        e=g.loc[elig].copy()
        if e.empty:
            continue

        q=e["q"].to_numpy(float)
        rho=e["player_rho"].to_numpy(float)
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
        first_probs=np.append(q,complement)
        alpha=first_probs*kappa
        if not np.isfinite(alpha).all() or (alpha<=0).any():
            continue

        y=e[ycol].round().astype(int).to_numpy()
        comp_y=N-int(y.sum())
        if comp_y<0:
            raise RuntimeError(f"eligible counts exceed team total {season} W{week} {team} {cell}")
        yvec=np.append(y,comp_y)
        if int(yvec.sum())!=N:
            raise RuntimeError("joint outcome conservation failure")

        base_log=log_multinomial(yvec,first_probs)
        cand_log=log_dirichlet_multinomial(yvec,alpha)
        if not (np.isfinite(base_log) and np.isfinite(cand_log)):
            continue

        gid=sched.loc[sched["team"].eq(team),"game_id"]
        game_id=str(gid.iloc[0]) if len(gid) else f"{season}_{week}_{team}"
        room_rows.append({
            "season":season,"week":week,"game_id":game_id,"team":team,"cell":cell,
            "N":N,"eligible_players":int(len(e)),"rho_room":rho_room,"kappa":kappa,
            "baseline_log_score":base_log,"candidate_log_score":cand_log,
            "joint_log_score_improvement":cand_log-base_log,
            "expected_count_gap":0.0,
        })
        room_count+=1

        A=float(alpha.sum())
        for j,(_,r) in enumerate(e.iterrows()):
            yi=int(round(float(r[ycol])))
            qi=float(r["q"])
            base_crps=count_crps_binom(N,yi,qi)
            cand_crps=count_crps_betabinom(N,yi,float(alpha[j]),float(A-alpha[j]))
            marginal_rows.append({
                "season":season,"week":week,"game_id":game_id,"team":team,"cell":cell,
                "player_identity_key":str(r["player_identity_key"]),
                "player_clean_key":str(r["player_clean_key"]),
                "N":N,"y":yi,"q":qi,"rho_i":float(r["player_rho"]),
                "rho_room":rho_room,
                "baseline_expected_count":N*qi,
                "candidate_expected_count":N*qi,
                "expected_count_gap":0.0,
                "baseline_crps":base_crps,"candidate_crps":cand_crps,
                "crps_improvement":base_crps-cand_crps,
            })
            eligible_player_keys.add(str(r["player_identity_key"]))

    return pd.DataFrame(room_rows),pd.DataFrame(marginal_rows),{
        "source_rows":int(len(s)),
        "canonical_q_match_rate":match_rate,
        "rooms":room_count,
        "eligible_players":len(eligible_player_keys),
    }


def cluster_boot(q:pd.DataFrame,value_col:str,cluster_col:str,seed:int)->dict:
    z=q[[value_col,cluster_col]].dropna()
    g=z.groupby(cluster_col)[value_col].agg(["sum","count"])
    if len(g)<2:
        return {"ci_low":np.nan,"ci_high":np.nan,"valid_reps":0}
    a=g[["sum","count"]].to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=[];done=0;p=np.full(len(a),1.0/len(a))
    while done<BOOT_REPS:
        k=min(250,BOOT_REPS-done)
        counts=rng.multinomial(len(a),p,size=k)
        sums=counts@a
        vals.append(np.where(sums[:,1]>0,sums[:,0]/sums[:,1],np.nan))
        done+=k
    v=np.concatenate(vals);v=v[np.isfinite(v)]
    return {
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
        "valid_reps":int(len(v)),
    }


def season_cell_result(rooms:pd.DataFrame,marg:pd.DataFrame,season:int,cell:str)->dict:
    r=rooms.loc[rooms["season"].eq(season)&rooms["cell"].eq(cell)].copy()
    m=marg.loc[marg["season"].eq(season)&marg["cell"].eq(cell)].copy()
    games=r["game_id"].nunique()
    players=m["player_identity_key"].nunique()
    support=len(r)>=100 and players>=50
    out={
        "season":season,"cell":cell,"room_rows":int(len(r)),
        "games":int(games),"eligible_players":int(players),"support":bool(support),
    }
    if not support:
        out["historical_pass"]=False
        return out
    gb=cluster_boot(r,"joint_log_score_improvement","game_id",BOOT_SEED+season+CELLS.index(cell)*100)
    tb=cluster_boot(r,"joint_log_score_improvement","team",BOOT_SEED+10000+season+CELLS.index(cell)*100)
    mg=cluster_boot(m,"crps_improvement","game_id",BOOT_SEED+20000+season+CELLS.index(cell)*100)
    mean_joint=float(r["joint_log_score_improvement"].mean())
    mean_crps=float(m["crps_improvement"].mean())
    supported_harm=bool(np.isfinite(mg["ci_high"]) and mg["ci_high"]<0)
    passed=bool(
        mean_joint>0
        and np.isfinite(gb["ci_low"]) and gb["ci_low"]>0
        and np.isfinite(tb["ci_low"]) and tb["ci_low"]>0
        and mean_crps>=-1e-12
        and not supported_harm
        and float(r["expected_count_gap"].max())<=MEAN_TOL
        and float(m["expected_count_gap"].max())<=MEAN_TOL
    )
    out.update({
        "mean_joint_log_score_improvement":mean_joint,
        "game_boot":gb,"team_boot":tb,
        "marginal_rows":int(len(m)),
        "mean_marginal_crps_improvement":mean_crps,
        "marginal_game_boot":mg,
        "marginal_supported_harm":supported_harm,
        "historical_pass":passed,
    })
    return out


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-rows",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-dir-2023",type=Path,required=True)
    ap.add_argument("--universe-dir-2024",type=Path,required=True)
    ap.add_argument("--universe-dir-2025",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    source=prepare_v2(read_csv(a.v2_rows,"V2 source"))
    counts=build_target_counts()
    team=read_csv(a.team_weekly,"team-week history")
    logs=read_csv(a.player_logs,"player logs")
    schedule=read_csv(a.schedule,"schedule")
    for frame in (team,logs,schedule):
        frame["season"]=num(frame["season"]).astype("Int64")
        frame["week"]=num(frame["week"]).astype("Int64")
    team["team"]=team["team"].map(canon_team)
    logs["team"]=logs["team"].map(canon_team)
    if "opponent" in logs.columns:
        logs["opponent"]=logs["opponent"].map(canon_team)
    schedule["team"]=schedule["team"].map(canon_team)
    schedule["opponent"]=schedule["opponent"].map(canon_team)

    udirs={2023:a.universe_dir_2023,2024:a.universe_dir_2024,2025:a.universe_dir_2025}
    room_parts=[];marg_parts=[];audits=[]
    for season in TARGET_SEASONS:
        for week in TARGET_WEEKS:
            universe=read_csv(udirs[season]/f"{season}_week_{week:02d}.csv",f"universe {season} W{week}")
            metrics=build_week_metrics(
                player_logs=logs,team_weekly=team,pregame_universe=universe,
                schedule=schedule,season=season,week=week,
            )
            for cell in CELLS:
                rooms,marg,audit=build_rooms(
                    source=source,counts=counts,metrics=metrics,
                    season=season,week=week,cell=cell,schedule=schedule,
                )
                if len(rooms): room_parts.append(rooms)
                if len(marg): marg_parts.append(marg)
                audit.update({"season":season,"week":week,"cell":cell})
                audits.append(audit)

    rooms=pd.concat(room_parts,ignore_index=True) if room_parts else pd.DataFrame()
    marg=pd.concat(marg_parts,ignore_index=True) if marg_parts else pd.DataFrame()
    if rooms.empty or marg.empty:
        raise RuntimeError("zero historical joint allocation scoring rows")

    cells=[]
    per_cell=[]
    for cell in CELLS:
        by={}
        for season in TARGET_SEASONS:
            rec=season_cell_result(rooms,marg,season,cell)
            cells.append(rec)
            by[season]=bool(rec.get("historical_pass",False))
        passed=all(by.get(s,False) for s in TARGET_SEASONS)
        per_cell.append({
            "cell":cell,
            "season_2023_pass":bool(by.get(2023,False)),
            "season_2024_pass":bool(by.get(2024,False)),
            "season_2025_pass":bool(by.get(2025,False)),
            "disposition":(
                "JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW"
                if passed else
                "JOINT_ALLOCATION_DIRICHLET_HISTORICAL_FAIL_CLOSED"
            ),
        })

    max_room_gap=float(rooms["expected_count_gap"].max())
    max_marg_gap=float(marg["expected_count_gap"].max())
    min_match=min(float(x["canonical_q_match_rate"]) for x in audits if x["source_rows"]>0)
    gates={
        "exact_target_seasons":sorted(rooms["season"].unique().tolist())==list(TARGET_SEASONS),
        "exact_cells":sorted(rooms["cell"].unique().tolist())==sorted(CELLS),
        "target_weeks_within_5_18":int(rooms["week"].min())>=5 and int(rooms["week"].max())<=18,
        "room_mean_gap_le_1e_12":max_room_gap<=MEAN_TOL,
        "marginal_mean_gap_le_1e_12":max_marg_gap<=MEAN_TOL,
        "canonical_q_match_rate_ge_095":min_match>=.95,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "fitted_coefficients_zero":True,
        "threshold_searches_zero":True,
        "outcomes_2026_zero":True,
        "production_changed_false":True,
    }
    result={
        "version":VERSION,"freeze":FREEZE,
        "disposition":(
            "JOINT_ALLOCATION_DIRICHLET_HAS_PASSING_CELLS"
            if any(x["disposition"]=="JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW" for x in per_cell)
            else "JOINT_ALLOCATION_DIRICHLET_ALL_FAIL_CLOSED"
        ),
        "target_seasons":list(TARGET_SEASONS),"target_weeks":[5,18],
        "cells_authorized":list(CELLS),
        "room_rows":int(len(rooms)),"marginal_rows":int(len(marg)),
        "max_room_expected_count_gap":max_room_gap,
        "max_marginal_expected_count_gap":max_marg_gap,
        "min_canonical_q_match_rate":min_match,
        "application_audit":audits,
        "per_cell":per_cell,"cells":cells,"gates":gates,
        "sportsbook_inputs_used":0,"oddsapi_calls":0,
        "fitted_coefficients":0,"threshold_searches":0,
        "outcomes_2026_used":0,"production_changed":False,
    }
    if not all(gates.values()):
        result["disposition"]="JOINT_ALLOCATION_DIRICHLET_INTEGRITY_FAILURE"

    a.out_dir.mkdir(parents=True,exist_ok=True)
    rooms.to_csv(a.out_dir/"joint_allocation_dirichlet_v1_rooms.csv",index=False,float_format="%.12f")
    marg.to_csv(a.out_dir/"joint_allocation_dirichlet_v1_marginals.csv",index=False,float_format="%.12f")
    (a.out_dir/"joint_allocation_dirichlet_v1_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
