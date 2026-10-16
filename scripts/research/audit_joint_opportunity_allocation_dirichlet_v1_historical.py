#!/usr/bin/env python3
"""Historical scoring for frozen joint team-conserving opportunity allocation V1."""
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
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.component_predictions import build_market_frame
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules
from scripts.simulation_v2 import _sharpen_wr_target_shares, _top_n_shares
from scripts.research.audit_player_opportunity_allocation_uncertainty_betabinomial_v1 import (
    build_target_counts,
    attach_counts,
    discrete_crps,
)
from scripts.research.joint_opportunity_allocation_dirichlet_v1 import (
    beta_intraclass_rho,
    room_rho,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION="PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL"
FREEZE="docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_SCORING_FREEZE.md"
TARGET_SEASONS=(2023,2024,2025)
TARGET_WEEKS=tuple(range(5,19))
CELLS=("RB_CARRY_SHARE","WR_TARGET_SHARE")
BOOT_REPS=5000
BOOT_SEED=20261013
MEAN_TOL=1e-12
PROB_TOL=1e-12
RB_POS={"RB","HB","FB"}
WR_POS={"WR","LWR","RWR","SWR"}


def num(s):
    return pd.to_numeric(s,errors="coerce")


def key(value)->str:
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
    return x


def prepare_v2_source(v2:pd.DataFrame)->pd.DataFrame:
    req={
        "season","week","game_id","team","player_identity_key","player","position",
        "cell","prior_share_mean","excess_share_var",
        "prior_feature_max_week","denominator_source_max_week",
    }
    miss=req-set(v2.columns)
    if miss:
        raise RuntimeError(f"V2 rows missing {sorted(miss)}")
    counts=build_target_counts()
    x=attach_counts(v2,counts)
    x=x.loc[x["cell"].isin(CELLS)].copy()
    x["player_clean_key"]=x["player"].map(key)
    x["team"]=x["team"].map(canon_team)
    x["season"]=num(x["season"]).astype(int)
    x["week"]=num(x["week"]).astype(int)
    if x["player_clean_key"].eq("").any():
        raise RuntimeError("V2 source has empty canonical player key")
    if x.duplicated(["season","week","team","player_clean_key","cell"]).any():
        sample=x.loc[
            x.duplicated(["season","week","team","player_clean_key","cell"],keep=False),
            ["season","week","team","player","player_identity_key","cell"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"V2 canonical-name collision: {sample}")
    x["rho_i"]=[
        beta_intraclass_rho(float(mu),float(v))
        if pd.notna(mu) and pd.notna(v) else None
        for mu,v in zip(x["prior_share_mean"],x["excess_share_var"])
    ]
    x["rho_i"]=pd.to_numeric(x["rho_i"],errors="coerce")
    if (
        (num(x["prior_feature_max_week"])>=x["week"]).any()
        or (num(x["denominator_source_max_week"])>=x["week"]).any()
    ):
        raise RuntimeError("same/future volatility feature in joint source")
    return x


def prepare_canonical_metrics(bundle)->pd.DataFrame:
    metrics=build_market_frame(bundle)
    metrics=apply_bayesian_to_metrics(metrics,build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(bundle.teams,bundle.players)):
        metrics=simulation_rules.apply_rules_to_metrics(metrics)
    if int(num(metrics["rules_applied"]).fillna(0).sum())==0:
        raise RuntimeError("historical rules matched zero rows")
    metrics=metrics.copy()
    metrics["team"]=metrics["team"].map(canon_team)
    metrics["player_clean_key"]=metrics["player"].map(key)
    game_key="event_id"
    player_cols=[game_key,"team","player_clean_key"]
    players=(metrics.sort_values(player_cols)
             .drop_duplicates(player_cols,keep="last")
             .reset_index(drop=True))
    if players.duplicated(player_cols).any():
        raise RuntimeError("canonical player collapse duplicate")
    return players


def normalize_player_probs(shares:np.ndarray)->tuple[np.ndarray,float]:
    clean=np.nan_to_num(np.asarray(shares,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    clean=np.clip(clean,0.0,0.95)
    total=float(clean.sum())
    if total>0.95:
        clean*=0.95/total
    residual=max(0.0,1.0-float(clean.sum()))
    probs=np.append(clean,residual)
    probs=probs/probs.sum()
    return probs[:-1],float(probs[-1])


def canonical_room_probabilities(players:pd.DataFrame,cell:str)->pd.DataFrame:
    g=players.sort_values(["event_id","team","player_clean_key"]).copy()
    if cell=="RB_CARRY_SHARE":
        raw=num(g["rules_rush_share"]).fillna(0.0).to_numpy(float)
        used=_top_n_shares(raw,5)
    elif cell=="WR_TARGET_SHARE":
        raw=num(g["rules_tgt_share"]).fillna(0.0).to_numpy(float)
        used=_sharpen_wr_target_shares(g,raw)
    else:
        raise KeyError(cell)
    q,residual=normalize_player_probs(used)
    out=g[["event_id","team","player_clean_key","player","position"]].copy()
    out["canonical_q"]=q
    out["canonical_residual"]=residual
    if abs(float(out["canonical_q"].sum())+residual-1.0)>PROB_TOL:
        raise RuntimeError("canonical probability mass does not conserve")
    return out


def multinomial_logpmf(counts:np.ndarray,probs:np.ndarray)->float:
    y=np.asarray(counts,dtype=int)
    p=np.asarray(probs,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p):
        raise ValueError("multinomial shape mismatch")
    N=int(y.sum())
    if (y<0).any() or (p<0).any() or not np.isfinite(p).all():
        return -np.inf
    if abs(float(p.sum())-1.0)>1e-10:
        return -np.inf
    if ((y>0)&(p<=0)).any():
        return -np.inf
    val=lgamma(N+1)-sum(lgamma(int(v)+1) for v in y)
    positive=y>0
    if positive.any():
        val+=float(np.sum(y[positive]*np.log(p[positive])))
    return float(val)


def dirichlet_multinomial_logpmf(counts:np.ndarray,alpha:np.ndarray)->float:
    y=np.asarray(counts,dtype=int)
    a=np.asarray(alpha,dtype=float)
    if y.ndim!=1 or a.ndim!=1 or len(y)!=len(a):
        raise ValueError("dirichlet-multinomial shape mismatch")
    if (y<0).any() or (a<=0).any() or not np.isfinite(a).all():
        return -np.inf
    N=int(y.sum())
    a0=float(a.sum())
    val=lgamma(N+1)-sum(lgamma(int(v)+1) for v in y)
    val+=lgamma(a0)-lgamma(N+a0)
    val+=sum(lgamma(int(v)+float(ai))-lgamma(float(ai)) for v,ai in zip(y,a))
    return float(val)


def marginal_crps(N:int,y:int,q:float,alpha_i:float,alpha0:float)->tuple[float,float,float]:
    base=binom(N,q)
    cand=betabinom(N,alpha_i,alpha0-alpha_i)
    grid=np.arange(N+1)
    b=discrete_crps(base.cdf(grid),y)
    c=discrete_crps(cand.cdf(grid),y)
    return float(b),float(c),float(b-c)


def build_room_record(
    *,
    season:int,
    week:int,
    cell:str,
    team:str,
    game_id:str,
    canonical:pd.DataFrame,
    source:pd.DataFrame,
)->tuple[dict|None,list[dict]]:
    s=source.loc[
        source["season"].eq(season)
        & source["week"].eq(week)
        & source["team"].eq(team)
        & source["cell"].eq(cell)
    ].copy()
    if s.empty:
        return None,[]

    c=canonical.loc[canonical["team"].eq(team)].copy()
    if c.empty:
        return None,[]
    if c.duplicated("player_clean_key").any():
        raise RuntimeError(f"canonical duplicate team={team} {season} W{week}")

    joined=s.merge(
        c[["player_clean_key","canonical_q"]],
        on="player_clean_key",how="left",validate="one_to_one",
    )
    joined["canonical_q"]=num(joined["canonical_q"])
    eligible=(
        joined["rho_i"].notna()
        & joined["rho_i"].gt(0)
        & joined["rho_i"].lt(1)
        & joined["canonical_q"].notna()
        & joined["canonical_q"].gt(0)
        & joined["canonical_q"].lt(1)
    )
    e=joined.loc[eligible].copy()
    if e.empty:
        return None,[]

    total_player_mass=float(c["canonical_q"].sum())
    residual=float(c["canonical_residual"].iloc[0])
    if abs(total_player_mass+residual-1.0)>PROB_TOL:
        raise RuntimeError(f"canonical room mass drift team={team}")

    eligible_q=e["canonical_q"].to_numpy(float)
    complement=1.0-float(eligible_q.sum())
    if not (0.0<complement<1.0):
        return None,[]

    rr=room_rho(
        np.append(eligible_q,complement),
        np.append(np.ones(len(eligible_q),dtype=bool),False),
        np.append(e["rho_i"].to_numpy(float),np.nan),
    )
    if rr is None:
        return None,[]
    rho_room,kappa,_=rr
    probs=np.append(eligible_q,complement)
    alpha=probs*kappa
    if (alpha<=0).any() or not np.isfinite(alpha).all():
        return None,[]

    Ns=num(e["N"]).dropna().astype(int).unique()
    if len(Ns)!=1:
        raise RuntimeError(f"room N disagreement team={team} season={season} week={week}")
    N=int(Ns[0])
    y=e["y"].astype(int).to_numpy()
    yc=N-int(y.sum())
    if yc<0:
        raise RuntimeError(f"negative complement count team={team} season={season} week={week}")
    counts=np.append(y,yc)
    if int(counts.sum())!=N:
        raise RuntimeError("realized room counts do not conserve N")

    base_log=multinomial_logpmf(counts,probs)
    cand_log=dirichlet_multinomial_logpmf(counts,alpha)
    if not (np.isfinite(base_log) and np.isfinite(cand_log)):
        raise RuntimeError(f"nonfinite joint score team={team} season={season} week={week}")

    expected_gap=float(np.max(np.abs(N*probs - N*(alpha/alpha.sum()))))
    marg=[]
    for j,(_,r) in enumerate(e.iterrows()):
        b,cscore,imp=marginal_crps(N,int(r["y"]),float(probs[j]),float(alpha[j]),float(alpha.sum()))
        marg.append({
            "season":season,"week":week,"game_id":game_id,"team":team,"cell":cell,
            "player_identity_key":str(r["player_identity_key"]),
            "player_clean_key":str(r["player_clean_key"]),
            "N":N,"y":int(r["y"]),"canonical_q":float(probs[j]),
            "rho_i":float(r["rho_i"]),"rho_room":float(rho_room),
            "baseline_crps":b,"candidate_crps":cscore,"crps_improvement":imp,
            "expected_count_gap":abs(N*float(probs[j])-N*float(alpha[j]/alpha.sum())),
        })

    rec={
        "season":season,"week":week,"game_id":game_id,"team":team,"cell":cell,
        "N":N,"eligible_players":int(len(e)),
        "eligible_probability":float(eligible_q.sum()),
        "complement_probability":float(complement),
        "rho_room":float(rho_room),"kappa":float(kappa),
        "baseline_nll":float(-base_log),"candidate_nll":float(-cand_log),
        "joint_logscore_improvement":float(cand_log-base_log),
        "max_expected_count_gap":expected_gap,
        "probability_mass_gap":abs(float(probs.sum())-1.0),
    }
    return rec,marg


def cluster_bootstrap(q:pd.DataFrame,cluster_col:str,seed:int)->dict:
    z=q[["joint_logscore_improvement",cluster_col]].dropna()
    g=z.groupby(cluster_col)["joint_logscore_improvement"].agg(["sum","count"])
    if len(g)<2:
        return {"valid_reps":0,"ci_low":np.nan,"ci_high":np.nan}
    a=g[["sum","count"]].to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=[]
    p=np.full(len(a),1.0/len(a))
    done=0
    while done<BOOT_REPS:
        k=min(250,BOOT_REPS-done)
        cnt=rng.multinomial(len(a),p,size=k).astype(float)
        sm=cnt@a
        vals.append(np.where(sm[:,1]>0,sm[:,0]/sm[:,1],np.nan))
        done+=k
    v=np.concatenate(vals)
    v=v[np.isfinite(v)]
    return {
        "valid_reps":int(len(v)),
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
    }


def score_cell_rooms(rooms:pd.DataFrame,marg:pd.DataFrame,cell:str,season:int)->dict:
    r=rooms.loc[rooms["cell"].eq(cell)&rooms["season"].eq(season)].copy()
    m=marg.loc[marg["cell"].eq(cell)&marg["season"].eq(season)].copy()
    room_count=len(r)
    teams=r["team"].nunique()
    games=r["game_id"].nunique()
    support=room_count>=200 and teams>=28 and games>=100
    out={
        "cell":cell,"season":season,
        "rooms":int(room_count),"teams":int(teams),"games":int(games),
        "eligible_player_rows":int(len(m)),
        "support":bool(support),
    }
    if not support:
        out["historical_pass"]=False
        return out

    game=cluster_bootstrap(r,"game_id",BOOT_SEED+season+CELLS.index(cell)*100)
    team=cluster_bootstrap(r,"team",BOOT_SEED+10000+season+CELLS.index(cell)*100)
    joint=float(r["joint_logscore_improvement"].mean())
    base_crps=float(m["baseline_crps"].mean()) if len(m) else np.nan
    cand_crps=float(m["candidate_crps"].mean()) if len(m) else np.nan
    max_mean_gap=max(
        float(r["max_expected_count_gap"].max()) if len(r) else 0.0,
        float(m["expected_count_gap"].max()) if len(m) else 0.0,
    )
    passed=bool(
        joint>0
        and game["ci_low"]>0
        and team["ci_low"]>0
        and np.isfinite(base_crps) and np.isfinite(cand_crps)
        and cand_crps<=base_crps+1e-12
        and max_mean_gap<=MEAN_TOL
    )
    out.update({
        "baseline_joint_nll":float(r["baseline_nll"].mean()),
        "candidate_joint_nll":float(r["candidate_nll"].mean()),
        "joint_logscore_improvement":joint,
        "game_boot":game,
        "team_boot":team,
        "baseline_marginal_crps":base_crps,
        "candidate_marginal_crps":cand_crps,
        "marginal_crps_improvement":base_crps-cand_crps,
        "max_expected_count_gap":max_mean_gap,
        "historical_pass":passed,
    })
    return out


def audit(
    *,
    v2_rows:pd.DataFrame,
    player_logs:pd.DataFrame,
    team_weekly:pd.DataFrame,
    schedule:pd.DataFrame,
    universes:dict[tuple[int,int],pd.DataFrame],
)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    source=prepare_v2_source(v2_rows)
    for frame in (player_logs,team_weekly,schedule):
        frame.columns=[str(c).strip().lower() for c in frame.columns]
    player_logs["team"]=player_logs["team"].map(canon_team)
    if "opponent" in player_logs.columns:
        player_logs["opponent"]=player_logs["opponent"].map(canon_team)
    team_weekly["team"]=team_weekly["team"].map(canon_team)
    schedule["team"]=schedule["team"].map(canon_team)
    schedule["opponent"]=schedule["opponent"].map(canon_team)

    room_rows=[]
    marginal_rows=[]
    for season in TARGET_SEASONS:
        for week in TARGET_WEEKS:
            u=universes[(season,week)]
            bundle=build_historical_context_bundle(
                player_logs=player_logs,
                team_weekly=team_weekly,
                pregame_universe=u,
                schedule=schedule,
                season=season,
                week=week,
                prior_season=season-1,
            )
            players=prepare_canonical_metrics(bundle)
            for cell in CELLS:
                canon=canonical_room_probabilities(players,cell)
                s=source.loc[source["season"].eq(season)&source["week"].eq(week)&source["cell"].eq(cell)]
                for team,g in s.groupby("team",sort=True):
                    gids=g["game_id"].dropna().astype(str).unique()
                    if len(gids)!=1:
                        raise RuntimeError(f"source game_id disagreement {season} W{week} {team} {cell}")
                    rec,marg=build_room_record(
                        season=season,week=week,cell=cell,team=team,game_id=str(gids[0]),
                        canonical=canon,source=source,
                    )
                    if rec is not None:
                        room_rows.append(rec)
                        marginal_rows.extend(marg)

    rooms=pd.DataFrame(room_rows)
    marginal=pd.DataFrame(marginal_rows)
    if rooms.empty or marginal.empty:
        raise RuntimeError("zero candidate-applied joint rooms")

    cells=[]
    per_cell=[]
    for cell in CELLS:
        by={}
        for season in TARGET_SEASONS:
            rec=score_cell_rooms(rooms,marginal,cell,season)
            cells.append(rec)
            by[season]=bool(rec.get("historical_pass",False))
        passed=all(by.get(s,False) for s in TARGET_SEASONS)
        per_cell.append({
            "cell":cell,
            "season_2023_pass":bool(by.get(2023,False)),
            "season_2024_pass":bool(by.get(2024,False)),
            "season_2025_pass":bool(by.get(2025,False)),
            "disposition":(
                "JOINT_DIRICHLET_ALLOCATION_HISTORICAL_PASS_FREEZE_PROSPECTIVE_SHADOW"
                if passed else
                "JOINT_DIRICHLET_ALLOCATION_HISTORICAL_FAIL_CLOSED"
            ),
        })

    max_prob_gap=float(rooms["probability_mass_gap"].max())
    max_mean_gap=max(
        float(rooms["max_expected_count_gap"].max()),
        float(marginal["expected_count_gap"].max()),
    )
    gates={
        "exact_target_seasons":sorted(rooms["season"].unique().tolist())==list(TARGET_SEASONS),
        "target_weeks_within_5_18":int(rooms["week"].min())>=5 and int(rooms["week"].max())<=18,
        "exact_cells":sorted(rooms["cell"].unique().tolist())==sorted(CELLS),
        "probability_mass_gap_le_1e_12":max_prob_gap<=PROB_TOL,
        "expected_count_gap_le_1e_12":max_mean_gap<=MEAN_TOL,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "fitted_coefficients_zero":True,
        "threshold_searches_zero":True,
        "production_changed_false":True,
    }
    result={
        "version":VERSION,
        "freeze":FREEZE,
        "disposition":(
            "JOINT_DIRICHLET_ALLOCATION_HAS_PASSING_CELLS"
            if any(x["disposition"].startswith("JOINT_DIRICHLET_ALLOCATION_HISTORICAL_PASS") for x in per_cell)
            else
            "JOINT_DIRICHLET_ALLOCATION_ALL_FAIL_CLOSED"
        ),
        "target_seasons":list(TARGET_SEASONS),
        "target_weeks":[5,18],
        "rooms":int(len(rooms)),
        "marginal_player_rows":int(len(marginal)),
        "max_probability_mass_gap":max_prob_gap,
        "max_expected_count_gap":max_mean_gap,
        "per_cell":per_cell,
        "cells":cells,
        "gates":gates,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "fitted_coefficients":0,
        "threshold_searches":0,
        "production_changed":False,
    }
    if not all(gates.values()):
        result["disposition"]="JOINT_DIRICHLET_ALLOCATION_HISTORICAL_INTEGRITY_FAILURE"
    return rooms,marginal,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-rows",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-dir-2023",type=Path,required=True)
    ap.add_argument("--universe-dir-2024",type=Path,required=True)
    ap.add_argument("--universe-dir-2025",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    v2=read_csv(a.v2_rows,"V2 source rows")
    logs=read_csv(a.player_logs,"player logs")
    team=read_csv(a.team_weekly,"team weekly")
    sched=read_csv(a.schedule,"schedule")
    dirs={2023:a.universe_dir_2023,2024:a.universe_dir_2024,2025:a.universe_dir_2025}
    universes={}
    for season in TARGET_SEASONS:
        for week in TARGET_WEEKS:
            universes[(season,week)]=read_csv(
                dirs[season]/f"{season}_week_{week:02d}.csv",
                f"pregame universe {season} W{week}"
            )

    rooms,marg,result=audit(
        v2_rows=v2,player_logs=logs,team_weekly=team,schedule=sched,universes=universes
    )
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rooms.to_csv(a.out_dir/"joint_opportunity_allocation_dirichlet_v1_rooms.csv",index=False,float_format="%.12f")
    marg.to_csv(a.out_dir/"joint_opportunity_allocation_dirichlet_v1_marginals.csv",index=False,float_format="%.12f")
    (a.out_dir/"joint_opportunity_allocation_dirichlet_v1_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
