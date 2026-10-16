#!/usr/bin/env python3
"""TE Player Allocation Joint Dirichlet V1.

Research-only finite-team joint target allocation diagnostic. Exact contract:
docs/research/TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_V1_FREEZE.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import binom, betabinom

VERSION="TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_V1"
SEASONS=(2024,2025)
MIN_ROOMS=200
MIN_TEAMS=28
MIN_GAMES=100
BOOT_REPS=5000
GAME_SEED=2026101421
TEAM_SEED=2026101422
PLAYER_SEED=2026101423
TOL=1e-8
PROB_TOL=1e-12


def num(s):
    return pd.to_numeric(s,errors="coerce")


def read_parent(path:Path)->pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    need={
        "season","week","game_id","team","player_identity_key","player",
        "position","actual_share","actual_targets","pred_team_targets",
        "promoted_share","promoted_targets","candidate_sigma",
    }
    miss=need-set(x.columns)
    if miss:
        raise RuntimeError(f"parent rows missing {sorted(miss)}")
    x=x.loc[x["position"].astype(str).str.upper().eq("TE")].copy()
    for c in [
        "season","week","actual_share","actual_targets","pred_team_targets",
        "promoted_share","promoted_targets","candidate_sigma",
    ]:
        x[c]=num(x[c])
    x=x.loc[x["season"].isin(SEASONS)].copy()
    x["season"]=x["season"].astype(int)
    x["week"]=x["week"].astype(int)
    x["player_identity_key"]=x["player_identity_key"].fillna("").astype(str).str.strip()
    x["team"]=x["team"].fillna("").astype(str).str.strip()
    x["game_id"]=x["game_id"].fillna("").astype(str).str.strip()
    if x["player_identity_key"].eq("").any():
        raise RuntimeError("blank stable TE identity")
    if x.duplicated(["season","week","team","player_identity_key"]).any():
        raise RuntimeError("duplicate TE player-week")
    return x.sort_values(["season","week","game_id","team","player_identity_key"]).reset_index(drop=True)


def add_rho(x:pd.DataFrame)->pd.DataFrame:
    q=x.copy()
    q["desired_var"]=q["candidate_sigma"]**2
    q["baseline_var_pred_n"]=(
        q["pred_team_targets"]*q["promoted_share"]*(1.0-q["promoted_share"])
    )
    denom=(q["pred_team_targets"]-1.0)*q["baseline_var_pred_n"]
    q["rho_i"]=np.where(
        q["pred_team_targets"].gt(1)
        & q["promoted_share"].gt(0)
        & q["promoted_share"].lt(1)
        & q["desired_var"].gt(q["baseline_var_pred_n"])
        & denom.gt(0),
        (q["desired_var"]/q["baseline_var_pred_n"]-1.0)/(q["pred_team_targets"]-1.0),
        np.nan,
    )
    q["eligible"]=(
        np.isfinite(q["rho_i"])
        & q["rho_i"].gt(0)
        & q["rho_i"].lt(1)
        & q["promoted_share"].gt(0)
        & q["promoted_share"].lt(1)
    )
    return q


def infer_room_n(g:pd.DataFrame)->tuple[int|None,str]:
    z=g.loc[
        g["actual_targets"].gt(0)
        & g["actual_share"].gt(0)
        & np.isfinite(g["actual_targets"])
        & np.isfinite(g["actual_share"])
    ].copy()
    if z.empty:
        return None,"NO_POSITIVE_TE_TARGET_FOR_N_INFERENCE"
    implied=z["actual_targets"]/z["actual_share"]
    if not np.isfinite(implied).all():
        return None,"NONFINITE_N_INFERENCE"
    n0=float(implied.iloc[0])
    if float(np.max(np.abs(implied.to_numpy(float)-n0)))>TOL:
        return None,"INCONSISTENT_N_INFERENCE"
    nr=round(n0)
    if abs(n0-nr)>TOL or nr<=0:
        return None,"NONINTEGER_N_INFERENCE"
    return int(nr),"OK"


def log_multinomial(counts:np.ndarray,probs:np.ndarray)->float:
    n=int(np.sum(counts))
    p=np.asarray(probs,dtype=float)
    c=np.asarray(counts,dtype=int)
    if (p<=0).any() or abs(float(p.sum())-1.0)>PROB_TOL:
        return np.nan
    return float(
        gammaln(n+1)-np.sum(gammaln(c+1))
        +np.sum(np.where(c>0,c*np.log(p),0.0))
    )


def log_dirichlet_multinomial(counts:np.ndarray,alpha:np.ndarray)->float:
    c=np.asarray(counts,dtype=int)
    a=np.asarray(alpha,dtype=float)
    n=int(c.sum())
    if (a<=0).any() or not np.isfinite(a).all():
        return np.nan
    a0=float(a.sum())
    return float(
        gammaln(n+1)-np.sum(gammaln(c+1))
        +gammaln(a0)-gammaln(a0+n)
        +np.sum(gammaln(c+a)-gammaln(a))
    )


def discrete_crps_binom(n:int,p:float,y:int)->float:
    k=np.arange(n+1,dtype=int)
    f=binom.cdf(k,n,p)
    obs=(k>=int(y)).astype(float)
    return float(np.sum((f-obs)**2))


def discrete_crps_betabinom(n:int,a:float,b:float,y:int)->float:
    k=np.arange(n+1,dtype=int)
    f=betabinom.cdf(k,n,a,b)
    obs=(k>=int(y)).astype(float)
    return float(np.sum((f-obs)**2))


def build_rooms(x:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    room_rows=[]
    marginal_rows=[]
    counters={
        "rooms_seen":0,
        "rooms_no_n":0,
        "rooms_no_eligible_te":0,
        "rooms_bad_probability":0,
        "rooms_bad_conservation":0,
        "rooms_scored":0,
        "n_inference_reasons":{},
    }

    keys=["season","week","game_id","team"]
    for key,g in x.groupby(keys,sort=True):
        counters["rooms_seen"]+=1
        n,reason=infer_room_n(g)
        if n is None:
            counters["rooms_no_n"]+=1
            counters["n_inference_reasons"][reason]=counters["n_inference_reasons"].get(reason,0)+1
            continue

        e=g.loc[g["eligible"]].copy()
        if e.empty:
            counters["rooms_no_eligible_te"]+=1
            continue

        probs=e["promoted_share"].to_numpy(float)
        rhos=e["rho_i"].to_numpy(float)
        if not np.isfinite(probs).all() or not np.isfinite(rhos).all():
            counters["rooms_bad_probability"]+=1
            continue
        complement=1.0-float(probs.sum())
        if not (complement>0.0 and complement<1.0):
            counters["rooms_bad_probability"]+=1
            continue

        weights=probs*(1.0-probs)
        den=float(weights.sum())
        if den<=0:
            counters["rooms_bad_probability"]+=1
            continue
        rho_room=float(np.sum(weights*rhos)/den)
        if not (0.0<rho_room<1.0):
            counters["rooms_bad_probability"]+=1
            continue
        kappa=1.0/rho_room-1.0

        p=np.append(probs,complement)
        if abs(float(p.sum())-1.0)>PROB_TOL or (p<=0).any():
            counters["rooms_bad_probability"]+=1
            continue
        alpha=p*kappa
        actual=num(e["actual_targets"]).to_numpy(float)
        if not np.isfinite(actual).all() or np.max(np.abs(actual-np.rint(actual)))>TOL:
            counters["rooms_bad_conservation"]+=1
            continue
        actual=np.rint(actual).astype(int)
        comp_count=int(n-int(actual.sum()))
        if comp_count<0:
            counters["rooms_bad_conservation"]+=1
            continue
        counts=np.append(actual,comp_count)
        if int(counts.sum())!=n:
            counters["rooms_bad_conservation"]+=1
            continue

        lb=log_multinomial(counts,p)
        lc=log_dirichlet_multinomial(counts,alpha)
        if not np.isfinite(lb) or not np.isfinite(lc):
            counters["rooms_bad_probability"]+=1
            continue
        gain=float(lc-lb)  # NLL_baseline - NLL_candidate

        season,week,game_id,team=key
        room_rows.append({
            "season":int(season),"week":int(week),"game_id":str(game_id),"team":str(team),
            "N":int(n),"eligible_tes":int(len(e)),"complement_probability":complement,
            "rho_room":rho_room,"kappa":kappa,
            "baseline_logprob":lb,"candidate_logprob":lc,"joint_logscore_gain":gain,
            "probability_sum":float(p.sum()),"count_sum":int(counts.sum()),
        })

        a0=float(alpha.sum())
        for j,r in enumerate(e.itertuples(index=False)):
            ai=float(alpha[j]); bi=float(a0-ai); y=int(actual[j]); qi=float(probs[j])
            bcrps=discrete_crps_binom(n,qi,y)
            ccrps=discrete_crps_betabinom(n,ai,bi,y)
            expected_base=n*qi
            expected_cand=n*ai/a0
            marginal_rows.append({
                "season":int(season),"week":int(week),"game_id":str(game_id),"team":str(team),
                "player_identity_key":str(r.player_identity_key),"player":str(r.player),
                "N":int(n),"actual_targets":y,"q_i":qi,"rho_i":float(r.rho_i),
                "alpha_i":ai,"alpha_rest":bi,
                "baseline_crps":bcrps,"candidate_crps":ccrps,
                "crps_gain":bcrps-ccrps,
                "expected_base":expected_base,"expected_candidate":expected_cand,
                "expected_count_gap":abs(expected_base-expected_cand),
            })
        counters["rooms_scored"]+=1

    return pd.DataFrame(room_rows),pd.DataFrame(marginal_rows),counters


def cluster_boot(q:pd.DataFrame,value_col:str,cluster_col:str,seed:int)->dict:
    z=q[[value_col,cluster_col]].dropna().copy()
    groups={str(k):g[value_col].to_numpy(float) for k,g in z.groupby(cluster_col,sort=True)}
    keys=np.array(sorted(groups),dtype=object)
    observed=float(z[value_col].mean()) if len(z) else np.nan
    if len(keys)<2:
        return {"reps":0,"observed":observed,"ci_low":np.nan,"ci_high":np.nan,"p_positive":np.nan}
    rng=np.random.default_rng(seed)
    vals=np.empty(BOOT_REPS,dtype=float)
    for i in range(BOOT_REPS):
        sample=rng.choice(keys,size=len(keys),replace=True)
        vals[i]=float(np.concatenate([groups[str(k)] for k in sample]).mean())
    return {
        "reps":BOOT_REPS,"observed":observed,
        "ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975)),
        "p_positive":float(np.mean(vals>0)),
    }


def season_metrics(rooms:pd.DataFrame,marg:pd.DataFrame,season:int)->dict:
    r=rooms.loc[rooms["season"].eq(season)].copy()
    m=marg.loc[marg["season"].eq(season)].copy()
    gb=cluster_boot(r,"joint_logscore_gain","game_id",GAME_SEED+season)
    tb=cluster_boot(r,"joint_logscore_gain","team",TEAM_SEED+season)
    return {
        "rooms":int(len(r)),
        "teams":int(r["team"].nunique()) if len(r) else 0,
        "games":int(r["game_id"].nunique()) if len(r) else 0,
        "eligible_te_marginals":int(len(m)),
        "eligible_tes":int(m["player_identity_key"].nunique()) if len(m) else 0,
        "mean_joint_gain":float(r["joint_logscore_gain"].mean()) if len(r) else np.nan,
        "game_cluster":gb,
        "team_cluster":tb,
        "baseline_marginal_crps":float(m["baseline_crps"].mean()) if len(m) else np.nan,
        "candidate_marginal_crps":float(m["candidate_crps"].mean()) if len(m) else np.nan,
        "marginal_crps_gain":float(m["crps_gain"].mean()) if len(m) else np.nan,
        "max_expected_count_gap":float(m["expected_count_gap"].max()) if len(m) else np.nan,
    }


def player_confirmation(marg:pd.DataFrame)->dict:
    q=marg.loc[marg["season"].eq(2025)].copy()
    boot=cluster_boot(q,"crps_gain","player_identity_key",PLAYER_SEED)
    vals=[]
    for _,g in q.groupby("player_identity_key"):
        if len(g)<6:
            continue
        vals.append(float(g["crps_gain"].mean()))
    a=np.asarray(vals,dtype=float)
    return {
        "player_cluster":boot,
        "qualifying_players_ge6":int(len(a)),
        "median_per_player_gain":float(np.median(a)) if len(a) else np.nan,
        "fraction_players_positive":float(np.mean(a>0)) if len(a) else np.nan,
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    parent=add_rho(read_parent(a.parent_rows))
    rooms,marg,counters=build_rooms(parent)
    if rooms.empty:
        raise RuntimeError("zero scoreable TE rooms")

    by={str(s):season_metrics(rooms,marg,s) for s in SEASONS}
    pc=player_confirmation(marg)

    support=all(
        by[str(s)]["rooms"]>=MIN_ROOMS
        and by[str(s)]["teams"]>=MIN_TEAMS
        and by[str(s)]["games"]>=MIN_GAMES
        for s in SEASONS
    )
    integrity={
        "exact_seasons":sorted(rooms["season"].unique().tolist())==list(SEASONS),
        "probability_sums_exact":bool((rooms["probability_sum"].sub(1).abs()<=PROB_TOL).all()),
        "count_conservation_exact":bool((rooms["count_sum"]==rooms["N"]).all()),
        "expected_count_gap_le_1e12":bool((marg["expected_count_gap"]<=1e-12).all()),
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "outcomes_2026_zero":True,
        "projection_mean_changes_zero":True,
        "production_changed_false":True,
    }
    gates={"support":bool(support)}
    for s in SEASONS:
        z=by[str(s)]
        gates[f"{s}_joint_gain_positive"]=bool(z["mean_joint_gain"]>0)
        gates[f"{s}_game_ci_low_positive"]=bool(z["game_cluster"]["ci_low"]>0)
        gates[f"{s}_team_ci_low_positive"]=bool(z["team_cluster"]["ci_low"]>0)
        gates[f"{s}_marginal_crps_nonworse"]=bool(z["candidate_marginal_crps"]<=z["baseline_marginal_crps"])
        gates[f"{s}_expected_count_gap_le_1e12"]=bool(z["max_expected_count_gap"]<=1e-12)
    gates["2025_player_ci_low_positive"]=bool(pc["player_cluster"]["ci_low"]>0)
    gates["2025_median_player_gain_positive"]=bool(pc["median_per_player_gain"]>0)
    gates["2025_fraction_players_positive_gt_half"]=bool(pc["fraction_players_positive"]>0.50)

    if not all(integrity.values()):
        disp="TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_INTEGRITY_FAILURE"
    elif not support:
        disp="TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_SOURCE_LIMITED"
    elif all(gates.values()):
        disp="TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_CONFIRMED"
    else:
        disp="NO_ACTIONABLE_TE_PLAYER_ALLOCATION_JOINT_DIRICHLET"

    result={
        "version":VERSION,"disposition":disp,
        "parent_rows":int(len(parent)),"parent_players":int(parent["player_identity_key"].nunique()),
        "eligible_player_rows":int(parent["eligible"].sum()),
        "eligible_players":int(parent.loc[parent["eligible"],"player_identity_key"].nunique()),
        "room_counters":counters,
        "season_metrics":by,
        "player_confirmation_2025":pc,
        "gates":gates,
        "failed_gates":[k for k,v in gates.items() if not v],
        "integrity":integrity,
        "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,
        "projection_mean_changes":0,"production_changed":False,
    }
    rooms.to_csv(a.out_dir/"te_player_allocation_joint_dirichlet_rooms.csv",index=False)
    marg.to_csv(a.out_dir/"te_player_allocation_joint_dirichlet_marginals.csv",index=False)
    (a.out_dir/"te_player_allocation_joint_dirichlet_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
