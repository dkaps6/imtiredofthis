#!/usr/bin/env python3
"""Score frozen mean-identical Beta-Binomial opportunity uncertainty candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom

from scripts.research.audit_player_opportunity_share_stability_v1 import (
    build_weekly_universe,
)
from scripts.research.audit_player_opportunity_share_stability_2023_holdout_v1 import (
    build_2023_weekly_universe,
)

VERSION="PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1"
FREEZE="docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_FREEZE.md"
TARGET_SEASONS=(2023,2024,2025)
TARGET_WEEKS=tuple(range(5,19))
CELLS=("RB_CARRY_SHARE","RB_TARGET_SHARE","WR_TARGET_SHARE")
BOOT_REPS=5000
BOOT_SEED=20261012
MEAN_TOL=1e-12
SHARE_RECON_TOL=1e-9


def _num(s)->pd.Series:
    return pd.to_numeric(s,errors="coerce")


def read_csv(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def build_target_counts()->pd.DataFrame:
    u=pd.concat(
        [build_2023_weekly_universe(),build_weekly_universe()],
        ignore_index=True,
    )
    u=u.loc[
        _num(u["season"]).isin(TARGET_SEASONS)
        & _num(u["week"]).isin(TARGET_WEEKS)
    ].copy()
    u["season"]=_num(u["season"]).astype(int)
    u["week"]=_num(u["week"]).astype(int)
    keep=[
        "season","week","game_id","team","player_identity_key","position",
        "targets","rushes","team_targets","team_rushes",
    ]
    missing=set(keep)-set(u.columns)
    if missing:
        raise RuntimeError(f"target-count universe missing {sorted(missing)}")
    out=u[keep].copy()
    if out.duplicated(["season","week","team","player_identity_key"]).any():
        raise RuntimeError("duplicate target-count player-week identity")
    return out


def attach_counts(source:pd.DataFrame,counts:pd.DataFrame)->pd.DataFrame:
    req={
        "season","week","game_id","team","player_identity_key","cell",
        "actual_share","prior_share_mean","excess_share_var",
        "prior_feature_max_week","denominator_source_max_week",
    }
    miss=req-set(source.columns)
    if miss:
        raise RuntimeError(f"V2 source missing {sorted(miss)}")
    x=source.loc[
        _num(source["season"]).isin(TARGET_SEASONS)
        & _num(source["week"]).isin(TARGET_WEEKS)
        & source["cell"].astype(str).isin(CELLS)
    ].copy()
    x["season"]=_num(x["season"]).astype(int)
    x["week"]=_num(x["week"]).astype(int)
    if x.duplicated(["season","week","team","player_identity_key","cell"]).any():
        raise RuntimeError("duplicate V2 source row")

    c=counts.drop(columns=["game_id","position"],errors="ignore").copy()
    x=x.merge(
        c,
        on=["season","week","team","player_identity_key"],
        how="left",
        validate="many_to_one",
    )
    if x[["targets","rushes","team_targets","team_rushes"]].isna().all(axis=1).any():
        raise RuntimeError(
            f"target-count identity misses={int(x[['targets','rushes','team_targets','team_rushes']].isna().all(axis=1).sum())}"
        )

    is_carry=x["cell"].eq("RB_CARRY_SHARE")
    x["N"]=np.where(is_carry,_num(x["team_rushes"]),_num(x["team_targets"]))
    x["y"]=np.where(is_carry,_num(x["rushes"]),_num(x["targets"]))
    x["N"]=_num(x["N"])
    x["y"]=_num(x["y"])

    valid=(
        x["N"].gt(0)
        & x["N"].notna()
        & x["y"].notna()
        & x["y"].ge(0)
        & x["y"].le(x["N"])
    )
    x=x.loc[valid].copy()
    x["N"]=np.rint(x["N"]).astype(int)
    x["y"]=np.rint(x["y"]).astype(int)

    recon=x["y"]/x["N"]
    x["share_reconstruction_gap"]=(recon-_num(x["actual_share"])).abs()
    max_gap=float(x["share_reconstruction_gap"].max()) if len(x) else np.inf
    if max_gap>SHARE_RECON_TOL:
        raise RuntimeError(f"target-count share reconstruction drift max_gap={max_gap}")

    if (
        (_num(x["prior_feature_max_week"])>=x["week"]).any()
        or (_num(x["denominator_source_max_week"])>=x["week"]).any()
    ):
        raise RuntimeError("same/future V2 feature source detected")
    return x.reset_index(drop=True)


def beta_parameters(mu:float,v:float)->tuple[float,float,float] | None:
    if not (np.isfinite(mu) and np.isfinite(v)):
        return None
    if not (0.0<mu<1.0) or v<=0:
        return None
    upper=mu*(1.0-mu)
    if not (0.0<v<upper):
        return None
    kappa=upper/v-1.0
    alpha=mu*kappa
    beta=(1.0-mu)*kappa
    if not (np.isfinite(kappa) and np.isfinite(alpha) and np.isfinite(beta)):
        return None
    if kappa<=0 or alpha<=0 or beta<=0:
        return None
    return alpha,beta,kappa


def discrete_crps(cdf:np.ndarray,y:int)->float:
    k=np.arange(len(cdf),dtype=int)
    obs=(k>=int(y)).astype(float)
    return float(np.sum(np.square(cdf-obs)))


def central_interval(dist,level:float)->tuple[int,int]:
    tail=(1.0-level)/2.0
    lo=int(dist.ppf(tail))
    hi=int(dist.ppf(1.0-tail))
    return lo,hi


def score_row(r:pd.Series)->dict[str,Any]:
    N=int(r["N"]); y=int(r["y"]); mu=float(r["prior_share_mean"])
    v=float(r["excess_share_var"]) if pd.notna(r["excess_share_var"]) else np.nan
    if not (0.0<=mu<=1.0):
        raise RuntimeError(f"invalid prior mean share {mu}")

    base=binom(N,mu)
    base_cdf=base.cdf(np.arange(N+1))
    base_crps=discrete_crps(base_cdf,y)
    base_log=float(base.logpmf(y))

    params=beta_parameters(mu,v)
    candidate_changed=params is not None
    invalid_reason=""
    if params is None:
        cand=base
        cand_cdf=base_cdf
        cand_crps=base_crps
        cand_log=base_log
        paired_nll_improvement=0.0
        alpha=beta=kappa=np.nan
        if not np.isfinite(v):
            invalid_reason="NONFINITE_EXCESS_VAR"
        elif v<=0:
            invalid_reason="NO_POSITIVE_EXCESS_VAR"
        elif not (0.0<mu<1.0):
            invalid_reason="DEGENERATE_MEAN_SHARE"
        else:
            invalid_reason="INVALID_METHOD_OF_MOMENTS_VARIANCE"
    else:
        alpha,beta,kappa=params
        cand=betabinom(N,alpha,beta)
        cand_cdf=cand.cdf(np.arange(N+1))
        cand_crps=discrete_crps(cand_cdf,y)
        cand_log=float(cand.logpmf(y))
        if not (np.isfinite(cand_crps) and np.isfinite(base_log) and np.isfinite(cand_log)):
            raise RuntimeError("non-finite changed-row analytic score")
        paired_nll_improvement=float(cand_log-base_log)

    expected_base=N*mu
    expected_cand=N*mu
    b80=central_interval(base,.80); c80=central_interval(cand,.80)
    b90=central_interval(base,.90); c90=central_interval(cand,.90)
    return {
        "candidate_changed":bool(candidate_changed),
        "candidate_ineligible_reason":invalid_reason,
        "alpha":alpha,"beta":beta,"kappa":kappa,
        "baseline_expected_count":expected_base,
        "candidate_expected_count":expected_cand,
        "expected_count_gap":abs(expected_base-expected_cand),
        "baseline_crps":base_crps,
        "candidate_crps":cand_crps,
        "crps_improvement":base_crps-cand_crps,
        "baseline_nll":(-base_log if np.isfinite(base_log) else np.inf),
        "candidate_nll":(-cand_log if np.isfinite(cand_log) else np.inf),
        "paired_nll_improvement":paired_nll_improvement,
        "baseline_80_low":b80[0],"baseline_80_high":b80[1],
        "candidate_80_low":c80[0],"candidate_80_high":c80[1],
        "baseline_90_low":b90[0],"baseline_90_high":b90[1],
        "candidate_90_low":c90[0],"candidate_90_high":c90[1],
        "baseline_80_hit":bool(b80[0]<=y<=b80[1]),
        "candidate_80_hit":bool(c80[0]<=y<=c80[1]),
        "baseline_90_hit":bool(b90[0]<=y<=b90[1]),
        "candidate_90_hit":bool(c90[0]<=y<=c90[1]),
    }


def score_rows(frame:pd.DataFrame)->pd.DataFrame:
    extras=[score_row(r) for _,r in frame.iterrows()]
    return pd.concat([frame.reset_index(drop=True),pd.DataFrame(extras)],axis=1)


def cluster_bootstrap(q:pd.DataFrame,cluster_col:str,seed:int)->dict[str,Any]:
    z=q[["crps_improvement",cluster_col]].dropna().copy()
    g=z.groupby(cluster_col)["crps_improvement"].agg(["sum","count"])
    if len(g)<2:
        return {"valid_reps":0,"ci_low":np.nan,"ci_high":np.nan}
    a=g[["sum","count"]].to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=[]
    done=0
    p=np.full(len(a),1.0/len(a))
    while done<BOOT_REPS:
        k=min(250,BOOT_REPS-done)
        counts=rng.multinomial(len(a),p,size=k).astype(float)
        sums=counts@a
        vals.append(np.where(sums[:,1]>0,sums[:,0]/sums[:,1],np.nan))
        done+=k
    v=np.concatenate(vals)
    v=v[np.isfinite(v)]
    return {
        "valid_reps":int(len(v)),
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
    }


def cell_result(q:pd.DataFrame,cell:str,season:int)->dict[str,Any]:
    z=q.loc[q["cell"].eq(cell)&q["season"].eq(season)].copy()
    rows=len(z)
    players=z["player_identity_key"].nunique()
    games=z["game_id"].nunique()
    support=rows>=300 and players>=100 and games>=50
    out={
        "cell":cell,"season":season,
        "rows":int(rows),"players":int(players),"games":int(games),
        "support":bool(support),
        "changed_rows":int(z["candidate_changed"].sum()) if rows else 0,
        "changed_rate":float(z["candidate_changed"].mean()) if rows else np.nan,
        "max_expected_count_gap":float(z["expected_count_gap"].max()) if rows else np.nan,
        "max_share_reconstruction_gap":float(z["share_reconstruction_gap"].max()) if rows else np.nan,
    }
    if not support:
        out["historical_pass"]=False
        return out
    game=cluster_bootstrap(z,"game_id",BOOT_SEED+season+CELLS.index(cell)*100)
    player=cluster_bootstrap(
        z,"player_identity_key",
        BOOT_SEED+10000+season+CELLS.index(cell)*100,
    )
    base_crps=float(z["baseline_crps"].mean())
    cand_crps=float(z["candidate_crps"].mean())
    paired_nll=float(z["paired_nll_improvement"].mean())
    changed=z.loc[z["candidate_changed"]].copy()
    changed_paired_nll=(
        float(changed["paired_nll_improvement"].mean()) if len(changed) else 0.0
    )
    pass_cell=bool(
        cand_crps<base_crps
        and game["ci_low"]>0
        and player["ci_low"]>0
        and paired_nll>=-1e-12
        and float(z["expected_count_gap"].max())<=MEAN_TOL
    )
    out.update({
        "baseline_crps":base_crps,
        "candidate_crps":cand_crps,
        "crps_improvement":base_crps-cand_crps,
        "game_boot":game,
        "player_boot":player,
        "mean_paired_nll_improvement":paired_nll,
        "changed_rows_mean_paired_nll_improvement":changed_paired_nll,
        "nonfinite_identical_baseline_rows":int(
            ((~z["candidate_changed"]) & ~np.isfinite(z["baseline_nll"])).sum()
        ),
        "baseline_80_coverage":float(z["baseline_80_hit"].mean()),
        "candidate_80_coverage":float(z["candidate_80_hit"].mean()),
        "baseline_90_coverage":float(z["baseline_90_hit"].mean()),
        "candidate_90_coverage":float(z["candidate_90_hit"].mean()),
        "historical_pass":pass_cell,
    })
    return out


def audit(source:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    counts=build_target_counts()
    x=attach_counts(source,counts)
    scored=score_rows(x)

    cells=[]
    per_cell=[]
    for cell in CELLS:
        by={}
        for season in TARGET_SEASONS:
            rec=cell_result(scored,cell,season)
            cells.append(rec)
            by[season]=bool(rec.get("historical_pass",False))
        passed=all(by.get(s,False) for s in TARGET_SEASONS)
        per_cell.append({
            "cell":cell,
            "season_2023_pass":bool(by.get(2023,False)),
            "season_2024_pass":bool(by.get(2024,False)),
            "season_2025_pass":bool(by.get(2025,False)),
            "disposition":(
                "BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_PASS"
                if passed else
                "BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_FAIL_CLOSED"
            ),
        })

    reasons=(
        scored.loc[~scored["candidate_changed"],"candidate_ineligible_reason"]
        .value_counts(dropna=False).to_dict()
    )
    max_mean_gap=float(scored["expected_count_gap"].max()) if len(scored) else np.inf
    max_recon=float(scored["share_reconstruction_gap"].max()) if len(scored) else np.inf
    gates={
        "exact_target_seasons":sorted(scored["season"].unique().tolist())==list(TARGET_SEASONS),
        "exact_cells":sorted(scored["cell"].unique().tolist())==sorted(CELLS),
        "target_weeks_within_5_18":int(scored["week"].min())>=5 and int(scored["week"].max())<=18,
        "mean_identity_gap_le_1e_12":max_mean_gap<=MEAN_TOL,
        "share_reconstruction_gap_le_1e_9":max_recon<=SHARE_RECON_TOL,
        "same_future_feature_violations_zero":bool(
            (_num(scored["prior_feature_max_week"])<scored["week"]).all()
            and (_num(scored["denominator_source_max_week"])<scored["week"]).all()
        ),
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
            "BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HAS_PASSING_CELLS"
            if any(x["disposition"]=="BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_PASS" for x in per_cell)
            else
            "BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_ALL_FAIL_CLOSED"
        ),
        "target_seasons":list(TARGET_SEASONS),
        "target_weeks":[5,18],
        "cells_authorized":list(CELLS),
        "rows":int(len(scored)),
        "players":int(scored["player_identity_key"].nunique()),
        "candidate_changed_rows":int(scored["candidate_changed"].sum()),
        "candidate_ineligible_reasons":{str(k):int(v) for k,v in reasons.items()},
        "max_expected_count_gap":max_mean_gap,
        "max_share_reconstruction_gap":max_recon,
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
        result["disposition"]="BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_INTEGRITY_FAILURE"
    return scored,result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    source=read_csv(a.v2_rows,"V2 mechanism rows")
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rows,result=audit(source)
    rows.to_csv(
        a.out_dir/"player_opportunity_allocation_uncertainty_betabinomial_v1_rows.csv",
        index=False,float_format="%.12f"
    )
    (a.out_dir/"player_opportunity_allocation_uncertainty_betabinomial_v1_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
