#!/usr/bin/env python3
"""PRISM player mechanism-instability trust V1.

Historical WR/TE reliability test only. The candidate is frozen before scoring.
No 2026 outcomes, sportsbook/OddsAPI, predictive model fit, projection change, or
production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

VERSION = "PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1"
DOMINANCE_RATIO = 1.5
MAX_PRIOR = 8
MIN_PRIOR = 4
N_BOOT = 5000
TOL = 1e-10

REQ = {
    "season", "week", "team", "player_clean_key",
    "prior8_abs_opportunity_error_mean", "prior8_abs_efficiency_error_mean",
    "current_opportunity_error", "current_efficiency_error", "current_total_error",
}


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"{label} missing {sorted(miss)}")
    return x


def mechanism_label(opp: float, eff: float) -> str:
    o=abs(float(opp)); e=abs(float(eff))
    if o >= DOMINANCE_RATIO * e:
        return "OPPORTUNITY_DOMINANT"
    if e >= DOMINANCE_RATIO * o:
        return "EFFICIENCY_DOMINANT"
    return "MIXED_OPPORTUNITY_EFFICIENCY"


def construct_state(x: pd.DataFrame, position: str) -> pd.DataFrame:
    y=x.copy()
    y["season"]=pd.to_numeric(y["season"],errors="coerce")
    y["week"]=pd.to_numeric(y["week"],errors="coerce")
    for c in (
        "prior8_abs_opportunity_error_mean","prior8_abs_efficiency_error_mean",
        "current_opportunity_error","current_efficiency_error","current_total_error",
    ):
        y[c]=pd.to_numeric(y[c],errors="coerce")
    if y[list(REQ)].isna().any().any():
        raise RuntimeError(f"{position} has nonfinite required fields")
    if y.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError(f"{position} duplicate player-week identity")

    y["decomposition_gap"]=(
        y["current_opportunity_error"] + y["current_efficiency_error"]
        - y["current_total_error"]
    ).abs()
    if float(y["decomposition_gap"].max()) > TOL:
        raise RuntimeError(f"{position} decomposition identity failure")

    y["current_mechanism_label"]=[
        mechanism_label(o,e)
        for o,e in zip(y["current_opportunity_error"],y["current_efficiency_error"])
    ]
    y["current_abs_total_error"]=y["current_total_error"].abs()
    y["current_abs_opportunity_component"]=y["current_opportunity_error"].abs()
    y["current_abs_efficiency_component"]=y["current_efficiency_error"].abs()

    y=y.sort_values(["season","player_clean_key","week","team"],kind="mergesort").reset_index(drop=True)
    rows=[]
    for (_, _), g in y.groupby(["season","player_clean_key"],sort=False):
        inds=list(g.index)
        for j, idx in enumerate(inds):
            prior_inds=inds[max(0,j-MAX_PRIOR):j]
            if len(prior_inds) < MIN_PRIOR:
                continue
            if not (y.loc[prior_inds,"week"] < y.loc[idx,"week"]).all():
                raise RuntimeError(f"{position} chronology violation for row {idx}")
            labels=y.loc[prior_inds,"current_mechanism_label"].tolist()
            switches=sum(a != b for a,b in zip(labels[:-1],labels[1:]))
            counts=pd.Series(labels).value_counts()
            rec=y.loc[idx].to_dict()
            rec.update({
                "position":position,
                "prior_component_rows":len(labels),
                "mechanism_switch_rate":float(switches/(len(labels)-1)),
                "prior_dominant_state_share":float(counts.max()/len(labels)),
                "prior_opportunity_dominant_count":int(counts.get("OPPORTUNITY_DOMINANT",0)),
                "prior_efficiency_dominant_count":int(counts.get("EFFICIENCY_DOMINANT",0)),
                "prior_mixed_count":int(counts.get("MIXED_OPPORTUNITY_EFFICIENCY",0)),
                "prior_component_min_week":int(y.loc[prior_inds,"week"].min()),
                "prior_component_max_week":int(y.loc[prior_inds,"week"].max()),
                "chronology_valid":True,
            })
            rows.append(rec)
    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"{position} zero candidate rows after strict-prior construction")
    return out


def rank_within_week(x: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    y=x.copy()
    for c in cols:
        y[c+"_rank"]=y.groupby(["season","week"])[c].rank(method="average",pct=True)
    return y


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    yy=np.asarray(y,dtype=float)
    cc=np.asarray(controls,dtype=float)
    X=np.column_stack([np.ones(len(yy)),cc])
    beta=np.linalg.lstsq(X,yy,rcond=None)[0]
    return yy-X@beta


def partial_corr(frame: pd.DataFrame, outcome: str) -> float:
    cols=[
        "mechanism_switch_rate_rank",
        outcome+"_rank",
        "prior8_abs_opportunity_error_mean_rank",
        "prior8_abs_efficiency_error_mean_rank",
    ]
    z=frame[cols].dropna()
    if len(z)<3:
        return np.nan
    controls=z[[
        "prior8_abs_opportunity_error_mean_rank",
        "prior8_abs_efficiency_error_mean_rank",
    ]].to_numpy(float)
    a=residualize(z["mechanism_switch_rate_rank"].to_numpy(float),controls)
    b=residualize(z[outcome+"_rank"].to_numpy(float),controls)
    if np.std(a)<=1e-12 or np.std(b)<=1e-12:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])


def raw_corr(frame: pd.DataFrame) -> float:
    z=frame[["mechanism_switch_rate_rank","current_abs_total_error_rank"]].dropna()
    if len(z)<3 or z.iloc[:,0].nunique()<2 or z.iloc[:,1].nunique()<2:
        return np.nan
    return float(z.iloc[:,0].corr(z.iloc[:,1]))


def clustered_bootstrap(
    frame: pd.DataFrame,
    cluster_col: str,
    seed: int,
) -> dict:
    clusters=frame[cluster_col].dropna().astype(str).unique()
    if len(clusters)<2:
        return {"reps":0,"ci_low":np.nan,"ci_high":np.nan,"p_positive":np.nan}
    groups={k:g for k,g in frame.groupby(frame[cluster_col].astype(str),sort=False)}
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(N_BOOT):
        sampled=rng.choice(clusters,size=len(clusters),replace=True)
        b=pd.concat([groups[str(k)] for k in sampled],ignore_index=True)
        v=partial_corr(b,"current_abs_total_error")
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"reps":0,"ci_low":np.nan,"ci_high":np.nan,"p_positive":np.nan}
    a=np.asarray(vals,float)
    return {
        "reps":int(len(a)),
        "ci_low":float(np.quantile(a,0.025)),
        "ci_high":float(np.quantile(a,0.975)),
        "p_positive":float(np.mean(a>0)),
    }


def score_split(x: pd.DataFrame, season: int, *, bootstrap: bool) -> dict:
    s=x.loc[x["season"].eq(season)].copy()
    if s.empty:
        raise RuntimeError(f"zero rows for season {season}")
    out={
        "season":season,
        "rows":int(len(s)),
        "players":int(s["player_clean_key"].nunique()),
        "teams":int(s["team"].nunique()),
        "primary_partial_rank_association":partial_corr(s,"current_abs_total_error"),
        "raw_rank_association":raw_corr(s),
        "opportunity_component_partial_association":partial_corr(s,"current_abs_opportunity_component"),
        "efficiency_component_partial_association":partial_corr(s,"current_abs_efficiency_component"),
        "mean_switch_rate":float(s["mechanism_switch_rate"].mean()),
        "median_switch_rate":float(s["mechanism_switch_rate"].median()),
        "mean_prior_dominant_state_share":float(s["prior_dominant_state_share"].mean()),
    }
    if bootstrap:
        out["player_cluster"]=clustered_bootstrap(s,"player_clean_key",20261017)
        out["team_week_cluster"]=clustered_bootstrap(s,"team_week_cluster",20261018)
    return out


def evaluate(x: pd.DataFrame, position: str, dev: int, conf: int) -> tuple[pd.DataFrame,dict]:
    cols=[
        "mechanism_switch_rate",
        "current_abs_total_error",
        "current_abs_opportunity_component",
        "current_abs_efficiency_component",
        "prior8_abs_opportunity_error_mean",
        "prior8_abs_efficiency_error_mean",
    ]
    r=rank_within_week(x,cols)
    r["team_week_cluster"]=(
        r["season"].astype(int).astype(str)+"-"
        +r["week"].astype(int).astype(str)+"-"
        +r["team"].astype(str)
    )
    development=score_split(r,dev,bootstrap=False)
    confirmation=score_split(r,conf,bootstrap=True)

    gates={
        "development_primary_positive":bool(development["primary_partial_rank_association"]>0),
        "confirmation_primary_positive":bool(confirmation["primary_partial_rank_association"]>0),
        "confirmation_player_cluster_ci_low_positive":bool(confirmation["player_cluster"]["ci_low"]>0),
        "confirmation_team_week_cluster_ci_low_positive":bool(confirmation["team_week_cluster"]["ci_low"]>0),
        "chronology_violations_zero":bool(r["chronology_valid"].all()),
        "decomposition_identity_pass":bool(float(r["decomposition_gap"].max())<=TOL),
    }
    qualified=all(gates.values())
    result={
        "position":position,
        "development":development,
        "confirmation":confirmation,
        "gates":gates,
        "qualified":qualified,
        "max_decomposition_identity_gap":float(r["decomposition_gap"].max()),
        "chronology_violations":int((~r["chronology_valid"]).sum()),
    }
    return r,result


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--wr",type=Path,required=True)
    ap.add_argument("--te",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    wr_state=construct_state(read_csv(args.wr,"WR mechanism panel"),"WR")
    te_state=construct_state(read_csv(args.te,"TE mechanism panel"),"TE")

    wr_rows,wr=evaluate(wr_state,"WR",2023,2024)
    te_rows,te=evaluate(te_state,"TE",2024,2025)

    if wr["qualified"] and te["qualified"]:
        status="PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_QUALIFIED_WR_TE"
    elif wr["qualified"] or te["qualified"]:
        status="PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_QUALIFIED_PARTIAL"
    else:
        status="PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_CLOSED"

    result={
        "version":VERSION,
        "status":status,
        "wr":wr,
        "te":te,
        "rb_status":"SOURCE_NOT_PROVEN_FOR_THIS_V1",
        "qb_status":"EXCLUDED_BY_ANTI_RETEST",
        "dominance_ratio":DOMINANCE_RATIO,
        "prior_window_max":MAX_PRIOR,
        "prior_window_min":MIN_PRIOR,
        "bootstrap_replicates":N_BOOT,
        "outcomes_2026_used":0,
        "week6_outcomes_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "predictive_models_fit":0,
        "threshold_searches":0,
        "projection_changed":False,
        "distribution_changed":False,
        "production_changed":False,
        "boundary":"historical trust/selectivity evidence only; no mean/width/selection change",
    }

    wr_rows.to_csv(args.out_dir/"wr_mechanism_instability_trust_rows.csv",index=False,float_format="%.12f")
    te_rows.to_csv(args.out_dir/"te_mechanism_instability_trust_rows.csv",index=False,float_format="%.12f")
    (args.out_dir/"mechanism_instability_trust_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
