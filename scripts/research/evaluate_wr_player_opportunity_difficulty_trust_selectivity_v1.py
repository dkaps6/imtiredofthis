#!/usr/bin/env python3
"""Frozen WR player opportunity-difficulty trust/selectivity V1.

Research-only trust label around the unchanged WR-R15 receiving-yard point
projection. No model value is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION="WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_V1"
QUALIFIED="WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED"
CLOSED="NO_ACTIONABLE_WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY"
INTEGRITY="WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_INTEGRITY_FAILURE"
WR_VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_ROWS=3224
EXPECTED_PLAYERS=211
BOOT_REPS=5000
POOL_PLAYER_SEED=2026101521
POOL_GAME_SEED=2026101522
Y24_PLAYER_SEED=2026101523
Y24_GAME_SEED=2026101524
TOL=1e-9


def read(path:Path)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing/empty input: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def num(s): return pd.to_numeric(s,errors="coerce")


def corr(a,b)->float:
    x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<3 or float(np.std(x))<=0 or float(np.std(y))<=0:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def rank_pct(s:pd.Series)->pd.Series:
    return num(s).rank(method="average",pct=True)


def prepare(parent_path:Path,wr_path:Path)->tuple[pd.DataFrame,dict]:
    p=read(parent_path)
    reqp={
      "season","week","team","player_clean_key","wr_rank","prior_games",
      "prior8_abs_opportunity_error_mean","prior8_abs_efficiency_error_mean",
      "current_opportunity_error","current_efficiency_error",
      "current_total_error","current_abs_total_error",
    }
    if reqp-set(p.columns):
        raise RuntimeError(f"parent missing {sorted(reqp-set(p.columns))}")
    for c in reqp-{"team","player_clean_key"}:
        p[c]=num(p[c])
    p=p.loc[p["season"].isin([2023,2024])&p["prior_games"].ge(4)].copy()

    w=read(wr_path)
    reqw={
      "variant","event_id","season","week","team","player_clean_key","player",
      "wr_rank","mc_rec_yards","actual_rec_yards",
    }
    if reqw-set(w.columns):
        raise RuntimeError(f"WR authority missing {sorted(reqw-set(w.columns))}")
    w=w.loc[w["variant"].astype(str).eq(WR_VARIANT)].copy()
    for c in ["season","week","wr_rank","mc_rec_yards","actual_rec_yards"]:
        w[c]=num(w[c])

    keys=["season","week","team","player_clean_key"]
    for label,x in [("parent",p),("WR authority",w)]:
        if x.duplicated(keys).any():
            sample=x.loc[x.duplicated(keys,keep=False),keys].head(20).to_dict("records")
            raise RuntimeError(f"{label} duplicate identities: {sample}")

    x=p.merge(
      w[keys+["event_id","player","mc_rec_yards","actual_rec_yards"]],
      on=keys,how="inner",validate="one_to_one"
    )
    signed=num(x["mc_rec_yards"])-num(x["actual_rec_yards"])
    absolute=signed.abs()
    signed_gap=(signed-num(x["current_total_error"])).abs()
    abs_gap=(absolute-num(x["current_abs_total_error"])).abs()
    decomp_gap=(
      num(x["current_opportunity_error"])+num(x["current_efficiency_error"])
      -num(x["current_total_error"])
    ).abs()

    parity={
      "matched_rows":int(len(x)),
      "players":int(x["player_clean_key"].nunique()),
      "max_signed_total_error_gap":float(signed_gap.max()) if len(x) else np.inf,
      "max_abs_total_error_gap":float(abs_gap.max()) if len(x) else np.inf,
      "max_decomposition_identity_gap":float(decomp_gap.max()) if len(x) else np.inf,
    }
    if (
      parity["matched_rows"]!=EXPECTED_ROWS
      or parity["players"]!=EXPECTED_PLAYERS
      or max(parity["max_signed_total_error_gap"],parity["max_abs_total_error_gap"],parity["max_decomposition_identity_gap"])>TOL
    ):
        raise RuntimeError(f"WR trust parent parity failed: {parity}")

    x["player_mae"]=num(x["prior8_abs_opportunity_error_mean"])
    x["efficiency_mae"]=num(x["prior8_abs_efficiency_error_mean"])
    x["abs_total_error"]=absolute
    g=x.groupby(["season","week"],sort=False)
    x["trust_fragility_score"]=g["player_mae"].transform(rank_pct)
    x["efficiency_fragility_score"]=g["efficiency_mae"].transform(rank_pct)
    x["error_badness_rank"]=g["abs_total_error"].transform(rank_pct)
    x["week_pool_mae"]=g["player_mae"].transform("median")
    x["difficulty_ratio_vs_pool"]=x["player_mae"]/x["week_pool_mae"]
    x["trust_bucket"]="MIDDLE"
    x.loc[x["trust_fragility_score"].le(.25),"trust_bucket"]="HIGH_TRUST"
    x.loc[x["trust_fragility_score"].ge(.75),"trust_bucket"]="LOW_TRUST"
    x["signed_total_error"]=signed
    x["role"]=np.where(num(x["wr_rank"]).eq(1),"WR1","WR2+")

    good=(
      x["player_mae"].gt(0)&x["week_pool_mae"].gt(0)
      &x["abs_total_error"].ge(0)
      &x["event_id"].fillna("").astype(str).str.strip().ne("")
      &x["player_clean_key"].fillna("").astype(str).str.strip().ne("")
      &np.isfinite(x["trust_fragility_score"])
      &np.isfinite(x["error_badness_rank"])
    )
    x=x.loc[good].copy()
    if len(x)!=EXPECTED_ROWS:
        raise RuntimeError(f"eligible WR trust row drift {len(x)} != {EXPECTED_ROWS}")
    return x.sort_values(keys).reset_index(drop=True),parity


def metrics(q:pd.DataFrame)->dict:
    if q.empty: return {"rows":0,"players":0,"games":0}
    hi=q.loc[q["trust_bucket"].eq("HIGH_TRUST")]
    lo=q.loc[q["trust_bucket"].eq("LOW_TRUST")]
    return {
      "rows":int(len(q)),
      "players":int(q["player_clean_key"].nunique()),
      "games":int(q["event_id"].nunique()),
      "primary_rank_association":corr(q["trust_fragility_score"],q["error_badness_rank"]),
      "efficiency_difficulty_rank_association_diagnostic":corr(q["efficiency_fragility_score"],q["error_badness_rank"]),
      "overall_abs_error":float(q["abs_total_error"].mean()),
      "high_trust_abs_error":float(hi["abs_total_error"].mean()) if len(hi) else np.nan,
      "low_trust_abs_error":float(lo["abs_total_error"].mean()) if len(lo) else np.nan,
      "overall_bias":float(q["signed_total_error"].mean()),
      "high_trust_bias":float(hi["signed_total_error"].mean()) if len(hi) else np.nan,
      "low_trust_bias":float(lo["signed_total_error"].mean()) if len(lo) else np.nan,
      "high_trust_rows":int(len(hi)),
      "low_trust_rows":int(len(lo)),
      "median_difficulty_ratio_vs_pool":float(q["difficulty_ratio_vs_pool"].median()),
      "high_trust_median_ratio_vs_pool":float(hi["difficulty_ratio_vs_pool"].median()) if len(hi) else np.nan,
      "low_trust_median_ratio_vs_pool":float(lo["difficulty_ratio_vs_pool"].median()) if len(lo) else np.nan,
    }


def cluster_boot(q:pd.DataFrame,cluster_col:str,seed:int)->dict:
    z=q[["trust_fragility_score","error_badness_rank",cluster_col]].dropna().copy()
    groups={str(k):g for k,g in z.groupby(cluster_col,sort=True)}
    keys=np.asarray(sorted(groups),dtype=object)
    observed=corr(z["trust_fragility_score"],z["error_badness_rank"])
    if len(keys)<2 or not np.isfinite(observed):
        return {"reps":0,"observed":observed,"ci_low":np.nan,"ci_high":np.nan,"p_positive":np.nan,"clusters":int(len(keys))}
    rng=np.random.default_rng(seed)
    vals=np.empty(BOOT_REPS,dtype=float)
    for i in range(BOOT_REPS):
        sample=rng.choice(keys,size=len(keys),replace=True)
        b=pd.concat([groups[str(k)] for k in sample],ignore_index=True)
        vals[i]=corr(b["trust_fragility_score"],b["error_badness_rank"])
    vals=vals[np.isfinite(vals)]
    return {
      "reps":int(len(vals)),"observed":observed,
      "ci_low":float(np.quantile(vals,.025,method="linear")),
      "ci_high":float(np.quantile(vals,.975,method="linear")),
      "p_positive":float(np.mean(vals>0)),
      "clusters":int(len(keys)),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-rows",type=Path,required=True)
    ap.add_argument("--wr-authority",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    try:
        x,parity=prepare(a.parent_rows,a.wr_authority)
    except Exception as exc:
        result={
          "version":VERSION,"disposition":INTEGRITY,"error":str(exc),
          "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,
          "models_fit":0,"projection_distribution_values_changed":0,"production_changed":False,
        }
        (a.out_dir/"wr_player_opportunity_difficulty_trust_selectivity_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        raise

    y23=x.loc[x["season"].eq(2023)].copy()
    y24=x.loc[x["season"].eq(2024)].copy()
    m23=metrics(y23); m24=metrics(y24); mp=metrics(x)
    pp=cluster_boot(x,"player_clean_key",POOL_PLAYER_SEED)
    pg=cluster_boot(x,"event_id",POOL_GAME_SEED)
    c24p=cluster_boot(y24,"player_clean_key",Y24_PLAYER_SEED)
    c24g=cluster_boot(y24,"event_id",Y24_GAME_SEED)
    wr1=metrics(x.loc[x["role"].eq("WR1")])
    wr2p=metrics(x.loc[x["role"].eq("WR2+")])

    support=(
      m23["rows"]>=1200 and m23["players"]>=140 and
      m24["rows"]>=1600 and m24["players"]>=160 and
      m23["high_trust_rows"]>=250 and m23["low_trust_rows"]>=250 and
      m24["high_trust_rows"]>=350 and m24["low_trust_rows"]>=350 and
      mp["rows"]>=3000 and mp["players"]>=190
    )
    gates={
      "source_parity_integrity":True,
      "support":bool(support),
      "2023_primary_rank_association_positive":bool(m23["primary_rank_association"]>0),
      "2024_primary_rank_association_positive":bool(m24["primary_rank_association"]>0),
      "pooled_player_cluster_ci_low_positive":bool(pp["ci_low"]>0),
      "pooled_game_cluster_ci_low_positive":bool(pg["ci_low"]>0),
      "2024_player_cluster_ci_low_positive":bool(c24p["ci_low"]>0),
      "2024_game_cluster_ci_low_positive":bool(c24g["ci_low"]>0),
      "2023_high_trust_better_than_overall":bool(m23["high_trust_abs_error"]<m23["overall_abs_error"]),
      "2024_high_trust_better_than_overall":bool(m24["high_trust_abs_error"]<m24["overall_abs_error"]),
      "2023_low_trust_worse_than_overall":bool(m23["low_trust_abs_error"]>m23["overall_abs_error"]),
      "2024_low_trust_worse_than_overall":bool(m24["low_trust_abs_error"]>m24["overall_abs_error"]),
      "2023_low_trust_worse_than_high_trust":bool(m23["low_trust_abs_error"]>m23["high_trust_abs_error"]),
      "2024_low_trust_worse_than_high_trust":bool(m24["low_trust_abs_error"]>m24["high_trust_abs_error"]),
      "sportsbook_inputs_zero":True,
      "oddsapi_calls_zero":True,
      "outcomes_2026_read_zero":True,
      "models_fit_zero":True,
      "projection_distribution_values_changed_zero":True,
      "production_unchanged":True,
    }
    disp=QUALIFIED if all(gates.values()) else CLOSED
    result={
      "version":VERSION,"disposition":disp,"parent_parity":parity,
      "season_metrics":{"2023":m23,"2024":m24},"pooled_metrics":mp,
      "pooled_player_cluster":pp,"pooled_game_cluster":pg,
      "2024_player_cluster":c24p,"2024_game_cluster":c24g,
      "wr1_diagnostic":wr1,"wr2plus_diagnostic":wr2p,
      "gates":gates,"failed_gates":[k for k,v in gates.items() if not v],
      "trust_score_formula":"within-season-week average percentile rank of strictly-prior WR opportunity difficulty",
      "primary_outcome":"within-season-week percentile rank of absolute exact WR-R15 receiving-yard point error",
      "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,
      "models_fit":0,"projection_distribution_values_changed":0,"production_changed":False,
      "bootstrap_reps":BOOT_REPS,
    }
    x.to_csv(a.out_dir/"wr_player_opportunity_difficulty_trust_selectivity_rows.csv",index=False)
    (a.out_dir/"wr_player_opportunity_difficulty_trust_selectivity_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
