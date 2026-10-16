#!/usr/bin/env python3
"""Frozen WR dual reliability trust integration V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION="WR_DUAL_RELIABILITY_TRUST_INTEGRATION_V1"
BOOT_REPS=5000
EXPECTED_ROWS=3224
EXPECTED_PLAYERS=211
TOL=1e-12

REQUIRED={
    "season","week","team","player_clean_key",
    "efficiency_fragility_rank","opportunity_fragility_rank",
    "total_error_badness_rank","current_abs_total_error",
}

def num(s): return pd.to_numeric(s,errors="coerce")

def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    x=x[m]; y=y[m]
    if len(x)<3 or np.std(x)<=0 or np.std(y)<=0:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def metrics(x:pd.DataFrame)->dict:
    opp=corr(x["opportunity_fragility_rank"],x["total_error_badness_rank"])
    eff=corr(x["efficiency_fragility_rank"],x["total_error_badness_rank"])
    dual=corr(x["dual_fragility_score"],x["total_error_badness_rank"])
    hi=x.loc[x["dual_trust_bucket"].eq("DUAL_HIGH_TRUST")]
    lo=x.loc[x["dual_trust_bucket"].eq("DUAL_LOW_TRUST")]
    return {
        "rows":int(len(x)),
        "players":int(x["player_clean_key"].nunique()),
        "opportunity_association":opp,
        "efficiency_association":eff,
        "dual_association":dual,
        "dual_minus_opportunity":float(dual-opp),
        "dual_high_raw_final_abs_error":float(hi["current_abs_total_error"].mean()) if len(hi) else np.nan,
        "dual_low_raw_final_abs_error":float(lo["current_abs_total_error"].mean()) if len(lo) else np.nan,
        "dual_high_rows":int(len(hi)),
        "dual_low_rows":int(len(lo)),
    }

def bootstrap_gain(x:pd.DataFrame,cluster_col:str,seed:int)->dict:
    z=x[[
      "opportunity_fragility_rank","dual_fragility_score",
      "total_error_badness_rank",cluster_col
    ]].dropna().reset_index(drop=True)
    labs=z[cluster_col].astype(str).to_numpy()
    uniq=np.unique(labs)
    groups={k:np.flatnonzero(labs==k) for k in uniq}
    rng=np.random.default_rng(seed)
    vals=np.empty(BOOT_REPS,float)
    for i in range(BOOT_REPS):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([groups[k] for k in sampled])
        b=z.iloc[idx]
        vals[i]=(
            corr(b["dual_fragility_score"],b["total_error_badness_rank"])
            -corr(b["opportunity_fragility_rank"],b["total_error_badness_rank"])
        )
    vals=vals[np.isfinite(vals)]
    observed=(
        corr(z["dual_fragility_score"],z["total_error_badness_rank"])
        -corr(z["opportunity_fragility_rank"],z["total_error_badness_rank"])
    )
    return {
        "clusters":int(len(uniq)),
        "reps":int(len(vals)),
        "observed_gain":float(observed),
        "ci_low":float(np.quantile(vals,.025)),
        "ci_high":float(np.quantile(vals,.975)),
        "p_positive":float(np.mean(vals>0)),
    }

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    try:
        x=pd.read_csv(a.rows,low_memory=False)
        x.columns=[str(c).strip().lower() for c in x.columns]
        miss=REQUIRED-set(x.columns)
        if miss: raise RuntimeError(f"missing columns {sorted(miss)}")
        if len(x)!=EXPECTED_ROWS: raise RuntimeError(f"row drift {len(x)}")
        if int(x["player_clean_key"].nunique())!=EXPECTED_PLAYERS:
            raise RuntimeError("player-count drift")
        seasons=set(num(x["season"]).dropna().astype(int).unique())
        if seasons!={2023,2024}: raise RuntimeError(f"season drift {seasons}")
        keys=["season","week","team","player_clean_key"]
        if x.duplicated(keys).any(): raise RuntimeError("duplicate identity")
        for c in [
          "efficiency_fragility_rank","opportunity_fragility_rank",
          "total_error_badness_rank","current_abs_total_error"
        ]:
            x[c]=num(x[c])
        if not np.isfinite(x[list(REQUIRED-{"season","week","team","player_clean_key"})].to_numpy(float)).all():
            raise RuntimeError("nonfinite inputs")
        for c in ["efficiency_fragility_rank","opportunity_fragility_rank","total_error_badness_rank"]:
            if x[c].lt(0).any() or x[c].gt(1+TOL).any():
                raise RuntimeError(f"rank out of range {c}")
        if x["current_abs_total_error"].lt(0).any():
            raise RuntimeError("negative absolute error")

        x["dual_fragility_score"]=(
            x["opportunity_fragility_rank"]+x["efficiency_fragility_rank"]
        )/2.0
        x["dual_trust_bucket"]="MIDDLE"
        x.loc[x["dual_fragility_score"].le(.25),"dual_trust_bucket"]="DUAL_HIGH_TRUST"
        x.loc[x["dual_fragility_score"].ge(.75),"dual_trust_bucket"]="DUAL_LOW_TRUST"
        x["team_week_cluster"]=(
            num(x["season"]).astype(int).astype(str)+"-"+
            num(x["week"]).astype(int).astype(str)+"-"+x["team"].astype(str)
        )

        dev=x.loc[num(x["season"]).eq(2023)].copy()
        conf=x.loc[num(x["season"]).eq(2024)].copy()
        md=metrics(dev); mc=metrics(conf); mp=metrics(x)

        pooled_player=bootstrap_gain(x,"player_clean_key",2026101561)
        pooled_team=bootstrap_gain(x,"team_week_cluster",2026101562)
        conf_player=bootstrap_gain(conf,"player_clean_key",2026101563)
        conf_team=bootstrap_gain(conf,"team_week_cluster",2026101564)

        support=(
            md["rows"]>=1200 and md["players"]>=140
            and mc["rows"]>=1600 and mc["players"]>=160
            and mp["rows"]>=3000 and mp["players"]>=190
        )
        gates={
            "source_integrity":True,
            "support":bool(support),
            "dual_beats_opportunity_2023":bool(md["dual_association"]>md["opportunity_association"]),
            "dual_beats_opportunity_2024":bool(mc["dual_association"]>mc["opportunity_association"]),
            "dual_beats_opportunity_pooled":bool(mp["dual_association"]>mp["opportunity_association"]),
            "pooled_player_cluster_gain_ci_low_positive":bool(pooled_player["ci_low"]>0),
            "pooled_team_week_gain_ci_low_positive":bool(pooled_team["ci_low"]>0),
            "confirmation_player_cluster_gain_ci_low_positive":bool(conf_player["ci_low"]>0),
            "confirmation_team_week_gain_ci_low_positive":bool(conf_team["ci_low"]>0),
            "dual_bucket_order_2023":bool(md["dual_high_raw_final_abs_error"]<md["dual_low_raw_final_abs_error"]),
            "dual_bucket_order_2024":bool(mc["dual_high_raw_final_abs_error"]<mc["dual_low_raw_final_abs_error"]),
            "sportsbook_inputs_zero":True,
            "oddsapi_calls_zero":True,
            "outcomes_2026_read_zero":True,
            "models_fit_zero":True,
            "projection_distribution_values_changed_zero":True,
            "production_unchanged":True,
        }
        disp=(
            "WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED"
            if all(gates.values()) else
            "NO_ACTIONABLE_WR_DUAL_RELIABILITY_TRUST_INTEGRATION"
        )
        result={
            "version":VERSION,
            "disposition":disp,
            "development_2023":md,
            "confirmation_2024":mc,
            "pooled":mp,
            "pooled_player_cluster_gain":pooled_player,
            "pooled_team_week_gain":pooled_team,
            "confirmation_player_cluster_gain":conf_player,
            "confirmation_team_week_gain":conf_team,
            "gates":gates,
            "failed_gates":[k for k,v in gates.items() if not v],
            "sportsbook_inputs_used":0,
            "oddsapi_calls":0,
            "outcomes_2026_read":0,
            "models_fit":0,
            "projection_distribution_values_changed":0,
            "production_changed":False,
        }
        x.to_csv(a.out_dir/"wr_dual_reliability_trust_rows.csv",index=False,float_format="%.12f")
    except Exception as exc:
        result={
            "version":VERSION,
            "disposition":"WR_DUAL_RELIABILITY_TRUST_INTEGRATION_INTEGRITY_FAILURE",
            "error":str(exc),
            "sportsbook_inputs_used":0,
            "oddsapi_calls":0,
            "outcomes_2026_read":0,
            "models_fit":0,
            "projection_distribution_values_changed":0,
            "production_changed":False,
        }
        (a.out_dir/"wr_dual_reliability_trust_integration_result.json").write_text(
            json.dumps(result,indent=2,sort_keys=True)+"\n"
        )
        print(json.dumps(result,indent=2,sort_keys=True))
        raise

    (a.out_dir/"wr_dual_reliability_trust_integration_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
