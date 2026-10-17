#!/usr/bin/env python3
"""PRISM pass-catcher x QB-regime allocation signal V1.

Frozen predictive target-allocation test. One no-fit candidate. 2024 development
and 2025 independent confirmation. No sportsbook, OddsAPI, 2026 outcomes,
receiving-yard translation, fitted coefficients, or production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION="PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1"
SEASONS=(2024,2025)
FAMILIES=("WR","TE")
BOOTSTRAPS=5000
SEED=20261017

def num(x:Any)->pd.Series:
    return pd.to_numeric(x,errors="coerce")

def read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def finite(v:Any)->bool:
    try:
        return bool(np.isfinite(float(v)))
    except Exception:
        return False

def bootstrap_cluster_ci(x:pd.DataFrame,cluster_col:str,value_col:str,reps:int=BOOTSTRAPS,seed:int=SEED)->tuple[float,float]:
    g={str(k):num(v[value_col]).dropna().to_numpy(float) for k,v in x.groupby(cluster_col,sort=True)}
    keys=[k for k,v in g.items() if len(v)]
    if not keys:
        return np.nan,np.nan
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,dtype=float)
    for i in range(reps):
        sampled=rng.choice(keys,size=len(keys),replace=True)
        chunks=[g[k] for k in sampled]
        vals[i]=float(np.concatenate(chunks).mean())
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def score_metrics(x:pd.DataFrame)->dict[str,float|int]:
    b=num(x["baseline_abs_error"]);c=num(x["candidate_abs_error"])
    d=c-b
    bcount=num(x["baseline_target_count_abs_error"]);ccount=num(x["candidate_target_count_abs_error"])
    signed_b=num(x["baseline_share"])-num(x["actual_target_share"])
    signed_c=num(x["candidate_share"])-num(x["actual_target_share"])
    return {
        "rows":int(len(x)),
        "players":int(x["player_id"].nunique()),
        "teamweeks":int(x["teamweek"].nunique()),
        "baseline_mae":float(b.mean()),
        "candidate_mae":float(c.mean()),
        "mae_delta":float(c.mean()-b.mean()),
        "baseline_median_ae":float(b.median()),
        "candidate_median_ae":float(c.median()),
        "baseline_rmse":float(np.sqrt(np.mean(np.square(num(x["baseline_share"])-num(x["actual_target_share"]))))),
        "candidate_rmse":float(np.sqrt(np.mean(np.square(num(x["candidate_share"])-num(x["actual_target_share"]))))),
        "baseline_p90_ae":float(b.quantile(.90)),
        "candidate_p90_ae":float(c.quantile(.90)),
        "candidate_closer_rate":float((c<b).mean()),
        "ties_rate":float((c.eq(b)).mean()),
        "baseline_bias":float(signed_b.mean()),
        "candidate_bias":float(signed_c.mean()),
        "baseline_target_count_mae":float(bcount.mean()),
        "candidate_target_count_mae":float(ccount.mean()),
        "target_count_mae_delta":float(ccount.mean()-bcount.mean()),
        "median_equal_player_mae_delta":float(x.groupby("player_id")["error_delta"].mean().median()),
    }

def build(state:pd.DataFrame,history:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    need_state={
        "season","week","team","player_id","family","stable_roster_identity",
        "qb_regime_source_valid","overall_prior_target_share_mean",
        "overall_prior_target_share_latest","overall_prior_target_share_latest2",
        "same_qb_prior_games","same_qb_prior_target_share_mean","chronology_violation",
    }
    need_hist={"season","week","team","player_id","targets","team_targets","target_share","target_identity_valid"}
    if need_state-set(state.columns):
        raise RuntimeError(f"state missing {sorted(need_state-set(state.columns))}")
    if need_hist-set(history.columns):
        raise RuntimeError(f"history missing {sorted(need_hist-set(history.columns))}")

    s=state.copy()
    s["season"]=num(s["season"]).astype("Int64");s["week"]=num(s["week"]).astype("Int64")
    s=s.loc[s["season"].isin(SEASONS)&s["family"].astype(str).str.upper().isin(FAMILIES)].copy()
    if s.empty: raise RuntimeError("zero target state rows")
    if s.duplicated(["season","week","team","player_id"]).any():
        raise RuntimeError("duplicate state identity")
    if bool(s["chronology_violation"].fillna(False).astype(bool).any()):
        raise RuntimeError("source state chronology violation")

    h=history.copy()
    h["season"]=num(h["season"]).astype("Int64");h["week"]=num(h["week"]).astype("Int64")
    outcome=h[["season","week","team","player_id","targets","team_targets","target_share","target_identity_valid"]].copy()
    if outcome.duplicated(["season","week","team","player_id"]).any():
        raise RuntimeError("duplicate target outcome identity")
    outcome=outcome.rename(columns={
        "targets":"actual_targets",
        "team_targets":"actual_team_targets",
        "target_share":"actual_target_share",
        "target_identity_valid":"actual_target_identity_valid",
    })
    x=s.merge(outcome,on=["season","week","team","player_id"],how="left",validate="one_to_one")

    for c in [
        "overall_prior_target_share_mean","overall_prior_target_share_latest",
        "overall_prior_target_share_latest2","same_qb_prior_target_share_mean",
        "actual_targets","actual_team_targets","actual_target_share"
    ]:
        x[c]=num(x[c])

    x["baseline_raw"]=x["overall_prior_target_share_latest2"]
    x["baseline_raw"]=x["baseline_raw"].where(x["baseline_raw"].notna(),x["overall_prior_target_share_latest"])
    x["baseline_raw"]=x["baseline_raw"].where(x["baseline_raw"].notna(),x["overall_prior_target_share_mean"])
    x["baseline_raw"]=x["baseline_raw"].fillna(0.0).clip(0.0,1.0)

    x["signal_eligible"]=(
        x["stable_roster_identity"].eq(1)
        & x["qb_regime_source_valid"].astype(bool)
        & num(x["same_qb_prior_games"]).ge(2)
        & x["overall_prior_target_share_mean"].notna()
        & x["overall_prior_target_share_latest2"].notna()
        & x["same_qb_prior_target_share_mean"].notna()
        & x["actual_target_identity_valid"].fillna(False).astype(bool)
        & x["actual_target_share"].notna()
        & x["actual_team_targets"].gt(0)
    )
    x["qb_regime_delta"]=0.0
    e=x["signal_eligible"]
    x.loc[e,"qb_regime_delta"]=(
        x.loc[e,"same_qb_prior_target_share_mean"]-x.loc[e,"overall_prior_target_share_mean"]
    )
    x["candidate_raw"]=(x["baseline_raw"]+x["qb_regime_delta"]).clip(0.0,1.0)

    x["baseline_share"]=np.nan;x["candidate_share"]=np.nan
    room_rows=[]
    for keys,g in x.groupby(["season","week","team","family"],sort=True):
        idx=g.index
        b=num(g["baseline_raw"]).fillna(0.0).to_numpy(float)
        r=num(g["candidate_raw"]).fillna(0.0).to_numpy(float)
        bmass=float(b.sum());rmass=float(r.sum())
        if bmass<0 or not np.isfinite(bmass) or rmass<0 or not np.isfinite(rmass):
            raise RuntimeError(f"nonfinite room mass {keys}")
        bp=b.copy()
        cp=(r*(bmass/rmass)) if rmass>0 else b.copy()
        if np.any(cp < -1e-12):
            raise RuntimeError(f"negative candidate share {keys}")
        # V1 does not clip after room normalization because clipping would violate
        # the frozen exact room-mass conservation identity. Any >1 row is an
        # integrity failure rather than a post-result rescue.
        x.loc[idx,"baseline_share"]=bp
        x.loc[idx,"candidate_share"]=cp
        room_rows.append({
            "season":int(keys[0]),"week":int(keys[1]),"team":str(keys[2]),"family":str(keys[3]),
            "players":int(len(g)),"signal_eligible_players":int(g["signal_eligible"].sum()),
            "baseline_room_mass":bmass,"candidate_raw_room_mass":rmass,
            "candidate_room_mass":float(cp.sum()),
            "room_mass_gap":float(abs(cp.sum()-bmass)),
            "candidate_share_max":float(cp.max()) if len(cp) else np.nan,
        })
    rooms=pd.DataFrame(room_rows)
    if rooms["room_mass_gap"].max()>1e-12:
        raise RuntimeError(f"room conservation gap {rooms['room_mass_gap'].max()}")
    if (x["candidate_share"]>1.0+1e-12).any():
        bad=x.loc[x["candidate_share"]>1.0+1e-12,["season","week","team","player_id","family","candidate_share"]]
        raise RuntimeError(f"candidate share >1 after conservation: {bad.head(10).to_dict('records')}")

    scored=x.loc[x["signal_eligible"]].copy()
    if scored.empty: raise RuntimeError("zero signal-eligible scored rows")
    scored["teamweek"]=(
        scored["season"].astype(str)+"_"+scored["week"].astype(str)+"_"+scored["team"].astype(str)
    )
    scored["baseline_abs_error"]=(scored["baseline_share"]-scored["actual_target_share"]).abs()
    scored["candidate_abs_error"]=(scored["candidate_share"]-scored["actual_target_share"]).abs()
    scored["error_delta"]=scored["candidate_abs_error"]-scored["baseline_abs_error"]
    scored["baseline_pred_targets"]=scored["baseline_share"]*scored["actual_team_targets"]
    scored["candidate_pred_targets"]=scored["candidate_share"]*scored["actual_team_targets"]
    scored["baseline_target_count_abs_error"]=(scored["baseline_pred_targets"]-scored["actual_targets"]).abs()
    scored["candidate_target_count_abs_error"]=(scored["candidate_pred_targets"]-scored["actual_targets"]).abs()

    metrics_rows=[]
    family_results={}
    for fam in FAMILIES:
        family_results[fam]={}
        for season in SEASONS:
            z=scored.loc[(scored["family"].eq(fam))&scored["season"].eq(season)].copy()
            m=score_metrics(z) if len(z) else {"rows":0}
            room=rooms.loc[(rooms["family"].eq(fam))&rooms["season"].eq(season)]
            m["max_room_mass_gap"]=float(room["room_mass_gap"].max()) if len(room) else np.nan
            m["max_candidate_share"]=float(room["candidate_share_max"].max()) if len(room) else np.nan
            if season==2025 and len(z):
                lo,hi=bootstrap_cluster_ci(z,"teamweek","error_delta",seed=SEED+(1 if fam=="WR" else 2))
                plo,phi=bootstrap_cluster_ci(z,"player_id","error_delta",seed=SEED+(11 if fam=="WR" else 12))
                m["teamweek_bootstrap_ci_low"]=lo;m["teamweek_bootstrap_ci_high"]=hi
                m["player_bootstrap_ci_low"]=plo;m["player_bootstrap_ci_high"]=phi
            family_results[fam][str(season)]=m
            metrics_rows.append({"family":fam,"season":season,**m})

    gates={}
    qualified={}
    for fam in FAMILIES:
        d=family_results[fam]["2024"];c=family_results[fam]["2025"]
        min_rows=1000 if fam=="WR" else 600
        g={
            "dev_support":int(d.get("rows",0))>=min_rows,
            "dev_mae_improves":d.get("candidate_mae",np.inf)<d.get("baseline_mae",-np.inf),
            "dev_p90_nonworse":d.get("candidate_p90_ae",np.inf)<=d.get("baseline_p90_ae",-np.inf),
            "dev_closer_gt_50":d.get("candidate_closer_rate",0.0)>0.50,
            "dev_target_count_mae_nonworse":d.get("candidate_target_count_mae",np.inf)<=d.get("baseline_target_count_mae",-np.inf),
            "dev_room_conservation":d.get("max_room_mass_gap",np.inf)<=1e-12,
            "confirm_support":int(c.get("rows",0))>=min_rows,
            "confirm_mae_improves":c.get("candidate_mae",np.inf)<c.get("baseline_mae",-np.inf),
            "confirm_p90_nonworse":c.get("candidate_p90_ae",np.inf)<=c.get("baseline_p90_ae",-np.inf),
            "confirm_closer_gt_50":c.get("candidate_closer_rate",0.0)>0.50,
            "confirm_target_count_mae_nonworse":c.get("candidate_target_count_mae",np.inf)<=c.get("baseline_target_count_mae",-np.inf),
            "confirm_teamweek_ci_high_lt_0":c.get("teamweek_bootstrap_ci_high",np.inf)<0.0,
            "confirm_player_ci_high_lt_0":c.get("player_bootstrap_ci_high",np.inf)<0.0,
            "confirm_median_equal_player_delta_lt_0":c.get("median_equal_player_mae_delta",np.inf)<0.0,
            "confirm_room_conservation":c.get("max_room_mass_gap",np.inf)<=1e-12,
        }
        gates[fam]=g;qualified[fam]=bool(all(g.values()))

    if all(qualified.values()):
        disp="PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_QUALIFIED_ALL"
    elif any(qualified.values()):
        disp="PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_QUALIFIED_PARTIAL"
    else:
        disp="PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_CLOSED"

    result={
        "version":VERSION,
        "disposition":disp,
        "candidate_variants":1,
        "fitted_coefficients":0,
        "threshold_searches":0,
        "families":list(FAMILIES),
        "development_season":2024,
        "confirmation_season":2025,
        "family_qualified":qualified,
        "gates":gates,
        "metrics":metrics_rows,
        "integrity":{
            "chronology_violations":int(x["chronology_violation"].fillna(False).astype(bool).sum()),
            "target_game_outcomes_used_in_predictor":0,
            "sportsbook_inputs_used":0,
            "oddsapi_calls":0,
            "outcomes_2026_used":0,
            "receiving_yards_scored":False,
            "market_translation_scored":False,
            "production_changed":False,
        },
        "boundary":"A pass authorizes only a separately frozen canonical-entitlement integration test; no direct production or market promotion.",
    }
    return scored,rooms,result

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--history",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    rows,rooms,result=build(read(a.state,"target state"),read(a.history,"completed-game history"))
    rows.to_csv(a.out_dir/"qb_regime_allocation_signal_rows.csv",index=False,float_format="%.12f")
    rooms.to_csv(a.out_dir/"qb_regime_allocation_room_conservation.csv",index=False,float_format="%.12f")
    pd.DataFrame(result["metrics"]).to_csv(a.out_dir/"qb_regime_allocation_signal_metrics.csv",index=False,float_format="%.12f")
    (a.out_dir/"qb_regime_allocation_signal_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
