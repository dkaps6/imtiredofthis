#!/usr/bin/env python3
"""TE Player Error Persistence V1.

Diagnostic only. Uses exact promoted TE-R5P OOS authority. No candidate,
sportsbook input, production change, or 2026 outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VERSION="TE_PLAYER_ERROR_PERSISTENCE_V1"
TARGET_SEASONS=(2024,2025)
HISTORY_N=8
MIN_PRIOR=4
MISS=30.0
BOOT_REPS=5000
BOOT_SEED=20261007
MIN_ROWS=400
MIN_PLAYERS=50

EXPECTED_ROWS=3214
EXPECTED_SEASONS={2023:1019,2024:1082,2025:1113}
EXPECTED_MAE=15.850607934418186
EXPECTED_RMSE=23.06142088072041
TOL=1e-9

def num(x): return pd.to_numeric(x,errors="coerce")

def read_casebook(path:Path)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing TE-R5P casebook: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={"season","week","team","player_clean_key","candidate_rec_yards_r5p","rec_yards"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"TE-R5P authority missing columns: {sorted(miss)}")
    x["season"]=num(x["season"]).astype(int)
    x["week"]=num(x["week"]).astype(int)
    x["pred"]=num(x["candidate_rec_yards_r5p"])
    x["actual"]=num(x["rec_yards"])
    if x[["pred","actual"]].isna().any().any():
        raise RuntimeError("nonfinite TE-R5P projection/actual")
    if len(x)!=EXPECTED_ROWS:
        raise RuntimeError(f"TE-R5P row drift {len(x)} != {EXPECTED_ROWS}")
    counts={int(k):int(v) for k,v in x.groupby("season").size().to_dict().items()}
    if counts!=EXPECTED_SEASONS:
        raise RuntimeError(f"TE-R5P season drift {counts} != {EXPECTED_SEASONS}")
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("duplicate TE-R5P player-game authority rows")
    err=x["pred"]-x["actual"]
    mae=float(err.abs().mean())
    rmse=float(np.sqrt(np.mean(np.square(err))))
    if abs(mae-EXPECTED_MAE)>TOL or abs(rmse-EXPECTED_RMSE)>TOL:
        raise RuntimeError(f"TE-R5P authority metric drift mae={mae} rmse={rmse}")
    x["signed_error"]=err
    x["abs_error"]=err.abs()
    x["miss30"]=x["abs_error"].ge(MISS).astype(int)
    return x.sort_values(["season","week","team","player_clean_key"]).reset_index(drop=True)

def prior_mask(g,season,week):
    return (g["season"]<season)|((g["season"]==season)&(g["week"]<week))

def build_features(authority:pd.DataFrame)->pd.DataFrame:
    groups={str(k):g.sort_values(["season","week"]).copy() for k,g in authority.groupby("player_clean_key",sort=False)}
    rows=[]
    targets=authority.loc[authority["season"].isin(TARGET_SEASONS)].copy()
    for r in targets.itertuples(index=False):
        pid=str(r.player_clean_key)
        g=groups[pid]
        h=g.loc[prior_mask(g,int(r.season),int(r.week))].tail(HISTORY_N)
        if len(h)<MIN_PRIOR:
            continue
        rows.append({
            "season":int(r.season),"week":int(r.week),"team":str(r.team),
            "player_clean_key":pid,
            "prior_games":int(len(h)),
            "prior8_signed_error_mean":float(h["signed_error"].mean()),
            "prior8_abs_error_mean":float(h["abs_error"].mean()),
            "prior8_miss30_rate":float(h["miss30"].mean()),
            "current_signed_error":float(r.signed_error),
            "current_abs_error":float(r.abs_error),
            "current_miss30":int(r.miss30),
        })
    return pd.DataFrame(rows)

def corr(a,b):
    z=pd.DataFrame({"a":num(a),"b":num(b)}).dropna()
    if len(z)<3 or z["a"].nunique()<2 or z["b"].nunique()<2:
        return np.nan
    return float(spearmanr(z["a"].to_numpy(float),z["b"].to_numpy(float)).statistic)

def sign_agreement(df):
    a=num(df["prior8_signed_error_mean"])
    b=num(df["current_signed_error"])
    z=pd.DataFrame({"a":a,"b":b}).dropna()
    z=z.loc[z["a"].ne(0)&z["b"].ne(0)]
    return float((np.sign(z["a"])==np.sign(z["b"])).mean()) if len(z) else np.nan

def season_metrics(df):
    return {
      "rows":int(len(df)),
      "players":int(df["player_clean_key"].nunique()),
      "signed_spearman":corr(df["prior8_signed_error_mean"],df["current_signed_error"]),
      "signed_sign_agreement":sign_agreement(df),
      "difficulty_spearman":corr(df["prior8_abs_error_mean"],df["current_abs_error"]),
      "miss30_spearman":corr(df["prior8_miss30_rate"],df["current_miss30"]),
      "current_mae":float(num(df["current_abs_error"]).mean()) if len(df) else np.nan,
      "current_miss30_rate":float(num(df["current_miss30"]).mean()) if len(df) else np.nan,
    }

def cluster_bootstrap(df):
    players=sorted(df["player_clean_key"].astype(str).unique().tolist())
    if len(players)<2:
        return {}
    groups={p:df.loc[df["player_clean_key"].astype(str).eq(p)].copy() for p in players}
    rng=np.random.default_rng(BOOT_SEED)
    vals={"signed":[],"difficulty":[],"miss30":[]}
    for _ in range(BOOT_REPS):
        sampled=rng.choice(players,size=len(players),replace=True)
        parts=[groups[p] for p in sampled]
        z=pd.concat(parts,ignore_index=True)
        vals["signed"].append(corr(z["prior8_signed_error_mean"],z["current_signed_error"]))
        vals["difficulty"].append(corr(z["prior8_abs_error_mean"],z["current_abs_error"]))
        vals["miss30"].append(corr(z["prior8_miss30_rate"],z["current_miss30"]))
    out={}
    for k,v in vals.items():
        a=np.asarray(v,dtype=float)
        a=a[np.isfinite(a)]
        out[k]={
          "valid_reps":int(len(a)),
          "p_positive":float(np.mean(a>0)) if len(a) else np.nan,
          "ci_low":float(np.quantile(a,.025)) if len(a) else np.nan,
          "ci_high":float(np.quantile(a,.975)) if len(a) else np.nan,
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--casebook",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    authority=read_casebook(a.casebook)
    f=build_features(authority)
    a.out_dir.mkdir(parents=True,exist_ok=True)

    by={str(s):season_metrics(f.loc[f["season"].eq(s)]) for s in TARGET_SEASONS}
    pooled=season_metrics(f)
    boot=cluster_bootstrap(f)

    support=all(by[str(s)]["rows"]>=MIN_ROWS and by[str(s)]["players"]>=MIN_PLAYERS for s in TARGET_SEASONS)
    A=bool(
      by["2024"]["signed_spearman"]>0 and by["2025"]["signed_spearman"]>0
      and boot["signed"]["p_positive"]>=.95
      and pooled["signed_sign_agreement"]>.50
    )
    B=bool(
      by["2024"]["difficulty_spearman"]>0 and by["2025"]["difficulty_spearman"]>0
      and boot["difficulty"]["p_positive"]>=.95
    )
    C=bool(
      by["2024"]["miss30_spearman"]>0 and by["2025"]["miss30_spearman"]>0
      and boot["miss30"]["p_positive"]>=.95
    )
    family_count=sum([A,B,C])
    if not support:
        disposition="TE_PLAYER_ERROR_PERSISTENCE_SOURCE_LIMITED"
    elif family_count>=2 and (B or C):
        disposition="TE_PLAYER_ERROR_PERSISTENCE_DETECTED"
    else:
        disposition="NO_ACTIONABLE_TE_PLAYER_ERROR_PERSISTENCE"

    result={
      "version":VERSION,
      "parent_authority":{
        "run":34152797603,
        "artifact":10029942404,
        "digest":"sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb",
        "rows":EXPECTED_ROWS,
      },
      "target_seasons":list(TARGET_SEASONS),
      "history_window":HISTORY_N,
      "min_prior_games":MIN_PRIOR,
      "scoreable_rows":int(len(f)),
      "scoreable_players":int(f["player_clean_key"].nunique()),
      "season_metrics":by,
      "pooled_metrics":pooled,
      "player_cluster_bootstrap":boot,
      "gates":{
        "support":bool(support),
        "directional_bias_family_A":A,
        "difficulty_family_B":B,
        "extreme_miss_family_C":C,
        "passing_family_count":int(family_count),
      },
      "disposition":disposition,
      "candidate_models_fit":0,
      "sportsbook_inputs_used":0,
      "outcomes_2026_read":0,
      "production_changed":False,
      "same_or_future_history_violations":0,
    }
    f.to_csv(a.out_dir/"te_player_error_persistence_rows.csv",index=False)
    (a.out_dir/"te_player_error_persistence_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
