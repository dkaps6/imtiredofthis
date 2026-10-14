#!/usr/bin/env python3
"""Player Allocation-Only Difficulty V1.

Mean-neutral research bridge from confirmed same-player opportunity difficulty
into target-allocation uncertainty after removing team pass-volume error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts._opponent_map import canon_team

VERSION="PLAYER_ALLOCATION_ONLY_DIFFICULTY_V1"
PRIOR_WINDOW=8
MIN_PRIOR=4
BOOT_REPS=5000
PLAYER_SEED=2026101411
GAME_SEED=2026101412
SQRT_PI_OVER_2=math.sqrt(math.pi/2.0)
EPS=1e-6
Z80=float(norm.ppf(.90))
Z90=float(norm.ppf(.95))
WR_VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"


def num(s):
    return pd.to_numeric(s,errors="coerce")


def norm_name(v)->str:
    s="" if v is None or pd.isna(v) else str(v)
    s=unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]","",s)


def read_csv(path:Path)->pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def normal_crps_zero(y,sigma):
    y=np.asarray(y,dtype=float)
    sigma=np.maximum(np.asarray(sigma,dtype=float),EPS)
    z=y/sigma
    return sigma*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))


def coverage_gap(x:pd.Series,nominal:float)->float:
    return abs(float(x.mean())-nominal)


def prep_v2(path:Path)->pd.DataFrame:
    x=read_csv(path)
    need={"season","week","game_id","team","player_identity_key","player","cell","actual_share","excess_share_var"}
    miss=need-set(x.columns)
    if miss:
        raise RuntimeError(f"V2 panel missing {sorted(miss)}")
    x=x.loc[x["cell"].isin(["WR_TARGET_SHARE","TE_TARGET_SHARE"])].copy()
    x["season"]=num(x["season"]).astype("Int64")
    x["week"]=num(x["week"]).astype("Int64")
    x["team"]=x["team"].map(canon_team)
    x["name_key"]=x["player"].map(norm_name)
    x["actual_share"]=num(x["actual_share"])
    x["excess_share_var"]=num(x["excess_share_var"])
    if x["name_key"].eq("").any():
        raise RuntimeError("blank normalized V2 player name")
    keys=["season","week","team","name_key","cell"]
    if x.duplicated(keys).any():
        d=x.loc[x.duplicated(keys,keep=False),keys+["player","player_identity_key"]].head(20)
        raise RuntimeError(f"ambiguous V2 normalized identity keys: {d.to_dict('records')}")
    return x


def prep_wr(path:Path)->pd.DataFrame:
    x=read_csv(path)
    need={"variant","season","week","event_id","team","player","player_clean_key","entitlement_tgt_share","pred_targets","mc_rec_yards","actual_targets"}
    miss=need-set(x.columns)
    if miss:
        raise RuntimeError(f"WR authority missing {sorted(miss)}")
    x=x.loc[x["variant"].astype(str).eq(WR_VARIANT)].copy()
    for c in ["season","week","entitlement_tgt_share","pred_targets","mc_rec_yards","actual_targets"]:
        x[c]=num(x[c])
    x["season"]=x["season"].astype("Int64"); x["week"]=x["week"].astype("Int64")
    x["team"]=x["team"].map(canon_team)
    x["name_key"]=x["player"].map(norm_name)
    x["pred_team_targets"]=x["pred_targets"]/x["entitlement_tgt_share"]
    x["promoted_targets"]=x["pred_targets"]
    x["promoted_share"]=x["entitlement_tgt_share"]
    x["position"]="WR"
    x["cell"]="WR_TARGET_SHARE"
    return x


def prep_te(path:Path)->pd.DataFrame:
    x=read_csv(path)
    need={"season","week","event_id","team","player","player_clean_key","b0_target_probability","b0_expected_targets","candidate_targets_r5p","candidate_rec_yards_r5p","targets"}
    miss=need-set(x.columns)
    if miss:
        raise RuntimeError(f"TE authority missing {sorted(miss)}")
    for c in ["season","week","b0_target_probability","b0_expected_targets","candidate_targets_r5p","candidate_rec_yards_r5p","targets"]:
        x[c]=num(x[c])
    x["season"]=x["season"].astype("Int64"); x["week"]=x["week"].astype("Int64")
    x["team"]=x["team"].map(canon_team)
    x["name_key"]=x["player"].map(norm_name)
    x["pred_team_targets"]=x["b0_expected_targets"]/x["b0_target_probability"]
    x["promoted_targets"]=x["candidate_targets_r5p"]
    x["promoted_share"]=x["promoted_targets"]/x["pred_team_targets"]
    x["actual_targets"]=x["targets"]
    x["mc_rec_yards"]=x["candidate_rec_yards_r5p"]
    x["position"]="TE"
    x["cell"]="TE_TARGET_SHARE"
    return x


def merge_authority(auth:pd.DataFrame,v2:pd.DataFrame)->pd.DataFrame:
    keys=["season","week","team","name_key","cell"]
    z=auth.merge(
        v2[[
            "season","week","team","name_key","cell","game_id","player_identity_key",
            "actual_share","excess_share_var"
        ]],
        on=keys,how="inner",validate="many_to_one",
    )
    if z.empty:
        raise RuntimeError(f"zero matched rows for {auth['position'].iloc[0]}")
    z=z.loc[
        z["pred_team_targets"].gt(0)
        & z["promoted_targets"].ge(0)
        & z["promoted_share"].between(0,1)
        & z["actual_share"].between(0,1)
        & z["player_identity_key"].astype(str).ne("")
    ].copy()
    z["allocation_actual_targets"]=z["pred_team_targets"]*z["actual_share"]
    z["allocation_error"]=z["promoted_targets"]-z["allocation_actual_targets"]
    z["ord"]=z["season"].astype(int)*100+z["week"].astype(int)
    return z.sort_values(["ord","game_id","player_identity_key"]).reset_index(drop=True)


def add_prior_state(z:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for pid,g in z.groupby("player_identity_key",sort=False):
        g=g.sort_values(["ord","game_id"]).copy()
        hist=[]
        for r in g.itertuples(index=False):
            if len(hist)>=MIN_PRIOR:
                use=hist[-PRIOR_WINDOW:]
                d=r._asdict()
                d["prior_games"]=len(use)
                d["prior8_abs_alloc_error_mean"]=float(np.mean(np.abs(use)))
                rows.append(d)
            hist.append(float(r.allocation_error))
    q=pd.DataFrame(rows)
    if q.empty:
        return q
    q["player_mae"]=q["prior8_abs_alloc_error_mean"].astype(float)
    pool=(q.groupby(["season","week","position"],as_index=False)["player_mae"]
            .median().rename(columns={"player_mae":"week_pool_mae"}))
    q=q.merge(pool,on=["season","week","position"],how="left",validate="many_to_one")
    q["candidate_sigma"]=np.maximum(EPS,q["player_mae"]*SQRT_PI_OVER_2)
    q["pool_sigma"]=np.maximum(EPS,q["week_pool_mae"]*SQRT_PI_OVER_2)
    y=q["allocation_error"].to_numpy(float)
    q["candidate_crps"]=normal_crps_zero(y,q["candidate_sigma"])
    q["pool_crps"]=normal_crps_zero(y,q["pool_sigma"])
    q["gain_vs_pool"]=q["pool_crps"]-q["candidate_crps"]
    ay=np.abs(y)
    q["candidate_cover80"]=ay<=Z80*q["candidate_sigma"]
    q["pool_cover80"]=ay<=Z80*q["pool_sigma"]
    q["candidate_cover90"]=ay<=Z90*q["candidate_sigma"]
    q["pool_cover90"]=ay<=Z90*q["pool_sigma"]

    basevar=(
        q["pred_team_targets"]*q["promoted_share"]*(1-q["promoted_share"])
        +q["pred_team_targets"]*np.maximum(q["pred_team_targets"]-1.0,0.0)
         *np.maximum(q["excess_share_var"].fillna(0.0),0.0)
    )
    q["v2_sigma"]=np.sqrt(np.maximum(EPS,basevar))
    q["v2_crps"]=normal_crps_zero(y,q["v2_sigma"])
    q["gain_vs_v2"]=q["v2_crps"]-q["candidate_crps"]

    q75=q.groupby("position")["player_mae"].transform(lambda s:s.quantile(.75))
    q["difficulty_q4"]=q["player_mae"].ge(q75)
    return q


def metric(q:pd.DataFrame)->dict:
    if q.empty:
        return {"rows":0,"players":0}
    return {
      "rows":int(len(q)),
      "players":int(q["player_identity_key"].nunique()),
      "pool_crps":float(q["pool_crps"].mean()),
      "candidate_crps":float(q["candidate_crps"].mean()),
      "gain_vs_pool":float(q["gain_vs_pool"].mean()),
      "v2_crps":float(q["v2_crps"].mean()),
      "gain_vs_v2":float(q["gain_vs_v2"].mean()),
      "pool_80_coverage":float(q["pool_cover80"].mean()),
      "candidate_80_coverage":float(q["candidate_cover80"].mean()),
      "pool_80_gap":coverage_gap(q["pool_cover80"],.80),
      "candidate_80_gap":coverage_gap(q["candidate_cover80"],.80),
      "pool_90_coverage":float(q["pool_cover90"].mean()),
      "candidate_90_coverage":float(q["candidate_cover90"].mean()),
      "pool_90_gap":coverage_gap(q["pool_cover90"],.90),
      "candidate_90_gap":coverage_gap(q["candidate_cover90"],.90),
    }


def cluster_boot(q:pd.DataFrame,gain_col:str,cluster_col:str,seed:int)->dict:
    z=q[[gain_col,cluster_col]].dropna().copy()
    groups={str(k):g[gain_col].to_numpy(float) for k,g in z.groupby(cluster_col,sort=True)}
    keys=np.array(sorted(groups),dtype=object)
    if len(keys)<2:
        return {"reps":0,"observed":float(z[gain_col].mean()) if len(z) else np.nan,"ci_low":np.nan,"ci_high":np.nan}
    rng=np.random.default_rng(seed)
    vals=np.empty(BOOT_REPS,dtype=float)
    for i in range(BOOT_REPS):
        sample=rng.choice(keys,size=len(keys),replace=True)
        vals[i]=float(np.concatenate([groups[str(k)] for k in sample]).mean())
    return {
      "reps":BOOT_REPS,
      "observed":float(z[gain_col].mean()),
      "ci_low":float(np.quantile(vals,.025)),
      "ci_high":float(np.quantile(vals,.975)),
      "p_positive":float(np.mean(vals>0)),
    }


def player_diag(q:pd.DataFrame,gain_col:str,confirm_season:int)->dict:
    vals=[]
    for _,g in q.loc[q["season"].eq(confirm_season)].groupby("player_identity_key"):
        if len(g)<6: continue
        vals.append(float(g[gain_col].mean()))
    a=np.asarray(vals,dtype=float)
    return {
      "qualifying_players_ge6":int(len(a)),
      "median_per_player_gain":float(np.median(a)) if len(a) else np.nan,
      "fraction_players_positive":float(np.mean(a>0)) if len(a) else np.nan,
    }


def position_result(q:pd.DataFrame,pos:str)->dict:
    z=q.loc[q["position"].eq(pos)].copy()
    seasons=(2023,2024) if pos=="WR" else (2024,2025)
    sm={str(s):metric(z.loc[z["season"].eq(s)]) for s in seasons}
    pooled=metric(z)
    q4=metric(z.loc[z["difficulty_q4"]])
    pb=cluster_boot(z,"gain_vs_pool","player_identity_key",PLAYER_SEED)
    gb=cluster_boot(z,"gain_vs_pool","game_id",GAME_SEED)
    confirm=2024 if pos=="WR" else 2025
    confirm_rows=z.loc[z["season"].eq(confirm)].copy()
    v2pb=cluster_boot(confirm_rows,"gain_vs_v2","player_identity_key",PLAYER_SEED)
    v2gb=cluster_boot(confirm_rows,"gain_vs_v2","game_id",GAME_SEED)
    pdg=player_diag(z,"gain_vs_pool",confirm)

    if pos=="WR":
        support=(
          sm["2023"]["rows"]>=700 and sm["2023"]["players"]>=100
          and sm["2024"]["rows"]>=1000 and sm["2024"]["players"]>=140
          and pooled["rows"]>=1800 and pooled["players"]>=150
        )
        baseg={
          "support":bool(support),
          "2023_pool_crps_improves":sm["2023"]["candidate_crps"]<sm["2023"]["pool_crps"],
          "2024_pool_crps_improves":sm["2024"]["candidate_crps"]<sm["2024"]["pool_crps"],
          "pooled_gain_positive":pooled["gain_vs_pool"]>0,
          "pool_player_ci_low_positive":pb["ci_low"]>0,
          "pool_game_ci_low_positive":gb["ci_low"]>0,
          "pooled_80_gap_nonworse":pooled["candidate_80_gap"]<=pooled["pool_80_gap"],
          "pooled_90_gap_nonworse":pooled["candidate_90_gap"]<=pooled["pool_90_gap"],
          "q4_pool_crps_improves":q4["candidate_crps"]<q4["pool_crps"],
        }
        inc={
          "2024_v2_crps_improves":sm["2024"]["candidate_crps"]<sm["2024"]["v2_crps"],
          "2024_v2_player_ci_low_positive":v2pb["ci_low"]>0,
          "2024_v2_game_ci_low_positive":v2gb["ci_low"]>0,
        }
        if all(baseg.values()) and all(inc.values()):
            disp="WR_PLAYER_ALLOCATION_DIFFICULTY_INCREMENTAL_CONFIRMED"
        elif all(baseg.values()) and not all(inc.values()):
            disp="WR_PLAYER_ALLOCATION_DIFFICULTY_REDUNDANT_WITH_SHARE_VOLATILITY"
        else:
            disp="NO_ACTIONABLE_WR_PLAYER_ALLOCATION_DIFFICULTY"
        gates={**baseg,**inc}
    else:
        support=(
          sm["2024"]["rows"]>=600 and sm["2024"]["players"]>=60
          and sm["2025"]["rows"]>=600 and sm["2025"]["players"]>=60
          and pooled["rows"]>=1300 and pooled["players"]>=90
        )
        gates={
          "support":bool(support),
          "2024_pool_crps_improves":sm["2024"]["candidate_crps"]<sm["2024"]["pool_crps"],
          "2025_pool_crps_improves":sm["2025"]["candidate_crps"]<sm["2025"]["pool_crps"],
          "pooled_gain_positive":pooled["gain_vs_pool"]>0,
          "pool_player_ci_low_positive":pb["ci_low"]>0,
          "pool_game_ci_low_positive":gb["ci_low"]>0,
          "pooled_80_gap_nonworse":pooled["candidate_80_gap"]<=pooled["pool_80_gap"],
          "pooled_90_gap_nonworse":pooled["candidate_90_gap"]<=pooled["pool_90_gap"],
          "q4_pool_crps_improves":q4["candidate_crps"]<q4["pool_crps"],
        }
        disp="TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED" if all(gates.values()) else "NO_ACTIONABLE_TE_PLAYER_ALLOCATION_DIFFICULTY"

    return {
      "position":pos,
      "disposition":disp,
      "season_metrics":sm,
      "pooled":pooled,
      "high_difficulty_q4":q4,
      "pool_player_cluster":pb,
      "pool_game_cluster":gb,
      "confirmation_v2_player_cluster":v2pb,
      "confirmation_v2_game_cluster":v2gb,
      "player_diagnostics":pdg,
      "gates":gates,
      "failed_gates":[k for k,v in gates.items() if not v],
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--wr",type=Path,required=True)
    ap.add_argument("--te",type=Path,required=True)
    ap.add_argument("--v2-rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    v2=prep_v2(a.v2_rows)
    wr=merge_authority(prep_wr(a.wr),v2)
    te=merge_authority(prep_te(a.te),v2)
    all_rows=pd.concat([wr,te],ignore_index=True)
    q=add_prior_state(all_rows)

    wrres=position_result(q,"WR")
    teres=position_result(q,"TE")
    if wrres["disposition"]=="WR_PLAYER_ALLOCATION_DIFFICULTY_INCREMENTAL_CONFIRMED" or teres["disposition"]=="TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED":
        disp="PLAYER_ALLOCATION_ONLY_DIFFICULTY_CONFIRMED"
    elif wrres["disposition"]=="WR_PLAYER_ALLOCATION_DIFFICULTY_REDUNDANT_WITH_SHARE_VOLATILITY":
        disp="PLAYER_ALLOCATION_ONLY_DIFFICULTY_PARTIAL_WR_REDUNDANT"
    else:
        disp="NO_ACTIONABLE_PLAYER_ALLOCATION_ONLY_DIFFICULTY"

    result={
      "version":VERSION,
      "disposition":disp,
      "wr":wrres,
      "te":teres,
      "matched_rows_before_prior":{"WR":int(len(wr)),"TE":int(len(te))},
      "scored_rows":{"WR":int((q["position"]=="WR").sum()),"TE":int((q["position"]=="TE").sum())},
      "scored_players":{"WR":int(q.loc[q["position"].eq("WR"),"player_identity_key"].nunique()),"TE":int(q.loc[q["position"].eq("TE"),"player_identity_key"].nunique())},
      "prior_window":PRIOR_WINDOW,
      "min_prior_games":MIN_PRIOR,
      "candidate_models_fit":0,
      "sportsbook_inputs_used":0,
      "oddsapi_calls":0,
      "outcomes_2026_read":0,
      "projection_mean_changes":0,
      "production_changed":False,
    }
    q.to_csv(a.out_dir/"player_allocation_only_difficulty_rows.csv",index=False)
    (a.out_dir/"player_allocation_only_difficulty_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
