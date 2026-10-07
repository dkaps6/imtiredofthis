#!/usr/bin/env python3
"""QB-Receiver Pair Efficiency V1.

No fitted model. Compares exact passer-receiver prior YPT against the same
receiver's own prior YPT under a frozen leakage-safe design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION="QB_RECEIVER_PAIR_EFFICIENCY_V1"
HISTORY_START=2022
TARGET_SEASONS=(2023,2024,2025)
TARGET_WEEKS=tuple(range(5,19))
MIN_RECEIVER_TARGETS=10
MIN_PAIR_TARGETS=5
BOOT_REPS=10000
BOOT_SEED=20261007

def to_pd(x):
    if isinstance(x,pd.DataFrame): return x.copy()
    if hasattr(x,"to_pandas"): return x.to_pandas()
    return pd.DataFrame(x)

def lower(x):
    y=to_pd(x)
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y

def num(x): return pd.to_numeric(x,errors="coerce")

def clean(v):
    if v is None or pd.isna(v): return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s

def team(v):
    try:
        t=canon_team(v)
        return "WAS" if t=="WSH" else t
    except Exception:
        return clean(v).upper()

def regular_only(df):
    x=df.copy()
    c="season_type" if "season_type" in x.columns else "game_type" if "game_type" in x.columns else None
    if c:
        s=x[c].astype(str).str.upper()
        keep=s.isin(["REG","REGULAR","RS",""])
        if keep.any(): x=x.loc[keep].copy()
    return x

def ensure(x,cols):
    for c in cols:
        if c not in x.columns: x[c]=np.nan
    return x

def load_pbp():
    import nflreadpy as nfl
    frames=[]
    for season in range(HISTORY_START,max(TARGET_SEASONS)+1):
        q=regular_only(lower(nfl.load_pbp(seasons=[season])))
        q=ensure(q,[
            "season","week","game_id","posteam",
            "passer_player_id","receiver_player_id",
            "pass_attempt","sack","two_point_attempt","complete_pass",
            "passing_yards",
        ])
        q["season"]=num(q["season"]).fillna(season).astype(int)
        q["week"]=num(q["week"]).astype("Int64")
        q["team"]=q["posteam"].map(team)
        raw=num(q["pass_attempt"]).fillna(0).eq(1)
        sack=num(q["sack"]).fillna(0).eq(1)
        two=num(q["two_point_attempt"]).fillna(0).eq(1)
        q["official_pass_attempt"]=raw & ~sack & ~two
        q["passer_id"]=q["passer_player_id"].map(clean)
        q["receiver_id"]=q["receiver_player_id"].map(clean)
        q["target_event"]=q["official_pass_attempt"] & q["receiver_id"].ne("")
        q["_complete"]=num(q["complete_pass"]).fillna(0).eq(1)
        q["_yards"]=num(q["passing_yards"]).fillna(0.0)
        frames.append(q)
    x=pd.concat(frames,ignore_index=True,sort=False)
    return x

def load_positions():
    import nflreadpy as nfl
    rows=[]
    for season in TARGET_SEASONS:
        x=lower(nfl.load_player_stats(seasons=[season],summary_level="week"))
        pid=None
        for c in ("player_id","gsis_id","player_gsis_id"):
            if c in x.columns: pid=c; break
        pos=None
        for c in ("position","position_group","pos"):
            if c in x.columns: pos=c; break
        if pid is None or pos is None:
            continue
        z=pd.DataFrame({
            "season":season,
            "receiver_id":x[pid].map(clean),
            "position":x[pos].astype(str).str.upper().str.strip(),
        })
        z=z.loc[z["receiver_id"].ne("")&z["position"].ne("")].drop_duplicates(["season","receiver_id"],keep="last")
        rows.append(z)
    if not rows:
        raise RuntimeError("no receiver position identity")
    return pd.concat(rows,ignore_index=True)

def chrono_before(df,season,week):
    return (num(df["season"])<season) | ((num(df["season"])==season)&(num(df["week"])<week))

def latest_team_games(pbp,team_id,season,week,n=3):
    q=pbp.loc[
        pbp["team"].eq(team_id)
        & num(pbp["season"]).eq(season)
        & num(pbp["week"]).lt(week)
        & pbp["official_pass_attempt"]
    ].copy()
    games=(q[["week","game_id"]].drop_duplicates()
           .sort_values(["week","game_id"]).tail(n))
    return set(games["game_id"].astype(str))

def passer_proxy(pbp,team_id,season,week):
    gids=latest_team_games(pbp,team_id,season,week,3)
    if not gids: return ""
    q=pbp.loc[
        pbp["team"].eq(team_id)
        & pbp["game_id"].astype(str).isin(gids)
        & pbp["official_pass_attempt"]
        & pbp["passer_id"].ne("")
    ].copy()
    if q.empty: return ""
    c=q["passer_id"].value_counts().rename_axis("passer_id").reset_index(name="attempts")
    c=c.sort_values(["attempts","passer_id"],ascending=[False,True])
    return str(c.iloc[0]["passer_id"])

def receiver_history_window(pbp,receiver_id,season,week,n=8):
    q=pbp.loc[
        pbp["target_event"]
        & pbp["receiver_id"].eq(receiver_id)
        & chrono_before(pbp,season,week)
    ].copy()
    if q.empty: return q
    games=(q[["season","week","game_id"]].drop_duplicates()
           .sort_values(["season","week","game_id"]).tail(n))
    keys=set(zip(games["season"].astype(int),games["week"].astype(int),games["game_id"].astype(str)))
    mask=[
        (int(s),int(w),str(g)) in keys
        for s,w,g in zip(q["season"],q["week"],q["game_id"])
    ]
    return q.loc[mask].copy()

def build_rows(pbp,positions):
    target=pbp.loc[
        pbp["target_event"]
        & pbp["season"].isin(TARGET_SEASONS)
        & pbp["week"].isin(TARGET_WEEKS)
    ].copy()
    grouped=(target.groupby(["season","week","game_id","team","receiver_id"],as_index=False)
             .agg(actual_targets=("receiver_id","size"),actual_yards=("_yards","sum")))
    grouped["actual_ypt"]=grouped["actual_yards"]/grouped["actual_targets"]

    rows=[]
    proxy_cache={}
    for r in grouped.itertuples(index=False):
        key=(r.team,int(r.season),int(r.week))
        if key not in proxy_cache:
            proxy_cache[key]=passer_proxy(pbp,r.team,int(r.season),int(r.week))
        proxy=proxy_cache[key]
        if not proxy: continue

        h=receiver_history_window(pbp,r.receiver_id,int(r.season),int(r.week),8)
        if h.empty: continue
        recv_targets=int(len(h))
        recv_yards=float(h["_yards"].sum())
        if recv_targets<MIN_RECEIVER_TARGETS: continue
        control=recv_yards/recv_targets

        pair=h.loc[h["passer_id"].eq(proxy)].copy()
        pair_targets=int(len(pair))
        if pair_targets<MIN_PAIR_TARGETS: continue
        pair_yards=float(pair["_yards"].sum())
        challenger=pair_yards/pair_targets

        rows.append({
            "season":int(r.season),"week":int(r.week),"game_id":str(r.game_id),
            "team":r.team,"receiver_id":r.receiver_id,
            "position":positions.get((int(r.season),str(r.receiver_id)),""),
            "proxy_passer_id":proxy,
            "receiver_prior_targets":recv_targets,
            "pair_prior_targets":pair_targets,
            "pair_target_fraction":pair_targets/recv_targets,
            "control_receiver_ypt":control,
            "pair_ypt":challenger,
            "actual_targets":int(r.actual_targets),
            "actual_yards":float(r.actual_yards),
            "actual_ypt":float(r.actual_ypt),
            "control_yards_at_actual_targets":control*float(r.actual_targets),
            "pair_yards_at_actual_targets":challenger*float(r.actual_targets),
        })
    return pd.DataFrame(rows)

def metric_block(z,pred):
    if z.empty:
        return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"pearson":np.nan,"spearman":np.nan,"miss20":0,"yard_mae_actual_targets":np.nan}
    e=num(z[pred])-num(z["actual_ypt"])
    yard_pred="control_yards_at_actual_targets" if pred=="control_receiver_ypt" else "pair_yards_at_actual_targets"
    ye=(num(z[yard_pred])-num(z["actual_yards"])).abs()
    return {
        "n":int(len(z)),
        "mae":float(e.abs().mean()),
        "rmse":float(np.sqrt(np.mean(np.square(e)))),
        "bias":float(e.mean()),
        "pearson":float(num(z[pred]).corr(num(z["actual_ypt"]),method="pearson")) if len(z)>2 else np.nan,
        "spearman":float(num(z[pred]).corr(num(z["actual_ypt"]),method="spearman")) if len(z)>2 else np.nan,
        "miss20":int(e.abs().ge(20).sum()),
        "yard_mae_actual_targets":float(ye.mean()),
    }

def score_subset(z):
    return {
        "rows":int(len(z)),
        "players":int(z["receiver_id"].nunique()) if len(z) else 0,
        "games":int(z["game_id"].nunique()) if len(z) else 0,
        "control":metric_block(z,"control_receiver_ypt"),
        "pair":metric_block(z,"pair_ypt"),
        "mean_pair_prior_targets":float(num(z["pair_prior_targets"]).mean()) if len(z) else np.nan,
        "mean_pair_target_fraction":float(num(z["pair_target_fraction"]).mean()) if len(z) else np.nan,
    }

def bootstrap(z):
    q=z[["game_id","actual_ypt","control_receiver_ypt","pair_ypt"]].dropna().copy()
    q["c"]=(q["control_receiver_ypt"]-q["actual_ypt"]).abs()
    q["p"]=(q["pair_ypt"]-q["actual_ypt"]).abs()
    g=q.groupby("game_id",as_index=False).agg(n=("actual_ypt","size"),cs=("c","sum"),ps=("p","sum"))
    if len(g)<2: return {"valid_reps":0,"p_improve":np.nan,"ci_low":np.nan,"ci_high":np.nan}
    a=g[["n","cs","ps"]].to_numpy(float)
    rng=np.random.default_rng(BOOT_SEED)
    probs=np.full(len(g),1/len(g))
    vals=[]
    done=0
    while done<BOOT_REPS:
        k=min(250,BOOT_REPS-done)
        counts=rng.multinomial(len(g),probs,size=k).astype(float)
        s=counts@a
        vals.append((s[:,1]-s[:,2])/s[:,0])
        done+=k
    v=np.concatenate(vals)
    return {
        "valid_reps":int(len(v)),
        "p_improve":float((v>0).mean()),
        "ci_low":float(np.quantile(v,.025)),
        "ci_high":float(np.quantile(v,.975)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    pbp=load_pbp()
    posdf=load_positions()
    posmap={(int(r.season),str(r.receiver_id)):str(r.position) for r in posdf.itertuples(index=False)}
    rows=build_rows(pbp,posmap)
    rows["position_group"]=np.where(rows["position"].eq("WR"),"WR",np.where(rows["position"].eq("TE"),"TE","OTHER"))

    primary=rows.loc[rows["position_group"].isin(["WR","TE"])].copy()
    season_scores={str(s):score_subset(primary.loc[primary["season"].eq(s)]) for s in TARGET_SEASONS}
    pooled=score_subset(primary)
    wr=score_subset(primary.loc[primary["position_group"].eq("WR")])
    te=score_subset(primary.loc[primary["position_group"].eq("TE")])
    boot=bootstrap(primary)

    support=all(season_scores[str(s)]["rows"]>=300 and season_scores[str(s)]["games"]>=100 for s in TARGET_SEASONS)
    season_improve=sum(
        season_scores[str(s)]["pair"]["mae"] < season_scores[str(s)]["control"]["mae"]
        for s in TARGET_SEASONS
    )
    gates={
        "season_support":bool(support),
        "pooled_mae_improves":bool(pooled["pair"]["mae"]<pooled["control"]["mae"]),
        "mae_improves_at_least_2_of_3_seasons":bool(season_improve>=2),
        "pooled_rmse_non_worse":bool(pooled["pair"]["rmse"]<=pooled["control"]["rmse"]),
        "bootstrap_p_ge_0p80":bool(boot["p_improve"]>=.80),
        "wr_pooled_mae_improves":bool(wr["pair"]["mae"]<wr["control"]["mae"]),
        "te_pooled_mae_non_worse":bool(te["pair"]["mae"]<=te["control"]["mae"]),
        "actual_target_held_fixed_yard_mae_improves":bool(pooled["pair"]["yard_mae_actual_targets"]<pooled["control"]["yard_mae_actual_targets"]),
        "miss20_non_worse":bool(pooled["pair"]["miss20"]<=pooled["control"]["miss20"]),
    }
    confirmed=all(gates.values())
    result={
        "version":VERSION,
        "disposition":"QB_RECEIVER_PAIR_EFFICIENCY_SIGNAL_CONFIRMED" if confirmed else "QB_RECEIVER_PAIR_EFFICIENCY_SIGNAL_CLOSED",
        "history_start":HISTORY_START,
        "target_seasons":list(TARGET_SEASONS),
        "target_weeks":[5,18],
        "control":"receiver YPT over same latest-8 receiver-game history",
        "challenger":"exact proxy-passer x receiver YPT inside same history horizon",
        "min_receiver_prior_targets":MIN_RECEIVER_TARGETS,
        "min_pair_prior_targets":MIN_PAIR_TARGETS,
        "season_scores":season_scores,
        "pooled_wr_te":pooled,
        "pooled_wr":wr,
        "pooled_te":te,
        "bootstrap":boot,
        "gates":gates,
        "models_fit":0,
        "sportsbook_inputs_used":0,
        "outcomes_2026_read":0,
        "production_changed":False,
    }
    rows.to_csv(a.out_dir/"qb_receiver_pair_efficiency_rows.csv",index=False)
    (a.out_dir/"qb_receiver_pair_efficiency_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
