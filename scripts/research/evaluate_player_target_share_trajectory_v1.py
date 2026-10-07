#!/usr/bin/env python3
"""Player Target Share Trajectory V1.

No-fit historical diagnostic against exact promoted WR and TE opportunity
authorities. Tests whether a player's strictly-prior recent2 target-share
movement vs his earlier same-season team history explains remaining signed
opportunity error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

VERSION="PLAYER_TARGET_SHARE_TRAJECTORY_V1"
BOOT_REPS=5000
BOOT_SEED=20261007
WR_VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"
WR_EXPECTED_ROWS=4193
TE_EXPECTED_ROWS=3214

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

def lower(x):
    y=_to_pandas(x).copy()
    y.columns=[str(c).strip().lower() for c in y.columns]
    return y

def regular_only(x):
    y=x.copy()
    c="season_type" if "season_type" in y.columns else "game_type" if "game_type" in y.columns else None
    if c:
        s=y[c].astype(str).str.upper()
        keep=s.isin(["REG","REGULAR","RS",""])
        if keep.any(): y=y.loc[keep].copy()
    return y

def identity_map():
    import nflreadpy as nfl
    rows=[]
    for season in (2023,2024,2025):
        raw=_to_pandas(nfl.load_player_stats(seasons=[season],summary_level="week"))
        x=_normalize_weekly(raw,season)
        z=x[["season","player_clean_key","player_id"]].copy()
        z["player_id"]=z["player_id"].map(clean)
        z=z.loc[z["player_clean_key"].astype(str).ne("")&z["player_id"].ne("")].drop_duplicates()
        rows.append(z)
    allx=pd.concat(rows,ignore_index=True)
    g=(allx.groupby(["season","player_clean_key"])["player_id"]
       .agg(lambda s: sorted(set(map(str,s)))).reset_index())
    g["id_count"]=g["player_id"].map(len)
    g["stable_receiver_id"]=g["player_id"].map(lambda v:v[0] if len(v)==1 else "")
    return g[["season","player_clean_key","stable_receiver_id","id_count"]]

def load_pbp():
    import nflreadpy as nfl
    frames=[]
    for season in (2023,2024,2025):
        x=regular_only(lower(nfl.load_pbp(seasons=[season])))
        for c in ["season","week","game_id","posteam","receiver_player_id",
                  "pass_attempt","sack","two_point_attempt"]:
            if c not in x.columns: x[c]=np.nan
        x["season"]=num(x["season"]).fillna(season).astype(int)
        x["week"]=num(x["week"]).astype("Int64")
        x["team"]=x["posteam"].map(team)
        x["receiver_id"]=x["receiver_player_id"].map(clean)
        raw=num(x["pass_attempt"]).fillna(0).eq(1)
        sack=num(x["sack"]).fillna(0).eq(1)
        two=num(x["two_point_attempt"]).fillna(0).eq(1)
        x["target_event"]=raw & ~sack & ~two & x["receiver_id"].ne("") & x["team"].ne("")
        frames.append(x[["season","week","game_id","team","receiver_id","target_event"]])
    out=pd.concat(frames,ignore_index=True,sort=False)
    if out["season"].max()>2025: raise RuntimeError("2026+ PBP forbidden")
    return out

def prepare_wr(path):
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={"variant","season","week","team","player_clean_key","pred_targets","mc_rec_yards","actual_targets"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"WR authority missing: {sorted(miss)}")
    x=x.loc[x["variant"].astype(str).eq(WR_VARIANT)].copy()
    if len(x)!=WR_EXPECTED_ROWS: raise RuntimeError(f"WR authority row drift: {len(x)}")
    x["season"]=num(x["season"]).astype(int); x["week"]=num(x["week"]).astype(int)
    x["team"]=x["team"].map(team)
    for c in ["pred_targets","mc_rec_yards","actual_targets"]: x[c]=num(x[c])
    x=x.loc[x["pred_targets"].gt(0)&x[["mc_rec_yards","actual_targets"]].notna().all(axis=1)].copy()
    x["pred_ypt"]=x["mc_rec_yards"]/x["pred_targets"]
    x["opportunity_error"]=(x["pred_targets"]-x["actual_targets"])*x["pred_ypt"]
    x["position_group"]="WR"
    return x[["season","week","team","player_clean_key","position_group","opportunity_error"]]

def prepare_te(path):
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={"season","week","team","player_clean_key","candidate_targets_r5p","candidate_rec_yards_r5p","targets"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"TE authority missing: {sorted(miss)}")
    if len(x)!=TE_EXPECTED_ROWS: raise RuntimeError(f"TE authority row drift: {len(x)}")
    x["season"]=num(x["season"]).astype(int); x["week"]=num(x["week"]).astype(int)
    x["team"]=x["team"].map(team)
    for c in ["candidate_targets_r5p","candidate_rec_yards_r5p","targets"]: x[c]=num(x[c])
    x=x.loc[x["candidate_targets_r5p"].gt(0)&x[["candidate_rec_yards_r5p","targets"]].notna().all(axis=1)].copy()
    x["pred_ypt"]=x["candidate_rec_yards_r5p"]/x["candidate_targets_r5p"]
    x["opportunity_error"]=(x["candidate_targets_r5p"]-x["targets"])*x["pred_ypt"]
    x["position_group"]="TE"
    return x[["season","week","team","player_clean_key","position_group","opportunity_error"]]

def attach_identity(authority,ids):
    x=authority.merge(ids,on=["season","player_clean_key"],how="left",validate="many_to_one")
    x["stable_receiver_id"]=x["stable_receiver_id"].fillna("").astype(str)
    x["identity_ok"]=x["stable_receiver_id"].ne("")&num(x["id_count"]).eq(1)
    return x

def build_game_indexes(pbp):
    # Team-game list comes from all offensive PBP rows, while target counts use
    # only official target events. This preserves zero-target player games.
    game=(pbp.loc[pbp["team"].ne("")]
          [["season","week","game_id","team"]].drop_duplicates()
          .sort_values(["season","team","week","game_id"]))
    targets=pbp.loc[pbp["target_event"]].copy()
    team_t=(targets.groupby(["season","week","game_id","team"],as_index=False)
            .size().rename(columns={"size":"team_targets"}))
    player_t=(targets.groupby(["season","week","game_id","team","receiver_id"],as_index=False)
              .size().rename(columns={"size":"player_targets"}))
    game=game.merge(team_t,on=["season","week","game_id","team"],how="left",validate="one_to_one")
    game["team_targets"]=num(game["team_targets"]).fillna(0.0)
    team_idx={}
    for (season,tm),g in game.groupby(["season","team"],sort=False):
        team_idx[(int(season),str(tm))]=g.sort_values(["week","game_id"]).copy()
    player_lookup={(int(r.season),int(r.week),str(r.game_id),str(r.team),str(r.receiver_id)):float(r.player_targets)
                   for r in player_t.itertuples(index=False)}
    return team_idx,player_lookup

def trajectory_state(team_idx,player_lookup,season,week,tm,pid):
    g=team_idx.get((int(season),str(tm)))
    if g is None or g.empty: return None
    h=g.loc[num(g["week"]).lt(int(week))].copy()
    if len(h)<4: return None
    recent=h.tail(2).copy()
    earlier=h.iloc[:-2].copy()
    if len(earlier)<2: return None
    recent_den=float(num(recent["team_targets"]).sum())
    earlier_den=float(num(earlier["team_targets"]).sum())
    if recent_den<=0 or earlier_den<=0: return None
    def player_sum(q):
        total=0.0
        for r in q.itertuples(index=False):
            total += player_lookup.get((int(r.season),int(r.week),str(r.game_id),str(r.team),str(pid)),0.0)
        return float(total)
    recent_num=player_sum(recent)
    earlier_num=player_sum(earlier)
    recent_share=recent_num/recent_den
    earlier_share=earlier_num/earlier_den
    return {
      "prior_team_games":int(len(h)),
      "earlier_team_games":int(len(earlier)),
      "recent2_team_targets":recent_den,
      "earlier_team_targets":earlier_den,
      "recent2_player_targets":recent_num,
      "earlier_player_targets":earlier_num,
      "recent2_share":recent_share,
      "earlier_share":earlier_share,
      "trajectory_delta":recent_share-earlier_share,
      "feature_max_week":int(num(recent["week"]).max()),
    }

def build_panel(authority,pbp):
    team_idx,player_lookup=build_game_indexes(pbp)
    rows=[]; violations=0
    cache={}
    for r in authority.loc[authority["identity_ok"]].itertuples(index=False):
        key=(int(r.season),int(r.week),str(r.team),str(r.stable_receiver_id))
        if key not in cache:
            cache[key]=trajectory_state(team_idx,player_lookup,*key)
        st=cache[key]
        if st is None: continue
        if int(st["feature_max_week"])>=int(r.week):
            violations+=1
        row={
          "season":int(r.season),"week":int(r.week),"team":str(r.team),
          "player_clean_key":str(r.player_clean_key),"receiver_id":str(r.stable_receiver_id),
          "position_group":str(r.position_group),"opportunity_error":float(r.opportunity_error),
        }
        row.update(st); rows.append(row)
    return pd.DataFrame(rows),violations

def rho(df):
    z=df[["trajectory_delta","opportunity_error"]].dropna()
    if len(z)<3 or z["trajectory_delta"].nunique()<2 or z["opportunity_error"].nunique()<2: return np.nan
    return float(spearmanr(z["trajectory_delta"].to_numpy(float),z["opportunity_error"].to_numpy(float)).statistic)

def metrics(df):
    d=num(df["trajectory_delta"]).dropna()
    z=df.loc[df["trajectory_delta"].notna()].copy()
    rising=z.loc[z["trajectory_delta"].gt(0),"opportunity_error"]
    falling=z.loc[z["trajectory_delta"].lt(0),"opportunity_error"]
    flat=z.loc[z["trajectory_delta"].eq(0),"opportunity_error"]
    return {
      "rows":int(len(z)),
      "players":int(z["player_clean_key"].nunique()) if len(z) else 0,
      "rho":rho(z),
      "trajectory_sd":float(d.std(ddof=0)) if len(d) else np.nan,
      "trajectory_p10":float(d.quantile(.10)) if len(d) else np.nan,
      "trajectory_p50":float(d.quantile(.50)) if len(d) else np.nan,
      "trajectory_p90":float(d.quantile(.90)) if len(d) else np.nan,
      "fraction_abs_ge_0p03":float(d.abs().ge(.03).mean()) if len(d) else np.nan,
      "fraction_abs_ge_0p05":float(d.abs().ge(.05).mean()) if len(d) else np.nan,
      "mean_opp_error_rising":float(num(rising).mean()) if len(rising) else np.nan,
      "mean_opp_error_falling":float(num(falling).mean()) if len(falling) else np.nan,
      "mean_opp_error_flat":float(num(flat).mean()) if len(flat) else np.nan,
    }

def bootstrap(df,scope):
    z=df[["trajectory_delta","opportunity_error","player_clean_key","position_group"]].dropna().copy()
    if scope=="WR": z=z.loc[z["position_group"].eq("WR")].copy()
    elif scope=="TE": z=z.loc[z["position_group"].eq("TE")].copy()
    z["cluster"]=z["position_group"].astype(str)+"|"+z["player_clean_key"].astype(str)
    clusters=sorted(z["cluster"].unique())
    if len(clusters)<2:
        return {"valid_reps":0,"p_negative":np.nan,"ci_low":np.nan,"ci_high":np.nan}
    groups=[]
    for c in clusters:
        q=z.loc[z["cluster"].eq(c)]
        groups.append((num(q["trajectory_delta"]).to_numpy(float),num(q["opportunity_error"]).to_numpy(float)))
    rng=np.random.default_rng(BOOT_SEED)
    vals=np.full(BOOT_REPS,np.nan)
    ncl=len(groups)
    for rep in range(BOOT_REPS):
        idx=rng.integers(0,ncl,size=ncl)
        xa=np.concatenate([groups[i][0] for i in idx])
        ya=np.concatenate([groups[i][1] for i in idx])
        if len(xa)>=3 and np.unique(xa).size>=2 and np.unique(ya).size>=2:
            vals[rep]=float(spearmanr(xa,ya).statistic)
    a=vals[np.isfinite(vals)]
    return {
      "valid_reps":int(len(a)),
      "p_negative":float((a<0).mean()) if len(a) else np.nan,
      "ci_low":float(np.quantile(a,.025)) if len(a) else np.nan,
      "ci_high":float(np.quantile(a,.975)) if len(a) else np.nan,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wr",type=Path,required=True)
    ap.add_argument("--te",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    ids=identity_map()
    wr=attach_identity(prepare_wr(a.wr),ids)
    te=attach_identity(prepare_te(a.te),ids)
    idcov={"WR":float(wr["identity_ok"].mean()),"TE":float(te["identity_ok"].mean())}
    authority=pd.concat([wr,te],ignore_index=True)
    pbp=load_pbp()
    panel,violations=build_panel(authority,pbp)

    cells={
      "WR_2023":metrics(panel.loc[(panel.position_group.eq("WR"))&(panel.season.eq(2023))]),
      "WR_2024":metrics(panel.loc[(panel.position_group.eq("WR"))&(panel.season.eq(2024))]),
      "WR_pooled":metrics(panel.loc[panel.position_group.eq("WR")]),
      "TE_2024":metrics(panel.loc[(panel.position_group.eq("TE"))&(panel.season.eq(2024))]),
      "TE_2025":metrics(panel.loc[(panel.position_group.eq("TE"))&(panel.season.eq(2025))]),
      "TE_pooled":metrics(panel.loc[panel.position_group.eq("TE")]),
      "ALL_pooled":metrics(panel),
    }
    bw=bootstrap(panel,"WR"); bt=bootstrap(panel,"TE"); ba=bootstrap(panel,"ALL")
    support=bool(
      idcov["WR"]>=.95 and idcov["TE"]>=.95
      and cells["WR_2023"]["rows"]>=500 and cells["WR_2024"]["rows"]>=500
      and cells["WR_pooled"]["players"]>=100
      and cells["TE_2024"]["rows"]>=250 and cells["TE_2025"]["rows"]>=250
      and cells["TE_pooled"]["players"]>=50
    )
    confirmed=bool(
      support
      and cells["WR_2023"]["rho"]<0 and cells["WR_2024"]["rho"]<0
      and cells["TE_2024"]["rho"]<0 and cells["TE_2025"]["rho"]<0
      and cells["WR_pooled"]["rho"]<=-.05 and cells["TE_pooled"]["rho"]<=-.05
      and bw["p_negative"]>=.95 and bt["p_negative"]>=.95 and ba["p_negative"]>=.99
      and violations==0
    )
    disp=("PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED" if confirmed else
          "NO_ACTIONABLE_PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL" if support else
          "PLAYER_TARGET_SHARE_TRAJECTORY_SOURCE_LIMITED")
    result={
      "version":VERSION,"disposition":disp,
      "identity_mapping_coverage":idcov,
      "authority_rows":{"WR":int(len(wr)),"TE":int(len(te))},
      "panel_rows":int(len(panel)),
      "panel_players":{"WR":int(panel.loc[panel.position_group.eq("WR"),"player_clean_key"].nunique()),
                       "TE":int(panel.loc[panel.position_group.eq("TE"),"player_clean_key"].nunique())},
      "cells":cells,
      "bootstrap_wr":bw,"bootstrap_te":bt,"bootstrap_combined":ba,
      "gates":{"support":support,"confirmed":confirmed},
      "same_or_future_feature_violations":int(violations),
      "recent_window_games":2,
      "minimum_prior_team_games":4,
      "candidate_models_fit":0,
      "sportsbook_inputs_used":0,
      "outcomes_2026_read":0,
      "production_changed":False,
      "target_game_feature_rows_read":0,
    }
    panel.to_csv(a.out_dir/"player_target_share_trajectory_rows.csv",index=False)
    (a.out_dir/"player_target_share_trajectory_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
