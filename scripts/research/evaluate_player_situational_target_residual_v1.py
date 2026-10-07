#!/usr/bin/env python3
"""Player Situational Target Residual V1.

No-fit historical diagnostic against exact promoted WR-R15/M38 and TE-R5P
opportunity authorities. No sportsbook inputs. No 2026 outcomes.
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

VERSION="PLAYER_SITUATIONAL_TARGET_RESIDUAL_V1"
HIST_SEASONS=(2022,2023,2024,2025)
CONTEXTS=("THIRD_DOWN","RED_ZONE","TWO_MINUTE")
HISTORY_GAMES=8
MIN_TARGETED_GAMES=4
MIN_PLAYER_TARGETS=5
MIN_CONTEXT_TEAM_TARGETS=5
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

def load_pbp():
    import nflreadpy as nfl
    frames=[]
    for season in HIST_SEASONS:
        x=regular_only(lower(nfl.load_pbp(seasons=[season])))
        for c in ["season","week","game_id","posteam","receiver_player_id",
                  "pass_attempt","sack","two_point_attempt","down","yardline_100",
                  "half_seconds_remaining"]:
            if c not in x.columns: x[c]=np.nan
        x["season"]=num(x["season"]).fillna(season).astype(int)
        x["week"]=num(x["week"]).astype("Int64")
        x["team"]=x["posteam"].map(team)
        x["receiver_id"]=x["receiver_player_id"].map(clean)
        raw=num(x["pass_attempt"]).fillna(0).eq(1)
        sack=num(x["sack"]).fillna(0).eq(1)
        two=num(x["two_point_attempt"]).fillna(0).eq(1)
        x["target_event"]=raw & ~sack & ~two & x["receiver_id"].ne("") & x["team"].ne("")
        d=num(x["down"]); y=num(x["yardline_100"]); hs=num(x["half_seconds_remaining"])
        x["ALL"]=x["target_event"]
        x["THIRD_DOWN"]=x["target_event"] & d.eq(3)
        x["RED_ZONE"]=x["target_event"] & y.le(20)
        x["TWO_MINUTE"]=x["target_event"] & hs.le(120)
        frames.append(x[["season","week","game_id","team","receiver_id","ALL",*CONTEXTS]])
    out=pd.concat(frames,ignore_index=True,sort=False)
    if out["season"].max()>2025:
        raise RuntimeError("2026+ PBP forbidden")
    return out

def identity_map():
    import nflreadpy as nfl
    rows=[]
    for season in HIST_SEASONS:
        raw=_to_pandas(nfl.load_player_stats(seasons=[season],summary_level="week"))
        x=_normalize_weekly(raw,season)
        z=x[["season","player_clean_key","player_id"]].copy()
        z["player_id"]=z["player_id"].map(clean)
        z=z.loc[z["player_clean_key"].astype(str).ne("")&z["player_id"].ne("")]
        rows.append(z.drop_duplicates())
    allx=pd.concat(rows,ignore_index=True).drop_duplicates()
    g=(allx.groupby(["season","player_clean_key"])["player_id"]
       .agg(lambda s: sorted(set(map(str,s)))).reset_index())
    g["id_count"]=g["player_id"].map(len)
    g["stable_receiver_id"]=g["player_id"].map(lambda v:v[0] if len(v)==1 else "")
    return g[["season","player_clean_key","stable_receiver_id","id_count"]]

def prepare_wr(path):
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={"variant","season","week","team","player_clean_key","pred_targets","mc_rec_yards","actual_targets"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"WR authority missing {sorted(miss)}")
    x=x.loc[x["variant"].astype(str).eq(WR_VARIANT)].copy()
    if len(x)!=WR_EXPECTED_ROWS: raise RuntimeError(f"WR row drift {len(x)}")
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
    if miss: raise RuntimeError(f"TE authority missing {sorted(miss)}")
    if len(x)!=TE_EXPECTED_ROWS: raise RuntimeError(f"TE row drift {len(x)}")
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

def chronology_before(df,season,week):
    return (num(df["season"])<season)|((num(df["season"])==season)&(num(df["week"])<week))

def aggregate_pbp(pbp):
    t=pbp.loc[pbp["ALL"]].copy()
    tg=(t.groupby(["season","week","game_id","team"],as_index=False)
        .agg(**{c:(c,"sum") for c in ("ALL",*CONTEXTS)}))
    pg=(t.groupby(["season","week","game_id","team","receiver_id"],as_index=False)
        .agg(**{c:(c,"sum") for c in ("ALL",*CONTEXTS)}))
    team_index={}
    for tm,g in tg.groupby("team",sort=False):
        team_index[str(tm)]=g.sort_values(["season","week","game_id"]).copy()
    player_index={}
    for (tm,pid),g in pg.groupby(["team","receiver_id"],sort=False):
        player_index[(str(tm),str(pid))]=g.sort_values(["season","week","game_id"]).copy()
    return team_index,player_index

def state_for_row(team_index,player_index,tm,pid,season,week):
    tg=team_index.get(str(tm))
    if tg is None or tg.empty: return None
    h=tg.loc[chronology_before(tg,int(season),int(week))].tail(HISTORY_GAMES).copy()
    if h.empty: return None
    gamekeys=set(zip(h["season"].astype(int),h["week"].astype(int),h["game_id"].astype(str)))
    pg=player_index.get((str(tm),str(pid)))
    if pg is None:
        return None
    mask=[
        (int(s),int(w),str(gid)) in gamekeys
        for s,w,gid in zip(pg["season"],pg["week"],pg["game_id"])
    ]
    ph=pg.loc[mask].copy()
    if ph["game_id"].nunique()<MIN_TARGETED_GAMES: return None
    player_all=float(ph["ALL"].sum())
    if player_all<MIN_PLAYER_TARGETS: return None
    team_all=float(h["ALL"].sum())
    if team_all<=0: return None
    overall=player_all/team_all
    out={"history_team_games":int(len(h)),"history_targeted_games":int(ph["game_id"].nunique()),
         "history_player_targets":player_all,"overall_share":overall}
    for ctx in CONTEXTS:
        den=float(h[ctx].sum())
        n=float(ph[ctx].sum())
        out[f"{ctx.lower()}_team_targets"]=den
        out[f"{ctx.lower()}_player_targets"]=n
        out[f"{ctx.lower()}_share"]=n/den if den>=MIN_CONTEXT_TEAM_TARGETS else np.nan
        out[f"{ctx.lower()}_delta"]=(n/den-overall) if den>=MIN_CONTEXT_TEAM_TARGETS else np.nan
    return out

def build_panel(authority,pbp):
    team_index,player_index=aggregate_pbp(pbp)
    rows=[]
    cache={}
    for r in authority.loc[authority["identity_ok"]].itertuples(index=False):
        key=(str(r.team),str(r.stable_receiver_id),int(r.season),int(r.week))
        if key not in cache:
            cache[key]=state_for_row(team_index,player_index,*key)
        st=cache[key]
        if st is None: continue
        row={
          "season":int(r.season),"week":int(r.week),"team":str(r.team),
          "player_clean_key":str(r.player_clean_key),"receiver_id":str(r.stable_receiver_id),
          "position_group":str(r.position_group),"opportunity_error":float(r.opportunity_error),
        }
        row.update(st)
        rows.append(row)
    return pd.DataFrame(rows)

def rho(df,ctx):
    z=df[[f"{ctx.lower()}_delta","opportunity_error"]].dropna()
    if len(z)<3 or z.iloc[:,0].nunique()<2 or z.iloc[:,1].nunique()<2: return np.nan
    return float(spearmanr(z.iloc[:,0].to_numpy(float),z.iloc[:,1].to_numpy(float)).statistic)

def scope_metrics(df,ctx):
    z=df[[f"{ctx.lower()}_delta","opportunity_error","player_clean_key","position_group"]].dropna()
    return {"rows":int(len(z)),"players":int(z["player_clean_key"].nunique()),"rho":rho(z,ctx)}

def bootstrap(df,ctx,scope):
    xcol=f"{ctx.lower()}_delta"
    z=df[[xcol,"opportunity_error","player_clean_key","position_group"]].dropna().copy()
    if scope=="WR": z=z.loc[z["position_group"].eq("WR")].copy()
    elif scope=="TE": z=z.loc[z["position_group"].eq("TE")].copy()
    z["cluster"]=z["position_group"].astype(str)+"|"+z["player_clean_key"].astype(str)
    clusters=sorted(z["cluster"].unique().tolist())
    if len(clusters)<2:
        return {"valid_reps":0,"p_negative":np.nan,"ci_low":np.nan,"ci_high":np.nan}
    # Mechanical performance optimization only: preserve the exact cluster
    # resample and exact scipy Spearman statistic without rebuilding pandas
    # frames inside each replicate.
    groups=[]
    for c in clusters:
        q=z.loc[z["cluster"].eq(c)]
        groups.append((
            num(q[xcol]).to_numpy(float),
            num(q["opportunity_error"]).to_numpy(float),
        ))
    rng=np.random.default_rng(BOOT_SEED)
    vals=np.empty(BOOT_REPS,dtype=float)
    vals.fill(np.nan)
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
    idcov={
      "WR":float(wr["identity_ok"].mean()),
      "TE":float(te["identity_ok"].mean()),
    }
    authority=pd.concat([wr,te],ignore_index=True,sort=False)
    pbp=load_pbp()
    panel=build_panel(authority,pbp)

    results={}
    confirmed=[]
    any_judgeable=False
    for ctx in CONTEXTS:
        cells={
          "WR_2023":scope_metrics(panel.loc[(panel.position_group.eq("WR"))&(panel.season.eq(2023))],ctx),
          "WR_2024":scope_metrics(panel.loc[(panel.position_group.eq("WR"))&(panel.season.eq(2024))],ctx),
          "TE_2024":scope_metrics(panel.loc[(panel.position_group.eq("TE"))&(panel.season.eq(2024))],ctx),
          "TE_2025":scope_metrics(panel.loc[(panel.position_group.eq("TE"))&(panel.season.eq(2025))],ctx),
          "WR_pooled":scope_metrics(panel.loc[panel.position_group.eq("WR")],ctx),
          "TE_pooled":scope_metrics(panel.loc[panel.position_group.eq("TE")],ctx),
          "ALL_pooled":scope_metrics(panel,ctx),
        }
        support=bool(
          idcov["WR"]>=.95 and idcov["TE"]>=.95
          and cells["WR_2023"]["rows"]>=500 and cells["WR_2024"]["rows"]>=500
          and cells["WR_pooled"]["players"]>=100
          and cells["TE_2024"]["rows"]>=250 and cells["TE_2025"]["rows"]>=250
          and cells["TE_pooled"]["players"]>=50
        )
        any_judgeable |= support
        bw=bootstrap(panel,ctx,"WR")
        bt=bootstrap(panel,ctx,"TE")
        ba=bootstrap(panel,ctx,"ALL")
        gate=bool(
          support
          and cells["WR_2023"]["rho"]<0 and cells["WR_2024"]["rho"]<0
          and cells["TE_2024"]["rho"]<0 and cells["TE_2025"]["rho"]<0
          and cells["WR_pooled"]["rho"]<=-.05
          and cells["TE_pooled"]["rho"]<=-.05
          and bw["p_negative"]>=.95 and bt["p_negative"]>=.95 and ba["p_negative"]>=.99
        )
        if gate: confirmed.append(ctx)
        results[ctx]={"cells":cells,"support":support,"bootstrap_wr":bw,"bootstrap_te":bt,
                      "bootstrap_combined":ba,"replicated_player_role_signal":gate}

    if confirmed:
        disp="PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL_CONFIRMED"
    elif any_judgeable:
        disp="NO_ACTIONABLE_PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL"
    else:
        disp="PLAYER_SITUATIONAL_TARGET_ROLE_SOURCE_LIMITED"

    out={
      "version":VERSION,
      "disposition":disp,
      "identity_mapping_coverage":idcov,
      "authority_rows":{"WR":int(len(wr)),"TE":int(len(te))},
      "panel_rows":int(len(panel)),
      "panel_players_by_position":{
        "WR":int(panel.loc[panel.position_group.eq("WR"),"player_clean_key"].nunique()),
        "TE":int(panel.loc[panel.position_group.eq("TE"),"player_clean_key"].nunique()),
      },
      "contexts":results,
      "confirmed_contexts":confirmed,
      "history_games":HISTORY_GAMES,
      "min_targeted_games":MIN_TARGETED_GAMES,
      "min_player_targets":MIN_PLAYER_TARGETS,
      "min_context_team_targets":MIN_CONTEXT_TEAM_TARGETS,
      "candidate_models_fit":0,
      "sportsbook_inputs_used":0,
      "outcomes_2026_read":0,
      "production_changed":False,
      "target_game_feature_rows_read":0,
    }
    panel.to_csv(a.out_dir/"player_situational_target_residual_rows.csv",index=False)
    (a.out_dir/"player_situational_target_residual_result.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
