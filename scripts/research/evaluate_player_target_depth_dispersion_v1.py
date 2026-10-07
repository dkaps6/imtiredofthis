#!/usr/bin/env python3
"""Player Target Depth Dispersion V1 diagnostic."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

VERSION="PLAYER_TARGET_DEPTH_DISPERSION_V1"
WR_VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"
WR_EXPECTED_ROWS=4193
TE_EXPECTED_ROWS=3214
TARGETS={"WR":(2023,2024),"TE":(2024,2025)}
BOOT_REPS=5000
BOOT_SEED=20261007

def num(x): return pd.to_numeric(x,errors="coerce")
def clean(v):
    if v is None or pd.isna(v): return ""
    s=str(v).strip()
    return "" if s.lower() in {"","nan","none","<na>"} else s

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
    for season in (2022,2023,2024,2025):
        raw=_to_pandas(nfl.load_player_stats(seasons=[season],summary_level="week"))
        x=_normalize_weekly(raw,season)
        z=x[["season","player_clean_key","player_id"]].copy()
        z["player_id"]=z["player_id"].map(clean)
        z=z.loc[z["player_clean_key"].astype(str).ne("")&z["player_id"].ne("")].drop_duplicates()
        rows.append(z)
    allx=pd.concat(rows,ignore_index=True)
    g=(allx.groupby(["season","player_clean_key"])["player_id"]
       .agg(lambda s:sorted(set(map(str,s)))).reset_index())
    g["id_count"]=g["player_id"].map(len)
    g["receiver_id"]=g["player_id"].map(lambda v:v[0] if len(v)==1 else "")
    return g[["season","player_clean_key","receiver_id","id_count"]]

def load_pbp():
    import nflreadpy as nfl
    frames=[]
    for season in (2022,2023,2024,2025):
        x=regular_only(lower(nfl.load_pbp(seasons=[season])))
        for c in ["season","week","game_id","receiver_player_id","pass_attempt","sack","two_point_attempt","air_yards"]:
            if c not in x.columns: x[c]=np.nan
        x["season"]=num(x["season"]).fillna(season).astype(int)
        x["week"]=num(x["week"]).astype("Int64")
        x["receiver_id"]=x["receiver_player_id"].map(clean)
        raw=num(x["pass_attempt"]).fillna(0).eq(1)
        sack=num(x["sack"]).fillna(0).eq(1)
        two=num(x["two_point_attempt"]).fillna(0).eq(1)
        x["target_event"]=raw & ~sack & ~two & x["receiver_id"].ne("")
        x["air_yards"]=num(x["air_yards"])
        frames.append(x[["season","week","game_id","receiver_id","target_event","air_yards"]])
    return pd.concat(frames,ignore_index=True,sort=False)

def prepare_wr(path):
    x=pd.read_csv(path,low_memory=False); x.columns=[str(c).lower().strip() for c in x.columns]
    req={"variant","season","week","player_clean_key","pred_targets","mc_rec_yards","actual_targets","actual_rec_yards"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"WR authority missing {sorted(miss)}")
    x=x.loc[x["variant"].astype(str).eq(WR_VARIANT)].copy()
    if len(x)!=WR_EXPECTED_ROWS: raise RuntimeError(f"WR row drift {len(x)}")
    x["season"]=num(x["season"]).astype(int); x["week"]=num(x["week"]).astype(int)
    for c in ["pred_targets","mc_rec_yards","actual_targets","actual_rec_yards"]: x[c]=num(x[c])
    x=x.loc[x["pred_targets"].gt(0)&x[["mc_rec_yards","actual_targets","actual_rec_yards"]].notna().all(axis=1)].copy()
    x["pred_ypt"]=x["mc_rec_yards"]/x["pred_targets"]
    x["actual_ypt"]=np.where(x["actual_targets"].gt(0),x["actual_rec_yards"]/x["actual_targets"],0.0)
    x["abs_efficiency_error"]=(x["actual_targets"]*(x["pred_ypt"]-x["actual_ypt"])).abs()
    x["position_group"]="WR"
    return x[["season","week","player_clean_key","position_group","abs_efficiency_error"]]

def prepare_te(path):
    x=pd.read_csv(path,low_memory=False); x.columns=[str(c).lower().strip() for c in x.columns]
    req={"season","week","player_clean_key","candidate_targets_r5p","candidate_rec_yards_r5p","targets","rec_yards"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"TE authority missing {sorted(miss)}")
    if len(x)!=TE_EXPECTED_ROWS: raise RuntimeError(f"TE row drift {len(x)}")
    x["season"]=num(x["season"]).astype(int); x["week"]=num(x["week"]).astype(int)
    for c in ["candidate_targets_r5p","candidate_rec_yards_r5p","targets","rec_yards"]: x[c]=num(x[c])
    x=x.loc[x["candidate_targets_r5p"].gt(0)&x[["candidate_rec_yards_r5p","targets","rec_yards"]].notna().all(axis=1)].copy()
    x["pred_ypt"]=x["candidate_rec_yards_r5p"]/x["candidate_targets_r5p"]
    x["actual_ypt"]=np.where(x["targets"].gt(0),x["rec_yards"]/x["targets"],0.0)
    x["abs_efficiency_error"]=(x["targets"]*(x["pred_ypt"]-x["actual_ypt"])).abs()
    x["position_group"]="TE"
    return x[["season","week","player_clean_key","position_group","abs_efficiency_error"]]

def attach_identity(authority,ids):
    x=authority.merge(ids,on=["season","player_clean_key"],how="left",validate="many_to_one")
    x["receiver_id"]=x["receiver_id"].fillna("").astype(str)
    x["identity_ok"]=x["receiver_id"].ne("")&num(x["id_count"]).eq(1)
    return x

def before(g,season,week):
    return (g["season"]<season)|((g["season"]==season)&(g["week"]<week))

def build_event_index(pbp):
    t=pbp.loc[pbp["target_event"]&pbp["air_yards"].notna()].copy()
    return {str(pid):g.sort_values(["season","week","game_id"]).copy() for pid,g in t.groupby("receiver_id",sort=False)}

def feature_for(index,pid,season,week):
    g=index.get(str(pid))
    if g is None or g.empty: return None
    h=g.loc[before(g,int(season),int(week))].copy()
    if h.empty: return None
    games=(h[["season","week","game_id"]].drop_duplicates()
           .sort_values(["season","week","game_id"]).tail(8))
    if len(games)<4: return None
    keys=set(zip(games["season"].astype(int),games["week"].astype(int),games["game_id"].astype(str)))
    m=[(int(s),int(w),str(gid)) in keys for s,w,gid in zip(h["season"],h["week"],h["game_id"])]
    q=h.loc[m].copy()
    air=num(q["air_yards"]).dropna().to_numpy(float)
    if len(air)<10: return None
    if int(q["week"].max())>=int(week) and int(q.loc[q["week"].eq(q["week"].max()),"season"].max())==int(season):
        raise RuntimeError("same/future feature row detected")
    return {
      "prior_receiver_games":int(len(games)),
      "prior_finite_air_targets":int(len(air)),
      "prior8_target_depth_sd":float(np.std(air,ddof=0)),
      "prior8_mean_air_yards":float(np.mean(air)),
      "prior8_deep_target_rate":float(np.mean(air>=20.0)),
      "feature_max_season":int(q["season"].max()),
      "feature_max_week":int(q.loc[q["season"].eq(q["season"].max()),"week"].max()),
    }

def build_panel(authority,pbp):
    idx=build_event_index(pbp); rows=[]; violations=0
    cache={}
    for r in authority.loc[authority["identity_ok"]].itertuples(index=False):
        key=(str(r.receiver_id),int(r.season),int(r.week))
        if key not in cache: cache[key]=feature_for(idx,*key)
        f=cache[key]
        if f is None: continue
        if (f["feature_max_season"]>int(r.season)) or (f["feature_max_season"]==int(r.season) and f["feature_max_week"]>=int(r.week)):
            violations+=1
        row={
          "season":int(r.season),"week":int(r.week),"player_clean_key":str(r.player_clean_key),
          "receiver_id":str(r.receiver_id),"position_group":str(r.position_group),
          "abs_efficiency_error":float(r.abs_efficiency_error),
        }
        row.update(f); rows.append(row)
    return pd.DataFrame(rows),violations

def rho(df):
    z=df[["prior8_target_depth_sd","abs_efficiency_error"]].dropna()
    if len(z)<3 or z.iloc[:,0].nunique()<2 or z.iloc[:,1].nunique()<2: return np.nan
    return float(spearmanr(z.iloc[:,0].to_numpy(float),z.iloc[:,1].to_numpy(float)).statistic)

def metrics(df):
    z=df.dropna(subset=["prior8_target_depth_sd","abs_efficiency_error"]).copy()
    d=num(z["prior8_target_depth_sd"])
    return {
      "rows":int(len(z)),"players":int(z["player_clean_key"].nunique()) if len(z) else 0,
      "rho":rho(z),
      "depth_sd_mean":float(d.mean()) if len(d) else np.nan,
      "depth_sd_p10":float(d.quantile(.10)) if len(d) else np.nan,
      "depth_sd_p50":float(d.quantile(.50)) if len(d) else np.nan,
      "depth_sd_p90":float(d.quantile(.90)) if len(d) else np.nan,
      "mean_abs_efficiency_error":float(num(z["abs_efficiency_error"]).mean()) if len(z) else np.nan,
    }

def bootstrap(df,scope):
    z=df[["prior8_target_depth_sd","abs_efficiency_error","player_clean_key","position_group"]].dropna().copy()
    if scope in {"WR","TE"}: z=z.loc[z["position_group"].eq(scope)].copy()
    z["cluster"]=z["position_group"].astype(str)+"|"+z["player_clean_key"].astype(str)
    clusters=sorted(z["cluster"].unique())
    if len(clusters)<2: return {"valid_reps":0,"p_positive":np.nan,"ci_low":np.nan,"ci_high":np.nan}
    groups=[]
    for c in clusters:
        q=z.loc[z["cluster"].eq(c)]
        groups.append((num(q["prior8_target_depth_sd"]).to_numpy(float),num(q["abs_efficiency_error"]).to_numpy(float)))
    rng=np.random.default_rng(BOOT_SEED)
    vals=np.full(BOOT_REPS,np.nan)
    n=len(groups)
    for rep in range(BOOT_REPS):
        ids=rng.integers(0,n,size=n)
        a=np.concatenate([groups[i][0] for i in ids]); b=np.concatenate([groups[i][1] for i in ids])
        if len(a)>=3 and np.unique(a).size>=2 and np.unique(b).size>=2:
            vals[rep]=float(spearmanr(a,b).statistic)
    v=vals[np.isfinite(vals)]
    return {"valid_reps":int(len(v)),"p_positive":float((v>0).mean()) if len(v) else np.nan,
            "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
            "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wr",type=Path,required=True); ap.add_argument("--te",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    ids=identity_map()
    wr=attach_identity(prepare_wr(a.wr),ids); te=attach_identity(prepare_te(a.te),ids)
    idcov={"WR":float(wr["identity_ok"].mean()),"TE":float(te["identity_ok"].mean())}
    authority=pd.concat([wr,te],ignore_index=True)
    panel,viol=build_panel(authority,load_pbp())
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
    support=bool(idcov["WR"]>=.95 and idcov["TE"]>=.95
      and cells["WR_2023"]["rows"]>=500 and cells["WR_2024"]["rows"]>=500 and cells["WR_pooled"]["players"]>=100
      and cells["TE_2024"]["rows"]>=250 and cells["TE_2025"]["rows"]>=250 and cells["TE_pooled"]["players"]>=50)
    confirmed=bool(support
      and cells["WR_2023"]["rho"]>0 and cells["WR_2024"]["rho"]>0
      and cells["TE_2024"]["rho"]>0 and cells["TE_2025"]["rho"]>0
      and cells["WR_pooled"]["rho"]>=.05 and cells["TE_pooled"]["rho"]>=.05 and cells["ALL_pooled"]["rho"]>=.05
      and bw["p_positive"]>=.95 and bt["p_positive"]>=.95 and ba["p_positive"]>=.99 and viol==0)
    disp=("PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED" if confirmed else
          "NO_ACTIONABLE_PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY" if support else
          "PLAYER_TARGET_DEPTH_DISPERSION_SOURCE_LIMITED")
    result={"version":VERSION,"disposition":disp,"identity_mapping_coverage":idcov,
      "panel_rows":int(len(panel)),"panel_players":{"WR":int(panel.loc[panel.position_group.eq("WR"),"player_clean_key"].nunique()),
      "TE":int(panel.loc[panel.position_group.eq("TE"),"player_clean_key"].nunique())},
      "cells":cells,"bootstrap_wr":bw,"bootstrap_te":bt,"bootstrap_combined":ba,
      "gates":{"support":support,"confirmed":confirmed},"same_or_future_feature_violations":int(viol),
      "candidate_models_fit":0,"sportsbook_inputs_used":0,"outcomes_2026_read":0,"production_changed":False}
    panel.to_csv(a.out_dir/"player_target_depth_dispersion_rows.csv",index=False)
    (a.out_dir/"player_target_depth_dispersion_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
