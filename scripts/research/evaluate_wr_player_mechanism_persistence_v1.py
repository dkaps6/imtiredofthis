#!/usr/bin/env python3
"""WR Player Mechanism Persistence V1 diagnostic."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VERSION="WR_PLAYER_MECHANISM_PERSISTENCE_V1"
TARGET_SEASONS=(2023,2024)
HISTORY_N=8
MIN_PRIOR=4
MIN_ROWS=800
MIN_PLAYERS=100
BOOT_REPS=5000
BOOT_SEED=20261007
TOL=1e-9

EXPECTED_ROWS=4193
EXPECTED_SEASONS={2023:2076,2024:2117}
EXPECTED_MAE=22.52635823935656
EXPECTED_RMSE=31.988194283296206
EXPECTED_TARGET_MAE=2.0438856296229853
VARIANT="WR_R15_WR1_ANCHORED_PARTICIPATION"

def num(x): return pd.to_numeric(x,errors="coerce")

def read(path:Path)->pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={"variant","season","week","team","player_clean_key","wr_rank",
         "pred_targets","mc_rec_yards","actual_targets","actual_rec_yards"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"missing authority columns: {sorted(miss)}")
    x=x.loc[x["variant"].astype(str).eq(VARIANT)].copy()
    for c in ["pred_targets","mc_rec_yards","actual_targets","actual_rec_yards","wr_rank"]:
        x[c]=num(x[c])
    x["season"]=num(x["season"]).astype(int)
    x["week"]=num(x["week"]).astype(int)
    if len(x)!=EXPECTED_ROWS: raise RuntimeError(f"row drift {len(x)} != {EXPECTED_ROWS}")
    counts={int(k):int(v) for k,v in x.groupby("season").size().to_dict().items()}
    if counts!=EXPECTED_SEASONS: raise RuntimeError(f"season drift {counts} != {EXPECTED_SEASONS}")
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("duplicate authority rows")
    err=x["mc_rec_yards"]-x["actual_rec_yards"]
    mae=float(err.abs().mean()); rmse=float(np.sqrt(np.mean(np.square(err))))
    tmae=float((x["pred_targets"]-x["actual_targets"]).abs().mean())
    if abs(mae-EXPECTED_MAE)>TOL or abs(rmse-EXPECTED_RMSE)>TOL or abs(tmae-EXPECTED_TARGET_MAE)>TOL:
        raise RuntimeError(f"authority metric drift mae={mae} rmse={rmse} target_mae={tmae}")
    if set(x["season"].unique())!={2023,2024}: raise RuntimeError("unexpected confirmation seasons")
    return x.sort_values(["season","week","team","player_clean_key"]).reset_index(drop=True)

def add_components(x):
    q=x.loc[
        x["pred_targets"].gt(0)
        & x["mc_rec_yards"].notna()
        & x["actual_targets"].notna()
        & x["actual_rec_yards"].notna()
    ].copy()
    q["pred_ypt"]=q["mc_rec_yards"]/q["pred_targets"]
    q["actual_ypt"]=np.where(q["actual_targets"].gt(0),q["actual_rec_yards"]/q["actual_targets"],0.0)
    q["opportunity_error"]=(q["pred_targets"]-q["actual_targets"])*q["pred_ypt"]
    q["efficiency_error"]=q["actual_targets"]*(q["pred_ypt"]-q["actual_ypt"])
    q["total_error"]=q["mc_rec_yards"]-q["actual_rec_yards"]
    q["decomp_gap"]=(q["opportunity_error"]+q["efficiency_error"]-q["total_error"]).abs()
    if q.empty: raise RuntimeError("zero decomposition rows")
    if float(q["decomp_gap"].max())>TOL:
        raise RuntimeError(f"decomposition identity failure {q['decomp_gap'].max()}")
    return q

def prior_mask(g,s,w):
    return (g["season"]<s)|((g["season"]==s)&(g["week"]<w))

def build_features(q):
    groups={str(k):g.sort_values(["season","week"]).copy() for k,g in q.groupby("player_clean_key",sort=False)}
    rows=[]
    for r in q.itertuples(index=False):
        g=groups[str(r.player_clean_key)]
        h=g.loc[prior_mask(g,int(r.season),int(r.week))].tail(HISTORY_N)
        if len(h)<MIN_PRIOR: continue
        rows.append({
          "season":int(r.season),"week":int(r.week),"team":str(r.team),
          "player_clean_key":str(r.player_clean_key),"wr_rank":int(r.wr_rank) if pd.notna(r.wr_rank) else 99,
          "prior_games":int(len(h)),
          "prior8_opportunity_error_mean":float(h["opportunity_error"].mean()),
          "prior8_abs_opportunity_error_mean":float(h["opportunity_error"].abs().mean()),
          "prior8_efficiency_error_mean":float(h["efficiency_error"].mean()),
          "prior8_abs_efficiency_error_mean":float(h["efficiency_error"].abs().mean()),
          "current_opportunity_error":float(r.opportunity_error),
          "current_abs_opportunity_error":abs(float(r.opportunity_error)),
          "current_efficiency_error":float(r.efficiency_error),
          "current_abs_efficiency_error":abs(float(r.efficiency_error)),
          "current_total_error":float(r.total_error),
          "current_abs_total_error":abs(float(r.total_error)),
        })
    return pd.DataFrame(rows)

def corr(a,b):
    z=pd.DataFrame({"a":num(a),"b":num(b)}).dropna()
    if len(z)<3 or z.a.nunique()<2 or z.b.nunique()<2: return np.nan
    return float(spearmanr(z.a.to_numpy(float),z.b.to_numpy(float)).statistic)

PAIRS={
 "O1":("prior8_opportunity_error_mean","current_opportunity_error"),
 "O2":("prior8_abs_opportunity_error_mean","current_abs_opportunity_error"),
 "E1":("prior8_efficiency_error_mean","current_efficiency_error"),
 "E2":("prior8_abs_efficiency_error_mean","current_abs_efficiency_error"),
}

def metrics(df):
    out={"rows":int(len(df)),"players":int(df.player_clean_key.nunique()) if len(df) else 0}
    for k,(a,b) in PAIRS.items(): out[k+"_spearman"]=corr(df[a],df[b])
    ao=num(df["current_abs_opportunity_error"]); ae=num(df["current_abs_efficiency_error"]); den=ao+ae
    valid=den.gt(0)
    out["mean_abs_total_error"]=float(num(df["current_abs_total_error"]).mean()) if len(df) else np.nan
    out["mean_abs_opportunity_component"]=float(ao.mean()) if len(df) else np.nan
    out["mean_abs_efficiency_component"]=float(ae.mean()) if len(df) else np.nan
    out["mean_normalized_opportunity_mass"]=float((ao[valid]/den[valid]).mean()) if valid.any() else np.nan
    out["mean_normalized_efficiency_mass"]=float((ae[valid]/den[valid]).mean()) if valid.any() else np.nan
    opp=num(df["current_opportunity_error"]); eff=num(df["current_efficiency_error"])
    out["component_sign_cancellation_rate"]=float(((np.sign(opp)!=np.sign(eff))&opp.ne(0)&eff.ne(0)).mean()) if len(df) else np.nan
    return out

def bootstrap(df):
    players=sorted(df.player_clean_key.astype(str).unique())
    groups={p:df.loc[df.player_clean_key.astype(str).eq(p)] for p in players}
    rng=np.random.default_rng(BOOT_SEED)
    vals={k:[] for k in PAIRS}
    for _ in range(BOOT_REPS):
        sample=rng.choice(players,size=len(players),replace=True)
        z=pd.concat([groups[p] for p in sample],ignore_index=True)
        for k,(a,b) in PAIRS.items(): vals[k].append(corr(z[a],z[b]))
    out={}
    for k,v in vals.items():
        a=np.asarray(v,float); a=a[np.isfinite(a)]
        out[k]={
          "valid_reps":int(len(a)),
          "p_positive":float((a>0).mean()) if len(a) else np.nan,
          "ci_low":float(np.quantile(a,.025)) if len(a) else np.nan,
          "ci_high":float(np.quantile(a,.975)) if len(a) else np.nan,
        }
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    q=add_components(read(a.predictions))
    f=build_features(q)
    by={str(s):metrics(f.loc[f.season.eq(s)]) for s in TARGET_SEASONS}
    pooled=metrics(f); boot=bootstrap(f)
    wr1=metrics(f.loc[f["wr_rank"].eq(1)])
    wr2p=metrics(f.loc[f["wr_rank"].ge(2)])
    support=all(by[str(s)]["rows"]>=MIN_ROWS and by[str(s)]["players"]>=MIN_PLAYERS for s in TARGET_SEASONS)
    gates={}
    for k in PAIRS:
        gates[k]=bool(by["2023"][k+"_spearman"]>0 and by["2024"][k+"_spearman"]>0 and boot[k]["p_positive"]>=.95)
    opp=gates["O1"] or gates["O2"]; eff=gates["E1"] or gates["E2"]
    if not support: disp="WR_PLAYER_PERSISTENCE_MECHANISM_SOURCE_LIMITED"
    elif eff and not opp: disp="WR_PLAYER_PERSISTENCE_EFFICIENCY_DOMINANT"
    elif opp and not eff: disp="WR_PLAYER_PERSISTENCE_OPPORTUNITY_DOMINANT"
    elif opp and eff: disp="WR_PLAYER_PERSISTENCE_MIXED_MECHANISM"
    else: disp="WR_PLAYER_PERSISTENCE_MECHANISM_UNRESOLVED"
    result={
      "version":VERSION,"disposition":disp,
      "parent_authority":{"run":34238301577,"artifact":10061328722,"digest":"sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce","rows":EXPECTED_ROWS},
      "scoreable_rows":int(len(f)),"scoreable_players":int(f.player_clean_key.nunique()),
      "season_metrics":by,"pooled_metrics":pooled,"wr1_diagnostic":wr1,"wr2plus_diagnostic":wr2p,
      "player_cluster_bootstrap":boot,"gates":{"support":support,**gates},
      "max_decomposition_identity_gap":float(q["decomp_gap"].max()),
      "candidate_models_fit":0,"sportsbook_inputs_used":0,"outcomes_2026_read":0,"production_changed":False,"same_or_future_history_violations":0,
    }
    f.to_csv(a.out_dir/"wr_player_mechanism_persistence_rows.csv",index=False)
    (a.out_dir/"wr_player_mechanism_persistence_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
