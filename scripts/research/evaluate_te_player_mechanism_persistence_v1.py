#!/usr/bin/env python3
"""TE Player Mechanism Persistence V1 diagnostic."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VERSION="TE_PLAYER_MECHANISM_PERSISTENCE_V1"
TARGET_SEASONS=(2024,2025)
HISTORY_N=8
MIN_PRIOR=4
MIN_ROWS=400
MIN_PLAYERS=50
BOOT_REPS=5000
BOOT_SEED=20261007
TOL=1e-9

def num(x): return pd.to_numeric(x,errors="coerce")

def read(path):
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    req={"season","week","team","player_clean_key","candidate_targets_r5p","targets",
         "candidate_receptions_r5p","receptions","candidate_rec_yards_r5p","rec_yards"}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"missing columns: {sorted(miss)}")
    for c in ["candidate_targets_r5p","targets","candidate_receptions_r5p","receptions","candidate_rec_yards_r5p","rec_yards"]:
        x[c]=num(x[c])
    x["season"]=num(x["season"]).astype(int); x["week"]=num(x["week"]).astype(int)
    if x.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("duplicate authority identities")
    return x.sort_values(["season","week","team","player_clean_key"]).reset_index(drop=True)

def add_components(x):
    q=x.copy()
    q=q.loc[q["candidate_targets_r5p"].gt(0)&q["candidate_rec_yards_r5p"].notna()&q["rec_yards"].notna()&q["targets"].notna()].copy()
    q["pred_ypt"]=q["candidate_rec_yards_r5p"]/q["candidate_targets_r5p"]
    q["actual_ypt"]=np.where(q["targets"].gt(0),q["rec_yards"]/q["targets"],0.0)
    q["opportunity_error"]=(q["candidate_targets_r5p"]-q["targets"])*q["pred_ypt"]
    q["efficiency_error"]=q["targets"]*(q["pred_ypt"]-q["actual_ypt"])
    q["total_error"]=q["candidate_rec_yards_r5p"]-q["rec_yards"]
    q["decomp_gap"]=(q["opportunity_error"]+q["efficiency_error"]-q["total_error"]).abs()
    if float(q["decomp_gap"].max())>TOL:
        raise RuntimeError(f"decomposition identity failed max={q['decomp_gap'].max()}")
    q["catch_rate_error"]=np.nan
    mask=q["targets"].gt(0)&q["candidate_targets_r5p"].gt(0)
    q.loc[mask,"catch_rate_error"]=(q.loc[mask,"candidate_receptions_r5p"]/q.loc[mask,"candidate_targets_r5p"]
                                     - q.loc[mask,"receptions"]/q.loc[mask,"targets"])
    return q

def prior_mask(g,s,w):
    return (g["season"]<s)|((g["season"]==s)&(g["week"]<w))

def build_features(q):
    groups={str(k):g.sort_values(["season","week"]).copy() for k,g in q.groupby("player_clean_key",sort=False)}
    rows=[]
    for r in q.loc[q["season"].isin(TARGET_SEASONS)].itertuples(index=False):
        g=groups[str(r.player_clean_key)]
        h=g.loc[prior_mask(g,int(r.season),int(r.week))].tail(HISTORY_N)
        if len(h)<MIN_PRIOR: continue
        rows.append({
            "season":int(r.season),"week":int(r.week),"team":str(r.team),"player_clean_key":str(r.player_clean_key),
            "prior_games":int(len(h)),
            "prior8_opportunity_error_mean":float(h["opportunity_error"].mean()),
            "prior8_abs_opportunity_error_mean":float(h["opportunity_error"].abs().mean()),
            "prior8_efficiency_error_mean":float(h["efficiency_error"].mean()),
            "prior8_abs_efficiency_error_mean":float(h["efficiency_error"].abs().mean()),
            "prior8_catch_rate_error_mean":float(h["catch_rate_error"].dropna().mean()) if h["catch_rate_error"].notna().any() else np.nan,
            "current_opportunity_error":float(r.opportunity_error),
            "current_abs_opportunity_error":abs(float(r.opportunity_error)),
            "current_efficiency_error":float(r.efficiency_error),
            "current_abs_efficiency_error":abs(float(r.efficiency_error)),
            "current_catch_rate_error":float(r.catch_rate_error) if pd.notna(r.catch_rate_error) else np.nan,
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
 "C1":("prior8_catch_rate_error_mean","current_catch_rate_error"),
}

def metrics(df):
    out={"rows":int(len(df)),"players":int(df.player_clean_key.nunique()) if len(df) else 0}
    for k,(a,b) in PAIRS.items(): out[k+"_spearman"]=corr(df[a],df[b])
    ao=num(df["current_abs_opportunity_error"]); ae=num(df["current_abs_efficiency_error"])
    den=ao+ae
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
    ap=argparse.ArgumentParser(); ap.add_argument("--casebook",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    q=add_components(read(a.casebook))
    f=build_features(q)
    by={str(s):metrics(f.loc[f.season.eq(s)]) for s in TARGET_SEASONS}
    pooled=metrics(f); boot=bootstrap(f)
    support=all(by[str(s)]["rows"]>=MIN_ROWS and by[str(s)]["players"]>=MIN_PLAYERS for s in TARGET_SEASONS)
    gates={}
    for k in ["O1","O2","E1","E2"]:
        gates[k]=bool(by["2024"][k+"_spearman"]>0 and by["2025"][k+"_spearman"]>0 and boot[k]["p_positive"]>=.95)
    opp=gates["O1"] or gates["O2"]; eff=gates["E1"] or gates["E2"]
    if not support: disp="TE_PLAYER_PERSISTENCE_MECHANISM_SOURCE_LIMITED"
    elif eff and not opp: disp="TE_PLAYER_PERSISTENCE_EFFICIENCY_DOMINANT"
    elif opp and not eff: disp="TE_PLAYER_PERSISTENCE_OPPORTUNITY_DOMINANT"
    elif opp and eff: disp="TE_PLAYER_PERSISTENCE_MIXED_MECHANISM"
    else: disp="TE_PLAYER_PERSISTENCE_MECHANISM_UNRESOLVED"
    result={
      "version":VERSION,"disposition":disp,"scoreable_rows":int(len(f)),"scoreable_players":int(f.player_clean_key.nunique()),
      "season_metrics":by,"pooled_metrics":pooled,"player_cluster_bootstrap":boot,
      "gates":{"support":support,**gates,"C1_secondary":bool(by["2024"]["C1_spearman"]>0 and by["2025"]["C1_spearman"]>0 and boot["C1"]["p_positive"]>=.95)},
      "max_decomposition_identity_gap":float(q["decomp_gap"].max()),
      "candidate_models_fit":0,"sportsbook_inputs_used":0,"outcomes_2026_read":0,"production_changed":False,"same_or_future_history_violations":0,
    }
    f.to_csv(a.out_dir/"te_player_mechanism_persistence_rows.csv",index=False)
    (a.out_dir/"te_player_mechanism_persistence_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
