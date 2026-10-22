#!/usr/bin/env python3
"""Frozen Stage-1 incremental market-information test for Market-Relative Selector V1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20260930
BOOT=10000
MIN_ROWS=100
MIN_CLUSTERS=20

def fit_beta(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    den=float(np.dot(x,x))
    if not np.isfinite(den) or den<=1e-12: return np.nan,np.nan
    raw=float(np.dot(x,y)/den)
    return raw,float(np.clip(raw,0.0,1.0))

def consensus(props):
    gcols=["game_id","player_clean_key","market"]
    p=props.copy()
    p["line"]=pd.to_numeric(p["line"],errors="coerce")
    p=p.loc[p["line"].notna()].copy()
    c=(p.groupby(gcols,as_index=False)
         .agg(consensus_line=("line","median"),
              consensus_book_count=("book","nunique")))
    return c

def bootstrap_ci(df):
    by=(df.groupby("game_id",as_index=False)
         .agg(sum_improvement=("paired_improvement","sum"),n=("paired_improvement","size")))
    if len(by)<MIN_CLUSTERS: return np.nan,np.nan
    sums=by.sum_improvement.to_numpy(float); ns=by.n.to_numpy(float)
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(by),size=(BOOT,len(by)))
    vals=sums[idx].sum(axis=1)/ns[idx].sum(axis=1)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def run_direction(g,fit_s,test_s):
    tr=g.loc[g.season.eq(fit_s)].copy(); te=g.loc[g.season.eq(test_s)].copy()
    base={"fit_season":fit_s,"test_season":test_s,"train_rows":len(tr),"test_rows":len(te),
          "train_clusters":tr.game_id.nunique(),"test_clusters":te.game_id.nunique()}
    if len(tr)<MIN_ROWS or len(te)<MIN_ROWS or tr.game_id.nunique()<MIN_CLUSTERS or te.game_id.nunique()<MIN_CLUSTERS:
        return {**base,"status":"INSUFFICIENT_SUPPORT"},pd.DataFrame()
    x=tr.model_projection-tr.consensus_line
    y=tr.actual-tr.consensus_line
    raw,beta=fit_beta(x,y)
    if not np.isfinite(beta):
        return {**base,"status":"DEGENERATE_MODEL_GAP","beta_raw":raw,"beta":beta},pd.DataFrame()
    te["beta"]=beta
    te["model_gap"]=te.model_projection-te.consensus_line
    te["market_relative_fair_line"]=te.consensus_line+beta*te.model_gap
    te["baseline_abs_error"]=(te.consensus_line-te.actual).abs()
    te["candidate_abs_error"]=(te.market_relative_fair_line-te.actual).abs()
    te["paired_improvement"]=te.baseline_abs_error-te.candidate_abs_error
    lo,hi=bootstrap_ci(te)
    bmae=float(te.baseline_abs_error.mean()); cmae=float(te.candidate_abs_error.mean())
    imp=float(te.paired_improvement.mean())
    passed=bool(beta>0 and cmae<bmae and imp>0 and np.isfinite(lo) and lo>0)
    status="DIRECTION_PASS" if passed else "DIRECTION_FAIL"
    return {**base,"status":status,"beta_raw":raw,"beta":beta,
            "baseline_mae":bmae,"candidate_mae":cmae,"mean_paired_improvement":imp,
            "bootstrap_ci_low":lo,"bootstrap_ci_high":hi},te

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--props",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    src=pd.read_csv(a.source,low_memory=False)
    props=pd.read_csv(a.props,low_memory=False)
    for d in (src,props): d.columns=[str(c).strip().lower() for c in d.columns]
    c=consensus(props)
    z=src.merge(c,on=["game_id","player_clean_key","market"],how="inner",validate="one_to_one")
    z["season"]=pd.to_numeric(z.season,errors="raise").astype(int)
    for col in ["model_projection","actual","consensus_line"]: z[col]=pd.to_numeric(z[col],errors="raise")
    a.out_dir.mkdir(parents=True,exist_ok=True)
    z.to_csv(a.out_dir/"stage1_matched_rows.csv",index=False)

    summaries=[]; details=[]
    for market,g in z.groupby("market"):
        dirs=[]
        for fs,ts in [(2024,2025),(2025,2024)]:
            s,d=run_direction(g,fs,ts); s["market"]=market; dirs.append(s); summaries.append(s)
            if not d.empty:
                d=d.copy(); d["fit_season"]=fs; d["test_season"]=ts; details.append(d)
        ok=len(dirs)==2 and all(x.get("status")=="DIRECTION_PASS" for x in dirs)
        summaries.append({"market":market,"fit_season":"BOTH","test_season":"BOTH",
                          "status":"STAGE1_INCREMENTAL_LEVEL_SIGNAL_PASS" if ok else "NO_VERIFIED_INCREMENTAL_MODEL_LEVEL_SIGNAL_V1"})
    summary=pd.DataFrame(summaries)
    summary.to_csv(a.out_dir/"stage1_summary.csv",index=False)
    if details: pd.concat(details,ignore_index=True).to_csv(a.out_dir/"stage1_detail.csv",index=False)
    result={}
    for market in sorted(z.market.unique()):
        row=summary.loc[(summary.market.eq(market)) & summary.fit_season.astype(str).eq("BOTH")].iloc[0]
        result[market]=row.status
    (a.out_dir/"stage1_disposition.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
