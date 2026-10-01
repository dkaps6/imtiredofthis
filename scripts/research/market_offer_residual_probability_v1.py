#!/usr/bin/env python3
"""Frozen Market Offer Residual Probability V1 execution."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES=[
    "model_gap",
    "abs_model_gap",
    "line_vs_crossbook_median",
    "line_range",
    "novig_over_centered",
]
SEED=20261001
BOOT=10000
MIN_IDENTITIES=100
MIN_CLUSTERS=20

def implied_prob(o):
    try: x=float(o)
    except Exception: return np.nan
    if not np.isfinite(x) or x==0: return np.nan
    return 100.0/(x+100.0) if x>0 else (-x)/((-x)+100.0)

def prepare_offers(source: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    s=source.copy(); p=props.copy()
    for d in (s,p): d.columns=[str(c).strip().lower() for c in d.columns]
    need={"game_id","player_clean_key","market","book","line","over_odds","under_odds"}
    miss=sorted(need-set(p.columns))
    if miss: raise RuntimeError(f"props missing {miss}")
    key=["game_id","player_clean_key","market","book","line"]
    p=p.drop_duplicates(key).copy()
    if p.duplicated(key).any(): raise RuntimeError("duplicate offer identities")
    for c in ["line","over_odds","under_odds"]: p[c]=pd.to_numeric(p[c],errors="coerce")
    p=p.loc[p[["line","over_odds","under_odds"]].notna().all(axis=1)].copy()
    p["p_over_raw"]=p.over_odds.map(implied_prob)
    p["p_under_raw"]=p.under_odds.map(implied_prob)
    den=p.p_over_raw+p.p_under_raw
    p["book_novig_over"]=p.p_over_raw/den
    p=p.loc[p.book_novig_over.notna() & p.book_novig_over.between(0,1)].copy()

    g=["game_id","player_clean_key","market"]
    stats=(p.groupby(g,as_index=False)
             .agg(crossbook_median_line=("line","median"),
                  line_min=("line","min"),
                  line_max=("line","max"),
                  offer_count=("book","nunique")))
    p=p.merge(stats,on=g,how="left",validate="many_to_one")
    p["line_range"]=p.line_max-p.line_min
    z=p.merge(s,on=g,how="inner",validate="many_to_one")
    for c in ["model_projection","actual"]: z[c]=pd.to_numeric(z[c],errors="coerce")
    z=z.loc[z[["model_projection","actual","line"]].notna().all(axis=1)].copy()
    z=z.loc[~np.isclose(z.actual,z.line,rtol=0,atol=1e-12)].copy()
    z["y_over"]=(z.actual>z.line).astype(int)
    z["model_gap"]=z.model_projection-z.line
    z["abs_model_gap"]=z.model_gap.abs()
    z["line_vs_crossbook_median"]=z.line-z.crossbook_median_line
    z["novig_over_centered"]=z.book_novig_over-0.5
    z["sample_weight"]=1.0/z.offer_count.clip(lower=1).astype(float)
    z["identity_key"]=z.game_id.astype(str)+"|"+z.player_clean_key.astype(str)+"|"+z.market.astype(str)
    if not np.isfinite(z[FEATURES+["sample_weight","book_novig_over"]].to_numpy(float)).all():
        raise RuntimeError("non-finite model features")
    return z

def weighted_brier(y,p,w):
    return float(np.average((np.asarray(p)-np.asarray(y))**2,weights=np.asarray(w)))

def weighted_logloss(y,p,w):
    return float(log_loss(y,np.clip(np.asarray(p,float),1e-6,1-1e-6),sample_weight=w,labels=[0,1]))

def weighted_auc(y,p,w):
    if len(set(np.asarray(y,int).tolist()))<2: return np.nan
    return float(roc_auc_score(y,p,sample_weight=w))

def fit_predict(train,test):
    scaler=StandardScaler()
    Xtr=scaler.fit_transform(train[FEATURES].to_numpy(float))
    Xte=scaler.transform(test[FEATURES].to_numpy(float))
    model=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000)
    model.fit(Xtr,train.y_over.to_numpy(int),sample_weight=train.sample_weight.to_numpy(float))
    return model.predict_proba(Xte)[:,1],model,scaler

def bootstrap_logloss_improvement(test):
    p0=np.clip(test.book_novig_over.to_numpy(float),1e-6,1-1e-6)
    p1=np.clip(test.candidate_p_over.to_numpy(float),1e-6,1-1e-6)
    y=test.y_over.to_numpy(int); w=test.sample_weight.to_numpy(float)
    row_imp=-(y*np.log(p0)+(1-y)*np.log(1-p0)) - (-(y*np.log(p1)+(1-y)*np.log(1-p1)))
    q=test[["game_id"]].copy()
    q["weighted_imp"]=row_imp*w; q["w"]=w
    by=q.groupby("game_id",as_index=False).agg(weighted_imp=("weighted_imp","sum"),w=("w","sum"))
    if len(by)<MIN_CLUSTERS: return np.nan,np.nan
    sums=by.weighted_imp.to_numpy(float); ws=by.w.to_numpy(float)
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(by),size=(BOOT,len(by)))
    vals=sums[idx].sum(axis=1)/ws[idx].sum(axis=1)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def run_direction(g,fit_s,test_s):
    tr=g.loc[g.season.eq(fit_s)].copy(); te=g.loc[g.season.eq(test_s)].copy()
    base={
      "fit_season":fit_s,"test_season":test_s,
      "train_offers":len(tr),"test_offers":len(te),
      "train_identities":tr.identity_key.nunique(),"test_identities":te.identity_key.nunique(),
      "train_clusters":tr.game_id.nunique(),"test_clusters":te.game_id.nunique(),
    }
    if tr.identity_key.nunique()<MIN_IDENTITIES or te.identity_key.nunique()<MIN_IDENTITIES or tr.game_id.nunique()<MIN_CLUSTERS or te.game_id.nunique()<MIN_CLUSTERS:
        return {**base,"status":"INSUFFICIENT_SUPPORT"},pd.DataFrame()
    if tr.y_over.nunique()<2 or te.y_over.nunique()<2:
        return {**base,"status":"DEGENERATE_TARGET"},pd.DataFrame()
    p,model,scaler=fit_predict(tr,te)
    te["candidate_p_over"]=p
    y=te.y_over.to_numpy(int); w=te.sample_weight.to_numpy(float); p0=te.book_novig_over.to_numpy(float)
    bll=weighted_logloss(y,p0,w); cll=weighted_logloss(y,p,w)
    bb=weighted_brier(y,p0,w); cb=weighted_brier(y,p,w)
    ba=weighted_auc(y,p0,w); ca=weighted_auc(y,p,w)
    lo,hi=bootstrap_logloss_improvement(te)
    imp=bll-cll
    passed=bool(cll<bll and cb<bb and np.isfinite(ca) and np.isfinite(ba) and ca>=ba and imp>0 and np.isfinite(lo) and lo>0)
    return {
      **base,"status":"DIRECTION_PASS" if passed else "DIRECTION_FAIL",
      "baseline_logloss":bll,"candidate_logloss":cll,"logloss_improvement":imp,
      "baseline_brier":bb,"candidate_brier":cb,
      "baseline_auc":ba,"candidate_auc":ca,
      "bootstrap_ci_low":lo,"bootstrap_ci_high":hi,
      "coef_model_gap":float(model.coef_[0][0]),
      "coef_abs_model_gap":float(model.coef_[0][1]),
      "coef_line_vs_crossbook_median":float(model.coef_[0][2]),
      "coef_line_range":float(model.coef_[0][3]),
      "coef_novig_over_centered":float(model.coef_[0][4]),
      "intercept":float(model.intercept_[0]),
    },te

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--props",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    src=pd.read_csv(a.source,low_memory=False)
    props=pd.read_csv(a.props,low_memory=False)
    z=prepare_offers(src,props)
    z["season"]=pd.to_numeric(z.season,errors="raise").astype(int)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    z.to_csv(a.out_dir/"offer_population.csv",index=False)
    summaries=[]; details=[]
    for market,g in z.groupby("market"):
        ds=[]
        for fs,ts in [(2024,2025),(2025,2024)]:
            s,d=run_direction(g,fs,ts); s["market"]=market; summaries.append(s); ds.append(s)
            if not d.empty:
                d=d.copy(); d["fit_season"]=fs; d["test_season"]=ts; details.append(d)
        ok=len(ds)==2 and all(x.get("status")=="DIRECTION_PASS" for x in ds)
        summaries.append({"market":market,"fit_season":"BOTH","test_season":"BOTH",
                          "status":"MARKET_OFFER_RESIDUAL_PROBABILITY_V1_PASS" if ok else "NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1"})
    out=pd.DataFrame(summaries)
    out.to_csv(a.out_dir/"summary.csv",index=False)
    if details: pd.concat(details,ignore_index=True).to_csv(a.out_dir/"detail.csv",index=False)
    disp={m:out.loc[(out.market.eq(m)) & out.fit_season.astype(str).eq("BOTH"),"status"].iloc[0] for m in sorted(z.market.unique())}
    (a.out_dir/"disposition.json").write_text(json.dumps(disp,indent=2,sort_keys=True)+"\n")
    print(out.to_string(index=False))
    print(json.dumps(disp,sort_keys=True))

if __name__=="__main__":
    main()
