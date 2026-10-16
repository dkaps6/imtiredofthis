#!/usr/bin/env python3
"""Q90 upper-tail hinge V1 development evaluator.

Uses exact parent historical football distributions. No sportsbook input and no
mean change. 2024/2025 are development-only because they established the parent
right-tail signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.grade_empirical_fair_prob_v1 import KEYS, _load_metadata, rescale_outcomes

VERSION="DISTRIBUTION_Q90_UPPER_TAIL_HINGE_V1"
MARKETS=("rush_yards","rec_yards","receptions","rush_rec_yards")
SEASONS=(2024,2025)
EXPECTED_DRAWS=2000
MIN_ROWS=200
MIN_GAMES=50
BOOT_REPS=5000
BOOT_SEED=20261013


def read(path: Path,label: str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def q90_hinge(arr: np.ndarray, mean: float)->np.ndarray:
    x=np.asarray(arr,dtype=float)
    if len(x)==0 or not np.isfinite(x).all():
        raise RuntimeError("invalid baseline draw array")
    if not np.isfinite(mean) or mean<0:
        raise RuntimeError(f"invalid football mean {mean}")
    baseline=rescale_outcomes(x,float(mean))
    if mean==0:
        return baseline.copy()
    q90=float(np.quantile(baseline,.90))
    pre=baseline+np.maximum(baseline-q90,0.0)
    pm=float(np.mean(pre))
    if not np.isfinite(pm) or pm<=0:
        raise RuntimeError("candidate pre-align mean invalid")
    candidate=pre*(float(mean)/pm)
    if not np.isfinite(candidate).all() or (candidate<0).any():
        raise RuntimeError("candidate draw invalid")
    if abs(float(np.mean(candidate))-float(mean))>1e-10:
        raise RuntimeError("candidate mean invariant failed")
    return candidate


def empirical_crps(draws: np.ndarray, actual: float)->float:
    x=np.sort(np.asarray(draws,dtype=float))
    n=len(x)
    if n==0 or not np.isfinite(x).all() or not np.isfinite(actual):
        return np.nan
    first=float(np.mean(np.abs(x-float(actual))))
    # 0.5 * E|X-X'| for the empirical distribution.
    i=np.arange(1,n+1,dtype=float)
    second=float(np.sum((2.0*i-n-1.0)*x)/(n*n))
    return first-second


def cluster_ci(q: pd.DataFrame,seed: int)->dict:
    z=q.reset_index(drop=True)
    groups=[g.index.to_numpy(int) for _,g in z.groupby("game_id",sort=True)]
    if len(groups)<2:
        return {"valid_reps":0,"ci_low":None,"ci_high":None,"p_gt_0":None}
    rng=np.random.default_rng(seed)
    vals=np.full(BOOT_REPS,np.nan)
    n=len(groups)
    for rep in range(BOOT_REPS):
        pick=rng.integers(0,n,size=n)
        idx=np.concatenate([groups[j] for j in pick])
        vals[rep]=float(z.iloc[idx]["crps_gain"].mean())
    vals=vals[np.isfinite(vals)]
    return {
        "valid_reps":int(len(vals)),
        "ci_low":float(np.quantile(vals,.025)) if len(vals) else None,
        "ci_high":float(np.quantile(vals,.975)) if len(vals) else None,
        "p_gt_0":float((vals>0).mean()) if len(vals) else None,
    }


def build_detail(proj: pd.DataFrame,dist_dir: Path)->pd.DataFrame:
    required=set(KEYS+["actual","ensemble_proj","mc_proj","game_id"])
    miss=sorted(required-set(proj.columns))
    if miss:
        raise RuntimeError(f"projection trace missing columns {miss}")
    x=proj.copy()
    x["market"]=x["market"].astype(str).str.lower()
    x["season"]=pd.to_numeric(x["season"],errors="raise").astype(int)
    x["week"]=pd.to_numeric(x["week"],errors="raise").astype(int)
    for c in ["actual","ensemble_proj","mc_proj"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.loc[
        x["season"].isin(SEASONS)
        & x["market"].isin(MARKETS)
        & x["actual"].notna()
        & x["ensemble_proj"].notna()
        & x["mc_proj"].notna()
    ].copy()
    if x.empty:
        raise RuntimeError("zero scoreable candidate rows")
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate projection identities")

    meta=_load_metadata(dist_dir)
    z=x.merge(
        meta[KEYS+["array_key","npz_file","draws","mc_mean"]],
        on=KEYS,how="left",validate="one_to_one"
    )
    if z["array_key"].isna().any():
        sample=z.loc[z["array_key"].isna(),KEYS].head(10).to_dict("records")
        raise RuntimeError(f"missing distribution lineage {sample}")

    cache={}
    rows=[]
    for r in z.itertuples(index=False):
        if int(r.draws)!=EXPECTED_DRAWS:
            raise RuntimeError(f"draw count drift {r.season} W{r.week} {r.market}")
        fname=str(r.npz_file)
        if Path(fname).name!=fname:
            raise RuntimeError(f"unsafe shard path {fname}")
        if fname not in cache:
            cache[fname]=np.load(dist_dir/fname,allow_pickle=False)
        raw=np.asarray(cache[fname][str(r.array_key)],dtype=float)
        if len(raw)!=EXPECTED_DRAWS or not np.isfinite(raw).all():
            raise RuntimeError("invalid raw distribution")
        if abs(float(np.mean(raw))-float(r.mc_proj))>1e-8:
            raise RuntimeError("parent MC mean lineage mismatch")
        mean=float(r.ensemble_proj)
        base=rescale_outcomes(raw,mean)
        cand=q90_hinge(raw,mean)
        if abs(float(np.mean(base))-mean)>1e-8:
            raise RuntimeError("baseline mean alignment failed")
        mean_shift=float(np.mean(cand)-mean)

        b05,b10,b90,b95=np.quantile(base,[.05,.10,.90,.95])
        c05,c10,c90,c95=np.quantile(cand,[.05,.10,.90,.95])
        y=float(r.actual)
        rows.append({
            "season":int(r.season),"week":int(r.week),"game_id":str(r.game_id),
            "team":str(r.team),"opponent":str(r.opponent),
            "player_clean_key":str(r.player_clean_key),"market":str(r.market),
            "actual":y,"final_mean":mean,
            "baseline_crps":empirical_crps(base,y),
            "candidate_crps":empirical_crps(cand,y),
            "crps_gain":empirical_crps(base,y)-empirical_crps(cand,y),
            "baseline_sd":float(np.std(base,ddof=1)),
            "candidate_sd":float(np.std(cand,ddof=1)),
            "mean_shift":mean_shift,
            "baseline_q05":float(b05),"baseline_q10":float(b10),
            "baseline_q90":float(b90),"baseline_q95":float(b95),
            "candidate_q05":float(c05),"candidate_q10":float(c10),
            "candidate_q90":float(c90),"candidate_q95":float(c95),
            "baseline_upper_q90":int(y>b90),"candidate_upper_q90":int(y>c90),
            "baseline_lower_q10":int(y<b10),"candidate_lower_q10":int(y<c10),
            "baseline_upper_q95":int(y>b95),"candidate_upper_q95":int(y>c95),
            "baseline_lower_q05":int(y<b05),"candidate_lower_q05":int(y<c05),
            "baseline_cov80":int(b10<=y<=b90),"candidate_cov80":int(c10<=y<=c90),
            "baseline_cov90":int(b05<=y<=b95),"candidate_cov90":int(c05<=y<=c95),
            "baseline_width80":float(b90-b10),"candidate_width80":float(c90-c10),
            "baseline_width90":float(b95-b05),"candidate_width90":float(c95-c05),
        })
    return pd.DataFrame(rows)


def compare_parent(detail: pd.DataFrame,parent: pd.DataFrame)->dict:
    req={"season","week","team","opponent","player_clean_key","market","final_mean","q10","q90"}
    miss=sorted(req-set(parent.columns))
    if miss:
        raise RuntimeError(f"parent detail missing {miss}")
    p=parent.copy()
    p["market"]=p["market"].astype(str).str.lower()
    p=p.loc[p["market"].isin(MARKETS)&pd.to_numeric(p["season"],errors="coerce").isin(SEASONS)].copy()
    keys=["season","week","team","opponent","player_clean_key","market"]
    a=detail[keys+["final_mean","baseline_q10","baseline_q90"]].merge(
        p[keys+["final_mean","q10","q90"]],
        on=keys,how="outer",indicator=True,suffixes=("_regen","_parent"),validate="one_to_one"
    )
    if not a["_merge"].eq("both").all():
        raise RuntimeError(f"parent row identity drift counts={a['_merge'].value_counts().to_dict()}")
    gaps={
        "final_mean":float((a["final_mean_regen"]-a["final_mean_parent"]).abs().max()),
        "q10":float((a["baseline_q10"]-a["q10"]).abs().max()),
        "q90":float((a["baseline_q90"]-a["q90"]).abs().max()),
    }
    if any((not np.isfinite(v)) or v>1e-8 for v in gaps.values()):
        raise RuntimeError(f"parent numerical parity failed {gaps}")
    return {"rows":int(len(a)),"max_abs_gaps":gaps}


def summarize(detail: pd.DataFrame)->tuple[pd.DataFrame,dict]:
    cells=[]
    per_market=[]
    for market in MARKETS:
        season_pass={}
        for season in SEASONS:
            q=detail.loc[detail["market"].eq(market)&detail["season"].eq(season)].copy()
            support=len(q)>=MIN_ROWS and q["game_id"].nunique()>=MIN_GAMES
            boot=cluster_ci(q,BOOT_SEED+season+MARKETS.index(market)*100) if support else {}
            base_upper=float(q["baseline_upper_q90"].mean()) if len(q) else np.nan
            cand_upper=float(q["candidate_upper_q90"].mean()) if len(q) else np.nan
            base80=float(q["baseline_cov80"].mean()) if len(q) else np.nan
            cand80=float(q["candidate_cov80"].mean()) if len(q) else np.nan
            base90=float(q["baseline_cov90"].mean()) if len(q) else np.nan
            cand90=float(q["candidate_cov90"].mean()) if len(q) else np.nan
            crps_gain=float(q["crps_gain"].mean()) if len(q) else np.nan
            max_shift=float(q["mean_shift"].abs().max()) if len(q) else np.nan
            passed=bool(
                support and crps_gain>0
                and boot.get("ci_low") is not None and boot["ci_low"]>0
                and cand_upper<base_upper
                and abs(cand_upper-.10)<abs(base_upper-.10)
                and abs(cand80-.80)<=abs(base80-.80)+.005
                and abs(cand90-.90)<=abs(base90-.90)+.005
                and max_shift<=1e-10
            )
            season_pass[season]=passed
            cells.append({
                "season":season,"market":market,"rows":int(len(q)),
                "games":int(q["game_id"].nunique()),"support":support,
                "baseline_crps":float(q["baseline_crps"].mean()) if len(q) else np.nan,
                "candidate_crps":float(q["candidate_crps"].mean()) if len(q) else np.nan,
                "crps_gain":crps_gain,
                "bootstrap_ci_low":boot.get("ci_low"),"bootstrap_ci_high":boot.get("ci_high"),
                "bootstrap_p_gt_0":boot.get("p_gt_0"),
                "baseline_upper_q90":base_upper,"candidate_upper_q90":cand_upper,
                "baseline_lower_q10":float(q["baseline_lower_q10"].mean()) if len(q) else np.nan,
                "candidate_lower_q10":float(q["candidate_lower_q10"].mean()) if len(q) else np.nan,
                "baseline_cov80":base80,"candidate_cov80":cand80,
                "baseline_cov90":base90,"candidate_cov90":cand90,
                "baseline_width80":float(q["baseline_width80"].mean()) if len(q) else np.nan,
                "candidate_width80":float(q["candidate_width80"].mean()) if len(q) else np.nan,
                "baseline_width90":float(q["baseline_width90"].mean()) if len(q) else np.nan,
                "candidate_width90":float(q["candidate_width90"].mean()) if len(q) else np.nan,
                "mean_sd_ratio":float((q["candidate_sd"]/q["baseline_sd"].replace(0,np.nan)).mean()) if len(q) else np.nan,
                "max_abs_mean_shift":max_shift,
                "cell_pass":passed,
            })
        qualified=all(season_pass.get(s,False) for s in SEASONS)
        per_market.append({
            "market":market,
            "season_2024_pass":bool(season_pass.get(2024,False)),
            "season_2025_pass":bool(season_pass.get(2025,False)),
            "disposition":(
                "Q90_UPPER_TAIL_HINGE_DEVELOPMENT_QUALIFIED"
                if qualified else "Q90_UPPER_TAIL_HINGE_DEVELOPMENT_NOT_QUALIFIED"
            )
        })
    anyq=any(r["disposition"]=="Q90_UPPER_TAIL_HINGE_DEVELOPMENT_QUALIFIED" for r in per_market)
    result={
        "version":VERSION,
        "disposition":(
            "Q90_UPPER_TAIL_HINGE_DEVELOPMENT_SURVIVOR_EXISTS"
            if anyq else "Q90_UPPER_TAIL_HINGE_DEVELOPMENT_NULL"
        ),
        "development_only":True,
        "historical_confirmation":False,
        "transform":"candidate_pre = x + max(x-q90,0); multiplicative exact-mean realignment",
        "tail_excess_multiplier":2.0,
        "hinge_quantile":0.90,
        "markets":list(MARKETS),
        "seasons":list(SEASONS),
        "bootstrap_reps":BOOT_REPS,
        "per_market":per_market,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "outcomes_2026_used":0,
        "parameters_fit":0,
        "threshold_searches":0,
        "football_mean_changed":False,
        "production_changed":False,
    }
    return pd.DataFrame(cells),result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file",type=Path,required=True)
    ap.add_argument("--distribution-dir",type=Path,required=True)
    ap.add_argument("--parent-detail",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    detail=build_detail(read(a.projection_file,"projection trace"),a.distribution_dir)
    parity=compare_parent(detail,read(a.parent_detail,"frozen parent detail"))
    cells,result=summarize(detail)
    result["parent_parity"]=parity
    result["scoreable_rows"]=int(len(detail))
    result["max_abs_mean_shift"]=float(detail["mean_shift"].abs().max())
    if result["max_abs_mean_shift"]>1e-10:
        result["disposition"]="Q90_UPPER_TAIL_HINGE_INTEGRITY_FAILURE"
    detail.to_csv(a.out_dir/"distribution_q90_upper_tail_hinge_detail.csv",index=False)
    cells.to_csv(a.out_dir/"distribution_q90_upper_tail_hinge_cells.csv",index=False)
    (a.out_dir/"distribution_q90_upper_tail_hinge_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    print(cells.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
