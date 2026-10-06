#!/usr/bin/env python3
"""Frozen RBDI-F7-1 scorer.

2024 fits one zero-intercept coefficient from opponent OUT/DOUBTFUL FRONT7
strictly-prior defensive snap mass. 2025 is untouched confirmation.
No sportsbook data, no 2026 outcomes, no production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION="RB_OPPONENT_DEFENDER_INJURY_CANDIDATE_V1"
TRAIN=2024
CONFIRM=2025
WEEKS=tuple(range(2,19))
RB_POS={"RB","HB","FB"}
BOOT_REPS=10000
BOOT_SEED=20261006
MIN_ROWS=500
MIN_GAMES=100
TOL=1e-10

FORBIDDEN=(
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x


def num(x):
    return pd.to_numeric(x,errors="coerce")


def key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def bval(x) -> pd.Series:
    if isinstance(x,pd.Series):
        if x.dtype==bool:
            return x
        return x.astype(str).str.strip().str.lower().isin({"true","1","yes","y","t"})
    return pd.Series(dtype=bool)


def check_forbidden(label: str, x: pd.DataFrame) -> None:
    bad=[c for c in x.columns if any(t in str(c).lower() for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sportsbook fields in {label}: {bad}")


def build_burden(evidence: pd.DataFrame) -> pd.DataFrame:
    x=evidence.copy()
    for c in ("season","week"):
        x[c]=num(x[c]).astype("Int64")
    x["team"]=x["team"].map(canon_team)
    x=x.loc[x["season"].isin([TRAIN,CONFIRM]) & x["week"].isin(WEEKS)].copy()
    x["front7_b"]=bval(x.get("front7",pd.Series(False,index=x.index)))
    x["out_doubtful_b"]=bval(x.get("out_doubtful",pd.Series(False,index=x.index)))
    x["snap_joined_b"]=bval(x.get("snap_joined",pd.Series(False,index=x.index)))
    x["chronology_valid_b"]=bval(x.get("chronology_valid",pd.Series(False,index=x.index)))
    x["prior_defense_pct_n"]=num(x.get("prior_defense_pct",np.nan))

    rel=x.loc[x["front7_b"] & x["out_doubtful_b"]].copy()
    if rel.empty:
        return pd.DataFrame(columns=[
            "season","week","team","front7_out_doubtful_count",
            "front7_missing_prior_snap_count","front7_out_doubtful_snap_mass",
            "candidate_team_week_complete",
        ])

    rows=[]
    for (season,week,team),g in rel.groupby(["season","week","team"],sort=True):
        valid=g["snap_joined_b"] & g["chronology_valid_b"] & g["prior_defense_pct_n"].notna()
        missing=int((~valid).sum())
        rows.append({
            "season":int(season),
            "week":int(week),
            "team":str(team),
            "front7_out_doubtful_count":int(len(g)),
            "front7_missing_prior_snap_count":missing,
            "front7_out_doubtful_snap_mass":float(g.loc[valid,"prior_defense_pct_n"].sum()) if missing==0 else np.nan,
            "candidate_team_week_complete":bool(missing==0),
        })
    return pd.DataFrame(rows)


def attach_positions(detail: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    x=detail.copy()
    for c in ("season","week"):
        x[c]=num(x[c]).astype("Int64")
    x=x.loc[x["season"].isin([TRAIN,CONFIRM]) & x["week"].isin(WEEKS)].copy()
    x["team"]=x["team"].map(canon_team)
    x["opponent"]=x["opponent"].map(canon_team)
    x["player_clean_key"]=x["player_clean_key"].map(key)
    x["market"]=x["market"].astype(str).str.lower()
    x["actual"]=num(x["actual"])
    x["baseline_projection"]=num(x["final_mean"])

    m=logs.copy()
    for c in ("season","week"):
        m[c]=num(m[c]).astype("Int64")
    m["team"]=m["team"].map(canon_team)
    m["player_clean_key"]=m["player_clean_key"].map(key)
    m["position"]=m["position"].astype(str).str.upper().str.strip()
    cols=["season","week","team","player_clean_key","position"]
    m=m[cols].drop_duplicates()
    dup=m.duplicated(["season","week","team","player_clean_key"],keep=False)
    if dup.any():
        # identical-position duplicates are harmless; conflicting positions fail.
        n=m.groupby(["season","week","team","player_clean_key"])["position"].nunique()
        if n.gt(1).any():
            raise RuntimeError("conflicting player positions in preserved history")
        m=m.drop_duplicates(["season","week","team","player_clean_key"],keep="last")

    x=x.merge(m,on=["season","week","team","player_clean_key"],how="left",validate="many_to_one")
    miss=x["position"].isna()
    if len(x) and float(miss.mean())>.02:
        raise RuntimeError(f"position coverage too low: {int(miss.sum())}/{len(x)}")
    return x


def candidate_rows(detail: pd.DataFrame, logs: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    p=attach_positions(detail,logs)
    q=p.loc[p["market"].eq("rush_yards") & p["position"].isin(RB_POS)].copy()
    q=q.loc[q["actual"].notna() & q["baseline_projection"].notna()].copy()

    burden=build_burden(evidence)
    b=burden.rename(columns={"team":"opponent"})
    q=q.merge(
        b,
        on=["season","week","opponent"],
        how="left",
        validate="many_to_one",
    )
    # No qualifying O/D FRONT7 injury row => exact zero burden and complete.
    q["front7_out_doubtful_count"]=num(q["front7_out_doubtful_count"]).fillna(0).astype(int)
    q["front7_missing_prior_snap_count"]=num(q["front7_missing_prior_snap_count"]).fillna(0).astype(int)
    q["candidate_team_week_complete"]=q["candidate_team_week_complete"].fillna(True).astype(bool)
    q["front7_out_doubtful_snap_mass"]=num(q["front7_out_doubtful_snap_mass"])
    zero=q["front7_out_doubtful_count"].eq(0)
    q.loc[zero,"front7_out_doubtful_snap_mass"]=0.0

    # An identified relevant injury with incomplete prior snap is not scoreable.
    q["candidate_scoreable"]=(
        q["candidate_team_week_complete"]
        & q["front7_out_doubtful_snap_mass"].notna()
    )
    return q,burden


def fit_beta(q: pd.DataFrame) -> dict:
    z=q.loc[q["season"].eq(TRAIN) & q["candidate_scoreable"]].copy()
    x=num(z["front7_out_doubtful_snap_mass"]).to_numpy(float)
    y=(num(z["actual"])-num(z["baseline_projection"])).to_numpy(float)
    good=np.isfinite(x)&np.isfinite(y)
    x,y=x[good],y[good]
    den=float(np.dot(x,x))
    beta=float(np.dot(x,y)/den) if den>0 else np.nan
    return {
        "rows":int(good.sum()),
        "games":int(z.loc[good,"game_id"].nunique()) if len(z)==len(good) else int(z["game_id"].nunique()),
        "beta_train":beta,
        "positive_beta":bool(np.isfinite(beta) and beta>0),
    }


def metrics(actual,pred) -> dict:
    z=pd.DataFrame({"actual":num(actual),"pred":num(pred)}).dropna()
    if z.empty:
        return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"tail75":0,"tail100":0}
    e=z["pred"]-z["actual"]
    return {
        "n":int(len(z)),
        "mae":float(e.abs().mean()),
        "rmse":float(np.sqrt(np.mean(np.square(e)))),
        "bias":float(e.mean()),
        "tail75":int(e.abs().ge(75).sum()),
        "tail100":int(e.abs().ge(100).sum()),
    }


def bootstrap(q: pd.DataFrame) -> dict:
    z=q[["game_id","actual","baseline_projection","candidate_projection"]].dropna().copy()
    if z["game_id"].nunique()<2:
        return {"ci_low":np.nan,"ci_high":np.nan,"p_improve":np.nan,"valid_reps":0}
    z["base_ae"]=(z["baseline_projection"]-z["actual"]).abs()
    z["cand_ae"]=(z["candidate_projection"]-z["actual"]).abs()
    g=z.groupby("game_id",as_index=False).agg(
        n=("actual","size"),base_sum=("base_ae","sum"),cand_sum=("cand_ae","sum")
    )
    a=g[["n","base_sum","cand_sum"]].to_numpy(float)
    rng=np.random.default_rng(BOOT_SEED)
    vals=[]
    done=0
    probs=np.full(len(g),1.0/len(g))
    while done<BOOT_REPS:
        k=min(250,BOOT_REPS-done)
        counts=rng.multinomial(len(g),probs,size=k).astype(float)
        s=counts@a
        good=s[:,0]>0
        vals.append((s[good,1]-s[good,2])/s[good,0])
        done+=k
    v=np.concatenate(vals)
    v=v[np.isfinite(v)]
    return {
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
        "p_improve":float((v>0).mean()) if len(v) else np.nan,
        "valid_reps":int(len(v)),
    }


def score(q: pd.DataFrame) -> tuple[dict,pd.DataFrame]:
    fit=fit_beta(q)
    beta=fit["beta_train"]
    out=q.copy()
    out["candidate_projection"]=out["baseline_projection"] + beta*out["front7_out_doubtful_snap_mass"]

    zero=out["candidate_scoreable"] & out["front7_out_doubtful_snap_mass"].eq(0)
    zero_gap=float((out.loc[zero,"candidate_projection"]-out.loc[zero,"baseline_projection"]).abs().max()) if zero.any() else 0.0

    train=out.loc[out["season"].eq(TRAIN)&out["candidate_scoreable"]].copy()
    confirm=out.loc[out["season"].eq(CONFIRM)&out["candidate_scoreable"]].copy()

    train_base=metrics(train["actual"],train["baseline_projection"])
    train_cand=metrics(train["actual"],train["candidate_projection"])
    base=metrics(confirm["actual"],confirm["baseline_projection"])
    cand=metrics(confirm["actual"],confirm["candidate_projection"])
    boot=bootstrap(confirm)

    support=bool(len(confirm)>=MIN_ROWS and confirm["game_id"].nunique()>=MIN_GAMES)
    gates={
        "positive_beta":bool(np.isfinite(beta) and beta>0),
        "support":support,
        "mae_improves":bool(cand["mae"]<base["mae"]) if support else False,
        "bootstrap_ci_low_positive":bool(np.isfinite(boot["ci_low"]) and boot["ci_low"]>0),
        "rmse_non_worse":bool(cand["rmse"]<=base["rmse"]) if support else False,
        "absolute_bias_non_worse":bool(abs(cand["bias"])<=abs(base["bias"])) if support else False,
        "tail75_non_worse":bool(cand["tail75"]<=base["tail75"]) if support else False,
        "tail100_non_worse":bool(cand["tail100"]<=base["tail100"]) if support else False,
        "zero_burden_noop":bool(zero_gap<=TOL),
    }
    scientific=all(gates.values())
    disposition="RBDI_F7_1_CONFIRMED" if scientific else "RBDI_F7_1_CLOSED"
    result={
        "version":VERSION,
        "candidate_id":"RBDI-F7-1",
        "train_season":TRAIN,
        "confirmation_season":CONFIRM,
        "weeks":[2,18],
        "feature":"front7_out_doubtful_snap_mass",
        "functional_form":"baseline + beta_2024 * raw_front7_out_doubtful_snap_mass",
        "train":{
            **fit,
            "baseline":train_base,
            "candidate":train_cand,
        },
        "confirmation":{
            "rows":int(len(confirm)),
            "games":int(confirm["game_id"].nunique()),
            "baseline":base,
            "candidate":cand,
            "mae_improvement":float(base["mae"]-cand["mae"]) if support else np.nan,
            "bootstrap":boot,
            "zero_burden_max_abs_gap":zero_gap,
        },
        "gates":gates,
        "disposition":disposition,
        "sportsbook_inputs_used":0,
        "outcomes_2026_read":0,
        "production_changed":False,
        "posthoc_subgroups_scored":0,
    }
    return result,out


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--right-tail-detail",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--injury-evidence",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    detail=read(a.right_tail_detail,"frozen right-tail detail")
    logs=read(a.player_logs,"preserved player logs")
    evidence=read(a.injury_evidence,"frozen injury readiness evidence")
    for label,x in (("detail",detail),("logs",logs),("injury",evidence)):
        check_forbidden(label,x)

    q,burden=candidate_rows(detail,logs,evidence)
    result,trace=score(q)

    a.out_dir.mkdir(parents=True,exist_ok=True)
    burden.to_csv(a.out_dir/"rb_opponent_defender_injury_candidate_team_week_feature.csv",index=False)
    trace.to_csv(a.out_dir/"rb_opponent_defender_injury_candidate_trace.csv",index=False)
    (a.out_dir/"rb_opponent_defender_injury_candidate_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
