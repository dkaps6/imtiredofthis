#!/usr/bin/env python3
"""Score the frozen public TE-YPT transmission candidate PUB-TEY1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION="PUBLIC_TE_YPT_INTEGRATION_CANDIDATE_V1"
TRAIN=2022
PRIMARY=2023
SECONDARY=(2024,2025)
WEEKS=tuple(range(2,19))
BOOT_REPS=5000
BOOT_SEED=20261006
MIN_TRAIN_ROWS=200
MIN_TRAIN_GAMES=50
FORBIDDEN=(
    "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
    "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob",
)


def _read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    bad=[c for c in x.columns if any(t in c for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sportsbook columns in {label}: {bad}")
    return x


def _num(x):
    return pd.to_numeric(x,errors="coerce")


def _key(v)->str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _position_group(v)->str:
    s=str(v or "").upper().strip()
    if s.startswith("TE"):
        return "TE"
    if s.startswith("WR"):
        return "WR"
    if s in {"RB","HB","FB"} or s.startswith("RB"):
        return "RB"
    return s


def build_public_te_ypt(logs:pd.DataFrame, target_universe:pd.DataFrame, seasons=(2022,2023))->pd.DataFrame:
    x=logs.copy()
    x["season"]=_num(x["season"]).astype("Int64")
    x["week"]=_num(x["week"]).astype("Int64")
    x["team"]=x["team"].map(canon_team)
    x["opponent"]=x["opponent"].map(canon_team)
    x["position_group"]=x["position"].map(_position_group)
    x=x.loc[x["season"].isin(seasons)&x["position_group"].eq("TE")].copy()
    x["targets"]=_num(x["targets"]).fillna(0.0)
    x["rec_yards"]=_num(x["rec_yards"]).fillna(0.0)
    x["defense"]=x["opponent"]
    weekly=x.groupby(["season","week","defense"],as_index=False).agg(
        te_targets=("targets","sum"),
        te_rec_yards=("rec_yards","sum"),
    )

    u=target_universe.copy()
    u["season"]=_num(u["season"]).astype("Int64")
    u["week"]=_num(u["week"]).astype("Int64")
    u["team"]=u["team"].map(canon_team)
    u["opponent"]=u["opponent"].map(canon_team)
    u=u.loc[u["season"].isin(seasons)&u["week"].isin(WEEKS),["season","week","team","opponent"]].drop_duplicates()

    rows=[]
    for r in u.itertuples(index=False):
        h=weekly.loc[
            weekly["season"].eq(int(r.season))
            & weekly["defense"].eq(r.opponent)
            & weekly["week"].lt(int(r.week))
        ].copy()
        if h.empty:
            ypt=np.nan; den=0.0; max_week=np.nan
        else:
            keep=sorted(h["week"].dropna().astype(int).unique().tolist())[-8:]
            h=h.loc[h["week"].isin(keep)]
            den=float(h["te_targets"].sum())
            ypt=float(h["te_rec_yards"].sum()/den) if den>0 else np.nan
            max_week=int(h["week"].max()) if len(h) else np.nan
        rows.append({
            "season":int(r.season),"week":int(r.week),"team":r.team,"opponent":r.opponent,
            "public_def_te_ypt_allowed":ypt,
            "public_def_te_targets_faced":den,
            "source_max_week":max_week,
        })
    out=pd.DataFrame(rows)
    out["weakness_z"]=np.nan
    for _,idx in out.groupby(["season","week"]).groups.items():
        v=_num(out.loc[idx,"public_def_te_ypt_allowed"])
        sd=float(v.std(ddof=0)) if v.notna().sum()>=2 else np.nan
        if np.isfinite(sd) and sd>0:
            out.loc[idx,"weakness_z"]=(v-float(v.mean()))/sd
    bad=out.loc[_num(out["source_max_week"]).ge(_num(out["week"]))]
    if len(bad):
        raise RuntimeError(f"public TE YPT chronology violation: {bad.head(20).to_dict('records')}")
    return out


def build_secondary_features(phase:pd.DataFrame)->pd.DataFrame:
    x=phase.copy()
    for c in ("season","week"):
        x[c]=_num(x[c]).astype("Int64")
    x["team"]=x["team"].map(canon_team)
    x["opponent"]=x["opponent"].map(canon_team)
    need={"season","week","team","opponent","def_te_ypt_allowed","def_te_ypt_allowed__z"}
    if not need.issubset(x.columns):
        raise RuntimeError(f"Phase B/C feature authority missing {sorted(need-set(x.columns))}")
    x=x.loc[x["season"].isin(SECONDARY)&x["week"].isin(WEEKS)].copy()
    return x[["season","week","team","opponent","def_te_ypt_allowed","def_te_ypt_allowed__z"]].rename(
        columns={
            "def_te_ypt_allowed":"public_def_te_ypt_allowed",
            "def_te_ypt_allowed__z":"weakness_z",
        }
    )


def prepare_rows(baseline:pd.DataFrame, logs:pd.DataFrame, phase:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    b=baseline.copy()
    for c in ("season","week"):
        b[c]=_num(b[c]).astype("Int64")
    b["team"]=b["team"].map(canon_team)
    b["opponent"]=b["opponent"].map(canon_team)
    b["player_clean_key"]=b["player_clean_key"].map(_key)
    b["market"]=b["market"].astype(str).str.lower().str.strip()
    b["actual"]=_num(b["actual"])
    b["baseline_projection"]=_num(b["ensemble_proj"])
    b=b.loc[b["season"].isin([2022,2023,2024,2025])&b["week"].isin(WEEKS)].copy()

    m=logs.copy()
    for c in ("season","week"):
        m[c]=_num(m[c]).astype("Int64")
    m["team"]=m["team"].map(canon_team)
    m["player_clean_key"]=m["player_clean_key"].map(_key)
    m["position"]=m["position"].map(_position_group)
    if "player_identity_key" not in m.columns:
        m["player_identity_key"]=m["player_clean_key"]
    meta=m[["season","week","team","player_clean_key","position","player_identity_key"]].drop_duplicates()
    n=meta.groupby(["season","week","team","player_clean_key"])["position"].nunique()
    if n.gt(1).any():
        raise RuntimeError("conflicting player positions in preserved player logs")
    meta=meta.drop_duplicates(["season","week","team","player_clean_key"],keep="last")

    b=b.merge(meta,on=["season","week","team","player_clean_key"],how="left",validate="many_to_one")
    missing=float(b["position"].isna().mean()) if len(b) else 1.0
    if missing>.02:
        raise RuntimeError(f"position identity coverage below 98%: missing={missing:.6f}")

    te=b.loc[b["market"].eq("rec_yards")&b["position"].eq("TE")].copy()
    te=te.loc[te["actual"].notna()&te["baseline_projection"].notna()].copy()

    early=build_public_te_ypt(m,te[["season","week","team","opponent"]],seasons=(2022,2023))
    secondary=build_secondary_features(phase)
    feat=pd.concat([early,secondary],ignore_index=True,sort=False)
    if feat.duplicated(["season","week","team","opponent"]).any():
        raise RuntimeError("duplicate candidate team features")
    te=te.merge(feat,on=["season","week","team","opponent"],how="left",validate="many_to_one")

    # Baseline provenance: 2024/25 must be the frozen right-tail authority.
    parity={}
    if "baseline_authority" in te.columns:
        for season in SECONDARY:
            vals=sorted(te.loc[te["season"].eq(season),"baseline_authority"].dropna().astype(str).unique().tolist())
            parity[str(season)]=vals
            if vals!=["FROZEN_RIGHT_TAIL_FINAL_MEAN"]:
                raise RuntimeError(f"secondary baseline authority drift season={season}: {vals}")

    scoreable=te["weakness_z"].notna()
    te["candidate_scoreable"]=scoreable
    return te,{
        "position_missing_fraction_all_baseline":missing,
        "secondary_baseline_authority":parity,
        "chronology_violations_2022_2023":0,
    }


def fit_beta(q:pd.DataFrame)->dict:
    z=q.loc[q["season"].eq(TRAIN)&q["candidate_scoreable"]].copy()
    x=_num(z["weakness_z"]).to_numpy(float)
    y=(_num(z["actual"])-_num(z["baseline_projection"])).to_numpy(float)
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good]; y=y[good]
    den=float(np.dot(x,x))
    beta=float(np.dot(x,y)/den) if den>0 else np.nan
    rows=int(good.sum())
    games=int(z.loc[good,"game_id"].nunique()) if len(z)==len(good) else int(z["game_id"].nunique())
    return {
        "rows":rows,"games":games,"beta_train":beta,
        "support":bool(rows>=MIN_TRAIN_ROWS and games>=MIN_TRAIN_GAMES),
        "positive_beta":bool(np.isfinite(beta) and beta>0),
    }


def metric_block(z:pd.DataFrame,pred_col:str)->dict:
    q=z[["actual",pred_col,"player_identity_key"]].copy().dropna()
    if q.empty:
        return {"n":0,"players":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"median_ae":np.nan,"corr":np.nan,"tail75":0,"tail100":0}
    e=_num(q[pred_col])-_num(q["actual"])
    return {
        "n":int(len(q)),
        "players":int(q["player_identity_key"].nunique()),
        "mae":float(e.abs().mean()),
        "rmse":float(np.sqrt(np.mean(np.square(e)))),
        "bias":float(e.mean()),
        "median_ae":float(e.abs().median()),
        "corr":float(_num(q["actual"]).corr(_num(q[pred_col]))) if len(q)>2 else np.nan,
        "tail75":int(e.abs().ge(75).sum()),
        "tail100":int(e.abs().ge(100).sum()),
    }


def spearman(z:pd.DataFrame,pred_col:str)->float:
    q=z[["actual",pred_col,"weakness_z"]].copy().replace([np.inf,-np.inf],np.nan).dropna()
    if len(q)<3:
        return np.nan
    residual=_num(q["actual"])-_num(q[pred_col])
    return float(residual.corr(_num(q["weakness_z"]),method="spearman"))


def bootstrap(z:pd.DataFrame,seed:int)->dict:
    q=z[["game_id","actual","baseline_projection","candidate_projection"]].copy().dropna()
    q["base_ae"]=(_num(q["baseline_projection"])-_num(q["actual"])).abs()
    q["cand_ae"]=(_num(q["candidate_projection"])-_num(q["actual"])).abs()
    g=q.groupby("game_id",as_index=False).agg(n=("actual","size"),b=("base_ae","sum"),c=("cand_ae","sum"))
    if len(g)<2:
        return {"p_improve":np.nan,"ci_low":np.nan,"ci_high":np.nan,"valid_reps":0}
    a=g[["n","b","c"]].to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=[]
    done=0
    p=np.full(len(g),1/len(g))
    while done<BOOT_REPS:
        k=min(250,BOOT_REPS-done)
        counts=rng.multinomial(len(g),p,size=k).astype(float)
        s=counts@a
        vals.append((s[:,1]-s[:,2])/s[:,0])
        done+=k
    v=np.concatenate(vals)
    v=v[np.isfinite(v)]
    return {
        "p_improve":float((v>0).mean()) if len(v) else np.nan,
        "ci_low":float(np.quantile(v,.025)) if len(v) else np.nan,
        "ci_high":float(np.quantile(v,.975)) if len(v) else np.nan,
        "valid_reps":int(len(v)),
    }


def season_score(q:pd.DataFrame,season:int,beta:float)->dict:
    z=q.loc[q["season"].eq(season)&q["candidate_scoreable"]].copy()
    z["candidate_projection"]=z["baseline_projection"]+beta*z["weakness_z"]
    base=metric_block(z,"baseline_projection")
    cand=metric_block(z,"candidate_projection")
    adj=(z["candidate_projection"]-z["baseline_projection"]).abs()
    return {
        "season":season,
        "rows":int(len(z)),
        "games":int(z["game_id"].nunique()),
        "baseline":base,
        "candidate":cand,
        "mae_improvement":float(base["mae"]-cand["mae"]) if len(z) else np.nan,
        "bootstrap":bootstrap(z,BOOT_SEED+season),
        "residual_spearman_before":spearman(z,"baseline_projection"),
        "residual_spearman_after":spearman(z,"candidate_projection"),
        "mean_abs_adjustment":float(adj.mean()) if len(adj) else np.nan,
        "p95_abs_adjustment":float(adj.quantile(.95)) if len(adj) else np.nan,
        "max_abs_adjustment":float(adj.max()) if len(adj) else np.nan,
    }


def pooled_secondary(q:pd.DataFrame,beta:float)->dict:
    z=q.loc[q["season"].isin(SECONDARY)&q["candidate_scoreable"]].copy()
    z["candidate_projection"]=z["baseline_projection"]+beta*z["weakness_z"]
    base=metric_block(z,"baseline_projection")
    cand=metric_block(z,"candidate_projection")
    return {
        "rows":int(len(z)),"games":int(z["game_id"].nunique()),
        "baseline":base,"candidate":cand,
        "mae_improvement":float(base["mae"]-cand["mae"]) if len(z) else np.nan,
    }


def score(q:pd.DataFrame,integrity:dict)->tuple[dict,pd.DataFrame]:
    fit=fit_beta(q)
    beta=fit["beta_train"]
    trace=q.copy()
    trace["candidate_projection"]=trace["baseline_projection"]+beta*trace["weakness_z"]

    scores={str(s):season_score(q,s,beta) for s in (2022,2023,2024,2025)}
    primary=scores["2023"]
    s24=scores["2024"]; s25=scores["2025"]
    pooled=pooled_secondary(q,beta)

    gates={
        "training_support":bool(fit["support"]),
        "positive_beta":bool(fit["positive_beta"]),
        "primary_mae_improves":bool(primary["candidate"]["mae"]<primary["baseline"]["mae"]),
        "primary_rmse_non_worse":bool(primary["candidate"]["rmse"]<=primary["baseline"]["rmse"]),
        "primary_bootstrap_p_ge_0p80":bool(primary["bootstrap"]["p_improve"]>=.80),
        "primary_tail75_non_worse":bool(primary["candidate"]["tail75"]<=primary["baseline"]["tail75"]),
        "primary_tail100_non_worse":bool(primary["candidate"]["tail100"]<=primary["baseline"]["tail100"]),
        "primary_residual_spearman_reduced":bool(
            np.isfinite(primary["residual_spearman_before"])
            and np.isfinite(primary["residual_spearman_after"])
            and abs(primary["residual_spearman_after"])<abs(primary["residual_spearman_before"])
        ),
        "secondary_2024_mae_non_worse":bool(s24["candidate"]["mae"]<=s24["baseline"]["mae"]),
        "secondary_2025_mae_non_worse":bool(s25["candidate"]["mae"]<=s25["baseline"]["mae"]),
        "secondary_pooled_mae_improves":bool(pooled["candidate"]["mae"]<pooled["baseline"]["mae"]),
        "secondary_pooled_tail75_non_worse":bool(pooled["candidate"]["tail75"]<=pooled["baseline"]["tail75"]),
        "secondary_pooled_tail100_non_worse":bool(pooled["candidate"]["tail100"]<=pooled["baseline"]["tail100"]),
        "chronology_clean":bool(integrity["chronology_violations_2022_2023"]==0),
    }
    disposition="PUB_TEY1_CONFIRMED" if all(gates.values()) else "PUB_TEY1_CLOSED"
    result={
        "version":VERSION,
        "candidate_id":"PUB-TEY1",
        "feature":"public_def_te_ypt_allowed",
        "train_season":2022,
        "primary_confirmation_season":2023,
        "secondary_consistency_seasons":[2024,2025],
        "evaluation_weeks":[2,18],
        "functional_form":"baseline + beta_2022 * week_z(public_def_te_ypt_allowed)",
        "train":fit,
        "season_scores":scores,
        "secondary_pooled":pooled,
        "gates":gates,
        "integrity":integrity,
        "disposition":disposition,
        "sportsbook_inputs_used":0,
        "outcomes_2026_read":0,
        "production_changed":False,
        "wr_candidate_scored":False,
        "rb_candidate_scored":False,
        "posthoc_rescue_scored":False,
    }
    return result,trace


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--composite-baseline",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--phase-bc-team-features",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    baseline=_read(a.composite_baseline,"corrected composite baseline")
    logs=_read(a.player_logs,"preserved player logs")
    phase=_read(a.phase_bc_team_features,"frozen Phase B/C team features")
    q,integrity=prepare_rows(baseline,logs,phase)
    result,trace=score(q,integrity)

    a.out_dir.mkdir(parents=True,exist_ok=True)
    trace.to_csv(a.out_dir/"public_te_ypt_candidate_trace.csv",index=False)
    (a.out_dir/"public_te_ypt_candidate_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
