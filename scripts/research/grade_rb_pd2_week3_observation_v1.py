#!/usr/bin/env python3
"""Attach Week-3 outcomes to the immutable RB-PD2 forward lock.

Implements the frozen forward-confirmation metrics without changing the
candidate. Week 3 is observation week #1; scientific PASS/FAIL is forbidden
until >=8 locked weeks and >=400 unique eligible player-games.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.utils.canonical_names import canonicalize_player_name_safe

SEASON=2026
WEEK=3
THRESHOLDS=(50.0,75.0,100.0)


def _to_pandas(x):
    return x.to_pandas() if hasattr(x,"to_pandas") else pd.DataFrame(x)


def _pick(df,names):
    for c in names:
        if c in df.columns:
            return c
    raise RuntimeError(f"none of {names} present; have={sorted(df.columns)}")


def _empirical_crps(draws: np.ndarray, actual: float) -> float:
    x=np.asarray(draws,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():
        raise RuntimeError("invalid empirical draw array")
    xs=np.sort(x)
    n=len(xs)
    coeff=2*np.arange(n,dtype=float)-n+1
    half_pairwise=float(np.dot(coeff,xs)/(n*n))
    return float(np.mean(np.abs(x-float(actual)))-half_pairwise)


def _verified_aliases() -> dict[tuple[str,str],str]:
    out={}
    for p in (Path("data/player_identity_aliases.csv"),Path("config/player_identity_current_aliases_v1.csv")):
        if not p.exists() or not p.stat().st_size:
            continue
        x=pd.read_csv(p,dtype="string").fillna("")
        for r in x.itertuples(index=False):
            team=canon_team(str(r.current_team))
            _,cur=canonicalize_player_name_safe(str(r.current_name))
            _,hist=canonicalize_player_name_safe(str(r.historical_name))
            if cur and hist:
                out[(team,cur)]=hist
                out[(team,hist)]=cur
    return out


def _load_actuals() -> tuple[pd.DataFrame,pd.DataFrame]:
    stats=load_weekly_player_stats(SEASON).copy()
    stats.columns=[str(c).strip().lower() for c in stats.columns]
    stats=stats.loc[pd.to_numeric(stats["week"],errors="coerce").eq(WEEK)].copy()
    tc=_pick(stats,("recent_team","team","team_abbr","club"))
    nc=_pick(stats,("player_display_name","player_name","player"))
    yc=_pick(stats,("rushing_yards","rush_yards"))
    stats["team"]=stats[tc].astype("string").fillna("").str.strip().map(canon_team)
    canon=stats[nc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    stats["player_clean_key"]=canon.map(lambda t:t[1])
    stats["actual_rush_yards"]=pd.to_numeric(stats[yc],errors="coerce").fillna(0.0)
    idc="player_id" if "player_id" in stats.columns else "gsis_id" if "gsis_id" in stats.columns else None
    stats["player_id_actual"]=stats[idc].astype("string").fillna("").str.strip() if idc else ""
    stats=stats[["team","player_clean_key","player_id_actual","actual_rush_yards"]].drop_duplicates()

    import nflreadpy as nfl
    roster=_to_pandas(nfl.load_rosters_weekly(SEASON)).copy()
    roster.columns=[str(c).strip().lower() for c in roster.columns]
    roster=roster.loc[pd.to_numeric(roster["week"],errors="coerce").eq(WEEK)].copy()
    rtc=_pick(roster,("team","team_abbr","club_code"))
    rnc=_pick(roster,("full_name","football_name","player_name","player"))
    roster["team"]=roster[rtc].astype("string").fillna("").str.strip().map(canon_team)
    rc=roster[rnc].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    roster["player_clean_key"]=rc.map(lambda t:t[1])
    ridc="gsis_id" if "gsis_id" in roster.columns else "player_id" if "player_id" in roster.columns else None
    roster["player_id_roster"]=roster[ridc].astype("string").fillna("").str.strip() if ridc else ""
    roster=roster[["team","player_clean_key","player_id_roster"]].drop_duplicates()
    return stats,roster


def _assert_games_final(rows: list[dict]) -> None:
    import nflreadpy as nfl
    sched=_to_pandas(nfl.load_schedules(seasons=[SEASON])).copy()
    sched.columns=[str(c).strip().lower() for c in sched.columns]
    if "game_type" in sched.columns:
        sched=sched.loc[sched["game_type"].astype(str).str.upper().eq("REG")]
    sched=sched.loc[pd.to_numeric(sched["week"],errors="coerce").eq(WEEK)].copy()
    sched["home_team"]=sched["home_team"].map(canon_team)
    sched["away_team"]=sched["away_team"].map(canon_team)
    frozen=set(canon_team(str(r["team"])) for r in rows)
    seen=set()
    for r in sched.itertuples(index=False):
        if str(r.home_team) in frozen or str(r.away_team) in frozen:
            hs=pd.to_numeric(pd.Series([r.home_score]),errors="coerce").iloc[0]
            aws=pd.to_numeric(pd.Series([r.away_score]),errors="coerce").iloc[0]
            if pd.isna(hs) or pd.isna(aws):
                raise RuntimeError(f"nonfinal frozen game {r.home_team}-{r.away_team}")
            seen.update([str(r.home_team),str(r.away_team)])
    missing=frozen-seen
    if missing:
        raise RuntimeError(f"frozen teams missing from Week-3 final schedule: {sorted(missing)}")


def _attach_outcomes(rows:list[dict])->pd.DataFrame:
    stats,roster=_load_actuals()
    aliases=_verified_aliases()
    out=[]
    unresolved=[]
    ambiguous=[]
    for r in rows:
        team=canon_team(str(r["team"]))
        key=str(r["player_clean_key"])
        q=stats.loc[stats["team"].eq(team)&stats["player_clean_key"].eq(key)].copy()
        if q.empty:
            alt=aliases.get((team,key))
            if alt:
                q=stats.loc[stats["team"].eq(team)&stats["player_clean_key"].eq(alt)].copy()
        if len(q)>1:
            ambiguous.append({"team":team,"player":r["player"],"key":key,"stats_rows":len(q)})
            continue
        if len(q)==1:
            actual=float(q.iloc[0]["actual_rush_yards"])
            source="weekly_stats"
        else:
            keys={key}
            alt=aliases.get((team,key))
            if alt: keys.add(alt)
            rq=roster.loc[roster["team"].eq(team)&roster["player_clean_key"].isin(keys)]
            if len(rq)!=1:
                unresolved.append({
                    "team":team,"player":r["player"],"key":key,
                    "roster_rows":len(rq),"verified_alias":aliases.get((team,key),"")
                })
                continue
            actual=0.0
            source="final_week_exact_roster_verified_zero"
        out.append({
            "event_id":str(r["event_id"]),
            "team":team,
            "opponent":canon_team(str(r["opponent"])),
            "player":str(r["player"]),
            "player_clean_key":key,
            "actual_rush_yards":actual,
            "actual_source":source,
        })
    if ambiguous or unresolved:
        raise RuntimeError(
            "RB-PD2 Week3 actual identity gate failed: "
            + json.dumps({"ambiguous":ambiguous,"unresolved":unresolved},sort_keys=True)
        )
    return pd.DataFrame(out)


def _coverage(draws:np.ndarray,actual:float,lo:float,hi:float)->bool:
    qlo,qhi=np.quantile(draws,[lo,hi],method="linear")
    return bool(float(actual)>=float(qlo) and float(actual)<=float(qhi))


def _brier(draws:np.ndarray,actual:float,threshold:float)->float:
    p=float(np.mean(np.asarray(draws)>=threshold))
    y=float(actual>=threshold)
    return float((p-y)**2)


def _summarize(detail:pd.DataFrame,prefix:str)->dict:
    crps=float(detail[f"{prefix}_crps"].mean())
    cov80=float(detail[f"{prefix}_cover80"].mean())
    cov90=float(detail[f"{prefix}_cover90"].mean())
    return {
        "rows":int(len(detail)),
        "mean_crps":crps,
        "coverage80":cov80,
        "coverage90":cov90,
        "coverage80_abs_gap":abs(cov80-.80),
        "coverage90_abs_gap":abs(cov90-.90),
        "brier_ge50":float(detail[f"{prefix}_brier_ge50"].mean()),
        "brier_ge75":float(detail[f"{prefix}_brier_ge75"].mean()),
        "brier_ge100":float(detail[f"{prefix}_brier_ge100"].mean()),
        "mean_point_abs_error":float(detail[f"{prefix}_point_abs_error"].mean()),
    }


def _game_cluster_bootstrap(detail:pd.DataFrame,reps:int=10000,seed:int=42027)->dict:
    by={k:v["crps_gain"].to_numpy(float) for k,v in detail.groupby("event_id")}
    keys=list(by)
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,float)
    for i in range(reps):
        sampled=rng.choice(keys,size=len(keys),replace=True)
        vals[i]=float(np.concatenate([by[k] for k in sampled]).mean())
    return {
        "games":len(keys),
        "reps":reps,
        "seed":seed,
        "observed_mean_crps_gain":float(detail["crps_gain"].mean()),
        "ci95_lo":float(np.quantile(vals,.025)),
        "ci95_hi":float(np.quantile(vals,.975)),
    }


def _crossed_bootstrap(detail:pd.DataFrame,reps:int=10000,seed:int=42027)->dict:
    players=sorted(detail["player_clean_key"].unique())
    games=sorted(detail["event_id"].unique())
    rng=np.random.default_rng(seed)
    vals=[]
    rows=detail[["player_clean_key","event_id","candidate_minus_baseline_crps"]].copy()
    while len(vals)<reps:
        ps=rng.choice(players,size=len(players),replace=True)
        gs=rng.choice(games,size=len(games),replace=True)
        pm=pd.Series(ps).value_counts().to_dict()
        gm=pd.Series(gs).value_counts().to_dict()
        w=np.array([pm.get(p,0)*gm.get(g,0) for p,g in rows[["player_clean_key","event_id"]].itertuples(index=False,name=None)],dtype=float)
        if w.sum()<=0:
            continue
        vals.append(float(np.average(rows["candidate_minus_baseline_crps"],weights=w)))
    a=np.asarray(vals)
    return {
        "reps":reps,
        "seed":seed,
        "prob_candidate_minus_baseline_crps_lt_zero":float(np.mean(a<0)),
        "mean_candidate_minus_baseline_crps":float(a.mean()),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--locks",type=Path,required=True)
    ap.add_argument("--arrays",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    receipt=json.loads(a.receipt.read_text(encoding="utf-8"))
    if receipt.get("version")!="RB_PD2_FORWARD_LOCK_SESSION_V1" or receipt.get("valid") is not True:
        raise RuntimeError("invalid frozen lock receipt")
    if receipt.get("target_season")!=SEASON or receipt.get("target_week")!=WEEK:
        raise RuntimeError("lock target drift")
    if receipt.get("locked_rows")!=46:
        raise RuntimeError(f"expected 46 locked rows, got {receipt.get('locked_rows')}")
    if receipt.get("outcome_present_at_lock") is not False:
        raise RuntimeError("target outcome present at lock")
    if receipt.get("sportsbook_inputs_used_in_candidate") is not False:
        raise RuntimeError("sportsbook input contamination")
    if receipt.get("production_changed") is not False:
        raise RuntimeError("production mutation detected")

    rows=[json.loads(line) for line in a.locks.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=46:
        raise RuntimeError(f"lock jsonl rows={len(rows)} expected=46")
    _assert_games_final(rows)
    actual=_attach_outcomes(rows)
    amap={(r.event_id,r.team,r.player_clean_key):r.actual_rush_yards for r in actual.itertuples(index=False)}
    smap={(r.event_id,r.team,r.player_clean_key):r.actual_source for r in actual.itertuples(index=False)}

    npz=np.load(a.arrays,allow_pickle=False)
    detail_rows=[]
    max_mean_gap=0.0
    for r in rows:
        key=(str(r["event_id"]),canon_team(str(r["team"])),str(r["player_clean_key"]))
        if key not in amap:
            raise RuntimeError(f"actual missing after attachment: {key}")
        actual_y=float(amap[key])
        b=np.asarray(npz[str(r["baseline_array_key"])],dtype=float)
        c=np.asarray(npz[str(r["candidate_array_key"])],dtype=float)
        if len(b)!=25000 or len(c)!=25000:
            raise RuntimeError(f"draw count drift {key}: {len(b)} {len(c)}")
        bmean=float(b.mean()); cmean=float(c.mean())
        gap=abs(bmean-cmean); max_mean_gap=max(max_mean_gap,gap)
        br=_empirical_crps(b,actual_y); cr=_empirical_crps(c,actual_y)
        rec={
            "season":SEASON,"week":WEEK,
            "event_id":key[0],"team":key[1],"opponent":canon_team(str(r["opponent"])),
            "player":str(r["player"]),"player_clean_key":key[2],
            "difficulty_score":float(r["difficulty_score"]),
            "width_multiplier":float(r["width_multiplier"]),
            "prior_games":int(r["prior_games"]),
            "prior8_yard_mae":float(r["prior8_yard_mae"]),
            "actual_rush_yards":actual_y,
            "actual_source":smap[key],
            "baseline_mean":bmean,"candidate_mean":cmean,
            "mean_abs_gap":gap,
            "baseline_crps":br,"candidate_crps":cr,
            "crps_gain":br-cr,
            "candidate_minus_baseline_crps":cr-br,
            "baseline_cover80":_coverage(b,actual_y,.10,.90),
            "candidate_cover80":_coverage(c,actual_y,.10,.90),
            "baseline_cover90":_coverage(b,actual_y,.05,.95),
            "candidate_cover90":_coverage(c,actual_y,.05,.95),
            "baseline_point_abs_error":abs(bmean-actual_y),
            "candidate_point_abs_error":abs(cmean-actual_y),
        }
        for t in THRESHOLDS:
            tag=int(t)
            rec[f"baseline_brier_ge{tag}"]=_brier(b,actual_y,t)
            rec[f"candidate_brier_ge{tag}"]=_brier(c,actual_y,t)
        detail_rows.append(rec)
    detail=pd.DataFrame(detail_rows)
    if max_mean_gap>1e-8:
        raise RuntimeError(f"mean-neutrality violated max_gap={max_mean_gap}")
    point_mae_gap=abs(detail["baseline_point_abs_error"].mean()-detail["candidate_point_abs_error"].mean())
    if point_mae_gap>1e-8:
        raise RuntimeError(f"point-MAE neutrality violated gap={point_mae_gap}")

    baseline=_summarize(detail,"baseline")
    candidate=_summarize(detail,"candidate")

    q75=float(np.quantile(detail["difficulty_score"],.75,method="linear"))
    high=detail.loc[detail["difficulty_score"].ge(q75)].copy()
    high_baseline=_summarize(high,"baseline")
    high_candidate=_summarize(high,"candidate")

    game_boot=_game_cluster_bootstrap(detail)
    crossed=_crossed_bootstrap(detail)

    guardrails={
        "pooled_80_gap_nonworse":candidate["coverage80_abs_gap"]<=baseline["coverage80_abs_gap"]+1e-15,
        "pooled_90_gap_nonworse":candidate["coverage90_abs_gap"]<=baseline["coverage90_abs_gap"]+1e-15,
        "pooled_at_least_one_gap_strictly_better":(
            candidate["coverage80_abs_gap"]<baseline["coverage80_abs_gap"]-1e-15
            or candidate["coverage90_abs_gap"]<baseline["coverage90_abs_gap"]-1e-15
        ),
        "high_q75_threshold":q75,
        "high_rows":int(len(high)),
        "high_crps_strictly_better":high_candidate["mean_crps"]<high_baseline["mean_crps"],
        "high_80_gap_nonworse":high_candidate["coverage80_abs_gap"]<=high_baseline["coverage80_abs_gap"]+1e-15,
        "high_90_gap_nonworse":high_candidate["coverage90_abs_gap"]<=high_baseline["coverage90_abs_gap"]+1e-15,
        "high_at_least_one_gap_strictly_better":(
            high_candidate["coverage80_abs_gap"]<high_baseline["coverage80_abs_gap"]-1e-15
            or high_candidate["coverage90_abs_gap"]<high_baseline["coverage90_abs_gap"]-1e-15
        ),
        "brier_ge100_strictly_better":candidate["brier_ge100"]<baseline["brier_ge100"],
        "brier_ge50_nonworse":candidate["brier_ge50"]<=baseline["brier_ge50"]+1e-15,
        "brier_ge75_nonworse":candidate["brier_ge75"]<=baseline["brier_ge75"]+1e-15,
    }

    support={
        "distinct_locked_weeks":1,
        "unique_eligible_player_games":int(len(detail)),
        "required_locked_weeks":8,
        "required_unique_player_games":400,
        "sufficient":False,
    }
    result={
        "status":"RB_PD2_WEEK3_OBSERVATION_ATTACHED",
        "interim_state":"SUPPORT_ACCUMULATING_OBSERVATION_WEEK_1",
        "frozen_scientific_disposition_issued":None,
        "support":support,
        "baseline":baseline,
        "candidate":candidate,
        "pooled_crps_gain_baseline_minus_candidate":float(detail["crps_gain"].mean()),
        "game_cluster_bootstrap":game_boot,
        "crossed_player_game_bootstrap":crossed,
        "guardrails_descriptive_only_until_support_floor":guardrails,
        "high_difficulty_baseline":high_baseline,
        "high_difficulty_candidate":high_candidate,
        "max_rowwise_mean_gap":max_mean_gap,
        "pooled_point_mae_difference_abs":point_mae_gap,
        "sportsbook_inputs_used":0,
        "production_changed":False,
        "scientific_pass_fail_issued":False,
        "season_end_insufficient_support_disposition_applicable":False,
    }

    a.out_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.out_dir/"rb_pd2_week3_observation_detail.csv",index=False)
    (a.out_dir/"rb_pd2_week3_observation_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
