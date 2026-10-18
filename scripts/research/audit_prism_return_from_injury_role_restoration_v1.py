#!/usr/bin/env python3
"""PRISM Return-from-Injury Role Restoration V1 source audit.

Builds named-player injury-interruption episodes and classifies the first later
positive-participation observation against the player's own pre-interruption
snap baseline. No target/carry/yard/TD performance scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "3eafb971c0a842750fa5af31a77a8b17adf890b0"
SEASONS = (2024, 2025)
FAMILIES = {"QB", "RB", "WR", "TE"}
MAX_PRIOR = 4
MIN_PRIOR = 3
RETENTION_MAX = 0.65
ABS_DROP_MIN = 0.20
TOL = 1e-10

STATE_COLLAPSE = "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED"
OBS_INJ_NO_SNAP = "REPORTED_INJURY_NO_OFFENSIVE_SNAPS"
RETURN_RESTORED = "RETURN_PARTICIPATION_RESTORED"
RETURN_REDUCED = "RETURN_PARTICIPATION_STILL_REDUCED"
NO_RETURN = "NO_RETURN_OBSERVED"

HEALTH_REQ = {
    "season","week","team","stable_id","family","positive_participation",
    "offense_snaps","offense_pct","injury_reported","observation_state",
}
ANOM_REQ = {
    "season","week","team","stable_id","participation_anomaly_state",
    "player_prior_snap_pct",
}


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s=str(v).strip()
    return "" if s.upper() in {"","NAN","NONE","<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true":True,"false":False,"1":True,"0":False})
        .fillna(False).astype(bool)
    )


def load_health(path: Path) -> tuple[pd.DataFrame,float]:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    miss=HEALTH_REQ-set(x.columns)
    if miss:
        raise RuntimeError(f"health parent missing {sorted(miss)}")

    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["stable_id"]=x["stable_id"].map(text)
    x["team"]=x["team"].astype("string").fillna("").str.strip()
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["positive_participation"]=as_bool(x["positive_participation"])
    x["injury_reported"]=as_bool(x["injury_reported"])
    x["offense_snaps"]=pd.to_numeric(x["offense_snaps"],errors="coerce")
    x["offense_pct"]=pd.to_numeric(x["offense_pct"],errors="coerce")

    x=x.loc[x["season"].isin(SEASONS) & x["family"].isin(FAMILIES)].copy()
    x["season"]=x["season"].astype(int)
    x["week"]=x["week"].astype(int)

    stable_cov=float(x["stable_id"].ne("").mean()) if len(x) else 0.0
    nonblank=x["stable_id"].ne("")
    keys=["season","week","team","stable_id"]
    if x.loc[nonblank].duplicated(keys).any():
        bad=x.loc[nonblank & x.duplicated(keys,keep=False),keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate health identities: {bad}")

    return x.sort_values(
        ["season","team","stable_id","week"],kind="mergesort"
    ).reset_index(drop=True),stable_cov


def load_anomaly(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    miss=ANOM_REQ-set(x.columns)
    if miss:
        raise RuntimeError(f"anomaly parent missing {sorted(miss)}")
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["stable_id"]=x["stable_id"].map(text)
    x["team"]=x["team"].astype("string").fillna("").str.strip()
    x["player_prior_snap_pct"]=pd.to_numeric(x["player_prior_snap_pct"],errors="coerce")
    x=x.loc[x["season"].isin(SEASONS)].copy()
    x["season"]=x["season"].astype(int)
    x["week"]=x["week"].astype(int)
    nonblank=x["stable_id"].ne("")
    keys=["season","week","team","stable_id"]
    if x.loc[nonblank].duplicated(keys).any():
        raise RuntimeError("duplicate anomaly identities")
    return x


def build_episodes(
    health: pd.DataFrame,
    anomaly: pd.DataFrame,
) -> tuple[pd.DataFrame,dict]:
    keys=["season","week","team","stable_id"]
    amap={
        tuple(r[k] for k in keys):r
        for r in anomaly.to_dict("records")
        if str(r["stable_id"])
    }

    episodes=[]
    baseline_crosscheck_gaps=[]
    chronology_violations=0
    overlap_violations=0

    source=health.loc[health["stable_id"].ne("")].copy()
    for (season,team,pid),g in source.groupby(
        ["season","team","stable_id"],sort=False
    ):
        g=g.sort_values("week",kind="mergesort").reset_index(drop=True)
        recs=g.to_dict("records")
        i=0
        episode_number=0

        while i < len(recs):
            r=recs[i]
            key=(int(season),int(r["week"]),str(team),str(pid))
            ar=amap.get(key)

            start_collapse=bool(
                ar is not None
                and str(ar.get("participation_anomaly_state","")) == STATE_COLLAPSE
            )
            start_no_snap=bool(str(r["observation_state"]) == OBS_INJ_NO_SNAP)

            if not (start_collapse or start_no_snap):
                i+=1
                continue

            prior=[
                q for q in recs[:i]
                if bool(q["positive_participation"])
                and pd.notna(q["offense_pct"])
                and float(q["offense_pct"])>0
            ][-MAX_PRIOR:]

            if len(prior)<MIN_PRIOR:
                i+=1
                continue

            if not all(int(q["week"]) < int(r["week"]) for q in prior):
                chronology_violations+=1
                i+=1
                continue

            baseline=float(np.median([float(q["offense_pct"]) for q in prior]))
            if not np.isfinite(baseline) or baseline<=0:
                raise RuntimeError("invalid pre-episode snap baseline")

            cross_gap=np.nan
            if start_collapse and ar is not None and pd.notna(ar.get("player_prior_snap_pct")):
                cross_gap=abs(float(ar["player_prior_snap_pct"])-baseline)
                baseline_crosscheck_gaps.append(cross_gap)

            j=i+1
            while j<len(recs) and not bool(recs[j]["positive_participation"]):
                j+=1

            episode_number+=1
            start_type=(
                "INJURY_CORROBORATED_PARTICIPATION_COLLAPSE"
                if start_collapse else
                "REPORTED_INJURY_NO_OFFENSIVE_SNAPS"
            )

            between=recs[i+1:j] if j<=len(recs) else []
            no_part_between=sum(not bool(q["positive_participation"]) for q in between)
            inj_no_part_between=sum(
                (not bool(q["positive_participation"])) and bool(q["injury_reported"])
                for q in between
            )

            base={
                "season":int(season),
                "team":str(team),
                "stable_id":str(pid),
                "family":str(r["family"]),
                "episode_number":int(episode_number),
                "episode_start_week":int(r["week"]),
                "episode_start_type":start_type,
                "episode_start_injury_reported":bool(r["injury_reported"]),
                "episode_start_positive_participation":bool(r["positive_participation"]),
                "pre_episode_positive_rows":int(len(prior)),
                "pre_episode_snap_pct":baseline,
                "parent_baseline_crosscheck_gap":cross_gap,
                "intervening_no_participation_rows":int(no_part_between),
                "intervening_injury_reported_no_participation_rows":int(inj_no_part_between),
            }

            if j>=len(recs):
                base.update({
                    "return_week":np.nan,
                    "return_injury_reported":False,
                    "return_offense_snaps":np.nan,
                    "return_offense_pct":np.nan,
                    "return_retention_ratio":np.nan,
                    "return_absolute_drop":np.nan,
                    "return_state":NO_RETURN,
                })
                episodes.append(base)
                break

            ret=recs[j]
            if int(ret["week"])<=int(r["week"]):
                chronology_violations+=1

            ret_pct=float(ret["offense_pct"]) if pd.notna(ret["offense_pct"]) else np.nan
            ret_snaps=float(ret["offense_snaps"]) if pd.notna(ret["offense_snaps"]) else np.nan
            if not np.isfinite(ret_pct) or ret_pct<=0:
                raise RuntimeError("positive return row has invalid offense_pct")
            retention=float(ret_pct/baseline)
            drop=float(baseline-ret_pct)
            still_reduced=bool(retention<=RETENTION_MAX and drop>=ABS_DROP_MIN)
            rstate=RETURN_REDUCED if still_reduced else RETURN_RESTORED

            base.update({
                "return_week":int(ret["week"]),
                "return_injury_reported":bool(ret["injury_reported"]),
                "return_offense_snaps":ret_snaps,
                "return_offense_pct":ret_pct,
                "return_retention_ratio":retention,
                "return_absolute_drop":drop,
                "return_state":rstate,
            })
            episodes.append(base)

            # End this episode at the first subsequent positive-participation row.
            # That return row cannot also become an overlapping start for the same
            # interruption.
            i=j+1

    out=pd.DataFrame(episodes)
    if out.empty:
        raise RuntimeError("zero supported injury-interruption episodes")

    if out.duplicated(["season","team","stable_id","episode_number"]).any():
        overlap_violations=int(
            out.duplicated(["season","team","stable_id","episode_number"]).sum()
        )

    audit={
        "chronology_violations":int(chronology_violations),
        "overlapping_episode_identity_violations":int(overlap_violations),
        "baseline_crosscheck_rows":int(len(baseline_crosscheck_gaps)),
        "baseline_crosscheck_max_abs_gap":(
            float(max(baseline_crosscheck_gaps))
            if baseline_crosscheck_gaps else 0.0
        ),
    }
    return out,audit


def summarize(ep: pd.DataFrame, season: int, stable_cov: float) -> dict:
    s=ep.loc[ep["season"].eq(season)].copy()
    restored=s["return_state"].eq(RETURN_RESTORED)
    reduced=s["return_state"].eq(RETURN_REDUCED)
    no_return=s["return_state"].eq(NO_RETURN)
    returned=s.loc[~no_return]

    fam=[]
    for family,g in s.groupby("family",sort=True):
        fam.append({
            "family":str(family),
            "episodes":int(len(g)),
            "players":int(g["stable_id"].nunique()),
            "restored_return_episodes":int(g["return_state"].eq(RETURN_RESTORED).sum()),
            "still_reduced_return_episodes":int(g["return_state"].eq(RETURN_REDUCED).sum()),
            "no_return_episodes":int(g["return_state"].eq(NO_RETURN).sum()),
            "median_pre_episode_snap_pct":float(g["pre_episode_snap_pct"].median()),
            "median_restored_return_snap_pct":(
                float(g.loc[g["return_state"].eq(RETURN_RESTORED),"return_offense_pct"].median())
                if g["return_state"].eq(RETURN_RESTORED).any() else np.nan
            ),
            "median_still_reduced_return_snap_pct":(
                float(g.loc[g["return_state"].eq(RETURN_REDUCED),"return_offense_pct"].median())
                if g["return_state"].eq(RETURN_REDUCED).any() else np.nan
            ),
        })

    gates={
        "stable_id_coverage_ge_099":bool(stable_cov>=0.99),
        "supported_episodes_ge_200":bool(len(s)>=200),
        "supported_named_players_ge_150":bool(s["stable_id"].nunique()>=150),
        "restored_return_episodes_ge_100":bool(restored.sum()>=100),
        "restored_return_named_players_ge_100":bool(
            s.loc[restored,"stable_id"].nunique()>=100
        ),
        "still_reduced_return_episodes_ge_25":bool(reduced.sum()>=25),
        "no_return_episodes_nonzero":bool(no_return.sum()>0),
        "baseline_support_ge_three_all":bool(s["pre_episode_positive_rows"].ge(3).all()),
        "baseline_snap_pct_finite_positive_all":bool(
            np.isfinite(s["pre_episode_snap_pct"]).all()
            and s["pre_episode_snap_pct"].gt(0).all()
        ),
        "return_week_after_start_all":bool(
            (
                no_return
                | (pd.to_numeric(s["return_week"],errors="coerce") > s["episode_start_week"])
            ).all()
        ),
        "return_offense_pct_finite_positive_all":bool(
            np.isfinite(returned["return_offense_pct"]).all()
            and returned["return_offense_pct"].gt(0).all()
        ),
        "return_retention_finite_nonnegative_all":bool(
            np.isfinite(returned["return_retention_ratio"]).all()
            and returned["return_retention_ratio"].ge(0).all()
        ),
    }

    return {
        "season":int(season),
        "episodes":int(len(s)),
        "players":int(s["stable_id"].nunique()),
        "restored_return_episodes":int(restored.sum()),
        "restored_return_players":int(s.loc[restored,"stable_id"].nunique()),
        "still_reduced_return_episodes":int(reduced.sum()),
        "still_reduced_return_players":int(s.loc[reduced,"stable_id"].nunique()),
        "no_return_episodes":int(no_return.sum()),
        "no_return_players":int(s.loc[no_return,"stable_id"].nunique()),
        "median_pre_episode_snap_pct":float(s["pre_episode_snap_pct"].median()),
        "median_restored_return_snap_pct":float(
            s.loc[restored,"return_offense_pct"].median()
        ) if restored.any() else np.nan,
        "median_still_reduced_return_snap_pct":float(
            s.loc[reduced,"return_offense_pct"].median()
        ) if reduced.any() else np.nan,
        "median_intervening_no_participation_rows":float(
            s["intervening_no_participation_rows"].median()
        ),
        "returned_still_injury_reported_share":float(
            returned["return_injury_reported"].mean()
        ) if len(returned) else np.nan,
        "family_summary":fam,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--health-parent",type=Path,required=True)
    ap.add_argument("--anomaly-parent",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    health,stable_cov=load_health(a.health_parent)
    anomaly=load_anomaly(a.anomaly_parent)
    ep,audit=build_episodes(health,anomaly)
    summaries=[summarize(ep,s,stable_cov) for s in SEASONS]

    integrity={
        "chronology_violations_zero":audit["chronology_violations"]==0,
        "overlap_violations_zero":audit["overlapping_episode_identity_violations"]==0,
        "baseline_crosscheck_pass":audit["baseline_crosscheck_max_abs_gap"]<=TOL,
    }
    ready=bool(all(s["source_ready"] for s in summaries) and all(integrity.values()))

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "health_parent_run":39154145666,
        "health_parent_artifact":12146625382,
        "health_parent_digest":"sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "anomaly_parent_run":39160554692,
        "anomaly_parent_artifact":12147549966,
        "anomaly_parent_digest":"sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449",
        "disposition":(
            "PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY"
            if ready else
            "PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_BLOCKED"
        ),
        "prior_window_max":MAX_PRIOR,
        "prior_window_min":MIN_PRIOR,
        "snap_retention_max":RETENTION_MAX,
        "absolute_snap_drop_min":ABS_DROP_MIN,
        "stable_id_coverage":stable_cov,
        "audit":audit,
        "integrity_gates":integrity,
        "season_summaries":summaries,
        "performance_fields_used":0,
        "predictive_relationships_scored":0,
        "projections_generated":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    ep.to_csv(
        a.out_dir/"return_from_injury_role_restoration_episodes.csv",
        index=False,float_format="%.12f"
    )
    pd.DataFrame([
        dict(season=s["season"],**f)
        for s in summaries for f in s["family_summary"]
    ]).to_csv(
        a.out_dir/"return_from_injury_role_restoration_family_summary.csv",
        index=False
    )
    (a.out_dir/"return_from_injury_role_restoration_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
