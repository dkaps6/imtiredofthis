#!/usr/bin/env python3
"""PRISM Restored-Return Injury-Report State Persistence V1.

Named-player health-state/reliability test only. After source-certified restored
participation, tests whether weekly injury-report presence at the return
distinguishes the immediately next clean no-report participation state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1"
FREEZE_COMMIT = "513a5e5ee29b909aeaff387396352e1a10d83633"
SEASONS = (2024, 2025)
FAMILIES = {"QB","RB","WR","TE"}
RETURN_RESTORED = "RETURN_PARTICIPATION_RESTORED"
STATE_REPORT = "RESTORED_RETURN_REPORT_PRESENT"
STATE_NO_REPORT = "RESTORED_RETURN_NO_REPORT"
NEXT_CLEAN = "NO_REPORT_PARTICIPATED"
ALLOWED_NEXT = {
    "NO_REPORT_PARTICIPATED",
    "REPORTED_INJURY_PARTICIPATED",
    "REPORTED_INJURY_NO_OFFENSIVE_SNAPS",
    "NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED",
}
N_BOOT = 5000
BOOT_SEED = 20261018

EP_REQ = {
    "season","team","stable_id","family","episode_number",
    "episode_start_week","return_week","return_state","return_injury_reported",
}
HEALTH_REQ = {
    "season","week","team","stable_id","family","positive_participation",
    "injury_reported","observation_state",
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


def load_episodes(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    miss=EP_REQ-set(x.columns)
    if miss:
        raise RuntimeError(f"episode source missing {sorted(miss)}")
    for c in ("season","episode_number","episode_start_week","return_week"):
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x["stable_id"]=x["stable_id"].map(text)
    x["team"]=x["team"].astype("string").fillna("").str.strip()
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["return_state"]=x["return_state"].astype("string").fillna("").str.strip()
    x["return_injury_reported"]=as_bool(x["return_injury_reported"])
    x=x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["stable_id"].ne("")
        & x["return_state"].eq(RETURN_RESTORED)
    ].copy()
    x["season"]=x["season"].astype(int)
    x["episode_number"]=x["episode_number"].astype(int)
    x["episode_start_week"]=x["episode_start_week"].astype(int)
    return x.sort_values(
        ["season","team","stable_id","episode_start_week","episode_number"],
        kind="mergesort"
    ).reset_index(drop=True)


def load_health(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    miss=HEALTH_REQ-set(x.columns)
    if miss:
        raise RuntimeError(f"health source missing {sorted(miss)}")
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["stable_id"]=x["stable_id"].map(text)
    x["team"]=x["team"].astype("string").fillna("").str.strip()
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["positive_participation"]=as_bool(x["positive_participation"])
    x["injury_reported"]=as_bool(x["injury_reported"])
    x["observation_state"]=x["observation_state"].astype("string").fillna("").str.strip()
    x=x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["stable_id"].ne("")
    ].copy()
    x["season"]=x["season"].astype(int)
    x["week"]=x["week"].astype(int)
    keys=["season","week","team","stable_id"]
    if x.duplicated(keys).any():
        raise RuntimeError("duplicate health identities")
    return x.sort_values(
        ["season","team","stable_id","week"],kind="mergesort"
    ).reset_index(drop=True)


def build_rows(
    episodes: pd.DataFrame,
    health: pd.DataFrame,
) -> tuple[pd.DataFrame,dict]:
    groups={
        (int(s),str(t),str(pid)):g.sort_values("week",kind="mergesort").reset_index(drop=True)
        for (s,t,pid),g in health.groupby(["season","team","stable_id"],sort=False)
    }
    rows=[]
    chronology_violations=0
    return_health_missing_or_nonunique=0
    return_report_mismatches=0
    skipped_next_violations=0
    no_next=0

    for e in episodes.to_dict("records"):
        season=int(e["season"])
        team=str(e["team"])
        pid=str(e["stable_id"])
        family=str(e["family"])
        start_week=int(e["episode_start_week"])
        if pd.isna(e["return_week"]):
            chronology_violations+=1
            continue
        return_week=int(e["return_week"])
        if return_week<=start_week:
            chronology_violations+=1
            continue

        g=groups.get((season,team,pid))
        if g is None:
            return_health_missing_or_nonunique+=1
            continue

        ret=g.loc[g["week"].eq(return_week)]
        if len(ret)!=1:
            return_health_missing_or_nonunique+=1
            continue
        rr=ret.iloc[0]
        if not bool(rr["positive_participation"]):
            return_health_missing_or_nonunique+=1
            continue

        parent_report=bool(e["return_injury_reported"])
        health_report=bool(rr["injury_reported"])
        if parent_report != health_report:
            return_report_mismatches+=1
            continue

        current_state=STATE_REPORT if parent_report else STATE_NO_REPORT

        future=g.loc[g["week"]>return_week].sort_values("week",kind="mergesort")
        if future.empty:
            no_next+=1
            continue
        nr=future.iloc[0]
        next_week=int(nr["week"])
        if next_week<=return_week:
            chronology_violations+=1
            continue
        first_later=int(g.loc[g["week"]>return_week,"week"].min())
        if next_week!=first_later:
            skipped_next_violations+=1

        next_state=str(nr["observation_state"])
        if next_state not in ALLOWED_NEXT:
            raise RuntimeError(f"unexpected next observation state {next_state!r}")

        rows.append({
            "season":season,
            "team":team,
            "stable_id":pid,
            "family":family,
            "episode_number":int(e["episode_number"]),
            "episode_start_week":start_week,
            "return_week":return_week,
            "return_injury_reported":parent_report,
            "current_return_report_state":current_state,
            "next_week":next_week,
            "next_observation_state":next_state,
            "next_positive_participation":bool(nr["positive_participation"]),
            "next_injury_reported":bool(nr["injury_reported"]),
            "next_clean_no_report_participation":int(next_state==NEXT_CLEAN),
        })

    out=pd.DataFrame(rows)
    if not out.empty:
        out=out.sort_values(
            ["season","next_week","team","stable_id","episode_number"],
            kind="mergesort"
        ).reset_index(drop=True)

    audit={
        "restored_source_episodes":int(len(episodes)),
        "scored_episodes":int(len(out)),
        "no_next_observation":int(no_next),
        "chronology_violations":int(chronology_violations),
        "return_health_missing_or_nonunique":int(return_health_missing_or_nonunique),
        "return_report_mismatches":int(return_report_mismatches),
        "skipped_immediate_next_violations":int(skipped_next_violations),
    }
    return out,audit


def state_player_means(z: pd.DataFrame, state: str) -> pd.Series:
    q=z.loc[z["current_return_report_state"].eq(state)]
    return q.groupby("stable_id")["next_clean_no_report_participation"].mean().astype(float)


def bootstrap_difference(z: pd.DataFrame) -> dict:
    players=sorted(z["stable_id"].unique())
    if len(players)<2:
        return {"requested_reps":N_BOOT,"valid_reps":0,"ci_low":np.nan,"ci_high":np.nan}

    per={}
    for pid,g in z.groupby("stable_id",sort=False):
        per[str(pid)]={}
        for state,h in g.groupby("current_return_report_state",sort=False):
            per[str(pid)][str(state)]=float(h["next_clean_no_report_participation"].mean())

    rng=np.random.default_rng(BOOT_SEED)
    vals=[]
    for _ in range(N_BOOT):
        sample=rng.choice(players,size=len(players),replace=True)
        no_report=[]
        report=[]
        for pid in sample:
            d=per[str(pid)]
            if STATE_NO_REPORT in d:
                no_report.append(d[STATE_NO_REPORT])
            if STATE_REPORT in d:
                report.append(d[STATE_REPORT])
        if not no_report or not report:
            continue
        vals.append(float(np.mean(no_report)-np.mean(report)))

    if not vals:
        return {"requested_reps":N_BOOT,"valid_reps":0,"ci_low":np.nan,"ci_high":np.nan}
    a=np.asarray(vals,dtype=float)
    return {
        "requested_reps":N_BOOT,
        "valid_reps":int(len(a)),
        "ci_low":float(np.quantile(a,0.025)),
        "ci_high":float(np.quantile(a,0.975)),
    }


def summarize(
    rows: pd.DataFrame,
    episodes: pd.DataFrame,
    season: int,
    bootstrap: bool,
    audit: dict,
    duplicate_episode_identities: int,
) -> dict:
    z=rows.loc[rows["season"].eq(season)].copy()
    e=episodes.loc[episodes["season"].eq(season)].copy()

    p_no=state_player_means(z,STATE_NO_REPORT)
    p_yes=state_player_means(z,STATE_REPORT)

    no_rate=float(p_no.mean()) if len(p_no) else np.nan
    yes_rate=float(p_yes.mean()) if len(p_yes) else np.nan
    gain=float(no_rate-yes_rate) if np.isfinite(no_rate) and np.isfinite(yes_rate) else np.nan

    ep_no=z["current_return_report_state"].eq(STATE_NO_REPORT)
    ep_yes=z["current_return_report_state"].eq(STATE_REPORT)

    fam=[]
    for family,g in z.groupby("family",sort=True):
        a=state_player_means(g,STATE_NO_REPORT)
        b=state_player_means(g,STATE_REPORT)
        fam.append({
            "family":str(family),
            "scored_episodes":int(len(g)),
            "named_players":int(g["stable_id"].nunique()),
            "no_report_episodes":int(g["current_return_report_state"].eq(STATE_NO_REPORT).sum()),
            "report_present_episodes":int(g["current_return_report_state"].eq(STATE_REPORT).sum()),
            "no_report_players":int(len(a)),
            "report_present_players":int(len(b)),
            "no_report_equal_player_clean_rate":float(a.mean()) if len(a) else np.nan,
            "report_present_equal_player_clean_rate":float(b.mean()) if len(b) else np.nan,
        })

    source_count=int(len(e))
    next_coverage=float(len(z)/source_count) if source_count else 0.0

    gates={
        "scored_episodes_ge_120":len(z)>=120,
        "scored_named_players_ge_100":z["stable_id"].nunique()>=100,
        "report_present_episodes_ge_30":int(ep_yes.sum())>=30,
        "report_present_named_players_ge_25":len(p_yes)>=25,
        "no_report_episodes_ge_30":int(ep_no.sum())>=30,
        "no_report_named_players_ge_25":len(p_no)>=25,
        "immediate_next_coverage_ge_075":next_coverage>=0.75,
        "stable_id_coverage_eq_100":bool(z["stable_id"].ne("").all()) if len(z) else False,
        "duplicate_episode_identities_zero":duplicate_episode_identities==0,
        "return_report_mismatches_zero":audit["return_report_mismatches"]==0,
        "chronology_violations_zero":audit["chronology_violations"]==0,
        "skipped_immediate_next_zero":audit["skipped_immediate_next_violations"]==0,
        "next_state_vocabulary_valid":bool(
            set(z["next_observation_state"].unique())<=ALLOWED_NEXT
        ) if len(z) else False,
    }

    out={
        "season":season,
        "restored_source_episodes":source_count,
        "scored_episodes":int(len(z)),
        "scored_named_players":int(z["stable_id"].nunique()),
        "immediate_next_coverage":next_coverage,
        "no_report_episodes":int(ep_no.sum()),
        "no_report_named_players":int(len(p_no)),
        "report_present_episodes":int(ep_yes.sum()),
        "report_present_named_players":int(len(p_yes)),
        "no_report_equal_player_clean_rate":no_rate,
        "report_present_equal_player_clean_rate":yes_rate,
        "report_state_clean_rate_gain":gain,
        "next_state_counts":{
            str(k):int(v)
            for k,v in z["next_observation_state"].value_counts().to_dict().items()
        },
        "family_diagnostics":fam,
        "support_gates":gates,
    }
    if bootstrap:
        out["player_cluster_bootstrap"]=bootstrap_difference(z)
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes",type=Path,required=True)
    ap.add_argument("--health",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    episodes=load_episodes(a.episodes)
    health=load_health(a.health)
    duplicate_episode_identities=int(
        episodes.duplicated(["season","team","stable_id","episode_number"]).sum()
    )
    rows,audit=build_rows(episodes,health)

    dev=summarize(rows,episodes,2024,False,audit,duplicate_episode_identities)
    conf=summarize(rows,episodes,2025,True,audit,duplicate_episode_identities)

    qualification={
        "development_gain_positive":bool(
            np.isfinite(dev["report_state_clean_rate_gain"])
            and dev["report_state_clean_rate_gain"]>0
        ),
        "development_no_report_rate_gt_report_present":bool(
            np.isfinite(dev["no_report_equal_player_clean_rate"])
            and np.isfinite(dev["report_present_equal_player_clean_rate"])
            and dev["no_report_equal_player_clean_rate"]>
                dev["report_present_equal_player_clean_rate"]
        ),
        "confirmation_gain_positive":bool(
            np.isfinite(conf["report_state_clean_rate_gain"])
            and conf["report_state_clean_rate_gain"]>0
        ),
        "confirmation_no_report_rate_gt_report_present":bool(
            np.isfinite(conf["no_report_equal_player_clean_rate"])
            and np.isfinite(conf["report_present_equal_player_clean_rate"])
            and conf["no_report_equal_player_clean_rate"]>
                conf["report_present_equal_player_clean_rate"]
        ),
        "confirmation_bootstrap_valid_reps_ge_4500":bool(
            conf["player_cluster_bootstrap"]["valid_reps"]>=4500
        ),
        "confirmation_bootstrap_ci_low_positive":bool(
            np.isfinite(conf["player_cluster_bootstrap"]["ci_low"])
            and conf["player_cluster_bootstrap"]["ci_low"]>0
        ),
    }

    integrity={
        "targets_carries_receptions_yards_tds_used_zero":True,
        "routes_inferred_from_snaps_false":True,
        "coefficients_fit_zero":True,
        "thresholds_searched_zero":True,
        "outcomes_2026_zero":True,
        "sportsbook_oddsapi_zero":True,
        "production_changed_false":True,
    }

    qualified=bool(
        all(dev["support_gates"].values())
        and all(conf["support_gates"].values())
        and all(qualification.values())
        and all(integrity.values())
    )

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "return_parent_run":39162437467,
        "return_parent_artifact":12149417234,
        "return_parent_digest":"sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e",
        "health_parent_run":39154145666,
        "health_parent_artifact":12146625382,
        "health_parent_digest":"sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "disposition":(
            "PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_QUALIFIED_RELIABILITY_ONLY"
            if qualified else
            "PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_CLOSED"
        ),
        "qualified_reliability_only":qualified,
        "audit":audit,
        "development_2024":dev,
        "confirmation_2025":conf,
        "qualification_gates":qualification,
        "integrity_gates":integrity,
        "bootstrap_replicates":N_BOOT,
        "bootstrap_seed":BOOT_SEED,
        "targets_used":0,
        "carries_used":0,
        "receptions_used":0,
        "yards_used":0,
        "touchdowns_used":0,
        "routes_inferred_from_snaps":False,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    rows.to_csv(
        a.out_dir/"restored_return_injury_report_state_persistence_rows.csv",
        index=False
    )
    (a.out_dir/"restored_return_injury_report_state_persistence_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
