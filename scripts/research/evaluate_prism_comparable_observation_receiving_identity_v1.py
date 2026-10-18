#!/usr/bin/env python3
"""PRISM Comparable-Observation Receiving Opportunity Identity V1.

Player-first historical test of current-season PlayerForm target-share identity.
The candidate prevents legally corroborated injury-partial observations from
updating ordinary current-season receiving-role history. Position is not the
candidate rule. No sportsbook, 2026 outcomes, or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
import scripts.player_form_v2 as pf

VERSION = "PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1"
FREEZE_COMMIT = "43e1c81a177e47118415d02fc072758e22f50825"
SEASONS = (2023, 2024, 2025)
TARGET_SEASONS = (2024, 2025)
FAMILIES = {"RB", "WR", "TE"}
EXCLUDE_HISTORY_STATE = "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED"
ABNORMAL_TARGET_STATES = {
    "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
    "PARTICIPATION_COLLAPSE_UNRESOLVED",
}
N_BOOT = 5000
BOOT_SEED = 20261018
TOL = 1e-12

REQ_ANOM = {
    "season", "week", "team", "stable_id", "family",
    "participation_anomaly_state", "corroboration_week",
}
REQ_HEALTH = {
    "season", "week", "team", "stable_id", "family",
    "positive_participation",
}


def pdx(v):
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def _bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def load_anomaly(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQ_ANOM - set(x.columns)
    if miss:
        raise RuntimeError(f"anomaly parent missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["participation_anomaly_state"] = (
        x["participation_anomaly_state"].astype("string").fillna("").str.strip()
    )
    x["corroboration_week"] = pd.to_numeric(x["corroboration_week"], errors="coerce")
    x = x.loc[x["season"].isin(TARGET_SEASONS) & x["family"].isin(FAMILIES)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "team", "stable_id"]
    nonblank = x["stable_id"].ne("")
    if x.loc[nonblank].duplicated(keys).any():
        raise RuntimeError("duplicate anomaly player-game identities")
    return x


def load_health(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQ_HEALTH - set(x.columns)
    if miss:
        raise RuntimeError(f"health parent missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["positive_participation"] = _bool(x["positive_participation"])
    x = x.loc[
        x["season"].isin(TARGET_SEASONS)
        & x["family"].isin(FAMILIES)
        & x["positive_participation"]
        & x["stable_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "team", "stable_id"]
    if x.duplicated(keys).any():
        raise RuntimeError("duplicate health target identities")
    return x


def load_weekly_ledgers() -> pd.DataFrame:
    import nflreadpy as nfl

    parts = []
    for season in SEASONS:
        raw = pdx(nfl.load_player_stats(seasons=[int(season)], summary_level="week"))
        if raw.empty:
            raise RuntimeError(f"weekly stats empty for {season}")
        q = pf._normalize_weekly(raw, int(season)).copy()
        q["season"] = int(season)
        q["week"] = pd.to_numeric(q["week"], errors="coerce").astype(int)
        q["team"] = q["team"].map(canon_team)
        q["stable_id"] = q["player_id"].map(text)
        q["targets"] = pd.to_numeric(q["targets"], errors="coerce")
        q["team_targets"] = pd.to_numeric(q["team_targets"], errors="coerce")
        q["family"] = q["position"].astype("string").fillna("").str.upper().str.strip()
        q = q.loc[q["stable_id"].ne("")].copy()
        keep = [
            "season", "week", "team", "stable_id", "family",
            "targets", "team_targets",
        ]
        q = q[keep]
        if q.duplicated(["season","week","team","stable_id"]).any():
            raise RuntimeError(f"duplicate admitted weekly identity {season}")
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def target_team_totals(ledger: pd.DataFrame) -> pd.DataFrame:
    # PlayerForm team_targets is repeated on each admitted player row. Verify
    # consistency and preserve one exact value per team-game.
    rows=[]
    for key,g in ledger.groupby(["season","week","team"], sort=False):
        vals=pd.to_numeric(g["team_targets"],errors="coerce").dropna().unique()
        if len(vals)!=1:
            raise RuntimeError(f"team_targets inconsistency {key}: {vals[:10]}")
        rows.append({
            "season":int(key[0]),"week":int(key[1]),"team":str(key[2]),
            "team_targets":float(vals[0]),
        })
    return pd.DataFrame(rows)


def season_share(rows: pd.DataFrame) -> tuple[float, int]:
    if rows.empty:
        return np.nan, 0
    den=float(pd.to_numeric(rows["team_targets"],errors="coerce").sum())
    num=float(pd.to_numeric(rows["targets"],errors="coerce").sum())
    games=int(pd.to_numeric(rows["week"],errors="coerce").nunique())
    if not np.isfinite(den) or den <= 0:
        return np.nan, games
    return num/den, games


def blend(prior_share: float, current_share: float, current_games: int) -> float:
    p=np.isfinite(prior_share)
    c=np.isfinite(current_share)
    if p and c:
        w=float(current_games)/(float(current_games)+4.0)
        return (1.0-w)*float(prior_share)+w*float(current_share)
    if c:
        return float(current_share)
    if p:
        return float(prior_share)
    return np.nan


def anomaly_lookup(anom: pd.DataFrame) -> dict:
    keys=["season","week","team","stable_id"]
    return {
        tuple(r[k] for k in keys): {
            "state": str(r["participation_anomaly_state"]),
            "corroboration_week": (
                None if pd.isna(r["corroboration_week"])
                else int(r["corroboration_week"])
            ),
        }
        for r in anom.to_dict("records")
        if str(r["stable_id"])
    }


def should_exclude_history_row(
    *,
    row_season: int,
    row_week: int,
    row_team: str,
    stable_id: str,
    target_week: int,
    anomalies: dict,
) -> bool:
    info=anomalies.get((int(row_season),int(row_week),str(row_team),str(stable_id)))
    if not info or info["state"] != EXCLUDE_HISTORY_STATE:
        return False
    cw=info["corroboration_week"]
    return cw is not None and int(cw) < int(target_week)


def target_state(
    *,
    season: int,
    week: int,
    team: str,
    stable_id: str,
    anomalies: dict,
) -> str:
    info=anomalies.get((int(season),int(week),str(team),str(stable_id)))
    return "" if not info else str(info["state"])


def construct_rows(
    *,
    health: pd.DataFrame,
    ledger: pd.DataFrame,
    anomalies: dict,
) -> tuple[pd.DataFrame, dict]:
    team_tot=target_team_totals(ledger)
    team_map={
        (int(r.season),int(r.week),str(r.team)):float(r.team_targets)
        for r in team_tot.itertuples(index=False)
    }
    led_keys=["season","week","team","stable_id"]
    target_value={
        tuple(r[k] for k in led_keys):float(r["targets"])
        for r in ledger.to_dict("records")
    }

    by_player={
        (int(season),str(pid)):g.sort_values(["week","team"],kind="mergesort").copy()
        for (season,pid),g in ledger.groupby(["season","stable_id"],sort=False)
    }

    rows=[]
    chronology_bad=0
    abnormal_target_excluded=0
    target_state_missing=0
    support_missing=0

    for r in health.sort_values(["season","week","team","stable_id"],kind="mergesort").itertuples(index=False):
        season=int(r.season); week=int(r.week); team=str(r.team); pid=str(r.stable_id)
        state=target_state(season=season,week=week,team=team,stable_id=pid,anomalies=anomalies)
        if not state:
            target_state_missing += 1
        if state in ABNORMAL_TARGET_STATES:
            abnormal_target_excluded += 1
            continue

        tt=team_map.get((season,week,team))
        if tt is None or not np.isfinite(tt) or tt<=0:
            support_missing += 1
            continue
        actual_targets=target_value.get((season,week,team,pid),0.0)
        actual_share=float(actual_targets)/float(tt)

        prior_rows=by_player.get((season-1,pid),pd.DataFrame())
        prior_share,_=season_share(prior_rows)

        current_all=by_player.get((season,pid),pd.DataFrame())
        if current_all.empty:
            base_hist=current_all.copy()
        else:
            base_hist=current_all.loc[current_all["week"].lt(week)].copy()
            if not base_hist.empty and not (base_hist["week"] < week).all():
                chronology_bad += 1

        baseline_cur_share,baseline_cur_games=season_share(base_hist)
        baseline=blend(prior_share,baseline_cur_share,baseline_cur_games)

        if base_hist.empty:
            cand_hist=base_hist.copy()
        else:
            keep=[]
            for h in base_hist.itertuples(index=False):
                keep.append(not should_exclude_history_row(
                    row_season=int(h.season), row_week=int(h.week),
                    row_team=str(h.team), stable_id=pid,
                    target_week=week, anomalies=anomalies,
                ))
            cand_hist=base_hist.loc[keep].copy()

        candidate_cur_share,candidate_cur_games=season_share(cand_hist)
        candidate=blend(prior_share,candidate_cur_share,candidate_cur_games)

        if not np.isfinite(baseline) or not np.isfinite(candidate):
            support_missing += 1
            continue

        changed=bool(abs(candidate-baseline)>TOL)
        improvement=float(abs(baseline-actual_share)-abs(candidate-actual_share))

        rows.append({
            "season":season,
            "week":week,
            "team":team,
            "stable_id":pid,
            "family":str(r.family),
            "target_state":state,
            "actual_targets":float(actual_targets),
            "team_targets":float(tt),
            "actual_target_share":actual_share,
            "prior_target_share":prior_share,
            "baseline_current_target_share":baseline_cur_share,
            "baseline_current_games":int(baseline_cur_games),
            "candidate_current_target_share":candidate_cur_share,
            "candidate_current_games":int(candidate_cur_games),
            "baseline_target_share":float(baseline),
            "candidate_target_share":float(candidate),
            "candidate_changed":changed,
            "player_game_improvement":improvement,
            "history_rows_removed":int(len(base_hist)-len(cand_hist)),
        })

    out=pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero scoreable comparable target rows")
    return out,{
        "chronology_violations":int(chronology_bad),
        "abnormal_target_games_excluded":int(abnormal_target_excluded),
        "target_state_missing_rows":int(target_state_missing),
        "support_missing_rows":int(support_missing),
    }


def bootstrap_players(affected: pd.DataFrame) -> dict:
    by=affected.groupby("stable_id")["player_game_improvement"].mean().astype(float)
    vals=by.to_numpy(float)
    if len(vals)<2:
        return {"reps":0,"ci_low":np.nan,"ci_high":np.nan}
    rng=np.random.default_rng(BOOT_SEED)
    boots=[]
    for _ in range(N_BOOT):
        boots.append(float(rng.choice(vals,size=len(vals),replace=True).mean()))
    a=np.asarray(boots,float)
    return {
        "reps":int(len(a)),
        "ci_low":float(np.quantile(a,0.025)),
        "ci_high":float(np.quantile(a,0.975)),
    }


def season_summary(rows: pd.DataFrame, season: int, *, bootstrap: bool) -> dict:
    s=rows.loc[rows["season"].eq(season)].copy()
    affected=s.loc[s["candidate_changed"]].copy()
    clean=s.loc[~s["candidate_changed"]].copy()
    player=affected.groupby("stable_id")["player_game_improvement"].mean().astype(float)

    fam=[]
    for family,g in affected.groupby("family",sort=True):
        pg=g.groupby("stable_id")["player_game_improvement"].mean().astype(float)
        fam.append({
            "family":str(family),
            "affected_target_games":int(len(g)),
            "affected_named_players":int(len(pg)),
            "equal_player_average_improvement":float(pg.mean()) if len(pg) else np.nan,
            "median_player_improvement":float(pg.median()) if len(pg) else np.nan,
            "improved_player_share":float((pg>0).mean()) if len(pg) else np.nan,
            "worsened_player_share":float((pg<0).mean()) if len(pg) else np.nan,
        })

    out={
        "season":int(season),
        "comparable_target_games":int(len(s)),
        "affected_target_games":int(len(affected)),
        "affected_named_players":int(len(player)),
        "equal_player_average_improvement":float(player.mean()) if len(player) else np.nan,
        "median_player_improvement":float(player.median()) if len(player) else np.nan,
        "improved_player_share":float((player>0).mean()) if len(player) else np.nan,
        "tied_player_share":float(np.isclose(player,0.0,atol=1e-15).mean()) if len(player) else np.nan,
        "worsened_player_share":float((player<0).mean()) if len(player) else np.nan,
        "p10_player_improvement":float(player.quantile(0.10)) if len(player) else np.nan,
        "p25_player_improvement":float(player.quantile(0.25)) if len(player) else np.nan,
        "p75_player_improvement":float(player.quantile(0.75)) if len(player) else np.nan,
        "p90_player_improvement":float(player.quantile(0.90)) if len(player) else np.nan,
        "mean_target_game_improvement":float(affected["player_game_improvement"].mean()) if len(affected) else np.nan,
        "clean_unaffected_max_drift":float(
            (clean["candidate_target_share"]-clean["baseline_target_share"]).abs().max()
        ) if len(clean) else 0.0,
        "family_guardrails":fam,
    }
    if bootstrap:
        out["player_cluster_bootstrap"]=bootstrap_players(affected)
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--anomaly-parent",type=Path,required=True)
    ap.add_argument("--health-parent",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)

    anomaly=load_anomaly(a.anomaly_parent)
    health=load_health(a.health_parent)
    ledger=load_weekly_ledgers()
    amap=anomaly_lookup(anomaly)

    rows,audit=construct_rows(health=health,ledger=ledger,anomalies=amap)
    dev=season_summary(rows,2024,bootstrap=False)
    conf=season_summary(rows,2025,bootstrap=True)

    gates={
        "stable_identity_coverage_ge_099":bool(
            health["stable_id"].astype(str).str.strip().ne("").mean()>=0.99
        ),
        "chronology_violations_zero":audit["chronology_violations"]==0,
        "clean_unaffected_drift_zero_2024":dev["clean_unaffected_max_drift"]<=TOL,
        "clean_unaffected_drift_zero_2025":conf["clean_unaffected_max_drift"]<=TOL,
        "primary_target_major_anomaly_rows_zero":True,
        "development_affected_games_nonzero":dev["affected_target_games"]>0,
        "development_affected_players_ge_20":dev["affected_named_players"]>=20,
        "development_equal_player_positive":dev["equal_player_average_improvement"]>0,
        "development_median_player_nonnegative":dev["median_player_improvement"]>=0,
        "development_improved_share_gt_worsened":dev["improved_player_share"]>dev["worsened_player_share"],
        "confirmation_affected_games_nonzero":conf["affected_target_games"]>0,
        "confirmation_affected_players_ge_20":conf["affected_named_players"]>=20,
        "confirmation_equal_player_positive":conf["equal_player_average_improvement"]>0,
        "confirmation_median_player_nonnegative":conf["median_player_improvement"]>=0,
        "confirmation_improved_share_gt_worsened":conf["improved_player_share"]>conf["worsened_player_share"],
        "confirmation_player_cluster_ci_low_positive":conf["player_cluster_bootstrap"]["ci_low"]>0,
        "outcomes_2026_zero":True,
        "sportsbook_oddsapi_zero":True,
        "coefficients_fit_zero":True,
        "thresholds_searched_zero":True,
    }
    qualified=bool(all(gates.values()))

    result={
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "disposition":(
            "PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_QUALIFIED"
            if qualified else
            "PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_CLOSED"
        ),
        "audit":audit,
        "development_2024":dev,
        "confirmation_2025":conf,
        "gates":gates,
        "qualified":qualified,
        "bootstrap_replicates":N_BOOT,
        "bootstrap_seed":BOOT_SEED,
        "anomaly_parent_run":39160554692,
        "anomaly_parent_artifact":12147549966,
        "health_parent_run":39154145666,
        "health_parent_artifact":12146625382,
        "player_form_blob_sha":"9246e4daf19f3f01cf61de3ad036b07873fd9561",
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    rows.to_csv(a.out_dir/"comparable_observation_receiving_identity_rows.csv",index=False,float_format="%.12f")
    (a.out_dir/"comparable_observation_receiving_identity_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
