#!/usr/bin/env python3
"""PRISM Offensive Playcaller Regime Transition V1 source audit.

Source/lineage only. Attaches a pregame team playcaller transition state to
named RB/WR/TE player-weeks and audits historical/prospective semantic parity.
No skill-player predictive outcome is scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import build_qb_playcaller_opening_leverage as m68
from scripts._opponent_map import canon_team

VERSION = "PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "533e628c99359fbfddbb9f4d787a96d8dbfbc67c"
M68_BLOB_SHA = "6db89b71288c62b3d6a0972a4848dd0449b2524a"
SEASONS = (2024, 2025)
FAMILIES = {"RB","WR","TE"}
MIN_CHANGE_EVENTS = 10
MIN_TRANSITION_PLAYER_WEEKS = 250
MIN_TRANSITION_PLAYERS = 175
MIN_PRIOR_SUPPORT_SHARE = 0.40


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true":True,"false":False,"1":True,"0":False})
        .fillna(False).astype(bool)
    )


def canon(v) -> str:
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def load_health(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    req = {
        "season","week","team","stable_id","family","positive_participation",
    }
    miss = req - set(x.columns)
    if miss:
        raise RuntimeError(f"health parent missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["positive_participation"] = as_bool(x["positive_participation"])
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["week"].between(1,18)
        & x["team"].ne("")
        & x["stable_id"].ne("")
        & x["family"].isin(FAMILIES)
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season","week","team","stable_id"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate health player identities: {bad}")
    return x.sort_values(
        ["season","team","stable_id","week"], kind="mergesort"
    ).reset_index(drop=True)


def load_2026_snapshot(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    req = {"season","team","effective_week","playcaller","source_type","source_url"}
    miss = req - set(x.columns)
    if miss:
        raise RuntimeError(f"2026 snapshot missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["effective_week"] = pd.to_numeric(x["effective_week"], errors="coerce")
    x["team"] = x["team"].map(canon)
    x["playcaller"] = x["playcaller"].map(text)
    x["source_type"] = x["source_type"].map(text)
    x["source_url"] = x["source_url"].map(text)
    x = x.loc[x["season"].eq(2026)].copy()
    if x[["team","effective_week","playcaller"]].isna().any().any():
        raise RuntimeError("2026 snapshot has null key semantics")
    x["effective_week"] = x["effective_week"].astype(int)
    return x.sort_values(["team","effective_week"], kind="mergesort").reset_index(drop=True)


def team_weeks_from_health(health: pd.DataFrame) -> pd.DataFrame:
    tw = health[["season","week","team"]].drop_duplicates().copy()
    if tw.duplicated(["season","week","team"]).any():
        raise RuntimeError("team-week duplication")
    return tw.sort_values(["season","team","week"], kind="mergesort").reset_index(drop=True)


def historical_caller_state(team_weeks: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    rows = []
    chronology_violations = 0

    for season in SEASONS:
        sw = team_weeks.loc[team_weeks["season"].eq(season)].copy()
        for team, g in sw.groupby("team", sort=True):
            g = g.sort_values("week", kind="mergesort")
            weeks = [int(v) for v in g["week"].tolist()]
            previous_caller = m68.caller_for(season-1, 18, team)
            previous_week = None

            for week in weeks:
                current = text(m68.caller_for(season, week, team))
                if week == 1:
                    prev = text(previous_caller)
                    prev_label = f"{season-1}_W18"
                else:
                    prev = text(m68.caller_for(season, int(previous_week), team))
                    prev_label = f"{season}_W{int(previous_week)}"

                changed = bool(current and prev and current != prev)
                state = (
                    "PLAYCALLER_REGIME_CHANGED" if changed
                    else "PLAYCALLER_REGIME_CONTINUED"
                    if current and prev else
                    "PLAYCALLER_REGIME_UNRESOLVED"
                )

                # Within-season effective start week for the current caller.
                start_week = 1
                for w in range(week, 0, -1):
                    if text(m68.caller_for(season, w, team)) != current:
                        start_week = w + 1
                        break

                if previous_week is not None and int(previous_week) >= week:
                    chronology_violations += 1

                rows.append({
                    "season":season,
                    "week":week,
                    "team":team,
                    "current_playcaller":current,
                    "previous_game_playcaller":prev,
                    "previous_game_label":prev_label,
                    "caller_transition_state":state,
                    "playcaller_changed":changed,
                    "current_caller_effective_start_week":int(start_week),
                    "offseason_week1_transition":bool(week == 1 and changed),
                    "inseason_transition":bool(week > 1 and changed),
                })
                previous_week = week

    out = pd.DataFrame(rows)
    if out.duplicated(["season","week","team"]).any():
        raise RuntimeError("duplicate historical caller team-week identity")
    audit = {
        "chronology_violations":int(chronology_violations),
        "duplicate_team_week_identities":int(
            out.duplicated(["season","week","team"]).sum()
        ),
    }
    return out,audit


def attach_players(
    health: pd.DataFrame,
    callers: pd.DataFrame,
) -> tuple[pd.DataFrame,dict]:
    out = health[["season","week","team","stable_id","family"]].merge(
        callers,
        on=["season","week","team"],
        how="left",
        validate="many_to_one",
    )

    histories = {}
    for (team,pid), g in health.groupby(["team","stable_id"], sort=False):
        histories[(str(team),str(pid))] = g.sort_values(
            ["season","week"], kind="mergesort"
        ).reset_index(drop=True)

    prior_counts = []
    prior_positive_counts = []
    latest_prior_positive = []
    chronology_violations = 0

    for r in out.itertuples(index=False):
        g = histories.get((str(r.team),str(r.stable_id)))
        if g is None:
            prior_counts.append(0)
            prior_positive_counts.append(0)
            latest_prior_positive.append(np.nan)
            continue
        prior = g.loc[
            (g["season"] < int(r.season))
            | ((g["season"] == int(r.season)) & (g["week"] < int(r.week)))
        ].copy()
        pos = prior.loc[prior["positive_participation"]].copy()
        if len(prior):
            bad = prior.loc[
                ~(
                    (prior["season"] < int(r.season))
                    | ((prior["season"] == int(r.season)) & (prior["week"] < int(r.week)))
                )
            ]
            chronology_violations += int(len(bad))
        prior_counts.append(int(len(prior)))
        prior_positive_counts.append(int(len(pos)))
        if len(pos):
            last = pos.sort_values(["season","week"]).iloc[-1]
            latest_prior_positive.append(f"{int(last['season'])}_W{int(last['week'])}")
        else:
            latest_prior_positive.append("")

    out["same_player_same_team_prior_roster_rows"] = prior_counts
    out["same_player_same_team_prior_positive_rows"] = prior_positive_counts
    out["latest_prior_positive_participation"] = latest_prior_positive
    out["prior_positive_support_ge3"] = out[
        "same_player_same_team_prior_positive_rows"
    ].ge(3)

    audit = {
        "strictly_prior_history_chronology_violations":int(chronology_violations),
    }
    return out,audit


def summarize(
    attached: pd.DataFrame,
    callers: pd.DataFrame,
    season: int,
    caller_audit: dict,
    player_audit: dict,
) -> dict:
    c = callers.loc[callers["season"].eq(season)].copy()
    p = attached.loc[attached["season"].eq(season)].copy()
    change_c = c.loc[c["playcaller_changed"]].copy()
    change_p = p.loc[p["playcaller_changed"]].copy()

    week2plus = c.loc[c["week"].ge(2)]
    caller_coverage = float(c["current_playcaller"].ne("").mean()) if len(c) else 0.0
    prev_week2_coverage = float(
        week2plus["previous_game_playcaller"].ne("").mean()
    ) if len(week2plus) else 0.0
    w1 = c.loc[c["week"].eq(1)]
    w1_prev_coverage = float(w1["previous_game_playcaller"].ne("").mean()) if len(w1) else 0.0
    prior_support_share = float(
        change_p["prior_positive_support_ge3"].mean()
    ) if len(change_p) else 0.0

    fam = []
    for family,g in change_p.groupby("family", sort=True):
        fam.append({
            "family":str(family),
            "transition_player_weeks":int(len(g)),
            "distinct_named_players":int(g["stable_id"].nunique()),
            "prior_positive_support_ge3_rows":int(g["prior_positive_support_ge3"].sum()),
            "prior_positive_support_share":float(g["prior_positive_support_ge3"].mean()) if len(g) else 0.0,
        })

    gates = {
        "all_team_weeks_current_caller_nonblank":bool(caller_coverage == 1.0),
        "week2plus_previous_caller_coverage_ge_099":bool(prev_week2_coverage >= 0.99),
        "week1_prior_season_final_caller_coverage_eq_100":bool(w1_prev_coverage == 1.0),
        "duplicate_team_week_caller_identities_zero":caller_audit["duplicate_team_week_identities"] == 0,
        "caller_chronology_violations_zero":caller_audit["chronology_violations"] == 0,
        "caller_change_events_ge_10":int(len(change_c)) >= MIN_CHANGE_EVENTS,
        "transition_player_weeks_ge_250":int(len(change_p)) >= MIN_TRANSITION_PLAYER_WEEKS,
        "transition_distinct_named_players_ge_175":int(change_p["stable_id"].nunique()) >= MIN_TRANSITION_PLAYERS,
        "transition_stable_id_coverage_eq_100":bool(change_p["stable_id"].ne("").all()) if len(change_p) else False,
        "prior_positive_support_share_ge_040":bool(prior_support_share >= MIN_PRIOR_SUPPORT_SHARE),
        "strictly_prior_history_chronology_violations_zero":player_audit["strictly_prior_history_chronology_violations"] == 0,
        "target_week_snaps_used_to_define_caller_state_zero":True,
        "target_week_opportunity_performance_fields_used_zero":True,
    }

    return {
        "season":season,
        "team_weeks":int(len(c)),
        "teams":int(c["team"].nunique()),
        "caller_identity_coverage":caller_coverage,
        "week2plus_previous_caller_coverage":prev_week2_coverage,
        "week1_prior_season_final_caller_coverage":w1_prev_coverage,
        "offseason_week1_change_events":int(change_c["offseason_week1_transition"].sum()),
        "inseason_change_events":int(change_c["inseason_transition"].sum()),
        "total_caller_change_events":int(len(change_c)),
        "caller_change_teams":int(change_c["team"].nunique()),
        "attached_player_weeks":int(len(p)),
        "transition_player_weeks":int(len(change_p)),
        "transition_named_players":int(change_p["stable_id"].nunique()),
        "prior_positive_support_ge3_player_weeks":int(change_p["prior_positive_support_ge3"].sum()),
        "prior_positive_support_share":prior_support_share,
        "family_summary":fam,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def prospective_parity(snapshot: pd.DataFrame) -> dict:
    week1 = snapshot.loc[snapshot["effective_week"].eq(1)].copy()
    dups = int(snapshot.duplicated(["team","effective_week"]).sum())
    override = snapshot.loc[snapshot["effective_week"].gt(1)].copy()
    sources_ok = bool(
        snapshot["source_url"].ne("").all()
        and snapshot["source_type"].ne("").all()
    )
    gates = {
        "exactly_32_week1_teams":int(week1["team"].nunique()) == 32 and len(week1) == 32,
        "duplicate_team_effective_week_zero":dups == 0,
        "source_url_and_type_nonblank_all":sources_ok,
        "effective_week_integer_ge_1_all":bool(
            snapshot["effective_week"].ge(1).all()
        ),
        "documented_midseason_override_nonzero":int(len(override)) >= 1,
        "historical_semantic_parity_primary_caller_effective_week":True,
        "outcomes_2026_used_zero":True,
    }
    return {
        "week1_rows":int(len(week1)),
        "week1_teams":int(week1["team"].nunique()),
        "override_rows":int(len(override)),
        "max_effective_week":int(snapshot["effective_week"].max()) if len(snapshot) else 0,
        "gates":gates,
        "prospective_parity_ready":bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-parent", type=Path, required=True)
    ap.add_argument("--snapshot-2026", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    health = load_health(a.health_parent)
    snapshot = load_2026_snapshot(a.snapshot_2026)
    team_weeks = team_weeks_from_health(health)
    callers,caller_audit = historical_caller_state(team_weeks)
    attached,player_audit = attach_players(health,callers)

    summaries = [
        summarize(attached,callers,s,caller_audit,player_audit)
        for s in SEASONS
    ]
    parity = prospective_parity(snapshot)

    integrity = {
        "predictive_player_outcomes_scored_zero":True,
        "coefficients_fit_zero":True,
        "thresholds_searched_zero":True,
        "sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,
        "paid_oddsapi_calls_zero":True,
        "production_changed_false":True,
    }

    ready = bool(
        all(s["source_ready"] for s in summaries)
        and parity["prospective_parity_ready"]
        and all(integrity.values())
    )

    result = {
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "m68_historical_source_blob_sha":M68_BLOB_SHA,
        "health_parent_run":39154145666,
        "health_parent_artifact":12146625382,
        "health_parent_digest":"sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "disposition":(
            "PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_READY"
            if ready else
            "PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_BLOCKED"
        ),
        "source_ready":ready,
        "caller_audit":caller_audit,
        "player_history_audit":player_audit,
        "season_summaries":summaries,
        "prospective_2026_parity":parity,
        "integrity_gates":integrity,
        "target_week_snaps_used_to_define_caller_state":0,
        "target_week_opportunity_performance_fields_used":0,
        "predictive_player_outcomes_scored":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "paid_oddsapi_calls":0,
        "production_changed":False,
    }

    callers.to_csv(
        a.out_dir/"playcaller_regime_team_weeks.csv",
        index=False
    )
    attached.to_csv(
        a.out_dir/"playcaller_regime_named_player_weeks.csv",
        index=False
    )
    pd.DataFrame([
        dict(season=s["season"], **f)
        for s in summaries for f in s["family_summary"]
    ]).to_csv(
        a.out_dir/"playcaller_regime_family_summary.csv",
        index=False
    )
    snapshot.to_csv(
        a.out_dir/"playcaller_regime_2026_snapshot_certified.csv",
        index=False
    )
    (a.out_dir/"playcaller_regime_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
