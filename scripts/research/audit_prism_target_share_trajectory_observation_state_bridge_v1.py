#!/usr/bin/env python3
"""PRISM Target-Share Trajectory Observation-State Bridge V1 source audit.

Source-only bridge between the immutable Target Share Trajectory V1 rows and
the immutable Player Observation Health-State V1 rows. No target-game outcome
field is loaded or scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_TARGET_SHARE_TRAJECTORY_OBSERVATION_STATE_BRIDGE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "cb4ec5210bc09b4eff265d0a032df1bed73d32e4"

TRAJ_COLS = [
    "season", "week", "team", "player_clean_key", "receiver_id",
    "position_group", "prior_team_games", "recent2_team_targets",
    "recent2_player_targets", "recent2_share", "earlier_share",
    "trajectory_delta", "feature_max_week",
]
HEALTH_COLS = [
    "season", "week", "team", "stable_id", "player", "family",
    "observation_state", "positive_participation",
]

PARTICIPATION_CERTIFIED = "PARTICIPATION_CERTIFIED"
REPORTED_NO_SNAPS = "REPORTED_NO_OFFENSIVE_SNAPS"
UNRESOLVED_NO_SNAPS = "UNRESOLVED_NO_OFFENSIVE_SNAPS"
NOT_ON_SCOPED_ROSTER = "NOT_ON_SCOPED_ROSTER"
RECENT_STATES = (
    PARTICIPATION_CERTIFIED,
    REPORTED_NO_SNAPS,
    UNRESOLVED_NO_SNAPS,
    NOT_ON_SCOPED_ROSTER,
)

HEALTH_REPORTED_NO = "REPORTED_INJURY_NO_OFFENSIVE_SNAPS"
HEALTH_UNRESOLVED_NO = "NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED"


def clean(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def team(v) -> str:
    s = clean(v).upper()
    return {"WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LAR", "JAC": "JAX", "ARZ": "ARI"}.get(s, s)


def as_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = clean(v).lower()
    return s in {"1", "true", "t", "yes", "y"}


def read_trajectory(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty trajectory source: {path}")
    x = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in set(TRAJ_COLS), low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = set(TRAJ_COLS) - set(x.columns)
    if missing:
        raise RuntimeError(f"trajectory source missing columns: {sorted(missing)}")
    if "opportunity_error" in x.columns:
        raise RuntimeError("forbidden trajectory outcome field was loaded")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["feature_max_week"] = pd.to_numeric(x["feature_max_week"], errors="raise").astype(int)
    x["team"] = x["team"].map(team)
    x["receiver_id"] = x["receiver_id"].map(clean)
    x["player_clean_key"] = x["player_clean_key"].map(clean)
    x["position_group"] = x["position_group"].astype("string").str.upper().str.strip()
    x = x.loc[
        ((x["position_group"].eq("WR") & x["season"].eq(2024))
         | (x["position_group"].eq("TE") & x["season"].isin([2024, 2025])))
    ].copy()
    if x.empty:
        raise RuntimeError("zero trajectory bridge rows after frozen scope")
    keys = ["season", "week", "team", "player_clean_key", "position_group"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate trajectory target identities: {bad}")
    if x["receiver_id"].eq("").any():
        raise RuntimeError("blank stable receiver_id in trajectory bridge scope")
    return x.sort_values(keys, kind="mergesort").reset_index(drop=True)


def read_health(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty health-state source: {path}")
    x = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in set(HEALTH_COLS), low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = set(HEALTH_COLS) - set(x.columns)
    if missing:
        raise RuntimeError(f"health-state source missing columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(team)
    x["stable_id"] = x["stable_id"].map(clean)
    x["family"] = x["family"].astype("string").str.upper().str.strip()
    x["observation_state"] = x["observation_state"].astype("string").str.strip()
    x["positive_participation"] = x["positive_participation"].map(as_bool)
    x = x.loc[x["season"].isin([2024, 2025])].copy()

    # Exact stable-ID rows must be unique. Blank stable IDs are irrelevant to the
    # receiver_id bridge and are kept only for team-week calendar derivation.
    z = x.loc[x["stable_id"].ne("")].copy()
    keys = ["season", "week", "team", "stable_id"]
    if z.duplicated(keys).any():
        bad = z.loc[z.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate certified health stable-ID identities: {bad}")
    return x.sort_values(["season", "week", "team", "stable_id"], kind="mergesort").reset_index(drop=True)


def recent_state(row: pd.Series | None) -> str:
    if row is None:
        return NOT_ON_SCOPED_ROSTER
    if bool(row["positive_participation"]):
        return PARTICIPATION_CERTIFIED
    state = str(row["observation_state"])
    if state == HEALTH_REPORTED_NO:
        return REPORTED_NO_SNAPS
    if state == HEALTH_UNRESOLVED_NO:
        return UNRESOLVED_NO_SNAPS
    raise RuntimeError(f"unexpected no-participation health state: {state}")


def build_bridge(traj: pd.DataFrame, health: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    team_weeks = {
        (int(s), str(t)): sorted(int(v) for v in g["week"].unique())
        for (s, t), g in health.groupby(["season", "team"], sort=False)
    }
    stable = {
        (int(r.season), int(r.week), str(r.team), str(r.stable_id)): r
        for r in health.loc[health["stable_id"].ne("")].itertuples(index=False)
    }

    rows = []
    chronology_violations = 0
    feature_max_mismatches = 0
    missing_two_prior = 0
    for r in traj.itertuples(index=False):
        key = (int(r.season), str(r.team))
        weeks = [w for w in team_weeks.get(key, []) if w < int(r.week)]
        if len(weeks) < 2:
            missing_two_prior += 1
            continue
        older_week, recent_week = weeks[-2], weeks[-1]
        if recent_week >= int(r.week) or older_week >= int(r.week):
            chronology_violations += 1
        if recent_week != int(r.feature_max_week):
            feature_max_mismatches += 1

        states = []
        for w in (older_week, recent_week):
            h = stable.get((int(r.season), int(w), str(r.team), str(r.receiver_id)))
            if h is None:
                states.append(NOT_ON_SCOPED_ROSTER)
            else:
                states.append(recent_state(pd.Series(h._asdict())))

        older_state, recent_state_name = states
        clean_recent2 = bool(
            older_state == PARTICIPATION_CERTIFIED
            and recent_state_name == PARTICIPATION_CERTIFIED
        )
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "player_clean_key": str(r.player_clean_key),
            "receiver_id": str(r.receiver_id),
            "position_group": str(r.position_group),
            "feature_max_week": int(r.feature_max_week),
            "recent2_older_week": int(older_week),
            "recent1_week": int(recent_week),
            "recent2_older_state": older_state,
            "recent1_state": recent_state_name,
            "recent2_participation_certified": clean_recent2,
            "recent2_observation_contaminated": not clean_recent2,
            "contamination_recent1_only": bool(recent_state_name != PARTICIPATION_CERTIFIED and older_state == PARTICIPATION_CERTIFIED),
            "contamination_recent2_older_only": bool(older_state != PARTICIPATION_CERTIFIED and recent_state_name == PARTICIPATION_CERTIFIED),
            "contamination_both_recent_games": bool(older_state != PARTICIPATION_CERTIFIED and recent_state_name != PARTICIPATION_CERTIFIED),
            "reported_no_snap_present": bool(REPORTED_NO_SNAPS in states),
            "unresolved_no_snap_present": bool(UNRESOLVED_NO_SNAPS in states),
            "not_on_scoped_roster_present": bool(NOT_ON_SCOPED_ROSTER in states),
            "trajectory_delta": float(r.trajectory_delta),
            "recent2_share": float(r.recent2_share),
            "earlier_share": float(r.earlier_share),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero bridge rows")
    keys = ["season", "week", "team", "player_clean_key", "position_group"]
    if out.duplicated(keys).any():
        raise RuntimeError("duplicate bridge output identities")
    audit = {
        "input_trajectory_rows": int(len(traj)),
        "output_bridge_rows": int(len(out)),
        "missing_two_prior_team_games": int(missing_two_prior),
        "chronology_violations": int(chronology_violations),
        "feature_max_week_mismatches": int(feature_max_mismatches),
    }
    return out, audit


def cell_summary(df: pd.DataFrame, position: str, season: int) -> dict:
    z = df.loc[df["position_group"].eq(position) & df["season"].eq(season)].copy()
    if z.empty:
        raise RuntimeError(f"zero source rows for {position} {season}")
    affected = z.loc[z["recent2_observation_contaminated"]].copy()
    per_player = affected.groupby("player_clean_key").size() if len(affected) else pd.Series(dtype=int)
    recent_states = pd.concat(
        [z["recent2_older_state"], z["recent1_state"]], ignore_index=True
    ).value_counts()
    return {
        "position": position,
        "season": int(season),
        "rows": int(len(z)),
        "players": int(z["player_clean_key"].nunique()),
        "clean_recent2_rows": int(z["recent2_participation_certified"].sum()),
        "contaminated_recent2_rows": int(z["recent2_observation_contaminated"].sum()),
        "contaminated_share": float(z["recent2_observation_contaminated"].mean()),
        "players_with_contamination": int(affected["player_clean_key"].nunique()),
        "median_contaminated_rows_per_affected_player": float(per_player.median()) if len(per_player) else 0.0,
        "max_contaminated_rows_one_player": int(per_player.max()) if len(per_player) else 0,
        "contamination_recent1_only": int(z["contamination_recent1_only"].sum()),
        "contamination_recent2_older_only": int(z["contamination_recent2_older_only"].sum()),
        "contamination_both_recent_games": int(z["contamination_both_recent_games"].sum()),
        "reported_no_snap_rows": int(z["reported_no_snap_present"].sum()),
        "unresolved_no_snap_rows": int(z["unresolved_no_snap_present"].sum()),
        "not_on_scoped_roster_rows": int(z["not_on_scoped_roster_present"].sum()),
        "recent_game_state_counts": {s: int(recent_states.get(s, 0)) for s in RECENT_STATES},
    }


def json_default(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    raise TypeError(f"Object of type {v.__class__.__name__} is not JSON serializable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--health", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = read_trajectory(a.trajectory)
    health = read_health(a.health)
    bridge, audit = build_bridge(trajectory, health)

    cells = [
        cell_summary(bridge, "WR", 2024),
        cell_summary(bridge, "TE", 2024),
        cell_summary(bridge, "TE", 2025),
    ]
    gates = {
        "trajectory_outcome_field_not_loaded": True,
        "frozen_scope_only": bool(
            set(zip(bridge["position_group"], bridge["season"]))
            <= {("WR", 2024), ("TE", 2024), ("TE", 2025)}
        ),
        "receiver_id_nonblank_all": bool(bridge["receiver_id"].astype(str).str.strip().ne("").all()),
        "team_week_history_available_all": bool(audit["missing_two_prior_team_games"] == 0),
        "exact_two_recent_games_all": bool(len(bridge) == len(trajectory)),
        "feature_max_week_match_all": bool(audit["feature_max_week_mismatches"] == 0),
        "strict_prior_all": bool(audit["chronology_violations"] == 0),
        "duplicate_trajectory_targets_zero": True,
        "duplicate_health_stable_ids_zero": True,
        "classification_coverage_all": bool(
            bridge["recent2_older_state"].isin(RECENT_STATES).all()
            and bridge["recent1_state"].isin(RECENT_STATES).all()
        ),
        "contaminated_rows_each_cell": bool(all(c["contaminated_recent2_rows"] > 0 for c in cells)),
        "chronology_violations_zero": bool(audit["chronology_violations"] == 0),
        "outcomes_2026_zero": True,
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "predictive_models_fit_zero": True,
        "thresholds_searched_zero": True,
        "production_changed_false": True,
    }
    ready = bool(all(gates.values()))
    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "disposition": (
            "PRISM_TARGET_SHARE_TRAJECTORY_OBSERVATION_STATE_BRIDGE_V1_SOURCE_READY"
            if ready else
            "PRISM_TARGET_SHARE_TRAJECTORY_OBSERVATION_STATE_BRIDGE_V1_SOURCE_BLOCKED"
        ),
        "source_audit": audit,
        "cells": cells,
        "gates": gates,
        "trajectory_artifact_run": 37638269235,
        "trajectory_artifact_id": 11490707699,
        "health_artifact_run": 39154145666,
        "health_artifact_id": 12146625382,
        "predictive_relationships_scored": 0,
        "target_yard_outcomes_loaded": 0,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }

    bridge.to_csv(a.out_dir / "trajectory_observation_state_bridge_rows.csv", index=False)
    pd.DataFrame(cells).to_csv(a.out_dir / "trajectory_observation_state_bridge_cells.csv", index=False)
    (a.out_dir / "trajectory_observation_state_bridge_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
