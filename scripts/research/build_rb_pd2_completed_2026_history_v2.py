#!/usr/bin/env python3
"""Certify cumulative completed 2026 RB rushing predictor history after Week 1.

This is the forward-history advance path for the frozen RB-PD2 shadow study.
For a completed week >= 2 it:

1. verifies an immutable Full Slate artifact was created before the first
   kickoff of that target week;
2. reconstructs the football-only full-roster rushing projection frame from
   the preserved pregame football universe, final target-entitlement state,
   ML/state diagnostics, and frozen ensemble weights;
3. mechanically proves parity against any preserved priced RB/FB rush-yards
   rows (sportsbook rows are parity evidence only, never projection inputs);
4. only after the projection frame is frozen, verifies the week is final and
   attaches realized rushing outcomes by GSIS/roster identity;
5. appends those rows to the previously certified cumulative 2026 history.

Week 1 remains governed by build_rb_pd2_completed_2026_history_v1.py because
its promoted P3/STACK1 mean authority is intentionally different.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.build._schedule_utils import get_nfl_schedule
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.operations.grade_market_track_record_gsis_v1 import (
    build_alias_index,
    load_actual_stats_unfiltered,
    load_roster_identity,
    resolve_gsis,
)
from scripts.research.build_rb_pd2_forward_history_v1 import validate_additional_history
from scripts.research.rb_pd2_forward_shadow_v1 import (
    ELIGIBLE_POSITIONS,
    verify_frozen_2025_weights,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.utils.player_identity_v3 import player_name_key

SEASON = 2026
FINAL_ENTITLEMENT_VERSION = "TEAM_TARGET_ENTITLEMENT_V1_PLUS_TE_R5P_PLUS_WR_R15_V1"
PARITY_TOLERANCE = 1e-8
SOURCE_REQUIRED = {
    "universe": Path("data/football_simulation_universe.csv"),
    "universe_audit": Path("data/football_simulation_universe_audit.json"),
    "entitlement_trace": Path("data/target_entitlement_v1_trace.csv"),
    "entitlement_audit": Path("data/target_entitlement_v1_audit.json"),
    "ml": Path("data/model_ml_diagnostics.csv"),
    "state": Path("data/model_state_diagnostics.csv"),
    "weights": Path("data/model_ensemble_weights.csv"),
    "priced": Path("outputs/props_priced_clean.csv"),
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    if out.empty:
        raise RuntimeError(f"empty {label}: {path}")
    return out


def _read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _full_name_key(value: Any) -> str:
    try:
        return str(player_name_key(value, strip_suffix=False) or "").strip()
    except Exception:
        return ""


def _suffix_safe_key(value: Any) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _source_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {name: root / rel for name, rel in SOURCE_REQUIRED.items()}


def validate_source_provenance(
    *,
    source_run_metadata: dict,
    source_artifact_metadata: dict,
    completed_week: int,
    schedule: pd.DataFrame,
) -> dict:
    """Prove the preserved artifact existed before the week's first kickoff."""
    if int(completed_week) <= 1:
        raise RuntimeError("V2 completed-history source is only valid for week >= 2")
    week_sched = schedule.loc[
        pd.to_numeric(schedule["season"], errors="coerce").eq(SEASON)
        & pd.to_numeric(schedule["week"], errors="coerce").eq(int(completed_week))
    ].copy()
    if week_sched.empty:
        raise RuntimeError(f"schedule has no {SEASON} Week {completed_week} games")
    kickoff = pd.to_datetime(week_sched["kickoff_utc"], utc=True, errors="coerce")
    if kickoff.isna().any():
        raise RuntimeError("source provenance schedule contains malformed kickoff timestamp")
    first_kickoff = kickoff.min()

    run_id = str(source_run_metadata.get("id") or "").strip()
    head_sha = str(source_run_metadata.get("head_sha") or "").strip()
    run_started = pd.to_datetime(
        source_run_metadata.get("run_started_at") or source_run_metadata.get("created_at"),
        utc=True, errors="coerce",
    )
    if not run_id or len(head_sha) != 40 or pd.isna(run_started):
        raise RuntimeError("source run metadata does not prove run id/head SHA/start time")
    if run_started >= first_kickoff:
        raise RuntimeError(
            f"source Full Slate run was not pregame: run_started={run_started.isoformat()} "
            f"first_kickoff={first_kickoff.isoformat()}"
        )

    artifact_id = str(source_artifact_metadata.get("id") or "").strip()
    artifact_name = str(source_artifact_metadata.get("name") or "").strip()
    digest = str(source_artifact_metadata.get("digest") or "").strip()
    created = pd.to_datetime(source_artifact_metadata.get("created_at"), utc=True, errors="coerce")
    expired = bool(source_artifact_metadata.get("expired"))
    if not artifact_id or not artifact_name or not digest.startswith("sha256:") or pd.isna(created):
        raise RuntimeError("source artifact metadata does not prove id/name/digest/creation time")
    if expired:
        raise RuntimeError("source pregame artifact is expired")
    if created >= first_kickoff:
        raise RuntimeError(
            f"source artifact was not frozen pregame: artifact_created={created.isoformat()} "
            f"first_kickoff={first_kickoff.isoformat()}"
        )
    workflow_run = source_artifact_metadata.get("workflow_run") or {}
    if str(workflow_run.get("id") or "") not in {"", run_id}:
        raise RuntimeError("source artifact workflow run does not match source run metadata")

    return {
        "source_run_id": run_id,
        "source_git_sha": head_sha,
        "source_run_started_at": run_started.isoformat(),
        "source_artifact_id": artifact_id,
        "source_artifact_name": artifact_name,
        "source_artifact_digest": digest,
        "source_artifact_created_at": created.isoformat(),
        "first_week_kickoff_utc": first_kickoff.isoformat(),
    }


def _validate_football_only_source(paths: dict[str, Path]) -> dict:
    universe_audit = _read_json(paths["universe_audit"], "football universe audit")
    entitlement_audit = _read_json(paths["entitlement_audit"], "target entitlement audit")

    if int(universe_audit.get("sportsbook_rows_used_to_define_player_universe", -1)) != 0:
        raise RuntimeError("sportsbook rows were used to define preserved football universe")
    if universe_audit.get("provider_event_ids_used_during_simulation") is not False:
        raise RuntimeError("provider event ids entered preserved football simulation")
    forbidden = universe_audit.get("sportsbook_line_odds_book_fields_present")
    if forbidden not in ([], None):
        raise RuntimeError(f"sportsbook fields present in preserved football universe: {forbidden}")
    if entitlement_audit.get("sportsbook_inputs_used") is not False:
        raise RuntimeError("sportsbook inputs entered preserved entitlement state")

    return {
        "football_universe_players": int(universe_audit.get("football_players", 0)),
        "football_universe_teams": int(universe_audit.get("football_teams", 0)),
        "football_universe_games": int(universe_audit.get("canonical_games", 0)),
        "sportsbook_rows_used_to_define_player_universe": 0,
        "sportsbook_inputs_used_for_entitlement": 0,
    }


def _merge_component(
    frame: pd.DataFrame,
    component: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    need = {"season", "week", "team", "player", "player_clean_key", value_col}
    missing = need - set(component.columns)
    if missing:
        raise RuntimeError(f"component {value_col} missing columns: {sorted(missing)}")
    if "player" not in frame.columns:
        raise RuntimeError(f"projection frame missing player display name for {value_col} identity bridge")

    left = frame.copy()
    c = component.copy()
    for x in (left, c):
        x["season"] = pd.to_numeric(x["season"], errors="coerce")
        x["week"] = pd.to_numeric(x["week"], errors="coerce")
        x["team"] = x["team"].map(canon_team)
        # Rebuild the football-only component attachment from the full display
        # name, preserving suffixes exactly. This mirrors football-source
        # identity without borrowing sportsbook identity.
        x["_component_identity_key"] = x["player"].map(_full_name_key)
        if x["_component_identity_key"].eq("").any():
            raise RuntimeError(f"blank full-name identity while joining component {value_col}")

    join = ["season", "week", "team", "_component_identity_key"]
    if left.duplicated(join).any():
        raise RuntimeError(f"projection frame has duplicate full-name identity for {value_col}")
    if c.duplicated(join).any():
        raise RuntimeError(f"component {value_col} has duplicate full-name identity")

    c = c[join + [value_col]].rename(columns={value_col: out_col})
    out = left.merge(c, on=join, how="left", validate="one_to_one")
    return out.drop(columns=["_component_identity_key"])

def build_projection_frame(
    *,
    source_root: Path,
    completed_week: int,
    provenance: dict,
) -> tuple[pd.DataFrame, dict]:
    """Freeze the full-roster Week-N football mean before any outcome is loaded."""
    paths = _source_paths(source_root)
    for label, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"preserved source missing {label}: {path}")
    football_audit = _validate_football_only_source(paths)

    universe = _read_csv(paths["universe"], "football simulation universe")
    trace = _read_csv(paths["entitlement_trace"], "final target entitlement trace")
    ml = _read_csv(paths["ml"], "ML diagnostics")
    state = _read_csv(paths["state"], "state diagnostics")
    weights = _read_csv(paths["weights"], "ensemble weights")
    priced = _read_csv(paths["priced"], "preserved priced parity rows")

    need = {
        "season", "week", "event_id", "player", "player_clean_key", "team",
        "opponent", "position",
    }
    missing = need - set(universe.columns)
    if missing:
        raise RuntimeError(f"football universe missing columns: {sorted(missing)}")
    season = pd.to_numeric(universe["season"], errors="coerce")
    week = pd.to_numeric(universe["week"], errors="coerce")
    if not season.eq(SEASON).all() or not week.eq(int(completed_week)).all():
        raise RuntimeError("preserved football universe season/week does not match requested completed week")
    universe["team"] = universe["team"].map(canon_team)
    universe["opponent"] = universe["opponent"].map(canon_team)
    universe["player_clean_key"] = universe["player_clean_key"].astype(str).str.strip()
    if universe["player_clean_key"].eq("").any():
        raise RuntimeError("preserved football universe has blank player identity")
    ident = ["event_id", "team", "player_clean_key"]
    if universe.duplicated(ident).any():
        raise RuntimeError("preserved football universe has duplicate player/game/team identity")

    trace_need = {
        "event_id", "team", "player_clean_key", "entitlement_tgt_share",
        "entitlement_version",
    }
    trace_missing = trace_need - set(trace.columns)
    if trace_missing:
        raise RuntimeError(f"entitlement trace missing columns: {sorted(trace_missing)}")
    if not trace["entitlement_version"].astype(str).eq(FINAL_ENTITLEMENT_VERSION).all():
        raise RuntimeError("preserved entitlement trace is not the current final TE-R5P + WR-R15 state")
    trace["team"] = trace["team"].map(canon_team)
    trace["player_clean_key"] = trace["player_clean_key"].astype(str).str.strip()
    if trace.duplicated(ident).any():
        raise RuntimeError("preserved entitlement trace has duplicate player/game/team identity")
    sim_input = universe.merge(
        trace[ident + ["entitlement_tgt_share"]],
        on=ident,
        how="left",
        validate="one_to_one",
    )
    ent = pd.to_numeric(sim_input["entitlement_tgt_share"], errors="coerce")
    if ent.isna().any() or not np.isfinite(ent.to_numpy(float)).all() or ent.lt(0).any():
        raise RuntimeError("preserved final target entitlement does not cover full football universe")

    sim_iterations_values = (
        pd.to_numeric(priced.get("simulation_iterations"), errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    if len(sim_iterations_values) != 1 or sim_iterations_values[0] <= 0:
        raise RuntimeError(
            f"preserved priced artifact does not prove one simulation iteration count: "
            f"{sim_iterations_values}"
        )
    sim_iterations = int(sim_iterations_values[0])
    sims = explicit_simulate(sim_input, iterations=sim_iterations)

    pos = universe["position"].fillna("").astype(str).str.upper().str.strip()
    eligible = universe.loc[pos.isin(ELIGIBLE_POSITIONS)].copy()
    eligible["position"] = pos.loc[eligible.index].to_numpy()
    if eligible.empty:
        raise RuntimeError("preserved full football universe contains zero RB/HB/FB rows")

    mc_values = []
    for row in eligible.itertuples(index=False):
        key = (str(row.event_id), str(row.player_clean_key), "rush_yards")
        values = sims.values.get(key)
        if values is None or len(values) == 0:
            raise RuntimeError(f"missing reconstructed rush_yards MC distribution: {key}")
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite reconstructed rush_yards MC distribution: {key}")
        mc_values.append(float(arr.mean()))
    eligible["mc_proj"] = mc_values

    eligible = _merge_component(
        eligible, ml, value_col="ml_rush_yards", out_col="ml_proj"
    )
    eligible = _merge_component(
        eligible, state, value_col="state_rush_yards", out_col="state_proj"
    )

    verify_frozen_2025_weights(weights)
    ensemble_input = eligible[
        ["event_id", "player", "player_clean_key", "team", "opponent", "position",
         "season", "week", "mc_proj", "ml_proj", "state_proj"]
    ].copy()
    ensemble_input["market"] = "rush_yards"
    scored = apply_ensemble(ensemble_input, weights=weights)
    projection = pd.DataFrame({
        "season": SEASON,
        "week": int(completed_week),
        "event_id": scored["event_id"].astype(str),
        "player": scored["player"].astype(str),
        "player_clean_key": scored["player_clean_key"].astype(str),
        "team": scored["team"].astype(str),
        "opponent": scored["opponent"].astype(str),
        "position": scored["position"].astype(str),
        "projection_mean": pd.to_numeric(scored["ensemble_proj"], errors="coerce"),
        "generic_mc_projection": pd.to_numeric(scored["mc_proj"], errors="coerce"),
        "generic_ml_projection": pd.to_numeric(scored["ml_proj"], errors="coerce"),
        "generic_state_projection": pd.to_numeric(scored["state_proj"], errors="coerce"),
        "pregame_lineage_certified": True,
        "projection_lineage": (
            f"{SEASON}_W{int(completed_week)}_GENERIC_ENSEMBLE_FULL_ROSTER|"
            f"run={provenance['source_run_id']}|artifact={provenance['source_artifact_id']}|"
            f"sha={provenance['source_git_sha']}"
        ),
    })
    if not np.isfinite(projection["projection_mean"]).all():
        raise RuntimeError("reconstructed full-roster generic ensemble contains non-finite projection")

    if "source_market" not in priced.columns:
        raise RuntimeError("preserved priced parity file missing source_market")
    q = priced.loc[priced["source_market"].astype(str).str.lower().eq("player_rush_yds")].copy()
    if q.empty:
        raise RuntimeError("preserved priced parity file contains zero rush-yards rows")
    q["team"] = q["team"].map(canon_team)
    q["canonical_player_key"] = q["player"].map(_suffix_safe_key)
    if q["canonical_player_key"].eq("").any():
        raise RuntimeError("preserved priced parity file has unresolved suffix-safe player identity")
    pkey = projection[
        ["team", "player", "player_clean_key", "position", "generic_mc_projection", "projection_mean"]
    ].copy()
    pkey["canonical_player_key"] = pkey["player"].map(_suffix_safe_key)
    if pkey["canonical_player_key"].eq("").any():
        raise RuntimeError("reconstructed projection has unresolved suffix-safe player identity")
    if pkey.duplicated(["team", "canonical_player_key"]).any():
        raise RuntimeError("reconstructed RB/FB projection has duplicate suffix-safe player identity")
    parity = q.merge(
        pkey.drop(columns=["player"]),
        on=["team", "canonical_player_key"],
        how="inner",
        validate="many_to_one",
    )
    parity = parity.loc[parity["position"].isin(ELIGIBLE_POSITIONS)].copy()
    if parity.empty:
        raise RuntimeError("zero preserved priced RB/FB rush rows matched full-roster projection frame")
    if "rb_synthesis_applied" in parity.columns:
        applied = pd.to_numeric(parity["rb_synthesis_applied"], errors="coerce").fillna(0)
        if not applied.eq(0).all():
            raise RuntimeError("non-Week-1 preserved rush rows unexpectedly consumed RB P3")

    mc_gap = (
        pd.to_numeric(parity["mc_proj"], errors="coerce")
        - pd.to_numeric(parity["generic_mc_projection"], errors="coerce")
    ).abs()
    if mc_gap.isna().any() or float(mc_gap.max()) > PARITY_TOLERANCE:
        raise RuntimeError(
            f"preserved priced mc parity failed max_abs_gap={float(mc_gap.max())}"
        )

    # Rebuild the preserved *priced* component path only as parity evidence.
    # These rows never feed projection_mean. This is necessary because the
    # Week-2 live board had a small downstream suffix-identity attachment gap:
    # the football-only full-roster projection may legitimately have a component
    # that the sportsbook-facing row failed to attach.
    priced_components = parity[["mc_proj", "ml_proj", "state_proj"]].copy()
    priced_components["market"] = "rush_yards"
    priced_rebuilt = apply_ensemble(priced_components, weights=weights)
    parity["priced_rebuilt_ensemble"] = pd.to_numeric(
        priced_rebuilt["ensemble_proj"], errors="coerce"
    ).to_numpy()

    ens_gap = (
        pd.to_numeric(parity["ensemble_proj"], errors="coerce")
        - parity["priced_rebuilt_ensemble"]
    ).abs()
    model_gap = (
        pd.to_numeric(parity["model_proj"], errors="coerce")
        - parity["priced_rebuilt_ensemble"]
    ).abs()
    for label, gap in [("ensemble", ens_gap), ("model", model_gap)]:
        if gap.isna().any() or float(gap.max()) > PARITY_TOLERANCE:
            raise RuntimeError(
                f"preserved priced {label} path parity failed max_abs_gap={float(gap.max())}"
            )

    full_vs_priced_gap = (
        pd.to_numeric(parity["projection_mean"], errors="coerce")
        - parity["priced_rebuilt_ensemble"]
    ).abs()
    component_identity_gap = full_vs_priced_gap.gt(PARITY_TOLERANCE)
    identity_gap_detail = parity.loc[
        component_identity_gap,
        ["player", "team", "canonical_player_key", "mc_proj", "ml_proj", "state_proj",
         "generic_mc_projection", "projection_mean", "priced_rebuilt_ensemble"],
    ].copy()
    identity_gap_detail = identity_gap_detail.astype(object).where(
        pd.notna(identity_gap_detail), None
    )

    audit = {
        **football_audit,
        **provenance,
        "projection_rows": int(len(projection)),
        "projection_players": int(projection["player_clean_key"].nunique()),
        "simulation_iterations": sim_iterations,
        "priced_parity_rows": int(len(parity)),
        "priced_parity_players": int(parity[["team", "canonical_player_key"]].drop_duplicates().shape[0]),
        "max_abs_priced_mc_parity_gap": float(mc_gap.max()),
        "max_abs_priced_ensemble_parity_gap": float(ens_gap.max()),
        "max_abs_priced_model_parity_gap": float(model_gap.max()),
        "max_abs_full_roster_vs_priced_mean_gap": float(full_vs_priced_gap.max()),
        "priced_component_identity_gap_rows": int(component_identity_gap.sum()),
        "priced_component_identity_gap_players": int(
            identity_gap_detail[["team", "canonical_player_key"]].drop_duplicates().shape[0]
        ),
        "priced_component_identity_gap_detail": identity_gap_detail.to_dict("records"),
        "sportsbook_inputs_used_for_projection": 0,
        "sportsbook_rows_used_for_parity_only": int(len(parity)),
        "weights_sha256": _file_sha256(paths["weights"]),
        "football_universe_sha256": _file_sha256(paths["universe"]),
        "entitlement_trace_sha256": _file_sha256(paths["entitlement_trace"]),
        "ml_diagnostics_sha256": _file_sha256(paths["ml"]),
        "state_diagnostics_sha256": _file_sha256(paths["state"]),
    }
    return projection, audit


def assert_week_complete_after_freeze(completed_week: int) -> dict:
    """Fail closed unless nflverse schedule has final-score evidence for all games."""
    import nflreadpy as nfl

    raw = nfl.load_schedules(seasons=[SEASON])
    x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError("nflreadpy schedule returned zero rows for completion gate")
    season = pd.to_numeric(x.get("season"), errors="coerce")
    week = pd.to_numeric(x.get("week"), errors="coerce")
    q = x.loc[season.eq(SEASON) & week.eq(int(completed_week))].copy()
    if "game_type" in q.columns:
        reg = q.loc[q["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            q = reg
    if q.empty:
        raise RuntimeError(f"no schedule rows for {SEASON} Week {completed_week} completion gate")
    for col in ("home_score", "away_score"):
        if col not in q.columns:
            raise RuntimeError(f"schedule completion gate missing {col}")
        score = pd.to_numeric(q[col], errors="coerce")
        if score.isna().any():
            raise RuntimeError(
                f"{SEASON} Week {completed_week} is not fully final: "
                f"{col} missing for {int(score.isna().sum())} game(s)"
            )
    if "result" in q.columns:
        result = pd.to_numeric(q["result"], errors="coerce")
        if result.isna().any():
            raise RuntimeError(
                f"{SEASON} Week {completed_week} is not fully final: "
                f"result missing for {int(result.isna().sum())} game(s)"
            )
    return {
        "scheduled_games": int(len(q)),
        "week_completion_score_rows": int(len(q)),
        "week_completion_gate": "PASS",
    }


def attach_verified_actuals(
    projections: pd.DataFrame,
    actual: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Attach completed outcomes after projection freeze; require exact coverage."""
    p = projections.copy()
    a = actual.copy()
    r = roster.copy()

    idx = build_alias_index(a, r)
    actual_by_id = (
        a[["season", "week", "gsis_id", "rush_yards"]]
        .drop_duplicates(["season", "week", "gsis_id"], keep="last")
        .set_index(["season", "week", "gsis_id"])["rush_yards"]
        .to_dict()
    )
    roster_keys = set(zip(r["season"], r["week"], r["team"], r["gsis_id"]))

    rows: list[dict] = []
    exclusions: list[dict] = []
    stats_n = zero_n = 0
    for rec in p.to_dict("records"):
        player_key = str(rec["player_clean_key"])
        team = str(rec["team"])
        gsis, status = resolve_gsis(player_key, team, idx)
        if status != "RESOLVED_GSIS" or not gsis:
            exclusions.append({
                "player_clean_key": rec["player_clean_key"],
                "team": rec["team"],
                "reason": status,
            })
            continue

        key = (int(rec["season"]), int(rec["week"]), str(gsis))
        roster_key = (int(rec["season"]), int(rec["week"]), team, str(gsis))
        if key in actual_by_id:
            value = float(actual_by_id[key])
            source = "stats_table"
            stats_n += 1
        elif roster_key in roster_keys:
            value = 0.0
            source = "roster_confirmed_verified_zero"
            zero_n += 1
        else:
            exclusions.append({
                "player_clean_key": rec["player_clean_key"],
                "team": rec["team"],
                "reason": "NO_VERIFIED_COMPLETED_OUTCOME",
            })
            continue

        if not np.isfinite(value):
            raise RuntimeError("non-finite verified rushing outcome")
        rec["actual_rush_yards"] = value
        rec["actual_source"] = source
        rec["gsis_id"] = str(gsis)
        rows.append(rec)

    if exclusions:
        raise RuntimeError(
            "unresolved completed rushing outcome(s) after stats/roster verification: "
            + json.dumps(exclusions, sort_keys=True)
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero completed-history rows survived verified actual join")
    if out.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate completed-history identity after actual join")
    if len(out) != len(p):
        raise RuntimeError(
            f"completed-history outcome coverage is not exact: projections={len(p)} verified={len(out)}"
        )
    return out, {
        "verified_rows": int(len(out)),
        "verified_stats_table": int(stats_n),
        "verified_roster_zero": int(zero_n),
        "excluded_rows": 0,
        "exclusions": [],
    }

def append_to_prior_history(
    *,
    prior: pd.DataFrame,
    new_week: pd.DataFrame,
    completed_week: int,
) -> pd.DataFrame:
    p = validate_additional_history(prior)
    weeks = sorted(pd.to_numeric(p["week"], errors="coerce").dropna().astype(int).unique().tolist())
    expected_prior = list(range(1, int(completed_week)))
    if weeks != expected_prior:
        raise RuntimeError(
            f"prior certified history must be contiguous through Week {int(completed_week)-1}: "
            f"found={weeks} expected={expected_prior}"
        )
    new_weeks = sorted(
        pd.to_numeric(new_week["week"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if new_weeks != [int(completed_week)]:
        raise RuntimeError(f"new completed history must contain only Week {completed_week}: {new_weeks}")
    out = pd.concat([p, new_week], ignore_index=True, sort=False)
    out = validate_additional_history(out)
    identity = ["season", "week", "team", "player_clean_key"]
    if out.duplicated(identity).any():
        raise RuntimeError("cumulative certified history contains duplicate football identity")
    final_weeks = sorted(
        pd.to_numeric(out["week"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if final_weeks != list(range(1, int(completed_week) + 1)):
        raise RuntimeError(f"cumulative certified history is noncontiguous: {final_weeks}")
    return out


def build_completed_history(
    *,
    source_root: Path,
    source_run_metadata_path: Path,
    source_artifact_metadata_path: Path,
    prior_history_path: Path,
    completed_week: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    if int(completed_week) <= 1:
        raise RuntimeError("completed_week must be >= 2; Week 1 uses the P3-specific V1 builder")
    schedule = get_nfl_schedule(SEASON)
    provenance = validate_source_provenance(
        source_run_metadata=_read_json(source_run_metadata_path, "source run metadata"),
        source_artifact_metadata=_read_json(source_artifact_metadata_path, "source artifact metadata"),
        completed_week=int(completed_week),
        schedule=schedule,
    )

    projections, projection_audit = build_projection_frame(
        source_root=source_root,
        completed_week=int(completed_week),
        provenance=provenance,
    )

    completion_audit = assert_week_complete_after_freeze(int(completed_week))
    actual = load_actual_stats_unfiltered(SEASON, [int(completed_week)])
    roster = load_roster_identity(SEASON, [int(completed_week)])
    completed, actual_audit = attach_verified_actuals(projections, actual, roster)

    prior = _read_csv(prior_history_path, "prior certified cumulative history")
    cumulative = append_to_prior_history(
        prior=prior,
        new_week=completed,
        completed_week=int(completed_week),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    cumulative_path = out_dir / "rb_pd2_completed_2026_history.csv"
    week_path = out_dir / f"rb_pd2_completed_2026_week{int(completed_week)}_history.csv"
    audit_path = out_dir / "rb_pd2_completed_2026_history_audit.json"
    cumulative.to_csv(cumulative_path, index=False)
    completed.to_csv(week_path, index=False)

    audit = {
        "version": "RB_PD2_COMPLETED_2026_HISTORY_V2",
        "disposition": "COMPLETED_2026_HISTORY_CERTIFIED",
        "completed_2026_through_week": int(completed_week),
        "certified_weeks": list(range(1, int(completed_week) + 1)),
        "prior_history_path": str(prior_history_path),
        "prior_history_sha256": _file_sha256(prior_history_path),
        "projection_audit": projection_audit,
        "completion_audit": completion_audit,
        "actual_audit": actual_audit,
        "projection_frame_frozen_before_completion_check": True,
        "outcomes_loaded_after_projection_freeze": True,
        "prospective_confirmation_observations_created": 0,
        "sportsbook_inputs_used_for_projection": 0,
        "cumulative_rows": int(len(cumulative)),
        "new_week_rows": int(len(completed)),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return cumulative, audit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-run-metadata", type=Path, required=True)
    p.add_argument("--source-artifact-metadata", type=Path, required=True)
    p.add_argument("--prior-certified-history", type=Path, required=True)
    p.add_argument("--completed-week", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    out, audit = build_completed_history(
        source_root=a.source_root,
        source_run_metadata_path=a.source_run_metadata,
        source_artifact_metadata_path=a.source_artifact_metadata,
        prior_history_path=a.prior_certified_history,
        completed_week=a.completed_week,
        out_dir=a.out_dir,
    )
    print(
        f"[rb-pd2-2026-history-v2] certified_through_week={a.completed_week} "
        f"cumulative_rows={len(out)}"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
