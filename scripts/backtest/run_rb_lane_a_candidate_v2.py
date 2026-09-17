#!/usr/bin/env python3
"""One-shot RB Lane A transition-gated allocation evaluation -- V2.

V1 (``run_rb_lane_a_candidate_v1.py``) is terminal and immutable: it failed
closed at Amendment-8 rush-yard-translation constructibility in both
rotations (run ``35167213629``, preserved artifact digest
``sha256:8c92c92ae713f4eedeb73acfdb94f3f0191c9fb7d5a79794581107eaa17e96ea``)
before any outcome was opened. V2
(``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V2_PLAN.md``, freeze
commit ``dfcd45ed``) is a new prospective experiment, not a V1 rescue.

The only scientific change from V1: the HHI-dampened carry pool is
reallocated only among "eligible recipients" -- active-room RB-room players
who also have an exact-identity, finite match in the dual-market promotion
comparator. Everything else (Gate 0, same-job authority, Rotation-2
mechanism-comparator diagnostic, conservation pool, prior-3 role weights,
pre-transition HHI, ``p = 1 + 2H``, Amendment-8 held-incumbent-efficiency
translation, both temporal rotations, all outcome gates) is inherited
unchanged from V1.

This runner deliberately separates construction from outcome scoring: no
actual/postgame field is accessed until Gate 0, authority reconstruction,
mechanism-comparator integrity, V2 recipient-universe integrity, V2
recipient-weight integrity, and rush-yard translation constructibility have
all passed for both rotations.

No model fitting, feature search, threshold tuning, sportsbook input, or
post-result rescue path exists here.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.rb_lane_a_candidate_v1 import (
    build_deployable_candidate,
    build_mechanism_diagnostic,
    check_rush_yard_translation_constructibility,
    check_stable_identity_gate,
    compute_conservation_pool,
    compute_historical_rb_room_rush_share,
    compute_hhi_dampened_reallocation,
    compute_incumbent_ypc,
    compute_role_weights_and_hhi,
    translate_candidate_rush_yards,
)
from scripts.backtest.rb_lane_a_candidate_v2 import (
    V2_RECIPIENT_UNIVERSE_FAILURE,
    V2_RECIPIENT_UNIVERSE_OK,
    V2_RECIPIENT_WEIGHT_FAILURE,
    V2_RECIPIENT_WEIGHT_OK,
    aggregate_recipient_integrity_disposition,
    check_recipient_universe_integrity,
    check_recipient_weight_integrity,
    filter_eligible_recipients,
)
from scripts.backtest.rb_lane_a_comparator_reconstruction_v1 import (
    blob_sha_of,
    build_dual_market_promotion_comparator,
    build_input_manifest,
    same_job_double_build_disposition,
)
from scripts.backtest.rb_lane_a_final_gates_v1 import (
    BASELINE_FAILURE,
    CONSTRUCTIBILITY_FAILURE,
    GATE0_BLOCKED,
    assemble_final_disposition,
    conservation_gate_from_event_meta,
    p90_catastrophic_protection_check,
    protected_cohort_gate_report,
    transition_mae_gate,
)
from scripts.backtest.rb_lane_a_gate_scoring_v1 import (
    adequacy_check,
    bootstrap_gate_report,
    compute_mae_delta_rows,
    per_season_nonregression_check,
    whole_season_deployable_safety_check,
)
from scripts.backtest.rb_lane_a_transition_detector_v1 import build_player_week_status


RB_POS = {"RB", "FB", "HB"}
ROTATIONS = {
    1: {"test_season": 2024, "prior_season": 2023},
    2: {"test_season": 2025, "prior_season": 2024},
}
STACK2_TEAM = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAX": "JAC", "ARZ": "ARI", "WSH": "WAS"}
V2_RECIPIENT_UNIVERSE_GATE_FAILURE = "V2_RECIPIENT_UNIVERSE_GATE_FAILURE"
V2_RECIPIENT_WEIGHT_GATE_FAILURE = "V2_RECIPIENT_WEIGHT_GATE_FAILURE"


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text())


def _name_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _stack2_team(value) -> str:
    s = str(value or "").upper().strip()
    return STACK2_TEAM.get(s, s)


def _git_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def build_authority_evidence(rotation: int, root: Path) -> tuple[pd.DataFrame, dict]:
    """Re-prove Amendment 6 inside the candidate run and return Build A."""
    paths = {
        "player_game_logs_history": root / "player_game_logs_history.csv",
        "team_weekly_history": root / "team_weekly_history.csv",
        "schedule_history": root / "schedule_history.csv",
        "injuries_history": root / "injuries_history.csv",
        "weather_history": root / "weather_history.csv",
    }
    universe = root / "pregame_universe"
    for p in sorted(universe.glob("*.csv")):
        paths[f"pregame_universe/{p.name}"] = p

    build_a = _read_csv(root / "component_predictions_build_a.csv", f"rotation {rotation} Build A")
    build_b = _read_csv(root / "component_predictions_build_b.csv", f"rotation {rotation} Build B")
    evidence = {
        "rotation": rotation,
        "test_season": ROTATIONS[rotation]["test_season"],
        "input_manifest_sha256": build_input_manifest(paths),
        "code_sha": {
            "repo_head": _git_sha(),
            "simulation_v2": blob_sha_of(Path(".").resolve(), "scripts/simulation_v2.py"),
            "walk_forward": blob_sha_of(Path(".").resolve(), "scripts/backtest/walk_forward.py"),
            "component_predictions": blob_sha_of(Path(".").resolve(), "scripts/backtest/component_predictions.py"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seed_policy": "42 + week",
        "iterations": 2000,
        "same_job_double_build": same_job_double_build_disposition(build_a, build_b),
    }
    return build_a, evidence


def _prepare_player_logs(root: Path) -> pd.DataFrame:
    logs = _read_csv(root / "player_game_logs_history.csv", "historical player logs")
    source = logs["player_clean_key"] if "player_clean_key" in logs.columns else logs.get("player", pd.Series("", index=logs.index))
    logs["name_key"] = source.map(_name_key)
    return logs


def _all_rb_rush_yards_rows(component_predictions: pd.DataFrame, dual: pd.DataFrame) -> pd.DataFrame:
    cp = component_predictions.copy()
    market = cp.get("market", pd.Series("", index=cp.index)).fillna("").astype(str).str.lower()
    pos = cp.get("position", pd.Series("", index=cp.index)).fillna("").astype(str).str.upper()
    week = pd.to_numeric(cp.get("week"), errors="coerce")
    x = cp.loc[market.eq("rush_yards") & pos.isin(RB_POS) & week.between(2, 18)].copy()
    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        raise RuntimeError("RB rush-yards Build-A rows contain duplicate player identities")
    x["actual_rush_yards"] = pd.to_numeric(x.get("actual"), errors="coerce")
    x["actual_rush_att"] = pd.to_numeric(x.get("actual_opportunities"), errors="coerce")
    x["name_key"] = x["player_clean_key"].map(_name_key)
    event_id = x.get("event_id", pd.Series("", index=x.index)).fillna("").astype(str)
    x["game_key"] = (
        x["season"].astype(str) + "-W" + x["week"].astype(int).astype(str).str.zfill(2) + "-" + event_id
    )
    keep = keys + [
        "player", "position", "opponent", "event_id", "name_key", "game_key",
        "actual_rush_yards", "actual_rush_att",
    ]
    keep = [c for c in keep if c in x.columns]
    x = x[keep].merge(
        dual[keys + ["promotion_rush_att", "promotion_rush_yards"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if x["promotion_rush_yards"].isna().any():
        raise RuntimeError("promotion rush-yards comparator missing on whole-season RB rows")
    return x


def construct_rotation_outcome_blind_v2(
    rotation: int,
    root: Path,
    component_predictions: pd.DataFrame,
    roster_state: pd.DataFrame,
    injury_state: pd.DataFrame,
    scored_events: pd.DataFrame,
) -> dict:
    """Build V2 Lane-A projections without accessing actual outcome columns.

    Identical to V1's construction except: for every scored event, the
    active room is first restricted to the V2 eligible-recipient
    intersection (this rotation's dual-market comparator, built first) before
    role weights/HHI/reallocation run. Pre-transition-room HHI is computed
    from the full (unfiltered) pre-transition room, per the V2 plan.
    """
    test_season = ROTATIONS[rotation]["test_season"]
    logs = _prepare_player_logs(root)
    dual = build_dual_market_promotion_comparator(component_predictions, rotation)

    status = build_player_week_status(roster_state, injury_state)
    events = scored_events.loc[
        pd.to_numeric(scored_events["season"], errors="coerce").eq(test_season)
        & pd.to_numeric(scored_events["week"], errors="coerce").between(2, 18)
    ].copy()

    room_share = compute_historical_rb_room_rush_share(logs)
    pool = compute_conservation_pool(component_predictions, room_share)

    candidate_att_chunks: list[pd.DataFrame] = []
    event_meta: list[dict] = []
    universe_reports: list[dict] = []
    weight_reports: list[dict] = []
    for e in events.itertuples(index=False):
        event_key = {"season": int(e.season), "week": int(e.week), "team": e.team}

        cur = status.loc[
            status["season"].eq(int(e.season))
            & status["week"].eq(int(e.week))
            & status["team"].eq(e.team)
        ].copy()
        cur = cur.loc[~cur["unavailable"].astype(bool)].copy()
        if cur.duplicated(["player_clean_key"]).any():
            raise RuntimeError(f"duplicate active-room player identity: {e.season} W{e.week} {e.team}")

        eligible_cur, excluded_cur = filter_eligible_recipients(cur, dual)
        universe_report = check_recipient_universe_integrity(event_key, eligible_cur)
        universe_reports.append({**universe_report, "n_excluded_ineligible": int(len(excluded_cur))})
        if universe_report["disposition"] == V2_RECIPIENT_UNIVERSE_FAILURE:
            continue

        pre = roster_state.loc[
            roster_state["season"].eq(int(e.prior_season))
            & roster_state["week"].eq(int(e.prior_week))
            & roster_state["team"].eq(e.team)
        ].copy()
        # Frozen HHI coordinate contract (head 216cd8d), unchanged by V2: HHI
        # is computed over the FULL pre-transition room, never the filtered
        # eligible-recipient set.
        pre["season"] = int(e.season)
        pre["week"] = int(e.week)

        pool_match = pool.loc[
            pool["season"].eq(int(e.season))
            & pool["week"].eq(int(e.week))
            & pool["team"].eq(e.team)
        ]
        if len(pool_match) != 1 or not np.isfinite(pd.to_numeric(pool_match["pool"], errors="coerce").iloc[0]):
            raise RuntimeError(f"unresolvable conservation pool: {e.season} W{e.week} {e.team}")
        pool_row = pool_match.iloc[0].to_dict()

        active_enriched, hhi_lookup = compute_role_weights_and_hhi(eligible_cur, pre, logs)
        weight_report = check_recipient_weight_integrity(event_key, active_enriched)
        weight_reports.append(weight_report)
        if weight_report["disposition"] == V2_RECIPIENT_WEIGHT_FAILURE:
            continue

        allocation, meta = compute_hhi_dampened_reallocation(active_enriched, hhi_lookup, pool_row)
        meta.update({"rotation": rotation, **event_key})
        delta = abs(float(pd.to_numeric(allocation["candidate_att"], errors="coerce").sum()) - float(pool_row["pool"]))
        meta["conservation_delta"] = delta
        event_meta.append(meta)
        candidate_att_chunks.append(allocation)

    recipient_universe_integrity = aggregate_recipient_integrity_disposition(
        universe_reports, V2_RECIPIENT_UNIVERSE_OK, V2_RECIPIENT_UNIVERSE_FAILURE
    )
    recipient_weight_integrity = aggregate_recipient_integrity_disposition(
        weight_reports, V2_RECIPIENT_WEIGHT_OK, V2_RECIPIENT_WEIGHT_FAILURE
    )

    candidate_att = pd.concat(candidate_att_chunks, ignore_index=True, sort=False) if candidate_att_chunks else pd.DataFrame(
        columns=["season", "week", "team", "player_clean_key", "name_key", "candidate_att"]
    )
    constructibility = check_rush_yard_translation_constructibility(candidate_att, dual)

    context = {
        "rotation": rotation,
        "test_season": test_season,
        "component_predictions": component_predictions,
        "player_logs": logs,
        "dual": dual,
        "events": events,
        "candidate_att": candidate_att,
        "event_meta": event_meta,
        "recipient_universe_integrity": recipient_universe_integrity,
        "recipient_weight_integrity": recipient_weight_integrity,
        "constructibility": constructibility,
    }
    if (
        recipient_universe_integrity["disposition"] != V2_RECIPIENT_UNIVERSE_OK
        or recipient_weight_integrity["disposition"] != V2_RECIPIENT_WEIGHT_OK
        or constructibility["disposition"] != "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE"
    ):
        return context

    incumbent = compute_incumbent_ypc(dual)
    context["incumbent_ypc"] = incumbent
    context["candidate"] = translate_candidate_rush_yards(candidate_att, incumbent)
    return context


def _mechanism_diagnostic_report(scored_eval: pd.DataFrame, casebook: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cand = scored_eval.copy()
    cand["team"] = cand["team"].map(_stack2_team)
    mech = casebook.copy()
    mech["team"] = mech["team"].map(_stack2_team)
    diag = build_mechanism_diagnostic(cand, mech)
    actual = pd.to_numeric(diag.get("actual_rush_yards"), errors="coerce")
    c = pd.to_numeric(diag.get("candidate_rush_yards"), errors="coerce")
    b = pd.to_numeric(diag.get("mechanism_comparator_rush_yards"), errors="coerce")
    ok = actual.notna() & c.notna() & b.notna()
    if not ok.any():
        return {"n": 0, "candidate_mae": None, "mechanism_comparator_mae": None}, diag
    return {
        "n": int(ok.sum()),
        "candidate_mae": float((c[ok] - actual[ok]).abs().mean()),
        "mechanism_comparator_mae": float((b[ok] - actual[ok]).abs().mean()),
    }, diag


def score_rotation(rotation: int, context: dict, out_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Outcome-touching stage. Called only after both rotations pass all
    pre-outcome integrity checks, including the two new V2 recipient checks.
    """
    candidate = context["candidate"].copy()
    all_rows = _all_rb_rush_yards_rows(context["component_predictions"], context["dual"])
    keys = ["season", "week", "team", "player_clean_key"]

    scored_eval = candidate.merge(
        all_rows[keys + ["actual_rush_yards", "actual_rush_att", "game_key"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if scored_eval[["actual_rush_yards", "actual_rush_att"]].isna().any().any():
        raise RuntimeError("constructed scored candidate has unresolved outcome row after constructibility passed")

    deployable = build_deployable_candidate(all_rows, candidate)
    deployable = deployable.merge(
        all_rows[keys + ["actual_rush_yards", "actual_rush_att", "game_key"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    cohort_masks = {
        "actual_carries_20_plus": pd.to_numeric(scored_eval["actual_rush_att"], errors="coerce") >= 20,
        "actual_rush_yards_100_plus": pd.to_numeric(scored_eval["actual_rush_yards"], errors="coerce") >= 100,
    }
    adequacy = adequacy_check(scored_eval, cohort_masks)
    transition = transition_mae_gate(scored_eval)
    protected = protected_cohort_gate_report(scored_eval)
    p90 = p90_catastrophic_protection_check(scored_eval)
    delta_rows = compute_mae_delta_rows(
        scored_eval,
        candidate_col="candidate_rush_yards",
        comparator_col="promotion_rush_yards",
        actual_col="actual_rush_yards",
    )
    bootstrap = bootstrap_gate_report(delta_rows)
    stable = check_stable_identity_gate(deployable)
    whole = whole_season_deployable_safety_check(deployable)
    conservation = conservation_gate_from_event_meta(context["event_meta"])

    scored_eval.to_csv(out_dir / f"rotation_{rotation}_scored_transition_rows.csv", index=False)
    deployable.to_csv(out_dir / f"rotation_{rotation}_deployable_whole_season_rows.csv", index=False)
    pd.DataFrame(context["event_meta"]).to_csv(out_dir / f"rotation_{rotation}_event_construction_meta.csv", index=False)
    delta_rows.to_csv(out_dir / f"rotation_{rotation}_paired_mae_deltas.csv", index=False)

    report = {
        "rotation": rotation,
        "test_season": context["test_season"],
        "scored_candidate_rows": int(len(scored_eval)),
        "scored_transition_team_weeks_before_all_zero_exclusion": int(len(context["events"])),
        "recipient_universe_integrity": context["recipient_universe_integrity"],
        "recipient_weight_integrity": context["recipient_weight_integrity"],
        "constructibility": context["constructibility"],
        "adequacy": adequacy,
        "transition_mae_gate": transition,
        "protected_cohorts": protected,
        "p90_protection": p90,
        "bootstrap": bootstrap,
        "stable_identity": stable,
        "whole_season_safety": whole,
        "conservation": conservation,
    }
    return report, delta_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate0-dir", type=Path, required=True)
    ap.add_argument("--rotation-1-root", type=Path, required=True)
    ap.add_argument("--rotation-2-root", type=Path, required=True)
    ap.add_argument("--mechanism-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gate0 = _read_json(args.gate0_dir / "gate0_report.json", "Gate 0 report")
    roster = _read_csv(args.gate0_dir / "gate0_3_roster_state.csv", "Gate 0 roster state")
    injuries = _read_csv(args.gate0_dir / "gate0_2_injury_state.csv", "Gate 0 injury state")
    events = _read_csv(args.gate0_dir / "gate0_scored_v1_events.csv", "Gate 0 scored V1 events")

    mech_reports = list(args.mechanism_dir.rglob("rb_lane_a_mechanism_comparator_reconstruction_report.json"))
    mech_casebooks = list(args.mechanism_dir.rglob("stack2_2025_casebook.csv"))
    if len(mech_reports) != 1 or len(mech_casebooks) != 1:
        raise RuntimeError(
            f"expected one accepted mechanism report/casebook, found reports={len(mech_reports)} casebooks={len(mech_casebooks)}"
        )
    mechanism_report = _read_json(mech_reports[0], "mechanism reconstruction report")
    mechanism_disposition = mechanism_report.get("mechanism_parity", {}).get("disposition", "")
    mechanism_casebook = _read_csv(mech_casebooks[0], "mechanism casebook")

    roots = {1: args.rotation_1_root, 2: args.rotation_2_root}
    build_a: dict[int, pd.DataFrame] = {}
    authority: dict[int, dict] = {}
    for rotation in (1, 2):
        build_a[rotation], authority[rotation] = build_authority_evidence(rotation, roots[rotation])
        _write_json(args.out_dir / f"rotation_{rotation}_authority.json", authority[rotation])

    authority_dispositions = {
        r: authority[r]["same_job_double_build"].get("disposition", "") for r in (1, 2)
    }
    gate0_disposition = gate0.get("gate0_overall_disposition", "")

    pre_science = {
        "stage": "pre_outcome_integrity",
        "candidate_version": "V2",
        "v1_terminal_disposition": "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE (both rotations, run 35167213629, preserved not rescued)",
        "repo_head": _git_sha(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "gate0_disposition": gate0_disposition,
        "authority_dispositions": authority_dispositions,
        "mechanism_rotation2_disposition": mechanism_disposition,
        "sportsbook_inputs_used": 0,
    }
    _write_json(args.out_dir / "pre_outcome_integrity.json", pre_science)

    # Fail before candidate construction/scoring if the frozen upstream authorities
    # are not intact.
    if gate0_disposition != "GATE0_PASS":
        pre_science["final_disposition"] = GATE0_BLOCKED
        _write_json(args.out_dir / "rb_lane_a_final_report.json", pre_science)
        return 0
    if any(v != "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS" for v in authority_dispositions.values()) or mechanism_disposition != "MECHANISM_AUTHORITY_RECONSTRUCTION_PASS":
        pre_science["final_disposition"] = BASELINE_FAILURE
        _write_json(args.out_dir / "rb_lane_a_final_report.json", pre_science)
        return 0

    contexts = {
        1: construct_rotation_outcome_blind_v2(1, roots[1], build_a[1], roster, injuries, events),
        2: construct_rotation_outcome_blind_v2(2, roots[2], build_a[2], roster, injuries, events),
    }
    for r in (1, 2):
        _write_json(args.out_dir / f"rotation_{r}_recipient_universe_integrity.json", contexts[r]["recipient_universe_integrity"])
        _write_json(args.out_dir / f"rotation_{r}_recipient_weight_integrity.json", contexts[r]["recipient_weight_integrity"])
        _write_json(args.out_dir / f"rotation_{r}_constructibility.json", contexts[r]["constructibility"])
        contexts[r]["candidate_att"].to_csv(args.out_dir / f"rotation_{r}_candidate_att_outcome_blind.csv", index=False)

    universe_dispositions = {r: contexts[r]["recipient_universe_integrity"].get("disposition", "") for r in (1, 2)}
    if any(v != V2_RECIPIENT_UNIVERSE_OK for v in universe_dispositions.values()):
        report = {
            **pre_science,
            "recipient_universe_dispositions": universe_dispositions,
            "final_disposition": V2_RECIPIENT_UNIVERSE_GATE_FAILURE,
            "note": "Outcome scoring not executed; V2 recipient-universe integrity failed closed.",
        }
        _write_json(args.out_dir / "rb_lane_a_final_report.json", report)
        return 0

    weight_dispositions = {r: contexts[r]["recipient_weight_integrity"].get("disposition", "") for r in (1, 2)}
    if any(v != V2_RECIPIENT_WEIGHT_OK for v in weight_dispositions.values()):
        report = {
            **pre_science,
            "recipient_weight_dispositions": weight_dispositions,
            "final_disposition": V2_RECIPIENT_WEIGHT_GATE_FAILURE,
            "note": "Outcome scoring not executed; V2 recipient-weight integrity failed closed.",
        }
        _write_json(args.out_dir / "rb_lane_a_final_report.json", report)
        return 0

    constructibility_dispositions = {r: contexts[r]["constructibility"].get("disposition", "") for r in (1, 2)}
    if any(v != "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE" for v in constructibility_dispositions.values()):
        report = {
            **pre_science,
            "constructibility_dispositions": constructibility_dispositions,
            "final_disposition": CONSTRUCTIBILITY_FAILURE,
            "note": "Outcome scoring not executed; Amendment-8 constructibility failed closed.",
        }
        _write_json(args.out_dir / "rb_lane_a_final_report.json", report)
        return 0

    # ------------------------------ OUTCOMES OPEN HERE ------------------------------
    # Everything above is candidate construction/integrity only.  From this line on,
    # actuals are used solely for the preregistered evaluation gates.
    rotation_reports: dict[int, dict] = {}
    delta_rows: dict[int, pd.DataFrame] = {}
    for r in (1, 2):
        rotation_reports[r], delta_rows[r] = score_rotation(r, contexts[r], args.out_dir)

    per_season = per_season_nonregression_check(delta_rows)

    # Rotation-2 mechanism evidence is informative only (Amendment 7); it never
    # participates in the qualification disposition.
    scored2 = pd.read_csv(args.out_dir / "rotation_2_scored_transition_rows.csv")
    mech_diag_report, mech_diag = _mechanism_diagnostic_report(scored2, mechanism_casebook)
    mech_diag.to_csv(args.out_dir / "rotation_2_mechanism_diagnostic_rows.csv", index=False)

    final_disposition = assemble_final_disposition(
        gate0_disposition=gate0_disposition,
        authority_dispositions=authority_dispositions,
        mechanism_rotation2_disposition=mechanism_disposition,
        constructibility_dispositions=constructibility_dispositions,
        adequacy_dispositions={r: rotation_reports[r]["adequacy"]["disposition"] for r in (1, 2)},
        transition_mae_dispositions={r: rotation_reports[r]["transition_mae_gate"]["disposition"] for r in (1, 2)},
        protected_cohort_dispositions={r: rotation_reports[r]["protected_cohorts"]["disposition"] for r in (1, 2)},
        bootstrap_dispositions={r: rotation_reports[r]["bootstrap"]["disposition"] for r in (1, 2)},
        p90_dispositions={r: rotation_reports[r]["p90_protection"]["disposition"] for r in (1, 2)},
        stable_identity_dispositions={r: rotation_reports[r]["stable_identity"]["disposition"] for r in (1, 2)},
        whole_season_safety_dispositions={r: rotation_reports[r]["whole_season_safety"]["disposition"] for r in (1, 2)},
        conservation_dispositions={r: rotation_reports[r]["conservation"]["disposition"] for r in (1, 2)},
        per_season_nonregression_disposition=per_season["disposition"],
        sportsbook_inputs_used=0,
    )

    final_report = {
        "repo_head": _git_sha(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "candidate_version": "V2",
        "candidate_exposure": "ONE_SHOT_FROZEN_EVALUATION",
        "sportsbook_inputs_used": 0,
        "gate0_disposition": gate0_disposition,
        "authority_dispositions": authority_dispositions,
        "mechanism_rotation2_integrity_disposition": mechanism_disposition,
        "mechanism_rotation1_disposition": "NOT_CONSTRUCTIBLE_NO_CASEBOOK (non-blocking, Amendment 7)",
        "recipient_universe_dispositions": universe_dispositions,
        "recipient_weight_dispositions": weight_dispositions,
        "constructibility_dispositions": constructibility_dispositions,
        "rotations": {str(r): rotation_reports[r] for r in (1, 2)},
        "per_season_nonregression": per_season,
        "mechanism_diagnostic_rotation2": mech_diag_report,
        "final_disposition": final_disposition,
        "stop_rule": "No rescue tuning, threshold changes, cohort search, or same-information retest after this result.",
    }
    _write_json(args.out_dir / "rb_lane_a_final_report.json", final_report)
    print(json.dumps(final_report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
