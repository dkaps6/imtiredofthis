#!/usr/bin/env python3
"""One-shot RB Workhorse-Transition-Gate V1 two-rotation evaluation.

Implements the frozen plan (``docs/research/
RB_WORKHORSE_TRANSITION_GATE_V1_PLAN.md``, as amended by Section 16's
two-rotation chronology). Wires Gate 0 (generalized to the evaluated
earlier seasons), the frozen 13-feature event builder, and the frozen
two-rotation classifier pipeline into one run.

Outcomes (the realized ``WORKHORSE_EVENT`` labels) are attached only after
Gate 0 and feature-construction integrity both pass for the WHOLE 2019-2023
scored population -- matching the frozen plan's whole-experiment
fail-closed semantics (Section 15 point 2): a single unconstructible event
anywhere in the population stops the run before any label is ever
attached, not just for that event.

2024-2025 no-tuning transport (plan Section 12) is NOT executed by this
runner. It is only authorized if both rotations independently confirm,
and the frozen plan does not specify which rotation's fitted classifier
should drive the transport router when both confirm -- that is a genuine
ambiguity to resolve prospectively (matching the "stop and post it"
discipline already used twice in this research thread) rather than guess,
and it only matters if this run's result ever reaches that state. This
runner stops at the two-rotation confirmation result and reports it.

No sportsbook input. No production change.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.backtest.rb_lane_a_gate0_v1 import (
    gate02_report,
    gate03_event_report,
    gate03_report,
    harmonize_injury_state,
    harmonize_roster_membership,
)
from scripts.backtest.rb_lane_a_transition_detector_v1 import (
    build_detected_transitions,
    build_scored_v1_event_population,
)
from scripts.backtest.rb_workhorse_gate_v1_adequacy_census import compute_workhorse_events
from scripts.backtest.rb_workhorse_gate_v1_classifier import CONFIRMED, INSUFFICIENT_EVIDENCE, run_rotation
from scripts.backtest.rb_workhorse_gate_v1_event_population import filter_scored_events_to_scheduled_games
from scripts.backtest.rb_workhorse_gate_v1_features import FEATURES_CONSTRUCTIBLE, build_event_features

SCORED_WEEK_FLOOR = 2
SCORED_WEEK_CEIL = 18
EVALUATED_SEASONS = [2019, 2020, 2021, 2022, 2023]

ROTATIONS = {
    "A": {"fit": [2019, 2020], "cutoff": 2021, "confirm": 2022},
    "B": {"fit": [2019, 2020, 2021], "cutoff": 2022, "confirm": 2023},
}

GATE0_BLOCKED = "RB_WORKHORSE_TRANSITION_GATE_V1_GATE0_BLOCKED"
BOTH_ROTATIONS_CONFIRMED = "RB_WORKHORSE_TRANSITION_GATE_V1_CONFIRMED_EARLY_OOS"
NOT_QUALIFIED = "RB_WORKHORSE_TRANSITION_GATE_V1_NOT_QUALIFIED"


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _name_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def _git_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def build_gate0_and_scored_population() -> dict:
    """Gate 0 for the full evaluated earlier-season window, plus the scored
    loss/vacancy event population restricted to weeks 2-18 -- the exact same
    definition used throughout Lane-A V1/V2 and the adequacy census.

    Additionally intersects that population with canonical scheduled target
    -game team-weeks (Issue #535 comment `5718356931`, after CI run 3
    exposed 5 non-game rows -- legacy 17-week-season fictitious W18 rows and
    a COVID-postponed bye-shifted week -- that the frozen transition
    detector can flag but that have no game, and thus no Build-A production
    inputs, to build a feature row from). This is pre-outcome, additive, and
    does not alter the transition trigger logic itself; excluded rows are
    preserved with their reason, never silently dropped.
    """
    roster_state = harmonize_roster_membership(EVALUATED_SEASONS)
    injury_state = harmonize_injury_state(EVALUATED_SEASONS)

    gate02 = gate02_report(injury_state, EVALUATED_SEASONS)
    gate03 = gate03_report(roster_state, EVALUATED_SEASONS, oos_test_seasons=EVALUATED_SEASONS)

    detected = build_detected_transitions(roster_state, injury_state)
    scored_raw = build_scored_v1_event_population(detected)
    scored_raw = scored_raw.loc[scored_raw["week"].between(SCORED_WEEK_FLOOR, SCORED_WEEK_CEIL)].reset_index(drop=True)

    schedule_domain = filter_scored_events_to_scheduled_games(scored_raw, EVALUATED_SEASONS)
    scored = schedule_domain["retained_events"]

    gate03_events = gate03_event_report(scored, roster_state)

    gate0_pass = (
        gate02["disposition"] == "PASS"
        and gate03["disposition"] == "PASS_STRUCTURAL_EVENT_CHECKS_PENDING"
        and not gate03["failures"]
        and gate03_events["disposition"] == "PASS"
    )

    return {
        "roster_state": roster_state,
        "injury_state": injury_state,
        "scored_events_pre_schedule_domain": scored_raw,
        "scored_events": scored,
        "excluded_non_game_events": schedule_domain["excluded_events"],
        "schedule_domain_counts": {
            "events_checked": schedule_domain["events_checked"],
            "events_retained": schedule_domain["events_retained"],
            "events_excluded": schedule_domain["events_excluded"],
        },
        "gate0_pass": gate0_pass,
        "gate02": gate02,
        "gate03": gate03,
        "gate03_events": gate03_events,
    }


def assemble_final_disposition(rotation_reports: dict[str, dict]) -> str:
    """No rescue, no selective use of a surviving rotation (plan Section
    10/14/16): either rotation failing adequacy is INSUFFICIENT_EVIDENCE;
    either rotation otherwise not confirming is NOT_QUALIFIED; only both
    confirming yields the confirmed disposition.
    """
    dispositions = {name: r["confirmation"]["disposition"] for name, r in rotation_reports.items()}
    if any(d == INSUFFICIENT_EVIDENCE for d in dispositions.values()):
        return INSUFFICIENT_EVIDENCE
    if all(d == CONFIRMED for d in dispositions.values()):
        return BOTH_ROTATIONS_CONFIRMED
    return NOT_QUALIFIED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--player-logs", type=Path, required=True,
        help="combined M95Q player_game_logs_history.csv covering 2018-2023 (2018 supplies trailing history only)",
    )
    ap.add_argument(
        "--component-predictions-build-a", type=Path, required=True,
        help="combined canonical same-job Build-A component_predictions.csv covering 2019-2023",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gate0 = build_gate0_and_scored_population()
    gate0["roster_state"].to_csv(args.out_dir / "roster_state.csv", index=False)
    gate0["injury_state"].to_csv(args.out_dir / "injury_state.csv", index=False)
    gate0["scored_events"].to_csv(args.out_dir / "scored_v1_events.csv", index=False)

    pre_science = {
        "stage": "pre_outcome_integrity",
        "repo_head": _git_sha(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "evaluated_seasons": EVALUATED_SEASONS,
        "rotations": ROTATIONS,
        "gate0_2_injury_disposition": gate0["gate02"]["disposition"],
        "gate0_3_roster_structural_disposition": gate0["gate03"]["disposition"],
        "gate0_3_roster_structural_failures": gate0["gate03"]["failures"],
        "gate0_3_event_checks_disposition": gate0["gate03_events"]["disposition"],
        "gate0_3_event_checks_failures": gate0["gate03_events"]["failures"],
        "scored_events_count": int(len(gate0["scored_events"])),
        "sportsbook_inputs_used": 0,
    }
    _write_json(args.out_dir / "pre_outcome_integrity.json", pre_science)

    if not gate0["gate0_pass"]:
        final = {**pre_science, "final_disposition": GATE0_BLOCKED}
        _write_json(args.out_dir / "rb_workhorse_gate_v1_final_report.json", final)
        print(json.dumps(final, indent=2, default=str))
        return 0

    player_logs = _read_csv(args.player_logs, "combined player logs")
    # compute_role_weights_and_hhi (reused unchanged from
    # rb_lane_a_candidate_v1.py) requires a name_key join column on
    # player_logs. The raw M95Q CSV only carries player_clean_key; every
    # other caller (e.g. run_rb_lane_a_candidate_v1.py's
    # _prepare_player_logs) derives name_key the same deterministic way
    # before use. This is that same derivation, not a new join rule.
    source = player_logs["player_clean_key"] if "player_clean_key" in player_logs.columns else player_logs.get(
        "player", pd.Series("", index=player_logs.index)
    )
    player_logs["name_key"] = source.map(_name_key)
    component_predictions = _read_csv(args.component_predictions_build_a, "combined Build-A component predictions")

    feature_result = build_event_features(
        scored_events=gate0["scored_events"],
        roster_state=gate0["roster_state"],
        player_logs=player_logs,
        component_predictions_build_a=component_predictions,
    )
    _write_json(
        args.out_dir / "feature_construction.json",
        {k: v for k, v in feature_result.items() if k != "feature_rows"},
    )
    if feature_result["disposition"] != FEATURES_CONSTRUCTIBLE:
        final = {
            **pre_science,
            "feature_construction_disposition": feature_result["disposition"],
            "events_failing": feature_result["events_failing"],
            "final_disposition": feature_result["disposition"],
            "note": "Outcomes were never opened; the whole-experiment fail-closed rule stopped the run at feature construction.",
        }
        _write_json(args.out_dir / "rb_workhorse_gate_v1_final_report.json", final)
        print(json.dumps(final, indent=2, default=str))
        return 0

    features = feature_result["feature_rows"]

    # ------------------------------ OUTCOMES OPEN HERE ------------------------------
    # Everything above is pre-outcome integrity and pregame feature
    # construction only. From this line on, realized WORKHORSE_EVENT labels
    # are attached and used solely for the preregistered fit/cutoff/confirm
    # sequence, exactly once per rotation.
    workhorse_rows = compute_workhorse_events(
        gate0["scored_events"], gate0["roster_state"], gate0["injury_state"], player_logs
    )
    labeled = features.merge(
        workhorse_rows[["season", "week", "team", "workhorse_event"]],
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    if labeled["workhorse_event"].isna().any():
        raise RuntimeError("unresolved workhorse_event label after feature constructibility passed")
    labeled["workhorse_event"] = labeled["workhorse_event"].astype(int)
    labeled.to_csv(args.out_dir / "labeled_feature_rows.csv", index=False)

    rotation_reports: dict[str, dict] = {}
    for name, spec in ROTATIONS.items():
        fit_mask = labeled["season"].isin(spec["fit"])
        cutoff_mask = labeled["season"].eq(spec["cutoff"])
        confirm_mask = labeled["season"].eq(spec["confirm"])
        result = run_rotation(
            fit_rows=labeled.loc[fit_mask],
            fit_labels=labeled.loc[fit_mask, "workhorse_event"],
            cutoff_rows=labeled.loc[cutoff_mask],
            cutoff_labels=labeled.loc[cutoff_mask, "workhorse_event"],
            confirm_rows=labeled.loc[confirm_mask],
            confirm_labels=labeled.loc[confirm_mask, "workhorse_event"],
            sportsbook_inputs_used=0,
        )
        rotation_reports[name] = result
        _write_json(
            args.out_dir / f"rotation_{name}_report.json",
            {k: v for k, v in result.items() if k != "fitted_objects"},
        )

    final_disposition = assemble_final_disposition(rotation_reports)

    final_report = {
        **pre_science,
        "feature_construction_disposition": feature_result["disposition"],
        "labeled_rows": int(len(labeled)),
        "rotation_a": {k: v for k, v in rotation_reports["A"].items() if k != "fitted_objects"},
        "rotation_b": {k: v for k, v in rotation_reports["B"].items() if k != "fitted_objects"},
        "final_disposition": final_disposition,
        "transport_2024_2025": (
            "NOT_EXECUTED -- authorized only if final_disposition is "
            f"{BOTH_ROTATIONS_CONFIRMED}; which rotation's classifier drives the "
            "transport router is not yet specified in the frozen plan and requires "
            "prospective clarification before execution, per the 'stop and post it' "
            "discipline already used for the plan's other ambiguities."
        ),
        "stop_rule": "No rescue tuning, feature changes, cutoff changes, or rotation substitution after this result.",
    }
    _write_json(args.out_dir / "rb_workhorse_gate_v1_final_report.json", final_report)
    print(json.dumps(final_report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
