"""RB Workhorse-Transition-Gate V1 -- bounded counts-only adequacy census.

Authorized by GPT-5.6 (Issue #535 comment `5714574380`): report, for each
season in the requested earlier-season window, on the SAME scored-V1/V2
loss/vacancy transition population the frozen Lane-A V2 mechanism is
eligible to fire on (never the broader `detected_transition` disclosure
population, per comment `5714574380` point 1):

* the number of scored transition events;
* the number of positive `WORKHORSE_EVENT` events;
* positive prevalence;
* source/timing integrity disposition needed to trust those counts.

This script fits NOTHING: no classifier, no probabilities, no
feature-outcome inspection, no cutoff selection, no V2 allocation. It is
hard-bounded to seasons <=2023 -- it must never touch 2024-25 outcomes,
which already informed the Workhorse-Gate hypothesis via Lane-A V2 and are
reserved for later no-tuning transport only.

`WORKHORSE_EVENT = 1` iff at least one post-transition active RB/FB/HB in
that scored transition records `actual carries >= 20` in that game.
Realized carries are target/evaluation only here -- this module builds no
pregame feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.backtest.historical_inputs import build_schedule_history
from scripts.backtest.historical_player_logs_m95q import build_logs as build_m95q_logs
from scripts.backtest.rb_lane_a_gate0_v1 import (
    gate02_report,
    gate03_event_report,
    gate03_report,
    harmonize_injury_state,
    harmonize_roster_membership,
)
from scripts.backtest.rb_lane_a_transition_detector_v1 import (
    build_detected_transitions,
    build_player_week_status,
    build_scored_v1_event_population,
)
from scripts.backtest.rb_workhorse_gate_v1_event_population import filter_scored_events_to_scheduled_games

SCORED_WEEK_FLOOR = 2
SCORED_WEEK_CEIL = 18
WORKHORSE_CARRY_FLOOR = 20
CENSUS_MAX_SEASON = 2023  # hard bound: 2024-25 must never be touched by this census


def compute_workhorse_events(
    scored_events: pd.DataFrame, roster_state: pd.DataFrame, injury_state: pd.DataFrame, player_logs: pd.DataFrame
) -> pd.DataFrame:
    """One row per scored transition event with the realized `workhorse_event`
    flag -- evaluation-only, never a pregame feature. Active room is the same
    unavailable-filtered post-transition room V1/V2 use.
    """
    status = build_player_week_status(roster_state, injury_state)
    logs = player_logs.copy()
    logs.columns = [str(c).strip().lower() for c in logs.columns]
    logs["rushes"] = pd.to_numeric(logs.get("rushes"), errors="coerce").fillna(0.0)

    rows = []
    for e in scored_events.itertuples(index=False):
        active = status.loc[
            status["season"].eq(e.season) & status["week"].eq(e.week) & status["team"].eq(e.team)
        ]
        active = active.loc[~active["unavailable"].astype(bool)]
        active_keys = set(active.get("player_clean_key", pd.Series(dtype=str)))

        week_logs = logs.loc[
            logs["season"].eq(e.season) & logs["week"].eq(e.week) & logs["team"].eq(e.team)
        ]
        active_week_logs = week_logs.loc[week_logs["player_clean_key"].isin(active_keys)]

        max_carries = float(active_week_logs["rushes"].max()) if len(active_week_logs) else 0.0
        rows.append(
            {
                "season": int(e.season),
                "week": int(e.week),
                "team": e.team,
                "n_active_room": int(len(active)),
                "n_active_room_with_realized_log": int(len(active_week_logs)),
                "max_active_realized_carries": max_carries,
                "workhorse_event": bool(max_carries >= WORKHORSE_CARRY_FLOOR),
            }
        )
    return pd.DataFrame(rows)


def season_census(workhorse_rows: pd.DataFrame, seasons: list[int]) -> dict:
    out = {}
    for season in seasons:
        g = workhorse_rows.loc[workhorse_rows["season"] == season]
        n = int(len(g))
        pos = int(g["workhorse_event"].sum()) if n else 0
        out[str(season)] = {
            "n_scored_transition_events": n,
            "n_positive_workhorse_events": pos,
            "positive_prevalence": (pos / n) if n else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RB Workhorse-Gate V1 adequacy census (counts only, no fitting)")
    ap.add_argument("--seasons", default="2019,2020,2021,2022,2023")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    seasons = sorted({int(s) for s in args.seasons.split(",") if s.strip()})
    if any(s > CENSUS_MAX_SEASON for s in seasons):
        raise RuntimeError(
            f"this census is hard-bounded to seasons <= {CENSUS_MAX_SEASON}; "
            f"2024-25 must never be touched here (Issue #535 comment 5714574380)"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roster_state = harmonize_roster_membership(seasons)
    roster_state.to_csv(args.out_dir / "roster_state.csv", index=False)
    injury_state = harmonize_injury_state(seasons)
    injury_state.to_csv(args.out_dir / "injury_state.csv", index=False)

    gate02 = gate02_report(injury_state, seasons)
    # Additive oos_test_seasons parameter (rb_lane_a_gate0_v1.py) -- exercises
    # real requirement-4 schedule-coverage completeness for these earlier
    # seasons instead of silently not running it, per the Workhorse-Gate V1
    # plan's mandatory Gate-0 repair (Section 6).
    gate03 = gate03_report(roster_state, seasons, oos_test_seasons=seasons)

    detected = build_detected_transitions(roster_state, injury_state)
    scored_raw = build_scored_v1_event_population(detected)
    scored_raw = scored_raw.loc[scored_raw["week"].between(SCORED_WEEK_FLOOR, SCORED_WEEK_CEIL)].reset_index(drop=True)
    scored_raw.to_csv(args.out_dir / "scored_v1_events_pre_schedule_domain.csv", index=False)

    # Schedule-domain correction (Issue #535 comment 5718356931): intersect
    # with canonical scheduled target-game team-weeks so the census -- and
    # everything downstream of it -- never counts a non-game row (legacy
    # 17-week-season fictitious W18, a COVID-postponed bye-shifted week,
    # etc.) as a scored event. Additive, pre-outcome; does not touch the
    # transition detector itself. Excluded rows are preserved, not dropped.
    schedule_domain = filter_scored_events_to_scheduled_games(scored_raw, seasons)
    scored = schedule_domain["retained_events"]
    scored.to_csv(args.out_dir / "scored_v1_events.csv", index=False)
    schedule_domain["excluded_events"].to_csv(args.out_dir / "excluded_non_game_events.csv", index=False)

    gate03_events = gate03_event_report(scored, roster_state)

    schedule = build_schedule_history(seasons)
    player_logs = build_m95q_logs(seasons, schedule)
    player_logs.to_csv(args.out_dir / "player_game_logs.csv", index=False)

    workhorse_rows = compute_workhorse_events(scored, roster_state, injury_state, player_logs)
    workhorse_rows.to_csv(args.out_dir / "workhorse_event_rows.csv", index=False)

    census = season_census(workhorse_rows, seasons)

    report = {
        "stage": "ADEQUACY_CENSUS_COUNTS_ONLY",
        "seasons": seasons,
        "scored_week_range": [SCORED_WEEK_FLOOR, SCORED_WEEK_CEIL],
        "workhorse_carry_floor": WORKHORSE_CARRY_FLOOR,
        "source_timing_integrity": {
            "gate0_2_injury_disposition": gate02["disposition"],
            "gate0_3_roster_structural_disposition": gate03["disposition"],
            "gate0_3_roster_structural_failures": gate03["failures"],
            "gate0_3_event_checks_disposition": gate03_events["disposition"],
            "gate0_3_event_checks_failures": gate03_events["failures"],
            "gate0_3_requirement_4_note": (
                "Schedule-coverage completeness (requirement 4) now runs for these "
                "evaluated earlier seasons via the additive oos_test_seasons "
                "parameter on gate03_report() (default unchanged for V1/V2)."
            ),
        },
        "schedule_domain_correction": {
            "events_checked_pre_schedule_domain": schedule_domain["events_checked"],
            "events_retained": schedule_domain["events_retained"],
            "events_excluded_non_game": schedule_domain["events_excluded"],
            "note": (
                "Scored population intersected with canonical scheduled "
                "target-game team-weeks before this census (Issue #535 "
                "comment 5718356931); excluded_non_game_events.csv preserves "
                "every dropped row with its reason."
            ),
        },
        "census_by_season": census,
        "no_fitting_performed": True,
        "no_probabilities_generated": True,
        "no_cutoff_selected": True,
        "no_v2_allocation_run": True,
        "no_2024_2025_outcomes_touched": True,
    }
    out_path = args.out_dir / "workhorse_gate_v1_adequacy_census.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
