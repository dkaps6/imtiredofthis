"""RB Workhorse-Transition-Gate V1 -- scored event population schedule-domain correction.

Authorized by GPT-5.6 (Issue #535 comment `5718356931`), after auditing CI run
3's (`35243854637`) `WORKHORSE_GATE_V1_WHOLE_EXPERIMENT_FAIL_CLOSED` result:
all 5 failing events were non-game team-weeks -- 2019/2020 W18 (those regular
seasons ended at Week 17) and DEN 2020 W5 (a COVID-postponed bye-shifted game
with no game that week). The frozen V1/V2 loss/vacancy transition detector
(``rb_lane_a_transition_detector_v1.py``, unmodified here) can flag a
membership change on any roster-snapshot week in its 1-18 range -- it does
not itself verify the target (season, week, team) is an actual scheduled
game, because Lane-A V1/V2 only ever evaluated 2024/2025, both full 18-week
seasons with no such gaps.

A gate that predicts whether an RB gets 20+ carries "in that game" cannot be
scored against a team-week with no game -- and Build-A production MC inputs
correctly do not exist for one, which is exactly why those 5 rows failed
feature construction. This module intersects the frozen scored population
with canonical scheduled target-game team-weeks (reusing, unchanged,
``rb_lane_a_gate0_v1.build_team_week_kickoffs``/``get_nfl_schedule`` --
the same already-audited schedule path used elsewhere in Gate 0). It is
additive and pre-outcome: it does not alter ``build_detected_transitions``,
``build_scored_v1_event_population``, the transition trigger logic, the
13-feature contract, or any classifier/gate/router. Every excluded row is
preserved with an explicit reason for full auditability -- no silent
downstream filtering.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from scripts.backtest.rb_lane_a_gate0_v1 import build_team_week_kickoffs

NO_SCHEDULED_TARGET_GAME = "NO_SCHEDULED_TARGET_GAME"


def build_scheduled_game_keys(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per (season, week, team) that is an actual scheduled game,
    reusing the already-audited kickoff reconstruction unchanged.
    """
    kickoffs = build_team_week_kickoffs(seasons)
    return kickoffs[["season", "week", "team"]].drop_duplicates().reset_index(drop=True)


def filter_scored_events_to_scheduled_games(scored_events: pd.DataFrame, seasons: Iterable[int]) -> dict:
    """Intersect ``scored_events`` (the frozen V1/V2 scored loss/vacancy
    population, already restricted to weeks 2-18) with canonical scheduled
    game team-weeks for ``seasons``.

    Returns retained/excluded frames plus counts. Raises if any retained row
    somehow fails the scheduled-game check (defensive integrity assertion --
    should be unreachable given the merge logic, but "no silent downstream
    filtering" is an explicit requirement here).
    """
    keys = build_scheduled_game_keys(seasons).assign(_scheduled=1)

    merged = scored_events.merge(keys, on=["season", "week", "team"], how="left")
    is_scheduled = merged["_scheduled"].eq(1).to_numpy()

    retained = scored_events.loc[is_scheduled].reset_index(drop=True)
    excluded = scored_events.loc[~is_scheduled].copy().reset_index(drop=True)
    excluded["exclusion_reason"] = NO_SCHEDULED_TARGET_GAME

    check = retained[["season", "week", "team"]].merge(keys, on=["season", "week", "team"], how="left")
    if check["_scheduled"].isna().any():
        raise RuntimeError(
            "filter_scored_events_to_scheduled_games: integrity check failed -- "
            "a retained scored event is not a scheduled game team-week"
        )

    return {
        "retained_events": retained,
        "excluded_events": excluded,
        "events_checked": int(len(scored_events)),
        "events_retained": int(len(retained)),
        "events_excluded": int(len(excluded)),
    }
