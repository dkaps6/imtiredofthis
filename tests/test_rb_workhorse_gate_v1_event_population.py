from unittest import mock

import pandas as pd

from scripts.backtest.rb_workhorse_gate_v1_event_population import (
    NO_SCHEDULED_TARGET_GAME,
    filter_scored_events_to_scheduled_games,
)

# KC @ SF in week 1 only -- 2019's only scheduled game/team-week pair for this fixture.
_FAKE_SCHEDULE = pd.DataFrame(
    {"week": [1], "home": ["SF"], "away": ["KC"], "kickoff_utc": ["2019-09-08T20:00:00Z"]}
)


def _event(season, week, team):
    return {"season": season, "week": week, "team": team, "prior_season": season, "prior_week": week - 1}


def test_filter_retains_scheduled_game_events_only():
    scored_events = pd.DataFrame(
        [_event(2019, 1, "KC"), _event(2019, 1, "SF"), _event(2019, 18, "KC")]
    )
    with mock.patch("scripts.backtest.rb_lane_a_gate0_v1.get_nfl_schedule", return_value=_FAKE_SCHEDULE):
        result = filter_scored_events_to_scheduled_games(scored_events, [2019])

    assert result["events_checked"] == 3
    assert result["events_retained"] == 2
    assert result["events_excluded"] == 1
    assert set(zip(result["retained_events"]["season"], result["retained_events"]["week"], result["retained_events"]["team"])) == {
        (2019, 1, "KC"), (2019, 1, "SF"),
    }
    excluded = result["excluded_events"]
    assert len(excluded) == 1
    assert excluded.iloc[0]["team"] == "KC"
    assert excluded.iloc[0]["week"] == 18
    assert excluded.iloc[0]["exclusion_reason"] == NO_SCHEDULED_TARGET_GAME


def test_filter_retains_everything_when_all_events_are_scheduled_games():
    scored_events = pd.DataFrame([_event(2019, 1, "KC"), _event(2019, 1, "SF")])
    with mock.patch("scripts.backtest.rb_lane_a_gate0_v1.get_nfl_schedule", return_value=_FAKE_SCHEDULE):
        result = filter_scored_events_to_scheduled_games(scored_events, [2019])

    assert result["events_excluded"] == 0
    assert len(result["retained_events"]) == 2
    assert result["excluded_events"].empty


def test_filter_excludes_everything_when_no_events_are_scheduled_games():
    scored_events = pd.DataFrame([_event(2020, 18, "BUF"), _event(2020, 5, "DEN")])
    with mock.patch("scripts.backtest.rb_lane_a_gate0_v1.get_nfl_schedule", return_value=_FAKE_SCHEDULE):
        result = filter_scored_events_to_scheduled_games(scored_events, [2019])

    assert result["events_retained"] == 0
    assert result["events_excluded"] == 2
    assert set(result["excluded_events"]["exclusion_reason"]) == {NO_SCHEDULED_TARGET_GAME}
