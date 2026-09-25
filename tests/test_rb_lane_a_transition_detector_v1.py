import pandas as pd

from scripts.backtest.rb_lane_a_transition_detector_v1 import (
    build_detected_transitions,
    build_scored_v1_event_population,
    disclosure_report,
)


def _roster(rows):
    return pd.DataFrame(rows, columns=["season", "week", "team", "player_key", "position", "status", "name_key"])


def _injuries(rows):
    return pd.DataFrame(rows, columns=["player", "team", "season", "week", "status", "practice_status", "body_part", "designation", "source", "report_available"])


def test_membership_shrink_is_scored_but_gain_is_disclosure_only():
    roster = _roster(
        [
            [2024, 1, "KC", "p1", "RB", "ACT", "back1"],
            [2024, 1, "KC", "p2", "RB", "ACT", "back2"],
            [2024, 2, "KC", "p2", "RB", "ACT", "back2"],
            [2024, 2, "KC", "p3", "RB", "ACT", "back3"],
        ]
    )
    injuries = _injuries([])
    out = build_detected_transitions(roster, injuries)
    row = out.loc[(out.season == 2024) & (out.week == 2) & (out.team == "KC")].iloc[0]
    assert row.trigger_membership_shrink  # p1 departed
    assert row.trigger_membership_gain  # p3 added
    assert row.scored_v1_transition  # shrink scores...
    assert row.detected_transition


def test_status_onset_loss_is_scored_status_onset_return_is_not():
    roster = _roster(
        [
            [2024, 1, "KC", "p1", "RB", "ACT", "back1"],
            [2024, 2, "KC", "p1", "RB", "ACT", "back1"],
        ]
    )
    injuries = _injuries(
        [
            ["Back1", "KC", 2024, 2, "Out", None, None, "Out", "nflverse_historical", 1],
        ]
    )
    out = build_detected_transitions(roster, injuries)
    row = out.loc[(out.season == 2024) & (out.week == 2) & (out.team == "KC")].iloc[0]
    assert row.trigger_status_onset_loss
    assert not row.trigger_status_onset_return
    assert row.scored_v1_transition
    assert not row.trigger_membership_shrink


def test_stable_week_has_no_triggers():
    roster = _roster(
        [
            [2024, 1, "KC", "p1", "RB", "ACT", "back1"],
            [2024, 2, "KC", "p1", "RB", "ACT", "back1"],
        ]
    )
    out = build_detected_transitions(roster, _injuries([]))
    row = out.iloc[0]
    assert not row.detected_transition
    assert not row.scored_v1_transition


def test_scored_v1_event_population_has_no_outcome_columns():
    roster = _roster(
        [
            [2024, 1, "KC", "p1", "RB", "ACT", "back1"],
            [2024, 2, "KC", "p2", "RB", "ACT", "back2"],
        ]
    )
    detected = build_detected_transitions(roster, _injuries([]))
    events = build_scored_v1_event_population(detected)
    assert list(events.columns) == ["season", "week", "team", "prior_season", "prior_week"]
    assert not events.empty


def test_disclosure_report_separates_scored_from_broader_detected_counts():
    roster = _roster(
        [
            [2024, 1, "KC", "p1", "RB", "ACT", "back1"],
            [2024, 1, "KC", "p2", "RB", "ACT", "back2"],
            [2024, 2, "KC", "p1", "RB", "ACT", "back1"],
            [2024, 2, "KC", "p2", "RB", "ACT", "back2"],
            [2024, 2, "KC", "p3", "RB", "ACT", "back3"],
        ]
    )
    detected = build_detected_transitions(roster, _injuries([]))
    report = disclosure_report(detected, [2024])
    assert report["2024"]["detected_transition_count"] == 1
    assert report["2024"]["scored_v1_transition_count"] == 0  # only a gain, not a shrink
    assert report["2024"]["trigger_membership_gain_count_disclosure_only"] == 1
