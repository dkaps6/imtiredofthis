import pandas as pd
import pytest

from scripts.backtest.rb_workhorse_gate_v1_features import (
    FEATURE_COLUMNS,
    FEATURES_CONSTRUCTIBLE,
    WHOLE_EXPERIMENT_FAIL_CLOSED,
    build_event_features,
)


def _roster_row(season, week, team, player_clean_key, name_key=None):
    return {
        "season": season, "week": week, "team": team,
        "player_clean_key": player_clean_key, "name_key": name_key or player_clean_key,
    }


def _player_log_row(season, week, team, position, rushes, rush_yards=0.0, name_key="p"):
    return {
        "season": season, "week": week, "team": team, "position": position,
        "rushes": rushes, "rush_yards": rush_yards, "name_key": name_key,
    }


def _cp_row(season, week, team, player_clean_key, market, mc_proj, mc_plays=60.0, mc_dropback=0.6):
    return {
        "season": season, "week": week, "team": team, "player_clean_key": player_clean_key,
        "market": market, "mc_proj": mc_proj,
        "mc_projected_plays": mc_plays, "mc_dropback_rate": mc_dropback,
    }


def _base_fixture():
    """Two active RBs (p1, p2) at week 3, p3 departed since week 2.
    History from weeks 1-3 feeds prior3_rb_share; week-3 team-week MC and
    Build-A rush_att rows are provided for p1/p2 only.
    """
    scored_events = pd.DataFrame(
        [{"season": 2019, "week": 3, "team": "TB", "prior_season": 2019, "prior_week": 2}]
    )
    roster_state = pd.DataFrame(
        [
            _roster_row(2019, 2, "TB", "p1"),
            _roster_row(2019, 2, "TB", "p2"),
            _roster_row(2019, 2, "TB", "p3"),
            _roster_row(2019, 3, "TB", "p1"),
            _roster_row(2019, 3, "TB", "p2"),
        ]
    )
    player_logs = pd.DataFrame(
        [
            _player_log_row(2019, 1, "TB", "RB", rushes=15, rush_yards=60, name_key="p1"),
            _player_log_row(2019, 1, "TB", "RB", rushes=5, rush_yards=20, name_key="p2"),
            _player_log_row(2019, 1, "TB", "RB", rushes=3, rush_yards=10, name_key="p3"),
            _player_log_row(2019, 2, "TB", "RB", rushes=14, rush_yards=55, name_key="p1"),
            _player_log_row(2019, 2, "TB", "RB", rushes=6, rush_yards=25, name_key="p2"),
            _player_log_row(2019, 2, "TB", "RB", rushes=2, rush_yards=8, name_key="p3"),
            _player_log_row(2019, 3, "TB", "RB", rushes=18, rush_yards=70, name_key="p1"),
        ]
    )
    component_predictions = pd.DataFrame(
        [
            _cp_row(2019, 3, "TB", "p1", "rush_att", mc_proj=14.0),
            _cp_row(2019, 3, "TB", "p1", "rush_yards", mc_proj=65.0),
            _cp_row(2019, 3, "TB", "p2", "rush_att", mc_proj=4.0),
        ]
    )
    return scored_events, roster_state, player_logs, component_predictions


def test_build_event_features_happy_path_produces_complete_finite_row():
    scored_events, roster_state, player_logs, component_predictions = _base_fixture()
    result = build_event_features(
        scored_events=scored_events, roster_state=roster_state,
        player_logs=player_logs, component_predictions_build_a=component_predictions,
    )
    assert result["disposition"] == FEATURES_CONSTRUCTIBLE
    assert result["events_failing"] == 0
    rows = result["feature_rows"]
    assert len(rows) == 1
    row = rows.iloc[0]
    for col in FEATURE_COLUMNS:
        assert col in row.index
        assert pd.notna(row[col])

    # p1 has more rushing history than p2 -> p1 is top, p2 is second.
    assert row["active_top_prior3_rb_share"] > row["active_second_prior3_rb_share"] > 0
    assert row["active_rb_room_size"] == 2
    assert row["prior_rb_room_size"] == 3
    # p3 departed -- its own prior3 share feeds the departed-room sum/max.
    assert row["departed_room_prior3_share_sum"] > 0
    assert row["departed_room_max_prior3_share"] == pytest.approx(row["departed_room_prior3_share_sum"])
    assert row["raw_mc_top_active_rush_att"] == pytest.approx(14.0)
    assert row["raw_mc_second_active_rush_att"] == pytest.approx(4.0)
    assert row["mc_projected_plays"] == pytest.approx(60.0)
    assert row["mc_dropback_rate"] == pytest.approx(0.6)
    assert row["raw_mc_team_rush_volume"] == pytest.approx(60.0 * 0.4)


def test_build_event_features_single_active_member_structural_zero_second_share():
    scored_events, roster_state, player_logs, component_predictions = _base_fixture()
    # Remove p2 from the active (week-3) room, keep everything else.
    roster_state = roster_state.loc[~((roster_state["week"] == 3) & (roster_state["player_clean_key"] == "p2"))]
    result = build_event_features(
        scored_events=scored_events, roster_state=roster_state,
        player_logs=player_logs, component_predictions_build_a=component_predictions,
    )
    assert result["disposition"] == FEATURES_CONSTRUCTIBLE
    row = result["feature_rows"].iloc[0]
    assert row["active_rb_room_size"] == 1
    assert row["active_second_prior3_rb_share"] == 0.0
    assert row["raw_mc_second_active_rush_att"] == 0.0
    assert row["raw_mc_top_active_rush_att"] == pytest.approx(14.0)


def test_build_event_features_zero_active_room_members_fails_whole_experiment():
    scored_events, roster_state, player_logs, component_predictions = _base_fixture()
    roster_state = roster_state.loc[roster_state["week"] != 3]  # no week-3 active room at all
    result = build_event_features(
        scored_events=scored_events, roster_state=roster_state,
        player_logs=player_logs, component_predictions_build_a=component_predictions,
    )
    assert result["disposition"] == WHOLE_EXPERIMENT_FAIL_CLOSED
    assert result["events_failing"] == 1
    assert result["failing_events"][0]["reason"] == "zero_active_room_members"
    assert result["feature_rows"].empty


def test_build_event_features_zero_matched_raw_mc_rows_fails_whole_experiment():
    scored_events, roster_state, player_logs, _ = _base_fixture()
    # No component_predictions rows at all for the event's team-week -- the
    # team-week MC lookup itself has nothing to resolve, so this must fail
    # the whole experiment closed rather than silently defaulting.
    component_predictions = pd.DataFrame(
        columns=["season", "week", "team", "player_clean_key", "market", "mc_proj",
                 "mc_projected_plays", "mc_dropback_rate"]
    )
    result = build_event_features(
        scored_events=scored_events, roster_state=roster_state,
        player_logs=player_logs, component_predictions_build_a=component_predictions,
    )
    assert result["disposition"] == WHOLE_EXPERIMENT_FAIL_CLOSED
    assert result["failing_events"][0]["reason"] == "missing_team_week_mc_or_history_share"


def test_build_event_features_zero_matched_active_players_but_team_week_mc_present():
    scored_events, roster_state, player_logs, component_predictions = _base_fixture()
    # Team-week MC values present (from p1's rows), but drop the per-player
    # rush_att rows entirely so no active player can be matched.
    component_predictions = component_predictions.loc[component_predictions["market"] != "rush_att"]
    result = build_event_features(
        scored_events=scored_events, roster_state=roster_state,
        player_logs=player_logs, component_predictions_build_a=component_predictions,
    )
    assert result["disposition"] == WHOLE_EXPERIMENT_FAIL_CLOSED
    assert result["failing_events"][0]["reason"] == "zero_active_players_matched_to_build_a_rush_att"


def test_build_event_features_preserves_all_failing_events_not_just_first():
    scored_events, roster_state, player_logs, component_predictions = _base_fixture()
    # Add a second scored event (week 4) whose active room is also empty.
    scored_events = pd.concat(
        [scored_events, pd.DataFrame([{"season": 2019, "week": 4, "team": "TB", "prior_season": 2019, "prior_week": 3}])],
        ignore_index=True,
    )
    roster_state = roster_state.loc[roster_state["week"] != 3]  # kills the first event's active room too
    result = build_event_features(
        scored_events=scored_events, roster_state=roster_state,
        player_logs=player_logs, component_predictions_build_a=component_predictions,
    )
    assert result["disposition"] == WHOLE_EXPERIMENT_FAIL_CLOSED
    assert result["events_failing"] == 2
    assert {f["week"] for f in result["failing_events"]} == {3, 4}
