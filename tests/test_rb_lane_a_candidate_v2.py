import pandas as pd
import pytest

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


def _active_room():
    return pd.DataFrame(
        [
            {"season": 2024, "week": 3, "team": "TB", "player_clean_key": "p1", "name_key": "p1"},
            {"season": 2024, "week": 3, "team": "TB", "player_clean_key": "p2", "name_key": "p2"},
            {"season": 2024, "week": 3, "team": "TB", "player_clean_key": "p3", "name_key": "p3"},
        ]
    )


def _dual_market():
    return pd.DataFrame(
        [
            # p1: fully eligible.
            {"season": 2024, "week": 3, "team": "TB", "player_clean_key": "p1",
             "promotion_rush_att": 5.0, "promotion_rush_yards": 20.0},
            # p2: below the att floor -- ineligible.
            {"season": 2024, "week": 3, "team": "TB", "player_clean_key": "p2",
             "promotion_rush_att": 0.10, "promotion_rush_yards": 1.0},
            # p3: no dual-market row at all -- ineligible (left_only after merge).
        ]
    )


def test_filter_eligible_recipients_splits_on_exact_identity_and_finite_floor():
    eligible, excluded = filter_eligible_recipients(_active_room(), _dual_market())
    assert eligible["player_clean_key"].tolist() == ["p1"]
    assert sorted(excluded["player_clean_key"].tolist()) == ["p2", "p3"]


def test_filter_eligible_recipients_excludes_nonfinite_promotion_yards():
    dual = _dual_market()
    dual.loc[dual["player_clean_key"] == "p1", "promotion_rush_yards"] = float("nan")
    eligible, excluded = filter_eligible_recipients(_active_room(), dual)
    assert eligible.empty
    assert sorted(excluded["player_clean_key"].tolist()) == ["p1", "p2", "p3"]


def test_filter_eligible_recipients_fails_closed_on_missing_active_room_columns():
    bad_active = pd.DataFrame([{"season": 2024, "week": 3, "team": "TB"}])
    with pytest.raises(RuntimeError, match="active_room missing"):
        filter_eligible_recipients(bad_active, _dual_market())


def test_filter_eligible_recipients_fails_closed_on_missing_comparator_columns():
    bad_dual = pd.DataFrame([{"season": 2024, "week": 3, "team": "TB", "player_clean_key": "p1"}])
    with pytest.raises(RuntimeError, match="dual_market_comparator missing"):
        filter_eligible_recipients(_active_room(), bad_dual)


def test_check_recipient_universe_integrity_ok_when_nonempty():
    eligible, _ = filter_eligible_recipients(_active_room(), _dual_market())
    report = check_recipient_universe_integrity({"season": 2024, "week": 3, "team": "TB"}, eligible)
    assert report["disposition"] == V2_RECIPIENT_UNIVERSE_OK
    assert report["n_eligible_recipients"] == 1


def test_check_recipient_universe_integrity_fails_closed_when_empty():
    report = check_recipient_universe_integrity(
        {"season": 2024, "week": 3, "team": "TB"}, pd.DataFrame(columns=["player_clean_key"])
    )
    assert report["disposition"] == V2_RECIPIENT_UNIVERSE_FAILURE
    assert report["n_eligible_recipients"] == 0


def test_check_recipient_weight_integrity_ok_when_positive_raw_w_sum():
    room = pd.DataFrame([{"player_clean_key": "p1", "raw_w": 0.4}])
    report = check_recipient_weight_integrity({"season": 2024, "week": 3, "team": "TB"}, room)
    assert report["disposition"] == V2_RECIPIENT_WEIGHT_OK
    assert report["eligible_raw_w_sum"] == pytest.approx(0.4)


def test_check_recipient_weight_integrity_fails_closed_when_all_zero():
    room = pd.DataFrame([{"player_clean_key": "p1", "raw_w": 0.0}, {"player_clean_key": "p2", "raw_w": 0.0}])
    report = check_recipient_weight_integrity({"season": 2024, "week": 3, "team": "TB"}, room)
    assert report["disposition"] == V2_RECIPIENT_WEIGHT_FAILURE


def test_aggregate_recipient_integrity_disposition_passes_when_all_events_ok():
    events = [
        {"season": 2024, "week": 3, "team": "TB", "disposition": V2_RECIPIENT_UNIVERSE_OK},
        {"season": 2024, "week": 4, "team": "TB", "disposition": V2_RECIPIENT_UNIVERSE_OK},
    ]
    agg = aggregate_recipient_integrity_disposition(events, V2_RECIPIENT_UNIVERSE_OK, V2_RECIPIENT_UNIVERSE_FAILURE)
    assert agg["disposition"] == V2_RECIPIENT_UNIVERSE_OK
    assert agg["events_failing"] == 0


def test_aggregate_recipient_integrity_disposition_fails_closed_on_any_failing_event():
    events = [
        {"season": 2024, "week": 3, "team": "TB", "disposition": V2_RECIPIENT_UNIVERSE_OK},
        {"season": 2024, "week": 4, "team": "KC", "disposition": V2_RECIPIENT_UNIVERSE_FAILURE},
    ]
    agg = aggregate_recipient_integrity_disposition(events, V2_RECIPIENT_UNIVERSE_OK, V2_RECIPIENT_UNIVERSE_FAILURE)
    assert agg["disposition"] == V2_RECIPIENT_UNIVERSE_FAILURE
    assert agg["events_failing"] == 1
    assert agg["failing_events"][0]["team"] == "KC"


def _player_log_row(season, week, team, position, rushes, rush_yards=0.0, name_key="p1"):
    return {
        "season": season, "week": week, "team": team, "position": position,
        "rushes": rushes, "rush_yards": rush_yards, "name_key": name_key,
    }


def test_construct_rotation_outcome_blind_v2_excludes_ineligible_recipient_before_reallocation(tmp_path):
    """End-to-end wiring check for the runner's core loop: this is V2's
    decisive fix over V1's real failure mode -- an active-room player (p2)
    that the production comparator cannot score this week must be excluded
    from the recipient set BEFORE reallocation, not merely reported after a
    constructibility failure. p2 has genuine positive prior3_rb_share
    history (so this isn't V1's old "history-less room" exclusion) -- the
    only reason it's excluded is the missing dual-market comparator row.
    """
    from scripts.backtest.run_rb_lane_a_candidate_v2 import construct_rotation_outcome_blind_v2

    logs = pd.DataFrame(
        [
            _player_log_row(2024, 1, "TB", "RB", rushes=12, rush_yards=50, name_key="p1"),
            _player_log_row(2024, 1, "TB", "RB", rushes=8, rush_yards=30, name_key="p2"),
            _player_log_row(2024, 1, "TB", "QB", rushes=2, name_key="qb1"),
            _player_log_row(2024, 2, "TB", "RB", rushes=14, rush_yards=60, name_key="p1"),
            _player_log_row(2024, 2, "TB", "RB", rushes=6, rush_yards=25, name_key="p2"),
            _player_log_row(2024, 2, "TB", "QB", rushes=1, name_key="qb1"),
            # Week 3 itself needs a row so compute_historical_rb_room_rush_share
            # knows this team-week exists to compute a trailing share for.
            _player_log_row(2024, 3, "TB", "RB", rushes=20, rush_yards=90, name_key="p1"),
        ]
    )
    logs["player_clean_key"] = logs["name_key"]
    root = tmp_path / "rotation1"
    root.mkdir()
    logs.to_csv(root / "player_game_logs_history.csv", index=False)

    roster_state = pd.DataFrame(
        [
            {"season": 2024, "week": 2, "team": "TB", "name_key": "p1", "player_clean_key": "p1"},
            {"season": 2024, "week": 2, "team": "TB", "name_key": "p2", "player_clean_key": "p2"},
            {"season": 2024, "week": 3, "team": "TB", "name_key": "p1", "player_clean_key": "p1"},
            {"season": 2024, "week": 3, "team": "TB", "name_key": "p2", "player_clean_key": "p2"},
        ]
    )
    injury_state = pd.DataFrame(columns=["season", "week", "team", "player", "status"])
    scored_events = pd.DataFrame(
        [{"season": 2024, "week": 3, "team": "TB", "prior_season": 2024, "prior_week": 2}]
    )

    cp_rows = []
    for market, mc, ml, state in [("rush_att", 15.0, 14.0, 16.0), ("rush_yards", 80.0, 75.0, 85.0)]:
        cp_rows.append(
            {
                "season": 2024, "week": 3, "team": "TB", "player_clean_key": "p1", "market": market,
                "mc_proj": mc, "ml_proj": ml, "state_proj": state,
                "mc_projected_plays": 60.0, "mc_dropback_rate": 0.6,
            }
        )
    # p2 deliberately has NO component_predictions row at all -- the
    # production projection universe cannot score them this week, the exact
    # structural gap that made V1 terminally fail closed at constructibility.
    component_predictions = pd.DataFrame(cp_rows)

    context = construct_rotation_outcome_blind_v2(
        rotation=1,
        root=root,
        component_predictions=component_predictions,
        roster_state=roster_state,
        injury_state=injury_state,
        scored_events=scored_events,
    )
    assert context["recipient_universe_integrity"]["disposition"] == "V2_RECIPIENT_UNIVERSE_OK"
    assert context["recipient_weight_integrity"]["disposition"] == "V2_RECIPIENT_WEIGHT_OK"
    assert context["constructibility"]["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE"

    candidate = context["candidate"]
    assert candidate["player_clean_key"].tolist() == ["p1"]
    # p1 is the sole ELIGIBLE recipient -- gets the entire pool despite p2
    # also carrying positive raw_w history; this is exactly the behavior
    # that differs from V1 (which put both in the room, then failed
    # constructibility because p2 had no comparator match). The pool's
    # team-week denominator includes the QB row too (RB room share of ALL
    # team rushes, not RB-only rushes): week1 total = 12+8+2=22, week2
    # total = 14+6+1=21.
    expected_pool = 60.0 * 0.4 * (((12 + 8) / 22) + ((14 + 6) / 21)) / 2
    assert candidate.iloc[0]["candidate_att"] == pytest.approx(expected_pool, rel=1e-6)
