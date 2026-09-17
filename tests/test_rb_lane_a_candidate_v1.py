import numpy as np
import pandas as pd
import pytest

from scripts.backtest.rb_lane_a_candidate_v1 import (
    build_active_and_pre_transition_rooms,
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


def _scored(rows):
    cols = ["season", "week", "team", "player_clean_key"]
    return pd.DataFrame(rows, columns=cols)


def _dual(rows):
    cols = ["season", "week", "team", "player_clean_key", "promotion_rush_att", "promotion_rush_yards"]
    return pd.DataFrame(rows, columns=cols)


def test_constructibility_passes_when_all_rows_have_finite_efficiency():
    scored = _scored([[2024, 3, "TB", "p1"], [2024, 3, "TB", "p2"]])
    dual = _dual(
        [
            [2024, 3, "TB", "p1", 12.0, 60.0],
            [2024, 3, "TB", "p2", 8.0, 30.0],
        ]
    )
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE"
    assert result["rows_failing"] == 0


def test_constructibility_fails_closed_on_missing_join():
    scored = _scored([[2024, 3, "TB", "p1"], [2024, 3, "TB", "p2"]])
    dual = _dual([[2024, 3, "TB", "p1", 12.0, 60.0]])
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"
    assert result["rows_failing"] == 1
    assert result["failing_rows"][0]["player_clean_key"] == "p2"


def test_constructibility_fails_closed_on_rush_att_at_or_below_floor():
    scored = _scored([[2024, 3, "TB", "p1"]])
    dual = _dual([[2024, 3, "TB", "p1", 0.20, 60.0]])
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"


def test_constructibility_fails_closed_on_nonfinite_rush_yards():
    scored = _scored([[2024, 3, "TB", "p1"]])
    dual = pd.DataFrame(
        [[2024, 3, "TB", "p1", 12.0, None]],
        columns=["season", "week", "team", "player_clean_key", "promotion_rush_att", "promotion_rush_yards"],
    )
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"


def test_constructibility_fails_closed_on_missing_columns():
    scored = pd.DataFrame({"season": [2024]})
    dual = _dual([[2024, 3, "TB", "p1", 12.0, 60.0]])
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"
    assert "missing required columns" in result["reason"]


def test_compute_incumbent_ypc():
    dual = _dual([[2024, 3, "TB", "p1", 10.0, 50.0]])
    out = compute_incumbent_ypc(dual)
    assert out.iloc[0]["incumbent_ypc"] == pytest.approx(5.0)


def test_translate_candidate_rush_yards():
    candidate_att = pd.DataFrame(
        [[2024, 3, "TB", "p1", 8.0]],
        columns=["season", "week", "team", "player_clean_key", "candidate_att"],
    )
    incumbent = compute_incumbent_ypc(_dual([[2024, 3, "TB", "p1", 10.0, 50.0]]))
    out = translate_candidate_rush_yards(candidate_att, incumbent)
    assert out.iloc[0]["candidate_rush_yards"] == pytest.approx(40.0)


def test_translate_candidate_rush_yards_fails_closed_on_unmatched_row():
    candidate_att = pd.DataFrame(
        [[2024, 3, "TB", "p_unknown", 8.0]],
        columns=["season", "week", "team", "player_clean_key", "candidate_att"],
    )
    incumbent = compute_incumbent_ypc(_dual([[2024, 3, "TB", "p1", 10.0, 50.0]]))
    with pytest.raises(RuntimeError, match="no incumbent_ypc match"):
        translate_candidate_rush_yards(candidate_att, incumbent)


def _player_log_row(season, week, team, position, rushes, rush_yards=0.0, name_key="p1"):
    return {
        "season": season, "week": week, "team": team, "position": position,
        "rushes": rushes, "rush_yards": rush_yards, "name_key": name_key,
    }


def test_compute_historical_rb_room_rush_share_uses_strictly_prior_weeks():
    logs = pd.DataFrame(
        [
            _player_log_row(2024, 1, "TB", "RB", rushes=10),
            _player_log_row(2024, 1, "TB", "WR", rushes=0),
            _player_log_row(2024, 1, "TB", "QB", rushes=5),
            _player_log_row(2024, 2, "TB", "RB", rushes=20),
            _player_log_row(2024, 2, "TB", "QB", rushes=0),
            _player_log_row(2024, 3, "TB", "RB", rushes=15),
            _player_log_row(2024, 3, "TB", "QB", rushes=0),
        ]
    )
    out = compute_historical_rb_room_rush_share(logs)
    w1 = out.loc[out.week == 1].iloc[0]
    assert np.isnan(w1["historical_rb_room_rush_share"])  # no prior weeks

    w2 = out.loc[out.week == 2].iloc[0]
    # week 1: rb=10, total=15 -> share=0.6667; only prior week available
    assert w2["historical_rb_room_rush_share"] == pytest.approx(10 / 15)

    w3 = out.loc[out.week == 3].iloc[0]
    # trailing mean of weeks 1 (0.6667) and 2 (20/20=1.0)
    assert w3["historical_rb_room_rush_share"] == pytest.approx(((10 / 15) + 1.0) / 2)


def test_compute_conservation_pool_formula():
    cp = pd.DataFrame(
        [
            {"season": 2024, "week": 3, "team": "TB", "mc_projected_plays": 60.0, "mc_dropback_rate": 0.6},
        ]
    )
    share = pd.DataFrame(
        [{"season": 2024, "week": 3, "team": "TB", "historical_rb_room_rush_share": 0.8}]
    )
    out = compute_conservation_pool(cp, share)
    assert out.iloc[0]["pool"] == pytest.approx(60.0 * (1 - 0.6) * 0.8)


def test_compute_conservation_pool_fails_closed_on_missing_columns():
    cp = pd.DataFrame([{"season": 2024, "week": 3, "team": "TB"}])
    share = pd.DataFrame([{"season": 2024, "week": 3, "team": "TB", "historical_rb_room_rush_share": 0.8}])
    with pytest.raises(RuntimeError, match="missing"):
        compute_conservation_pool(cp, share)


def test_hhi_dampened_reallocation_even_split_at_zero_hhi():
    room = pd.DataFrame([{"raw_w": 0.5}, {"raw_w": 0.5}])
    hhi_lookup = pd.DataFrame(
        [{"season": 2024, "week": 3, "team": "TB", "prior_backfield_hhi": 0.0}]
    )
    pool_row = {"season": 2024, "week": 3, "team": "TB", "pool": 20.0}
    out, meta = compute_hhi_dampened_reallocation(room, hhi_lookup, pool_row)
    assert meta["all_zero_weights"] is False
    assert out["candidate_att"].sum() == pytest.approx(20.0)
    # even raw weights + H=0 (p=1) -> even split
    assert out["candidate_att"].tolist() == pytest.approx([10.0, 10.0])


def test_hhi_dampened_reallocation_concentrates_toward_dominant_back_as_hhi_rises():
    room = pd.DataFrame([{"raw_w": 0.8}, {"raw_w": 0.2}])
    hhi_lookup = pd.DataFrame(
        [{"season": 2024, "week": 3, "team": "TB", "prior_backfield_hhi": 0.9}]
    )
    pool_row = {"season": 2024, "week": 3, "team": "TB", "pool": 20.0}
    out, meta = compute_hhi_dampened_reallocation(room, hhi_lookup, pool_row)
    assert out["candidate_att"].sum() == pytest.approx(20.0)
    # dominant back's share of the pool exceeds its raw share (concentration effect)
    dominant_share = out["candidate_att"].iloc[0] / 20.0
    assert dominant_share > 0.8


def test_hhi_dampened_reallocation_all_zero_weights_excluded():
    room = pd.DataFrame([{"raw_w": 0.0}, {"raw_w": 0.0}])
    hhi_lookup = pd.DataFrame(columns=["season", "week", "team", "prior_backfield_hhi"])
    pool_row = {"season": 2024, "week": 3, "team": "TB", "pool": 20.0}
    out, meta = compute_hhi_dampened_reallocation(room, hhi_lookup, pool_row)
    assert meta["all_zero_weights"] is True
    assert out.empty


def test_compute_role_weights_and_hhi_end_to_end():
    logs = pd.DataFrame(
        [
            _player_log_row(2024, 1, "TB", "RB", rushes=12, rush_yards=50, name_key="rb1"),
            _player_log_row(2024, 1, "TB", "RB", rushes=3, rush_yards=10, name_key="rb2"),
            _player_log_row(2024, 1, "TB", "QB", rushes=2, name_key="qb1"),
            _player_log_row(2024, 2, "TB", "RB", rushes=14, rush_yards=60, name_key="rb1"),
            _player_log_row(2024, 2, "TB", "RB", rushes=2, rush_yards=8, name_key="rb2"),
            _player_log_row(2024, 2, "TB", "QB", rushes=1, name_key="qb1"),
        ]
    )
    active_room = pd.DataFrame([{"season": 2024, "week": 3, "team": "TB", "name_key": "rb1"}])
    # Membership drawn from the prior week's (week 2) roster, but tagged with
    # the TRANSITION week's own coordinates (week 3) -- see the corrected
    # compute_role_weights_and_hhi() contract: this makes enrich_history()'s
    # cutoff use history strictly before the transition week, and makes the
    # resulting hhi_lookup key match pool_row's own (season, week, team).
    pre_transition_room = pd.DataFrame(
        [
            {"season": 2024, "week": 3, "team": "TB", "name_key": "rb1"},
            {"season": 2024, "week": 3, "team": "TB", "name_key": "rb2"},
        ]
    )
    active_enriched, hhi_lookup = compute_role_weights_and_hhi(active_room, pre_transition_room, logs)
    assert "raw_w" in active_enriched.columns
    assert active_enriched.iloc[0]["raw_w"] > 0
    assert len(hhi_lookup) == 1
    assert hhi_lookup.iloc[0]["season"] == 2024
    assert hhi_lookup.iloc[0]["week"] == 3
    assert hhi_lookup.iloc[0]["prior_backfield_hhi"] > 0


def _keys(season, week, team, player_clean_key):
    return {"season": season, "week": week, "team": team, "player_clean_key": player_clean_key}


def test_build_deployable_candidate_uses_candidate_on_scored_rows_and_comparator_elsewhere():
    all_rows = pd.DataFrame(
        [
            {**_keys(2024, 3, "TB", "p1"), "promotion_rush_yards": 50.0},
            {**_keys(2024, 4, "TB", "p1"), "promotion_rush_yards": 55.0},
        ]
    )
    scored = pd.DataFrame([{**_keys(2024, 3, "TB", "p1"), "candidate_rush_yards": 62.0}])
    out = build_deployable_candidate(all_rows, scored)
    row3 = out.loc[out.week == 3].iloc[0]
    row4 = out.loc[out.week == 4].iloc[0]
    assert row3["is_scored_v1_transition_row"] is True or bool(row3["is_scored_v1_transition_row"])
    assert row3["deployable_candidate_rush_yards"] == pytest.approx(62.0)
    assert bool(row4["is_scored_v1_transition_row"]) is False
    assert row4["deployable_candidate_rush_yards"] == pytest.approx(55.0)


def test_check_stable_identity_gate_passes_when_non_scored_rows_match_exactly():
    deployable = pd.DataFrame(
        [
            {"is_scored_v1_transition_row": True, "deployable_candidate_rush_yards": 62.0, "promotion_rush_yards": 50.0},
            {"is_scored_v1_transition_row": False, "deployable_candidate_rush_yards": 55.0, "promotion_rush_yards": 55.0},
        ]
    )
    result = check_stable_identity_gate(deployable)
    assert result["disposition"] == "STABLE_IDENTITY_GATE_PASS"
    assert result["rows_checked"] == 1


def test_check_stable_identity_gate_fails_closed_on_non_scored_mismatch():
    deployable = pd.DataFrame(
        [
            {"is_scored_v1_transition_row": False, "deployable_candidate_rush_yards": 55.1, "promotion_rush_yards": 55.0},
        ]
    )
    result = check_stable_identity_gate(deployable)
    assert result["disposition"] == "STABLE_IDENTITY_GATE_FAILURE"
    assert result["max_abs_delta"] == pytest.approx(0.1)


def test_build_mechanism_diagnostic_joins_on_name_key():
    scored = pd.DataFrame(
        [{"season": 2025, "week": 5, "team": "TB", "name_key": "p1", "candidate_rush_yards": 62.0}]
    )
    mechanism = pd.DataFrame(
        [{"season": 2025, "week": 5, "team": "TB", "name_key": "p1", "arch_enriched_opp_stack_eff_yards": 58.0}]
    )
    out = build_mechanism_diagnostic(scored, mechanism)
    assert out.iloc[0]["mechanism_comparator_rush_yards"] == pytest.approx(58.0)
    assert out.iloc[0]["candidate_rush_yards"] == pytest.approx(62.0)


def test_build_mechanism_diagnostic_fails_closed_on_missing_columns():
    scored = pd.DataFrame([{"season": 2025}])
    mechanism = pd.DataFrame(
        [{"season": 2025, "week": 5, "team": "TB", "name_key": "p1", "arch_enriched_opp_stack_eff_yards": 58.0}]
    )
    with pytest.raises(RuntimeError, match="candidate frame missing"):
        build_mechanism_diagnostic(scored, mechanism)


def test_build_active_and_pre_transition_rooms_retags_prior_week_onto_transition_week():
    roster_state = pd.DataFrame(
        [
            {"season": 2024, "week": 2, "team": "TB", "name_key": "p1", "player_clean_key": "p1"},
            {"season": 2024, "week": 2, "team": "TB", "name_key": "p2", "player_clean_key": "p2"},
            {"season": 2024, "week": 3, "team": "TB", "name_key": "p1", "player_clean_key": "p1"},
        ]
    )
    scored_events = pd.DataFrame(
        [{"season": 2024, "week": 3, "team": "TB", "prior_season": 2024, "prior_week": 2}]
    )
    active, pre = build_active_and_pre_transition_rooms(scored_events, roster_state)
    assert active["player_clean_key"].tolist() == ["p1"]
    assert set(pre["player_clean_key"]) == {"p1", "p2"}
    # pre-transition room is re-tagged onto the transition week's own coordinates
    assert set(pre["week"]) == {3}


def test_run_candidate_mechanism_for_rotation_end_to_end():
    from scripts.backtest.rb_lane_a_candidate_v1 import run_candidate_mechanism_for_rotation

    roster_state = pd.DataFrame(
        [
            {"season": 2024, "week": 2, "team": "TB", "name_key": "p1", "player_clean_key": "p1"},
            {"season": 2024, "week": 2, "team": "TB", "name_key": "p2", "player_clean_key": "p2"},
            {"season": 2024, "week": 3, "team": "TB", "name_key": "p1", "player_clean_key": "p1"},
        ]
    )
    scored_events = pd.DataFrame(
        [{"season": 2024, "week": 3, "team": "TB", "prior_season": 2024, "prior_week": 2}]
    )
    player_logs = pd.DataFrame(
        [
            _player_log_row(2024, 1, "TB", "RB", rushes=12, rush_yards=50, name_key="p1"),
            _player_log_row(2024, 1, "TB", "RB", rushes=3, rush_yards=10, name_key="p2"),
            _player_log_row(2024, 1, "TB", "QB", rushes=2, name_key="qb1"),
            _player_log_row(2024, 2, "TB", "RB", rushes=14, rush_yards=60, name_key="p1"),
            _player_log_row(2024, 2, "TB", "RB", rushes=2, rush_yards=8, name_key="p2"),
            _player_log_row(2024, 2, "TB", "QB", rushes=1, name_key="qb1"),
            # week 3 itself needs a row so compute_historical_rb_room_rush_share
            # knows this team-week exists to compute a trailing share for --
            # its own rushes don't matter, only prior weeks feed the mean.
            _player_log_row(2024, 3, "TB", "RB", rushes=16, rush_yards=70, name_key="p1"),
        ]
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
    component_predictions = pd.DataFrame(cp_rows)

    result = run_candidate_mechanism_for_rotation(
        scored_events=scored_events,
        roster_state=roster_state,
        player_logs=player_logs,
        component_predictions=component_predictions,
        rotation=1,
    )
    assert result["excluded_all_zero_weight_team_weeks"] == []
    rows = result["candidate_rows"]
    assert len(rows) == 1
    assert rows.iloc[0]["player_clean_key"] == "p1"
    assert rows.iloc[0]["candidate_rush_yards"] > 0
    # sole active-room player after the departure gets the whole conservation
    # pool: pool = 60 * (1-0.6) * mean(week1_rb_share=15/17, week2_rb_share=16/17)
    expected_pool = 60.0 * 0.4 * (((12 + 3) / 17) + ((14 + 2) / 17)) / 2
    assert rows.iloc[0]["candidate_att"] == pytest.approx(expected_pool, rel=1e-6)
