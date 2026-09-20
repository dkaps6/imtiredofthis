import numpy as np
import pandas as pd

from scripts.research.evaluate_historical_analog_reliability_v1 import (
    FAMILIES,
    NOVELTY_THRESHOLD,
    _cohort_from_risk,
    _evaluate_family,
    _primary_gate,
    _replication_gate,
    audit_integrity,
    build_novelty_scores,
)


def _state_row(season, week, player, mean_d, support, position="RB", state="VALID_ANALOG"):
    return {
        "season": season,
        "week": week,
        "team": "A",
        "player_identity_key": player,
        "position": position,
        "analog_state": state,
        "mean_k_distance": mean_d,
        "effective_analog_count": support,
    }


def test_expanding_percentiles_are_strict_prior_position_specific_and_same_week_safe():
    states = pd.DataFrame(
        [
            _state_row(2020, 1, "rb1", 1.0, 10.0),
            _state_row(2020, 2, "rb2", 2.0, 9.0),
            _state_row(2020, 2, "rb3", 0.5, 11.0),
            _state_row(2020, 3, "rb4", 3.0, 8.0),
            _state_row(2020, 1, "wr1", 5.0, 5.0, position="WR"),
            _state_row(2020, 2, "wr2", 4.0, 6.0, position="WR"),
        ]
    )
    out = build_novelty_scores(states)

    rb1 = out[out.player_identity_key.eq("rb1")].iloc[0]
    assert rb1.analog_scoring_state == "NO_PERCENTILE_SUPPORT"

    rb2 = out[out.player_identity_key.eq("rb2")].iloc[0]
    rb3 = out[out.player_identity_key.eq("rb3")].iloc[0]
    assert rb2.percentile_reference_rows == 1
    assert rb3.percentile_reference_rows == 1
    assert rb2.mean_k_distance_pct == 1.0
    assert rb2.effective_analog_support_pct == 0.0
    assert rb2.analog_novelty_risk == 1.0
    assert rb3.mean_k_distance_pct == 0.0
    assert rb3.effective_analog_support_pct == 1.0
    assert rb3.analog_novelty_risk == 0.0

    rb4 = out[out.player_identity_key.eq("rb4")].iloc[0]
    assert rb4.percentile_reference_rows == 3

    wr2 = out[out.player_identity_key.eq("wr2")].iloc[0]
    assert wr2.percentile_reference_rows == 1
    assert wr2.mean_k_distance_pct == 0.0


def test_no_analog_support_is_never_silently_scored():
    states = pd.DataFrame(
        [
            _state_row(2020, 1, "p1", np.nan, 0.0, state="NO_ANALOG_SUPPORT"),
            _state_row(2020, 2, "p2", 1.0, 9.0),
        ]
    )
    out = build_novelty_scores(states)
    first = out[out.player_identity_key.eq("p1")].iloc[0]
    assert first.analog_scoring_state == "NO_ANALOG_SUPPORT"
    assert np.isnan(first.analog_novelty_risk)


def test_novelty_threshold_is_exactly_point_eight():
    assert NOVELTY_THRESHOLD == 0.80
    assert _cohort_from_risk(0.80) == "HIGH_NOVELTY"
    assert _cohort_from_risk(0.799999) == "COMPARISON"


def _m(rows, mae=1.0, rmse=1.0, p90=1.0, cat=0.10):
    return {
        "rows": rows,
        "mae": mae,
        "rmse": rmse,
        "bias": 0.0,
        "p90_ae": p90,
        "catastrophic_miss_rate": cat,
    }


def test_primary_gate_locks_row_floors_mae_and_two_of_three_tail_rule():
    comparison = _m(250)
    high = _m(100, mae=1.03, rmse=1.03, p90=1.03, cat=0.10)
    gate = _primary_gate(high, comparison, True)
    assert gate["high_novelty_rows_gate"]
    assert gate["comparison_rows_gate"]
    assert gate["mae_gate"]
    assert gate["tail_three_pct_metric_count"] == 2
    assert gate["two_of_three_tail_gate"]
    assert gate["season_gate_pass"]

    too_few = _primary_gate(_m(99, mae=1.10, rmse=1.10, p90=1.10, cat=0.20), comparison, True)
    assert not too_few["high_novelty_rows_gate"]
    assert not too_few["season_gate_pass"]

    weak_mae = _primary_gate(_m(100, mae=1.029, rmse=1.10, p90=1.10, cat=0.20), comparison, True)
    assert not weak_mae["mae_gate"]
    assert not weak_mae["season_gate_pass"]


def test_replication_gate_locks_75_200_floors_and_same_mae_direction():
    comparison = _m(200)
    high = _m(75, mae=1.04, rmse=1.04, p90=1.04, cat=0.10)
    gate = _replication_gate(high, comparison, primary_mae_delta=0.05, integrity_gate=True)
    assert gate["high_novelty_rows_gate"]
    assert gate["comparison_rows_gate"]
    assert gate["mae_gate"]
    assert gate["two_of_three_tail_gate"]
    assert gate["same_mae_direction_gate"]
    assert gate["season_gate_pass"]

    reverse = _replication_gate(
        _m(75, mae=0.97, rmse=1.04, p90=1.04, cat=0.20),
        comparison,
        primary_mae_delta=0.05,
        integrity_gate=True,
    )
    assert not reverse["mae_gate"]
    assert not reverse["same_mae_direction_gate"]
    assert not reverse["season_gate_pass"]


def _prepared_rush_frame(primary_high_share):
    rows = []
    features = [
        "prod_rush_prior_share",
        "prod_rush_prior_games",
        "prod_rush_current_share",
        "prod_rush_current_games",
        "prod_rush_playerform_blend",
    ]

    def add(season, cohort, count, share):
        for i in range(count):
            row = {
                "season": season,
                "position": "RB",
                "rushes": int(round(share * 100)),
                "team_rushes": 100,
                "targets": 0,
                "team_targets": 100,
                "analog_scoring_state": "SCORED",
                "novelty_cohort": cohort,
                "analog_novelty_risk": 0.9 if cohort == "HIGH_NOVELTY" else 0.2,
            }
            for f in features:
                row[f] = 0.0
            rows.append(row)

    for season in range(2019, 2024):
        add(season, "COMPARISON", 20, 0.0)
    add(2024, "COMPARISON", 250, 0.10)
    add(2024, "HIGH_NOVELTY", 100, primary_high_share)
    add(2025, "COMPARISON", 200, 0.10)
    add(2025, "HIGH_NOVELTY", 75, 0.11)
    return pd.DataFrame(rows)


def test_2025_is_not_exposed_when_2024_primary_fails():
    frame = _prepared_rush_frame(primary_high_share=0.10)
    metrics, gates = _evaluate_family(
        frame,
        "RB_RUSH_OPPORTUNITY",
        FAMILIES["RB_RUSH_OPPORTUNITY"],
        True,
    )
    assert {row["evaluation_season"] for row in metrics} == {2024}
    assert {row["evaluation_season"] for row in gates} == {2024}
    assert gates[0]["replication_inspected"] is False
    assert gates[0]["family_disposition"] == "FAILED_CLOSED_PRIMARY"


def test_2025_is_exposed_only_after_legitimate_2024_pass():
    frame = _prepared_rush_frame(primary_high_share=0.11)
    metrics, gates = _evaluate_family(
        frame,
        "RB_RUSH_OPPORTUNITY",
        FAMILIES["RB_RUSH_OPPORTUNITY"],
        True,
    )
    assert {row["evaluation_season"] for row in metrics} == {2024, 2025}
    assert {row["evaluation_season"] for row in gates} == {2024, 2025}
    assert gates[0]["replication_inspected"] is True
    assert gates[-1]["family_disposition"] == "ANALOG_RELIABILITY_SIGNAL_REPLICATED"


def test_integrity_recomputes_strict_chronology_instead_of_trusting_flag():
    history = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "team": "A",
                "player_identity_key": "p1",
                "position": "RB",
            }
        ]
    )
    states = pd.DataFrame([_state_row(2024, 2, "p1", 1.0, 9.0)])
    neighbors = pd.DataFrame(
        [
            {
                "target_season": 2024,
                "target_week": 2,
                "target_team": "A",
                "target_player_identity_key": "p1",
                "position": "RB",
                "neighbor_rank": 1,
                "analog_season": 2024,
                "analog_week": 2,
                "analog_team": "B",
                "analog_player_identity_key": "p2",
                "strict_prior": True,
            }
        ]
    )
    integrity = audit_integrity(history, states, neighbors)
    assert integrity["chronology_violations"] == 1
    assert integrity["declared_strict_prior_mismatches"] == 1
    assert not integrity["integrity_gate"]
