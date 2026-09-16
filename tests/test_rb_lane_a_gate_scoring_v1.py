import numpy as np
import pandas as pd
import pytest

from scripts.backtest.rb_lane_a_gate_scoring_v1 import (
    ADEQUACY_FLOOR,
    adequacy_check,
    bootstrap_gate_report,
    compute_mae_delta_rows,
    crossed_player_game_bootstrap_probability,
    per_season_nonregression_check,
    player_cluster_bootstrap_probability,
    whole_season_deployable_safety_check,
)


def test_adequacy_check_passes_when_all_cohorts_meet_floor():
    population = pd.DataFrame({"x": range(40)})
    cohorts = {"carries_20plus": pd.Series([True] * 30 + [False] * 10)}
    result = adequacy_check(population, cohorts)
    assert result["disposition"] == "ADEQUATE"
    assert result["overall_n"] == 40


def test_adequacy_check_fails_closed_on_thin_cohort():
    population = pd.DataFrame({"x": range(40)})
    cohorts = {"yards_100plus": pd.Series([True] * 10 + [False] * 30)}
    result = adequacy_check(population, cohorts)
    assert result["disposition"] == "INSUFFICIENT_EVIDENCE"
    assert result["cohorts"]["yards_100plus"]["adequate"] is False


def test_adequacy_check_fails_closed_on_thin_overall_population():
    population = pd.DataFrame({"x": range(ADEQUACY_FLOOR - 1)})
    result = adequacy_check(population, {})
    assert result["disposition"] == "INSUFFICIENT_EVIDENCE"


def test_compute_mae_delta_rows_negative_when_candidate_closer_to_actual():
    scored = pd.DataFrame(
        [
            {
                "player_clean_key": "p1", "game_key": "g1",
                "candidate": 52.0, "comparator": 60.0, "actual": 50.0,
            }
        ]
    )
    out = compute_mae_delta_rows(
        scored, candidate_col="candidate", comparator_col="comparator", actual_col="actual"
    )
    # candidate error = 2, comparator error = 10 -> delta = -8 (candidate wins)
    assert out.iloc[0]["delta"] == pytest.approx(-8.0)


def test_compute_mae_delta_rows_fails_closed_on_missing_columns():
    scored = pd.DataFrame([{"player_clean_key": "p1"}])
    with pytest.raises(RuntimeError, match="missing columns"):
        compute_mae_delta_rows(scored, candidate_col="c", comparator_col="m", actual_col="a")


def _synthetic_delta_rows(n_players=10, rows_per_player=4, candidate_always_wins=True, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        for g in range(rows_per_player):
            base = rng.normal(0, 5)
            delta = base - 3.0 if candidate_always_wins else base + 3.0
            rows.append({"player_key": f"p{p}", "game_key": f"g{p}_{g}", "delta": delta})
    return pd.DataFrame(rows)


def test_player_cluster_bootstrap_high_probability_when_candidate_dominates():
    rows = _synthetic_delta_rows(candidate_always_wins=True)
    prob = player_cluster_bootstrap_probability(rows, boot_n=500, seed=1)
    assert prob > 0.90


def test_player_cluster_bootstrap_low_probability_when_candidate_loses():
    rows = _synthetic_delta_rows(candidate_always_wins=False)
    prob = player_cluster_bootstrap_probability(rows, boot_n=500, seed=1)
    assert prob < 0.50


def test_crossed_player_game_bootstrap_high_probability_when_candidate_dominates():
    rows = _synthetic_delta_rows(candidate_always_wins=True)
    prob = crossed_player_game_bootstrap_probability(rows, boot_n=500, seed=1)
    assert prob > 0.90


def test_bootstrap_gate_report_requires_both_bootstraps():
    rows = _synthetic_delta_rows(candidate_always_wins=True)
    report = bootstrap_gate_report(rows, boot_n=500, seed=1)
    assert report["disposition"] == "BOOTSTRAP_GATE_PASS"
    assert report["player_cluster_bootstrap_probability"] >= 0.90
    assert report["crossed_player_game_bootstrap_probability"] >= 0.90


def test_bootstrap_gate_report_fails_closed_when_one_bootstrap_fails():
    rows = _synthetic_delta_rows(candidate_always_wins=False)
    report = bootstrap_gate_report(rows, boot_n=500, seed=1)
    assert report["disposition"] == "BOOTSTRAP_GATE_FAILURE"


def test_per_season_nonregression_passes_when_both_rotations_win():
    rows = {
        1: pd.DataFrame({"delta": [-1.0, -2.0, -3.0]}),
        2: pd.DataFrame({"delta": [-0.5, -1.5]}),
    }
    result = per_season_nonregression_check(rows)
    assert result["disposition"] == "PER_SEASON_NONREGRESSION_PASS"


def test_per_season_nonregression_fails_closed_on_one_rotation_regressing():
    rows = {
        1: pd.DataFrame({"delta": [-1.0, -2.0]}),
        2: pd.DataFrame({"delta": [1.0, 2.0]}),
    }
    result = per_season_nonregression_check(rows)
    assert result["disposition"] == "PER_SEASON_NONREGRESSION_FAILURE"
    assert result["rotations"]["2"]["nonregressed"] is False


def test_whole_season_safety_passes_when_deployable_non_worse():
    rows = pd.DataFrame(
        [
            {"deployable_candidate_rush_yards": 51.0, "promotion_rush_yards": 55.0, "actual_rush_yards": 50.0},
            {"deployable_candidate_rush_yards": 60.0, "promotion_rush_yards": 60.0, "actual_rush_yards": 58.0},
        ]
    )
    result = whole_season_deployable_safety_check(rows)
    assert result["disposition"] == "WHOLE_SEASON_SAFETY_PASS"


def test_whole_season_safety_fails_closed_when_deployable_worse():
    rows = pd.DataFrame(
        [
            {"deployable_candidate_rush_yards": 80.0, "promotion_rush_yards": 51.0, "actual_rush_yards": 50.0},
        ]
    )
    result = whole_season_deployable_safety_check(rows)
    assert result["disposition"] == "WHOLE_SEASON_SAFETY_FAILURE"
