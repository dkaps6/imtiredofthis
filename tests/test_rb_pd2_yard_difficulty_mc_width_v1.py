import numpy as np
import pandas as pd

from scripts.research.evaluate_rb_pd2_yard_difficulty_mc_width_v1 import (
    BOOTSTRAP_REPS,
    crossed_player_game_bootstrap_probability,
    empirical_crps,
    player_cluster_bootstrap_probability,
    strict_prior_difficulty_scores,
    widen_mean_neutral,
    width_multiplier,
)


def test_width_multiplier_frozen_mapping():
    assert width_multiplier(0.25) == 1.0
    assert width_multiplier(0.50) == 1.0
    assert np.isclose(width_multiplier(0.75), 1.15)
    assert np.isclose(width_multiplier(1.00), 1.30)


def test_mean_neutral_width_preserves_nonnegative_mean():
    base = np.array([0.0, 10.0, 20.0, 50.0, 100.0])
    cand = widen_mean_neutral(base, 1.30)
    assert np.all(cand >= 0.0)
    assert np.isclose(cand.mean(), base.mean(), atol=1e-12)
    assert cand.max() > base.max()


def test_width_one_is_identity():
    base = np.array([0.0, 1.5, 7.0, 25.0])
    cand = widen_mean_neutral(base, 1.0)
    np.testing.assert_allclose(cand, base, atol=1e-12, rtol=0.0)


def test_empirical_crps_degenerate_distribution_is_absolute_error():
    draws = np.full(2000, 40.0)
    assert np.isclose(empirical_crps(draws, 55.0), 15.0)


def test_strict_prior_percentile_scores_week_before_inserting_current_week():
    rows = []
    # 100 earlier scoreable rows establish the reference pool, one per week group.
    for i in range(100):
        rows.append({
            "season": 2021,
            "week": i + 1,
            "team": "X",
            "player": f"p{i}",
            "player_key": f"p{i}",
            "prior_games": 4,
            "prior8_yard_mae": float(i + 1),
        })
    # Both current-week rows must see the same 100-row reference pool.
    rows.extend([
        {"season": 2022, "week": 1, "team": "A", "player": "a", "player_key": "a", "prior_games": 4, "prior8_yard_mae": 50.0},
        {"season": 2022, "week": 1, "team": "B", "player": "b", "player_key": "b", "prior_games": 4, "prior8_yard_mae": 100.0},
    ])
    scored = strict_prior_difficulty_scores(pd.DataFrame(rows))
    q = scored.loc[(scored.season == 2022) & (scored.week == 1)].set_index("player_key")
    assert q.loc["a", "difficulty_reference_n"] == 100
    assert q.loc["b", "difficulty_reference_n"] == 100
    assert np.isclose(q.loc["a", "difficulty_score"], 0.50)
    assert np.isclose(q.loc["b", "difficulty_score"], 1.00)


def test_player_cluster_bootstrap_is_paired_and_detects_uniform_improvement():
    rows = pd.DataFrame({
        "player_key": ["a", "a", "b", "b", "c"],
        "baseline_crps": [10.0, 11.0, 12.0, 13.0, 9.0],
        "candidate_crps": [9.0, 10.0, 11.0, 12.0, 8.0],
    })
    assert BOOTSTRAP_REPS == 10_000
    assert player_cluster_bootstrap_probability(rows) == 1.0


def test_crossed_player_game_bootstrap_detects_uniform_improvement():
    rows = pd.DataFrame({
        "player_key": ["a", "a", "b", "b", "c"],
        "game_key": ["g1", "g2", "g1", "g3", "g2"],
        "baseline_crps": [10.0, 11.0, 12.0, 13.0, 9.0],
        "candidate_crps": [9.0, 10.0, 11.0, 12.0, 8.0],
    })
    assert crossed_player_game_bootstrap_probability(rows) == 1.0


def test_crossed_player_game_bootstrap_detects_uniform_regression():
    rows = pd.DataFrame({
        "player_key": ["a", "a", "b", "b", "c"],
        "game_key": ["g1", "g2", "g1", "g3", "g2"],
        "baseline_crps": [9.0, 10.0, 11.0, 12.0, 8.0],
        "candidate_crps": [10.0, 11.0, 12.0, 13.0, 9.0],
    })
    assert crossed_player_game_bootstrap_probability(rows) == 0.0
