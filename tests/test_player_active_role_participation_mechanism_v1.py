import numpy as np
import pandas as pd

from scripts.research.evaluate_player_active_role_participation_v1 import (
    BASELINE_MIN,
    BOOT_REPS,
    BOOT_SEED,
    PLAYER_MIN,
    PLAYER_WINDOW,
    PRIOR_STRENGTH,
    baseline_probability,
    candidate_probability,
    score_frame,
    support_gates,
)


def test_frozen_constants():
    assert PLAYER_WINDOW == 4
    assert PLAYER_MIN == 3
    assert BASELINE_MIN == 100
    assert PRIOR_STRENGTH == 2.0
    assert BOOT_REPS == 5000
    assert BOOT_SEED == 20261016


def test_probability_mapping_is_exact_and_interior():
    p0 = baseline_probability(87, 100)
    assert np.isclose(p0, 87.5 / 101.0)
    p1 = candidate_probability(2, 3, p0)
    assert np.isclose(p1, (2.0 + 2.0 * p0) / 5.0)
    assert 0.0 < p0 < 1.0
    assert 0.0 < p1 < 1.0


def test_candidate_player_history_is_majority_weight_at_min_support():
    p0 = 0.5
    p1 = candidate_probability(3, 3, p0)
    # 3 real player games vs two fixed prior pseudo-games.
    assert np.isclose(p1, 0.8)


def test_brier_delta_positive_when_candidate_is_better():
    df = pd.DataFrame({
        "y": [1, 1, 0, 0],
        "p_baseline": [0.6, 0.6, 0.6, 0.6],
        "p_candidate": [0.8, 0.8, 0.2, 0.2],
        "player_id": ["a", "a", "b", "b"],
    })
    out = score_frame(df)
    assert out["delta_brier"] > 0
    assert out["delta_logloss"] > 0


def test_support_gates_fail_closed_without_zero_outcomes():
    metrics = {
        "rows": 1000,
        "players": 100,
        "positive": 1000,
        "zero": 0,
    }
    gates = support_gates("RB", metrics, eval_players=80)
    assert gates["zero"] is False
    assert all(v for k, v in gates.items() if k != "zero")
