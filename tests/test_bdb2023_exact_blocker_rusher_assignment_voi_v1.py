import numpy as np
import pandas as pd

from scripts.football_context.evaluate_bdb2023_exact_blocker_rusher_assignment_voi_v1 import (
    add_strict_prior_rates,
    boolish,
    cluster_bootstrap_gain,
)


def test_boolish_handles_common_encodings():
    s = pd.Series([1, 0, True, False, "yes", "no", None])
    known, pos = boolish(s)
    assert known.tolist() == [True, True, True, True, True, True, False]
    assert pos.tolist() == [True, False, True, False, True, False, False]


def test_strict_prior_rates_do_not_use_same_week():
    edges = pd.DataFrame([
        {"week": 1, "blocker_nfl_id": "b1", "defender_nfl_id": "d1", "pressure_allowed_edge": 1.0},
        {"week": 1, "blocker_nfl_id": "b1", "defender_nfl_id": "d1", "pressure_allowed_edge": 0.0},
        {"week": 2, "blocker_nfl_id": "b1", "defender_nfl_id": "d1", "pressure_allowed_edge": 0.0},
    ])
    out, audit = add_strict_prior_rates(edges)
    assert audit["same_or_future_week_history_violations"] == 0
    w1 = out.loc[out["week"].eq(1)]
    w2 = out.loc[out["week"].eq(2)].iloc[0]
    assert w1["prior_blocker_n"].eq(0).all()
    assert int(w2["prior_blocker_n"]) == 2
    assert int(w2["prior_defender_n"]) == 2
    assert float(w2["blocker_prior_allow_rate"]) == 0.5
    assert float(w2["defender_prior_pressure_rate"]) == 0.5


def test_game_cluster_bootstrap_keeps_both_team_rows_together():
    holdout = pd.DataFrame([
        {"game_id": 1, "target_pressure_allowed_edge_rate": 0.10},
        {"game_id": 1, "target_pressure_allowed_edge_rate": 0.20},
        {"game_id": 2, "target_pressure_allowed_edge_rate": 0.30},
        {"game_id": 2, "target_pressure_allowed_edge_rate": 0.40},
    ])
    base = np.array([0.20, 0.30, 0.40, 0.50])
    cand = np.array([0.10, 0.20, 0.30, 0.40])
    out = cluster_bootstrap_gain(holdout, base, cand)
    assert out["clusters"] == 2
    assert out["ci95_lower"] > 0
