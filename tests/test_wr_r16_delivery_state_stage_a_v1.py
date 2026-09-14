import numpy as np
import pandas as pd

from scripts.research.evaluate_wr_r16_delivery_state_stage_a_v1 import (
    _receiver_state,
    _team_state,
    add_signals,
    score_development,
)


def _target_rows():
    rows = []
    for week, cpoe, air, complete in [
        (1, 1.0, 10.0, True),
        (2, 2.0, 12.0, True),
        (3, 3.0, 14.0, False),
        (4, 4.0, 16.0, True),
        (5, 99.0, 60.0, True),  # target week: must never enter W5 feature history
    ]:
        rows.append({
            "season": 2023,
            "week": week,
            "game_id": f"g{week}",
            "team": "IND",
            "receiver_key": "testreceiver",
            "cpoe_num": cpoe,
            "air": air,
            "completed_air": air if complete else 0.0,
            "complete": complete,
            "passing_yards": air if complete else 0.0,
            "yards_after_catch": 0.0,
        })
    return pd.DataFrame(rows)


def _attempt_rows():
    rows = []
    for week, cpoe, air, complete in [
        (1, 1.0, 10.0, True),
        (2, 2.0, 20.0, True),
        (3, 3.0, 30.0, False),
        (4, 4.0, 40.0, True),
        (5, 99.0, 80.0, True),  # target week: forbidden
    ]:
        rows.append({
            "season": 2023,
            "week": week,
            "game_id": f"g{week}",
            "team": "IND",
            "cpoe_num": cpoe,
            "air": air,
            "completed_air": air if complete else 0.0,
            "deep15": air >= 15,
            "deep15_complete": (air >= 15 and complete),
        })
    return pd.DataFrame(rows)


def test_receiver_state_is_strictly_prior_to_target_week():
    state = _receiver_state(_target_rows(), "testreceiver", 2023, 5)
    assert state["wr_prior_target_games"] == 4
    assert state["wr_target_cpoe_mean8"] == 2.5
    assert state["wr_target_cpoe_mean8"] < 10


def test_team_state_is_strictly_prior_to_target_week():
    state = _team_state(_attempt_rows(), "IND", 2023, 5)
    assert state["team_prior_pass_games"] == 4
    assert state["team_cpoe_mean8"] == 2.5
    assert state["team_cpoe_mean8"] < 10


def test_signal_standardization_is_fit_from_2023_only():
    panel = pd.DataFrame({
        "season": [2023, 2023, 2024],
        "wr_target_cpoe_mean8": [1.0, 3.0, 1000.0],
        "team_cpoe_mean8": [2.0, 4.0, 1000.0],
        "wr_completed_air_yards_per_target8": [5.0, 7.0, 1000.0],
        "team_completed_air_per_attempt8": [4.0, 8.0, 1000.0],
        "team_deep15_completion_rate8": [0.4, 0.6, 1.0],
        "wr_cpoe_recent3_minus8": [0.1, 0.2, 100.0],
        "team_cpoe_recent3_minus8": [0.2, 0.3, 100.0],
    })
    _, params = add_signals(panel)
    assert params["wr_target_cpoe_mean8"]["mean"] == 2.0
    assert params["team_cpoe_mean8"]["mean"] == 3.0


def test_stage_a_scoring_ignores_2024_rows(monkeypatch):
    import scripts.research.evaluate_wr_r16_delivery_state_stage_a_v1 as m
    monkeypatch.setitem(m.EXPECTED_ROWS, 2023, 8)

    rows = []
    for season in [2023, 2024]:
        for i in range(8):
            resid = float(i * 10) if season == 2023 else float(10000 + i)
            rows.append({
                "season": season,
                "wr_rank": 1 if i % 2 == 0 else 2,
                "actual_rec_yards": 100.0 + resid,
                "yard_residual": resid,
                "DEEP_DELIVERY": float(i),
                "COMPLETED_AIR": float(i),
                "DELIVERY_CPOE": float(i),
                "DELIVERY_MOMENTUM": float(i),
            })
    score, _ = score_development(pd.DataFrame(rows))
    assert set(score["n"]) == {8}


def test_signal_priority_is_fixed_not_best_metric(monkeypatch):
    import scripts.research.evaluate_wr_r16_delivery_state_stage_a_v1 as m
    monkeypatch.setitem(m.EXPECTED_ROWS, 2023, 200)
    monkeypatch.setattr(m, "MIN_SLICE_N", 10)

    rows = []
    for i in range(200):
        v = float(i)
        resid = v
        rows.append({
            "season": 2023,
            "wr_rank": 1 if i % 2 == 0 else 2,
            "actual_rec_yards": 100.0 + resid,
            "yard_residual": resid,
            "DEEP_DELIVERY": v,
            "COMPLETED_AIR": v * 2,
            "DELIVERY_CPOE": v * 3,
            "DELIVERY_MOMENTUM": v * 4,
        })
    score, decision = score_development(pd.DataFrame(rows))
    assert score["supported"].all()
    assert decision["advancing_signal"] == "DEEP_DELIVERY"
