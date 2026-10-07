from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.research.lock_rb_target_share_trajectory_shadow_v1 as s


def _frame():
    return pd.DataFrame([
        {
            "season": 2026, "week": 5, "event_id": "G1", "team": "IND",
            "opponent": "HOU", "player": "RB One", "player_clean_key": "rbone",
            "position": "RB", "entitlement_tgt_share": 0.10,
            "trajectory_delta": 0.20,
        },
        {
            "season": 2026, "week": 5, "event_id": "G1", "team": "IND",
            "opponent": "HOU", "player": "RB Two", "player_clean_key": "rbtwo",
            "position": "RB", "entitlement_tgt_share": 0.06,
            "trajectory_delta": 0.00,
        },
        {
            "season": 2026, "week": 5, "event_id": "G1", "team": "IND",
            "opponent": "HOU", "player": "WR One", "player_clean_key": "wrone",
            "position": "WR", "entitlement_tgt_share": 0.24,
            "trajectory_delta": 0.50,
        },
        {
            "season": 2026, "week": 5, "event_id": "G1", "team": "IND",
            "opponent": "HOU", "player": "TE One", "player_clean_key": "teone",
            "position": "TE", "entitlement_tgt_share": 0.14,
            "trajectory_delta": -0.30,
        },
    ])


def test_positive_rb_trajectory_increases_share_within_fixed_room():
    x, rooms, team_gap, non_rb_gap = s.apply_shadow(_frame())
    rb1 = x.loc[x["player_clean_key"].eq("rbone")].iloc[0]
    rb2 = x.loc[x["player_clean_key"].eq("rbtwo")].iloc[0]

    assert rb1["trajectory_weight"] == pytest.approx(0.10 * np.exp(0.20))
    assert rb2["trajectory_weight"] == pytest.approx(0.06)
    assert rb1["shadow_entitlement_tgt_share"] > 0.10
    assert rb2["shadow_entitlement_tgt_share"] < 0.06

    baseline_pool = 0.10 + 0.06
    assert x.loc[x["position_family"].eq("RB"), "shadow_entitlement_tgt_share"].sum() == pytest.approx(
        baseline_pool, abs=1e-12
    )
    assert rooms.iloc[0]["pool_gap"] <= 1e-12
    assert team_gap <= 1e-12
    assert non_rb_gap <= 1e-12


def test_non_rb_entitlement_is_exact_noop_even_with_nonzero_delta():
    x, _, _, non_rb_gap = s.apply_shadow(_frame())
    non_rb = x.loc[~x["position_family"].eq("RB")]
    assert np.allclose(
        non_rb["shadow_entitlement_tgt_share"],
        non_rb["baseline_entitlement_tgt_share"],
        rtol=0,
        atol=0,
    )
    assert non_rb_gap == pytest.approx(0.0, abs=1e-15)


def test_single_player_rb_room_is_exact_noop():
    f = pd.DataFrame([{
        "season": 2026, "week": 5, "event_id": "G1", "team": "IND",
        "opponent": "HOU", "player": "Only Back", "player_clean_key": "onlyback",
        "position": "RB", "entitlement_tgt_share": 0.18,
        "trajectory_delta": 0.40,
    }])
    x, rooms, team_gap, non_rb_gap = s.apply_shadow(f)
    row = x.iloc[0]
    assert row["trajectory_weight"] == pytest.approx(0.18 * np.exp(0.40))
    assert row["shadow_entitlement_tgt_share"] == pytest.approx(0.18, abs=1e-12)
    assert row["entitlement_delta"] == pytest.approx(0.0, abs=1e-12)
    assert rooms.iloc[0]["pool_gap"] <= 1e-12
    assert team_gap <= 1e-12
    assert non_rb_gap <= 1e-12


def test_zero_delta_multi_player_room_is_exact_noop():
    f = _frame().copy()
    f.loc[f["position"].eq("RB"), "trajectory_delta"] = 0.0
    x, _, _, _ = s.apply_shadow(f)
    rb = x.loc[x["position_family"].eq("RB")]
    assert np.allclose(
        rb["shadow_entitlement_tgt_share"],
        rb["baseline_entitlement_tgt_share"],
        rtol=0,
        atol=1e-12,
    )


def test_zero_pool_fails_safe_to_zero_without_creating_mass():
    f = pd.DataFrame([
        {
            "event_id": "G1", "team": "IND", "player": "A", "player_clean_key": "a",
            "position": "RB", "entitlement_tgt_share": 0.0, "trajectory_delta": 0.5,
        },
        {
            "event_id": "G1", "team": "IND", "player": "B", "player_clean_key": "b",
            "position": "FB", "entitlement_tgt_share": 0.0, "trajectory_delta": -0.5,
        },
    ])
    x, rooms, team_gap, _ = s.apply_shadow(f)
    assert x["shadow_entitlement_tgt_share"].eq(0.0).all()
    assert rooms.iloc[0]["shadow_pool"] == pytest.approx(0.0)
    assert team_gap <= 1e-12


def test_trajectory_state_uses_recent2_minus_earlier_only():
    team_idx = {
        "IND": pd.DataFrame([
            {"season": 2026, "week": 1, "game_id": "g1", "team": "IND", "team_targets": 20.0},
            {"season": 2026, "week": 2, "game_id": "g2", "team": "IND", "team_targets": 20.0},
            {"season": 2026, "week": 3, "game_id": "g3", "team": "IND", "team_targets": 20.0},
            {"season": 2026, "week": 4, "game_id": "g4", "team": "IND", "team_targets": 20.0},
        ])
    }
    lookup = {
        (1, "g1", "IND", "P1"): 1.0,
        (2, "g2", "IND", "P1"): 1.0,
        (3, "g3", "IND", "P1"): 4.0,
        (4, "g4", "IND", "P1"): 4.0,
    }
    st = s.trajectory_state(team_idx, lookup, "IND", "P1")
    assert st is not None
    assert st["earlier_share"] == pytest.approx(2.0 / 40.0)
    assert st["recent2_share"] == pytest.approx(8.0 / 40.0)
    assert st["trajectory_delta"] == pytest.approx(0.15)
    assert st["trajectory_feature_max_week"] == 4
    assert st["prior_team_games"] == 4


def test_trajectory_state_fails_closed_below_four_prior_team_games():
    team_idx = {
        "IND": pd.DataFrame([
            {"season": 2026, "week": 1, "game_id": "g1", "team": "IND", "team_targets": 20.0},
            {"season": 2026, "week": 2, "game_id": "g2", "team": "IND", "team_targets": 20.0},
            {"season": 2026, "week": 3, "game_id": "g3", "team": "IND", "team_targets": 20.0},
        ])
    }
    assert s.trajectory_state(team_idx, {}, "IND", "P1") is None
