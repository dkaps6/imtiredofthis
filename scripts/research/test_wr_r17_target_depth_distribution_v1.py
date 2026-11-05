#!/usr/bin/env python3
"""Synthetic mechanical tests for frozen WR-R17 before a valid Stage-A result."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r17_target_depth_distribution_stage_a_v1b as base
from scripts.research import evaluate_wr_r17_target_depth_distribution_stage_a_v1c as roster_bridge


def _row(week: int, air: float, rid: str = "00-TEST", name: str = "Test Receiver") -> dict:
    return {"season": 2023, "week": week, "game_id": f"2023_{week:02d}_AAA_BBB", "posteam": "AAA", "receiver_player_name": name, "receiver_player_id": rid, "pass_attempt": 1, "sack": 0, "two_point_attempt": 0, "air_yards": air}


def test_feature_math_and_strict_cutoff() -> None:
    airs = [0.0, 5.0, 10.0, 5.0, 10.0, 15.0, 10.0, 15.0, 20.0, 15.0, 20.0, 25.0]
    rows, k = [], 0
    for week in [1, 2, 3, 4]:
        for _ in range(3):
            rows.append(_row(week, airs[k])); k += 1
    rows.append(_row(5, 999.0))
    targets = base.prepare_target_events(pd.DataFrame(rows))
    state = base.receiver_state(targets, "testreceiver", 2023, 5)
    assert state["identity_mode"] == "id", state
    assert state["prior_target_games"] == base.MIN_PRIOR_TARGET_GAMES, state
    assert state["prior_target_events"] == base.MIN_PRIOR_TARGET_EVENTS, state
    arr = np.array(airs, dtype=float)
    assert math.isclose(state["DEPTH_SD8"], float(arr.std(ddof=0)), rel_tol=0, abs_tol=1e-12)
    expected_iqr = float(np.quantile(arr, 0.75, method="linear") - np.quantile(arr, 0.25, method="linear"))
    assert math.isclose(state["DEPTH_IQR8"], expected_iqr, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(state["DEEP15_TARGET_SHARE8"], float((arr >= 15).mean()), rel_tol=0, abs_tol=1e-12)
    assert state["history_max_week"] == 4, state
    assert state["mean_air_yards_per_target8"] < 100.0


def test_event_preservation() -> None:
    rows = []
    for week in [1, 2, 3, 4]:
        rows.extend([_row(week, 10.0), _row(week, 10.0), _row(week, 20.0)])
    state = base.receiver_state(base.prepare_target_events(pd.DataFrame(rows)), "testreceiver", 2023, 5)
    assert state["prior_target_events"] == 12, state


def test_prior_roster_gsis_bridge_handles_abbreviated_pbp_names() -> None:
    rows = []
    for week in [1, 2, 3, 4]:
        rows.extend([_row(week, 5.0, "00-ID", "T.Receiver"), _row(week, 15.0, "00-ID", "T.Receiver"), _row(week, 25.0, "00-ID", "T.Receiver")])
    targets = base.prepare_target_events(pd.DataFrame(rows))
    rosters = pd.DataFrame([
        {"season": 2022, "week": 1, "team": "AAA", "player_clean_key": "testreceiver", "player_id": "00-ID"},
        {"season": 2022, "week": 8, "team": "AAA", "player_clean_key": "testreceiver", "player_id": "00-ID"},
        {"season": 2023, "week": 4, "team": "AAA", "player_clean_key": "testreceiver", "player_id": "00-ID"},
        {"season": 2023, "week": 5, "team": "AAA", "player_clean_key": "testreceiver", "player_id": "00-LEAK"},
    ])
    state = roster_bridge.receiver_state(targets, rosters, "testreceiver", "AAA", 2023, 5)
    assert state["identity_mode"] == "id", state
    assert state["identity_source"] == "weekly_roster", state
    assert state["resolved_receiver_id"] == "00-ID", state
    assert state["prior_target_events"] == 12, state
    assert math.isfinite(state["DEPTH_SD8"]), state


def test_target_week_roster_not_used() -> None:
    rosters = pd.DataFrame([{"season": 2023, "week": 5, "team": "AAA", "player_clean_key": "rookiereceiver", "player_id": "00-ROOKIE"}])
    out = roster_bridge.resolve_roster_player_id(rosters, "rookiereceiver", "AAA", 2023, 5)
    assert out["roster_identity_mode"] == "unmatched", out


def test_ambiguous_roster_identity_fails_closed() -> None:
    rosters = pd.DataFrame([
        {"season": 2022, "week": 1, "team": "BBB", "player_clean_key": "sameplayer", "player_id": "00-A"},
        {"season": 2022, "week": 2, "team": "CCC", "player_clean_key": "sameplayer", "player_id": "00-B"},
    ])
    out = roster_bridge.resolve_roster_player_id(rosters, "sameplayer", "AAA", 2023, 5)
    assert out["roster_identity_mode"] == "ambiguous", out


def main() -> int:
    test_feature_math_and_strict_cutoff()
    test_event_preservation()
    test_prior_roster_gsis_bridge_handles_abbreviated_pbp_names()
    test_target_week_roster_not_used()
    test_ambiguous_roster_identity_fails_closed()
    assert base.SIGNAL_PRIORITY == ["DEPTH_SD8", "DEPTH_IQR8", "DEEP15_TARGET_SHARE8"]
    print("WR-R17 synthetic mechanical + prior-roster identity tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
