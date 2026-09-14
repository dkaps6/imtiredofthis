#!/usr/bin/env python3
"""Synthetic mechanical tests for WR-R17 before any real Stage-A result."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.research.evaluate_wr_r17_target_depth_distribution_stage_a_v1 import (
    MIN_PRIOR_TARGET_EVENTS,
    MIN_PRIOR_TARGET_GAMES,
    SIGNAL_PRIORITY,
    prepare_target_events,
    receiver_state,
    resolve_prior_receiver_history,
)


def _row(week: int, air: float, rid: str = "00-TEST", name: str = "Test Receiver") -> dict:
    return {
        "season": 2023,
        "week": week,
        "game_id": f"2023_0{week}_AAA_BBB",
        "posteam": "AAA",
        "receiver_player_name": name,
        "receiver_player_id": rid,
        "pass_attempt": 1,
        "sack": 0,
        "two_point_attempt": 0,
        "air_yards": air,
    }


def test_feature_math_and_strict_cutoff() -> None:
    # Exactly 4 prior games x 3 valid targets = frozen 12-event support floor.
    airs = [0.0, 5.0, 10.0, 5.0, 10.0, 15.0, 10.0, 15.0, 20.0, 15.0, 20.0, 25.0]
    rows = []
    k = 0
    for week in [1, 2, 3, 4]:
        for _ in range(3):
            rows.append(_row(week, airs[k]))
            k += 1
    # Target-game event must never enter the feature. Its extreme value makes leakage obvious.
    rows.append(_row(5, 999.0))
    targets = prepare_target_events(pd.DataFrame(rows))
    state = receiver_state(targets, "testreceiver", 2023, 5)
    assert state["identity_mode"] == "id", state
    assert state["prior_target_games"] == MIN_PRIOR_TARGET_GAMES, state
    assert state["prior_target_events"] == MIN_PRIOR_TARGET_EVENTS, state
    arr = np.array(airs, dtype=float)
    assert math.isclose(state["DEPTH_SD8"], float(arr.std(ddof=0)), rel_tol=0, abs_tol=1e-12)
    expected_iqr = float(np.quantile(arr, 0.75, method="linear") - np.quantile(arr, 0.25, method="linear"))
    assert math.isclose(state["DEPTH_IQR8"], expected_iqr, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(state["DEEP15_TARGET_SHARE8"], float((arr >= 15).mean()), rel_tol=0, abs_tol=1e-12)
    assert state["history_max_week"] == 4, state
    assert state["mean_air_yards_per_target8"] < 100.0, "target-week 999-yard sentinel leaked"


def test_id_first_and_name_fallback() -> None:
    rows = [_row(w, 10 + w, rid="00-ID") for w in [1, 2, 3, 4]]
    targets = prepare_target_events(pd.DataFrame(rows))
    hist, audit = resolve_prior_receiver_history(targets, "testreceiver", 2023, 5)
    assert audit["identity_mode"] == "id"
    assert audit["resolved_receiver_id"] == "00-ID"
    assert len(hist) == 4

    fallback = pd.DataFrame([_row(w, 10 + w, rid="") for w in [1, 2, 3, 4]])
    fallback_targets = prepare_target_events(fallback)
    hist2, audit2 = resolve_prior_receiver_history(fallback_targets, "testreceiver", 2023, 5)
    assert audit2["identity_mode"] == "name_fallback"
    assert len(hist2) == 4


def test_ambiguous_alias_fails_closed() -> None:
    rows = [
        _row(1, 10.0, rid="00-A"),
        _row(2, 20.0, rid="00-B"),
        _row(3, 15.0, rid="00-A"),
        _row(4, 25.0, rid="00-B"),
    ]
    targets = prepare_target_events(pd.DataFrame(rows))
    hist, audit = resolve_prior_receiver_history(targets, "testreceiver", 2023, 5)
    assert audit["identity_mode"] == "ambiguous", audit
    assert hist.empty


def test_frozen_signal_priority() -> None:
    assert SIGNAL_PRIORITY == ["DEPTH_SD8", "DEPTH_IQR8", "DEEP15_TARGET_SHARE8"]


def main() -> int:
    test_feature_math_and_strict_cutoff()
    test_id_first_and_name_fallback()
    test_ambiguous_alias_fails_closed()
    test_frozen_signal_priority()
    print("WR-R17 synthetic mechanical tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
