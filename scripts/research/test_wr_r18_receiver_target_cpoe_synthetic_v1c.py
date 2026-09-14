#!/usr/bin/env python3
"""Final pre-result synthetic mechanics suite for WR-R18 v1c."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r18_receiver_target_cpoe_stage_a_v1c as r18
from scripts.research import test_wr_r18_receiver_target_cpoe_synthetic_v1 as original


def test_original_suite_passes_under_v1c() -> None:
    original.test_identity_support_and_leakage()
    original.test_ambiguous_identity_fails_closed()
    original.test_cpoe_missingness_audit()
    original.test_raw_gate_direction_is_frozen_positive()


def test_zero_denominator_tail_gate() -> None:
    assert np.isinf(r18._ratio(0.92, 0.0))
    assert r18._tail_ratio_pass(float("inf")) is True
    assert np.isnan(r18._ratio(0.0, 0.0))
    assert r18._tail_ratio_pass(np.nan) is False
    assert abs(r18._ratio(0.24, 0.12) - 2.0) < 1e-12


def test_last8_target_bearing_window_literal_contract() -> None:
    rows = []
    for week in range(1, 10):
        for j in range(2):
            if week == 1:
                cpoe = 999.0
            elif week == 9:
                cpoe = np.nan
            else:
                cpoe = float(week * 10 + j)
            rows.append(original._pbp_row(2022, week, f"g{week}", cpoe, 5 + j))

    targets, _ = r18.prepare_pbp_sources(pd.DataFrame(rows))
    key = r18._name_key("John Doe")
    rosters = pd.DataFrame([
        {"season": 2022, "week": w, "team": "IND", "player_clean_key": key, "player_id": "00-TEST1"}
        for w in range(1, 10)
    ])
    state = r18.receiver_state(targets, rosters, key, "IND", 2023, 1)

    assert state["prior_target_games"] == 8
    assert state["prior_otherwise_eligible_targets"] == 16
    assert state["prior_null_cpoe_targets"] == 2
    assert state["prior_valid_cpoe_targets"] == 14
    assert np.isnan(state["WR_TARGET_CPOE_MEAN8"])


def main() -> int:
    test_original_suite_passes_under_v1c()
    test_zero_denominator_tail_gate()
    test_last8_target_bearing_window_literal_contract()
    print("WR-R18 synthetic mechanics v1c: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
