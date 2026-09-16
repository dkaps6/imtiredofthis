#!/usr/bin/env python3
"""Synthetic mechanics tests for corrected WR-R18 v1b before real Stage A."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r18_receiver_target_cpoe_stage_a_v1b as r18
from scripts.research import test_wr_r18_receiver_target_cpoe_synthetic_v1 as original


def test_original_suite_under_corrected_mechanics() -> None:
    # Importing v1b patches the base module object used by the original suite.
    original.test_identity_support_and_leakage()
    original.test_ambiguous_identity_fails_closed()
    original.test_cpoe_missingness_audit()
    original.test_raw_gate_direction_is_frozen_positive()


def test_zero_denominator_tail_ratio_semantics() -> None:
    assert np.isinf(r18._ratio(0.92, 0.0))
    assert np.isnan(r18._ratio(0.0, 0.0))
    assert abs(r18._ratio(0.24, 0.12) - 2.0) < 1e-12


def test_last8_is_target_bearing_not_valid_cpoe_bearing() -> None:
    rows = []
    # Nine prior target-bearing games. Game 1 is intentionally very old with
    # valid extreme CPOE. Game 9 is the newest target-bearing game but all CPOE
    # is null. The frozen contract says the window is games 2..9, so game 1
    # must not sneak in merely to replace missing CPOE from game 9.
    for week in range(1, 10):
        for j in range(2):
            if week == 1:
                cpoe = 999.0
            elif week == 9:
                cpoe = np.nan
            else:
                cpoe = float(week * 10 + j)
            rows.append(original._pbp_row(2022, week, f"g{week}", cpoe, 5 + j))

    raw = pd.DataFrame(rows)
    targets, _ = r18.prepare_pbp_sources(raw)
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
    # If the implementation incorrectly selected the last 8 valid-CPOE games,
    # game 1 would enter and the 16-event floor would pass. Literal plan logic
    # correctly leaves the signal unsupported here.
    assert np.isnan(state["WR_TARGET_CPOE_MEAN8"])


def main() -> int:
    test_original_suite_under_corrected_mechanics()
    test_zero_denominator_tail_ratio_semantics()
    test_last8_is_target_bearing_not_valid_cpoe_bearing()
    print("WR-R18 synthetic mechanics v1b: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
