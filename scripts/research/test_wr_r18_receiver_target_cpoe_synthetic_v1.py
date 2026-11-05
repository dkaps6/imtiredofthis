#!/usr/bin/env python3
"""Synthetic mechanics tests for WR-R18 before any real-data Stage A run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r18_receiver_target_cpoe_stage_a_v1 as r18


def _pbp_row(season, week, game_id, cpoe, air, receiver_id="00-TEST1", name="J.Doe", team="IND"):
    return {
        "season": season,
        "week": week,
        "game_id": game_id,
        "posteam": team,
        "receiver_player_name": name,
        "receiver_player_id": receiver_id,
        "pass_attempt": 1,
        "sack": 0,
        "two_point_attempt": 0,
        "cpoe": cpoe,
        "air_yards": air,
        "season_type": "REG",
    }


def test_identity_support_and_leakage() -> None:
    rows = []
    cpoes = list(range(1, 17))
    k = 0
    for week in range(1, 5):
        for _ in range(4):
            rows.append(_pbp_row(2022, week, f"2022_{week}", cpoes[k], 5 + (k % 3)))
            k += 1
    # Otherwise-eligible target with null CPOE must survive source preparation
    # for the selection audit but must not count toward the 16-event support floor.
    rows.append(_pbp_row(2022, 4, "2022_4", np.nan, 25))
    # Target-game event must never enter prior history.
    rows.append(_pbp_row(2023, 1, "2023_1", 999.0, 50))
    raw = pd.DataFrame(rows)
    targets, attempts = r18.prepare_pbp_sources(raw)
    assert len(targets) == 18, len(targets)
    assert targets["target_event_seq"].nunique() == 18

    key = r18._name_key("John Doe")
    rosters = pd.DataFrame([
        {"season": 2022, "week": w, "team": "IND", "player_clean_key": key, "player_id": "00-TEST1"}
        for w in range(1, 5)
    ] + [
        # Different target-week identity must be excluded by the strict-prior resolver.
        {"season": 2023, "week": 1, "team": "IND", "player_clean_key": key, "player_id": "00-FUTURE"}
    ])

    resolved = r18.resolve_roster_player_id(rosters, key, "IND", 2023, 1)
    assert resolved["roster_identity_mode"] == "id"
    assert resolved["roster_player_id"] == "00-TEST1"

    state = r18.receiver_state(targets, rosters, key, "IND", 2023, 1)
    assert state["prior_target_games"] == 4
    assert state["prior_valid_cpoe_targets"] == 16
    assert state["prior_otherwise_eligible_targets"] == 17
    assert state["prior_null_cpoe_targets"] == 1
    assert abs(state["WR_TARGET_CPOE_MEAN8"] - 8.5) < 1e-12
    assert state["history_max_season"] == 2022
    assert state["history_max_week"] == 4

    team = r18.team_state(attempts, "IND", 2023, 1)
    # 999 target-game CPOE must be excluded from the team control too.
    expected_team = float(np.mean(cpoes))
    assert abs(team["team_cpoe_mean8"] - expected_team) < 1e-12
    assert team["team_history_max_season"] == 2022

    # Exactly 15 valid CPOE events must fail the frozen 16-event support floor.
    one_seq = int(targets.loc[targets["cpoe_num"].eq(1), "target_event_seq"].iloc[0])
    targets15 = targets.loc[targets["target_event_seq"].ne(one_seq)].copy()
    state15 = r18.receiver_state(targets15, rosters, key, "IND", 2023, 1)
    assert state15["prior_valid_cpoe_targets"] == 15
    assert np.isnan(state15["WR_TARGET_CPOE_MEAN8"])


def test_ambiguous_identity_fails_closed() -> None:
    key = r18._name_key("John Doe")
    rosters = pd.DataFrame([
        {"season": 2022, "week": 1, "team": "IND", "player_clean_key": key, "player_id": "00-A"},
        {"season": 2022, "week": 2, "team": "IND", "player_clean_key": key, "player_id": "00-B"},
    ])
    got = r18.resolve_roster_player_id(rosters, key, "IND", 2023, 1)
    assert got["roster_identity_mode"] == "ambiguous"
    assert got["roster_player_id"] == ""


def test_cpoe_missingness_audit() -> None:
    raw = pd.DataFrame([
        _pbp_row(2022, 1, "g1", 5.0, 5),
        _pbp_row(2022, 1, "g1", 6.0, 7),
        _pbp_row(2022, 1, "g1", np.nan, 25),
    ])
    targets, _ = r18.prepare_pbp_sources(raw)
    panel = pd.DataFrame({"resolved_receiver_id": ["00-TEST1"]})
    audit = r18.cpoe_missingness_audit(targets, panel)
    assert audit["resolved_wr_target_events"] == 3
    assert audit["nonnull_cpoe_events"] == 2
    assert audit["null_cpoe_events"] == 1
    assert abs(audit["null_cpoe_rate"] - (1 / 3)) < 1e-12
    assert audit["null_cpoe_mean_air_yards"] == 25.0


def test_raw_gate_direction_is_frozen_positive() -> None:
    n = 400
    signal = np.linspace(-2, 2, n)
    roles = np.where(np.arange(n) % 2 == 0, "WR1", "WR2PLUS")
    residual = 20.0 * signal + np.where(np.arange(n) % 7 == 0, 2.0, -2.0)
    actual = 80.0 + residual
    panel = pd.DataFrame({
        "WR_TARGET_CPOE_MEAN8": signal,
        "yard_residual": residual,
        "actual_rec_yards": actual,
        "wr_rank_bucket": roles,
    })
    metrics, _ = r18.raw_stage_a(panel)
    assert metrics["spearman"] > 0.08
    assert metrics["q4_minus_q1_residual_gap"] > 5.0
    assert metrics["wr1_gap"] > 0
    assert metrics["wr2plus_gap"] > 0
    assert metrics["supported_raw"] is True

    panel["yard_residual"] = -residual
    panel["actual_rec_yards"] = 80.0 - residual
    metrics_neg, _ = r18.raw_stage_a(panel)
    assert metrics_neg["spearman"] < 0
    assert metrics_neg["supported_raw"] is False


def main() -> int:
    test_identity_support_and_leakage()
    test_ambiguous_identity_fails_closed()
    test_cpoe_missingness_audit()
    test_raw_gate_direction_is_frozen_positive()
    print("WR-R18 synthetic mechanics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
