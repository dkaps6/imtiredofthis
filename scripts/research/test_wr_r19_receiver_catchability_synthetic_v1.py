#!/usr/bin/env python3
"""Synthetic mechanics tests for WR-R19 before any real 2023 outcome run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r19_receiver_catchability_stage_a_v1 as r19


def _target(season, week, game_id, catchable, air, complete, receiver_id="00-TEST1", name="J.Doe", team="IND", seq=0):
    return {
        "season": season, "week": week, "game_id": game_id, "team": team,
        "receiver_id": receiver_id, "receiver_name_key": r19._name_key(name),
        "catchable": catchable, "air": air, "complete_pass_num": complete,
        "target_event_seq": seq,
    }


def _rosters(key: str):
    return pd.DataFrame([
        {"season": 2022, "week": w, "team": "IND", "player_clean_key": key, "player_id": "00-TEST1"}
        for w in range(1, 10)
    ] + [
        # Target-week row must never override the prior identity.
        {"season": 2023, "week": 1, "team": "IND", "player_clean_key": key, "player_id": "00-FUTURE"}
    ])


def test_receiver_support_all_targets_and_target_game_exclusion() -> None:
    key = r19._name_key("John Doe")
    rows, seq = [], 0
    vals = [1, 1, 0, 1] * 4
    for week in range(1, 5):
        for j in range(4):
            # Alternate completion state: catchability must include both successes and failures.
            rows.append(_target(2022, week, f"g{week}", vals[seq], 5 + j, seq % 2, seq=seq))
            seq += 1
    rows.append(_target(2023, 1, "TARGET_GAME", 0, 40, 0, seq=seq))
    targets = pd.DataFrame(rows)
    state = r19.receiver_state(targets, _rosters(key), key, "IND", 2023, 1)
    assert state["prior_target_games"] == 4
    assert state["prior_target_events"] == 16
    assert state["prior_valid_catchable_targets"] == 16
    assert abs(state["WR_TARGET_CATCHABLE_RATE8"] - np.mean(vals)) < 1e-12
    assert state["history_max_season"] == 2022 and state["history_max_week"] == 4

    panel = pd.DataFrame({"resolved_receiver_id": ["00-TEST1"]})
    pop = r19.target_population_audit(targets, panel)
    assert pop["completed_target_events"] > 0
    assert pop["incomplete_target_events"] > 0
    assert pop["both_completions_and_incompletions_present"] is True


def test_last8_games_selected_before_catchability_filter() -> None:
    key = r19._name_key("John Doe")
    rows, seq = [], 0
    # Nine target-bearing games, two events each. The newest game's catchability is null.
    # Literal last-8 means weeks 2-9 => only 14 valid labels, so the 16-event floor fails.
    # An invalid implementation that filters nulls first would incorrectly choose weeks 1-8 and pass.
    for week in range(1, 10):
        for _ in range(2):
            catch = np.nan if week == 9 else 1.0
            rows.append(_target(2022, week, f"g{week}", catch, 8, 0, seq=seq))
            seq += 1
    state = r19.receiver_state(pd.DataFrame(rows), _rosters(key), key, "IND", 2023, 1)
    assert state["prior_target_games"] == 8
    assert state["prior_target_events"] == 16
    assert state["prior_valid_catchable_targets"] == 14
    assert np.isnan(state["WR_TARGET_CATCHABLE_RATE8"])
    assert state["history_max_week"] == 9


def test_exact_16_support_boundary() -> None:
    key = r19._name_key("John Doe")
    rows = []
    for i in range(16):
        week = 1 + i // 4
        rows.append(_target(2022, week, f"g{week}", float(i % 2), 7, i % 2, seq=i))
    targets = pd.DataFrame(rows)
    ok = r19.receiver_state(targets, _rosters(key), key, "IND", 2023, 1)
    assert ok["prior_valid_catchable_targets"] == 16
    assert np.isfinite(ok["WR_TARGET_CATCHABLE_RATE8"])
    bad = r19.receiver_state(targets.iloc[1:].copy(), _rosters(key), key, "IND", 2023, 1)
    assert bad["prior_valid_catchable_targets"] == 15
    assert np.isnan(bad["WR_TARGET_CATCHABLE_RATE8"])


def test_ambiguous_identity_fails_closed() -> None:
    key = r19._name_key("John Doe")
    rosters = pd.DataFrame([
        {"season": 2022, "week": 1, "team": "IND", "player_clean_key": key, "player_id": "00-A"},
        {"season": 2022, "week": 2, "team": "IND", "player_clean_key": key, "player_id": "00-B"},
    ])
    got = r19.resolve_roster_player_id(rosters, key, "IND", 2023, 1)
    assert got["roster_identity_mode"] == "ambiguous"
    hist, audit = r19.resolve_prior_receiver_history(pd.DataFrame(columns=["season","week","receiver_name_key","receiver_id"]), rosters, key, "IND", 2023, 1)
    assert hist.empty and audit["identity_mode"] == "ambiguous"


def test_team_control_support_and_leakage() -> None:
    rows, seq = [], 0
    for week in range(1, 5):
        for j in range(10):
            rows.append(_target(2022, week, f"g{week}", 1.0 if j < 8 else 0.0, 5, j % 2, receiver_id=f"R{j}", seq=seq))
            seq += 1
    # Target-week team events are excluded.
    for j in range(20):
        rows.append(_target(2023, 1, "TARGET", 0.0, 5, 0, receiver_id=f"X{j}", seq=seq))
        seq += 1
    st = r19.team_state(pd.DataFrame(rows), "IND", 2023, 1)
    assert st["team_prior_target_games"] == 4
    assert st["team_valid_catchable_targets"] == 40
    assert abs(st["TEAM_TARGET_CATCHABLE_RATE8"] - 0.8) < 1e-12
    assert st["team_history_max_season"] == 2022


def test_raw_gate_positive_direction_and_zero_denominator_tail() -> None:
    n = 400
    signal = np.linspace(0.01, 0.99, n)
    roles = np.where(np.arange(n) % 2 == 0, "WR1", "WR2PLUS")
    residual = 45.0 * (signal - 0.5)
    # Force Q1 to have zero 100+ games while Q4 has positive rate; ratio should be +inf and pass.
    actual = 60.0 + residual
    actual[signal >= np.quantile(signal, .75)] = 110.0
    actual[signal <= np.quantile(signal, .25)] = 70.0
    panel = pd.DataFrame({
        "WR_TARGET_CATCHABLE_RATE8": signal, "yard_residual": residual,
        "actual_rec_yards": actual, "wr_rank_bucket": roles,
    })
    metrics, _ = r19.raw_stage_a(panel)
    assert metrics["spearman"] > r19.MIN_SPEARMAN
    assert metrics["q4_minus_q1_residual_gap"] > r19.MIN_RESIDUAL_GAP
    assert np.isposinf(metrics["actual100_rate_ratio"])
    assert metrics["wr1_gap"] > 0 and metrics["wr2plus_gap"] > 0
    assert metrics["supported_raw"] is True

    panel["yard_residual"] = -residual
    metrics_neg, _ = r19.raw_stage_a(panel)
    assert metrics_neg["spearman"] < 0
    assert metrics_neg["supported_raw"] is False


def test_mediation_gate_mechanics() -> None:
    n = 400
    team = np.linspace(.55, .80, n)
    depth = np.tile(np.linspace(5, 15, 20), 20)
    entitlement = np.tile(np.linspace(.1, .35, 25), 16)
    wr1 = np.arange(n) % 2 == 0
    # Receiver-specific component intentionally remains after controls.
    unique = np.sin(np.linspace(0, 8 * np.pi, n)) * .08
    sig = team * .5 + depth * .005 + entitlement * .1 + wr1.astype(float) * .02 + unique
    residual = unique * 150.0
    panel = pd.DataFrame({
        "WR_TARGET_CATCHABLE_RATE8": sig,
        "TEAM_TARGET_CATCHABLE_RATE8": team,
        "mean_air_yards_per_target8": depth,
        "entitlement_tgt_share": entitlement,
        "wr_rank_bucket": np.where(wr1, "WR1", "WR2PLUS"),
        "yard_residual": residual,
    })
    got = r19.mediation_robustness(panel)
    assert got["receiver_specific_spearman"] > r19.MEDIATION_MIN_SPEARMAN
    assert got["receiver_specific_q4_minus_q1_residual_gap"] > r19.MEDIATION_MIN_RESIDUAL_GAP
    assert got["supported"] is True


def main() -> int:
    test_receiver_support_all_targets_and_target_game_exclusion()
    test_last8_games_selected_before_catchability_filter()
    test_exact_16_support_boundary()
    test_ambiguous_identity_fails_closed()
    test_team_control_support_and_leakage()
    test_raw_gate_positive_direction_and_zero_denominator_tail()
    test_mediation_gate_mechanics()
    print("WR-R19 synthetic mechanics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
