#!/usr/bin/env python3
"""Synthetic mechanics tests for WR-R20 before any real 2023 outcome run."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r20_early_no_extended_stage_a_v1 as r20


def _target(
    season, week, game_id, state, receiver_id="00-TEST1", name="John Doe", team="IND", seq=0
):
    return {
        "season": season,
        "week": week,
        "game_id": game_id,
        "team": team,
        "receiver_id": receiver_id,
        "receiver_name_key": r20._name_key(name),
        "read_norm": state,
        "progression_classifiable": state in {"EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"},
        "is_early_no_extended": float(state == "EARLY_NO_EXTENDED"),
        "target_event_seq": seq,
    }


def _rosters(key: str):
    return pd.DataFrame([
        {"season": 2022, "week": w, "team": "IND", "player_clean_key": key, "player_id": "00-TEST1"}
        for w in range(1, 10)
    ] + [
        # Target-week evidence must never override the strictly-prior identity.
        {"season": 2023, "week": 1, "team": "IND", "player_clean_key": key, "player_id": "00-FUTURE"}
    ])


def test_frozen_progression_semantics() -> None:
    assert r20.normalize_progression(np.nan, 2022) == "EARLY_NO_EXTENDED"
    assert r20.normalize_progression(np.nan, 2023) == "MISSING"
    assert r20.normalize_progression("0", 2023) == "EARLY_NO_EXTENDED"
    assert r20.normalize_progression("DES", 2023) == "EARLY_NO_EXTENDED"
    assert r20.normalize_progression("1", 2023) == "EXTENDED_PROGRESS"
    assert r20.normalize_progression("2", 2023) == "EXTENDED_PROGRESS"
    assert r20.normalize_progression("CHK", 2023) == "CHECKDOWN"
    assert r20.normalize_progression("SD", 2023) == "SCRAMBLE_DRILL"


def test_exact_ftn_pbp_merge_and_parity() -> None:
    ftn = pd.DataFrame([
        {"nflverse_game_id": "G1", "nflverse_play_id": 1, "season": 2023, "week": 1, "read_thrown": "DES"},
        {"nflverse_game_id": "G1", "nflverse_play_id": 2, "season": 2023, "week": 1, "read_thrown": "2"},
    ])
    pbp = pd.DataFrame([
        {"game_id": "G1", "play_id": 1, "season": 2023, "week": 1, "season_type": "REG", "posteam": "IND", "pass_attempt": 1, "sack": 0, "two_point_attempt": 0, "receiver_player_id": "00-A", "receiver_player_name": "John Doe", "air_yards": 2.0},
        {"game_id": "G1", "play_id": 2, "season": 2023, "week": 1, "season_type": "REG", "posteam": "IND", "pass_attempt": 1, "sack": 0, "two_point_attempt": 0, "receiver_player_id": "00-A", "receiver_player_name": "John Doe", "air_yards": 18.0},
    ])
    got, join_rate = r20._merge_ftn_pbp_progression_targets(ftn, pbp, 2023)
    assert join_rate == 1.0 and len(got) == 2
    assert list(got["read_norm"]) == ["EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"]
    bad = pbp.copy()
    bad.loc[1, "week"] = 2
    try:
        r20._merge_ftn_pbp_progression_targets(ftn, bad, 2023)
    except RuntimeError as exc:
        assert "season-week parity" in str(exc)
    else:
        raise AssertionError("season/week mismatch did not fail closed")


def test_receiver_support_and_target_week_exclusion() -> None:
    key = r20._name_key("John Doe")
    rows = []
    for week in range(1, 5):
        for j in range(4):
            state = "EARLY_NO_EXTENDED" if j < 3 else "EXTENDED_PROGRESS"
            rows.append(_target(2022, week, f"G{week}", state, seq=len(rows)))
    rows.append(_target(2023, 1, "TARGET_GAME", "EXTENDED_PROGRESS", seq=len(rows)))
    st = r20.receiver_state(pd.DataFrame(rows), _rosters(key), key, "IND", 2023, 1)
    assert st["prior_target_games"] == 4
    assert st["prior_classifiable_progression_targets"] == 16
    assert abs(st["EARLY_NO_EXTENDED_SHARE8"] - 0.75) < 1e-12
    assert st["history_max_season"] == 2022 and st["history_max_week"] == 4
    assert st["resolved_receiver_id"] == "00-TEST1"


def test_last8_games_selected_before_progression_filter() -> None:
    key = r20._name_key("John Doe")
    rows = []
    for week in range(1, 10):
        for _ in range(2):
            state = "CHECKDOWN" if week == 9 else "EARLY_NO_EXTENDED"
            rows.append(_target(2022, week, f"G{week}", state, seq=len(rows)))
    st = r20.receiver_state(pd.DataFrame(rows), _rosters(key), key, "IND", 2023, 1)
    # Literal last eight target-bearing games are weeks 2-9: 16 targets, only 14 classifiable.
    assert st["prior_target_games"] == 8
    assert st["prior_target_events"] == 16
    assert st["prior_classifiable_progression_targets"] == 14
    assert st["prior_checkdown_targets"] == 2
    assert np.isnan(st["EARLY_NO_EXTENDED_SHARE8"])
    assert st["history_max_week"] == 9


def test_exact_16_classifiable_support_boundary() -> None:
    key = r20._name_key("John Doe")
    rows = []
    for i in range(16):
        week = 1 + i // 4
        state = "EARLY_NO_EXTENDED" if i % 2 == 0 else "EXTENDED_PROGRESS"
        rows.append(_target(2022, week, f"G{week}", state, seq=i))
    ok = r20.receiver_state(pd.DataFrame(rows), _rosters(key), key, "IND", 2023, 1)
    assert ok["prior_classifiable_progression_targets"] == 16
    assert np.isfinite(ok["EARLY_NO_EXTENDED_SHARE8"])
    bad = r20.receiver_state(pd.DataFrame(rows[1:]), _rosters(key), key, "IND", 2023, 1)
    assert bad["prior_classifiable_progression_targets"] == 15
    assert np.isnan(bad["EARLY_NO_EXTENDED_SHARE8"])


def test_ambiguous_identity_fails_closed() -> None:
    key = r20._name_key("John Doe")
    rosters = pd.DataFrame([
        {"season": 2022, "week": 1, "team": "IND", "player_clean_key": key, "player_id": "00-A"},
        {"season": 2022, "week": 2, "team": "IND", "player_clean_key": key, "player_id": "00-B"},
    ])
    got = r20.resolve_roster_player_id(rosters, key, "IND", 2023, 1)
    assert got["roster_identity_mode"] == "ambiguous"


def test_team_control_support_and_target_week_exclusion() -> None:
    rows = []
    for week in range(1, 5):
        for j in range(10):
            state = "EARLY_NO_EXTENDED" if j < 8 else "EXTENDED_PROGRESS"
            rows.append(_target(2022, week, f"G{week}", state, receiver_id=f"R{j}", seq=len(rows)))
    for j in range(20):
        rows.append(_target(2023, 1, "TARGET", "EXTENDED_PROGRESS", receiver_id=f"X{j}", seq=len(rows)))
    st = r20.team_state(pd.DataFrame(rows), "IND", 2023, 1)
    assert st["team_classifiable_progression_targets"] == 40
    assert abs(st["TEAM_EARLY_NO_EXTENDED_SHARE8"] - 0.8) < 1e-12
    assert st["team_history_max_season"] == 2022


def test_air_control_independent_window_and_support() -> None:
    rows = []
    for week in range(1, 5):
        for j in range(3):
            rows.append({"season": 2022, "week": week, "game_id": f"A{week}", "team": "IND", "receiver_id": "00-TEST1", "air_yards": float(5 + j)})
    rows.append({"season": 2023, "week": 1, "game_id": "TARGET", "team": "IND", "receiver_id": "00-TEST1", "air_yards": 99.0})
    air = pd.DataFrame(rows)
    st = r20.air_state(air, "00-TEST1", 2023, 1)
    assert st["valid_air_targets"] == 12
    assert abs(st["MEAN_AIR_YARDS_PER_TARGET8"] - 6.0) < 1e-12
    assert st["air_history_max_season"] == 2022
    bad = r20.air_state(air.iloc[1:].copy(), "00-TEST1", 2023, 1)
    assert bad["valid_air_targets"] == 11
    assert np.isnan(bad["MEAN_AIR_YARDS_PER_TARGET8"])


def test_negative_direction_raw_gates_and_zero_denominator_tail() -> None:
    n = 400
    signal = np.linspace(0.01, 0.99, n)
    roles = np.where(np.arange(n) % 2 == 0, "WR1", "WR2PLUS")
    residual = -50.0 * (signal - 0.5)
    actual = np.full(n, 70.0)
    q25 = np.quantile(signal, 0.25)
    q75 = np.quantile(signal, 0.75)
    actual[signal <= q25] = 110.0
    actual[signal >= q75] = 70.0
    panel = pd.DataFrame({
        "EARLY_NO_EXTENDED_SHARE8": signal,
        "yard_residual": residual,
        "actual_rec_yards": actual,
        "wr_rank_bucket": roles,
    })
    got, _ = r20.raw_stage_a(panel)
    assert got["spearman"] < r20.MAX_SPEARMAN
    assert got["q4_minus_q1_residual_gap"] < r20.MAX_RESIDUAL_GAP
    assert np.isposinf(got["actual100_q1_over_q4_ratio"])
    assert got["wr1_gap"] < 0 and got["wr2plus_gap"] < 0
    assert got["supported_raw"] is True

    panel["yard_residual"] = -residual
    failed, _ = r20.raw_stage_a(panel)
    assert failed["spearman"] > 0
    assert failed["supported_raw"] is False


def test_negative_direction_mediation_gate() -> None:
    n = 400
    team = np.linspace(0.2, 0.8, n)
    depth = np.tile(np.linspace(5, 15, 20), 20)
    entitlement = np.tile(np.linspace(0.1, 0.35, 25), 16)
    wr1 = np.arange(n) % 2 == 0
    unique = np.sin(np.linspace(0, 8 * np.pi, n)) * 0.08
    signal = team * 0.4 - depth * 0.01 + entitlement * 0.1 + wr1.astype(float) * 0.02 + unique
    yard_residual = -unique * 150.0
    panel = pd.DataFrame({
        "EARLY_NO_EXTENDED_SHARE8": signal,
        "TEAM_EARLY_NO_EXTENDED_SHARE8": team,
        "MEAN_AIR_YARDS_PER_TARGET8": depth,
        "entitlement_tgt_share": entitlement,
        "wr_rank_bucket": np.where(wr1, "WR1", "WR2PLUS"),
        "yard_residual": yard_residual,
    })
    got = r20.mediation_robustness(panel)
    assert got["receiver_specific_spearman"] < r20.MEDIATION_MAX_SPEARMAN
    assert got["receiver_specific_q4_minus_q1_residual_gap"] < r20.MEDIATION_MAX_RESIDUAL_GAP
    assert got["supported"] is True


def test_holdout_authority_values_are_never_parsed() -> None:
    original = r20.EXPECTED_ROWS
    r20.EXPECTED_ROWS = {2023: 1, 2024: 1}
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "authority.csv"
            fields = [
                "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
                "entitlement_tgt_share", "mc_rec_yards", "season", "week", "actual_rec_yards",
            ]
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "variant": r20.AUTHORITY_VARIANT, "team": "IND", "player_clean_key": "john doe",
                    "player": "John Doe", "wr_rank": "1", "pred_targets": "7.5",
                    "entitlement_tgt_share": "0.25", "mc_rec_yards": "60", "season": "2023",
                    "week": "1", "actual_rec_yards": "70",
                })
                writer.writerow({
                    "variant": r20.AUTHORITY_VARIANT, "team": "IND", "player_clean_key": "john doe",
                    "player": "John Doe", "wr_rank": "DO_NOT_PARSE", "pred_targets": "DO_NOT_PARSE",
                    "entitlement_tgt_share": "DO_NOT_PARSE", "mc_rec_yards": "DO_NOT_PARSE", "season": "2024",
                    "week": "1", "actual_rec_yards": "DO_NOT_PARSE",
                })
            got = r20.load_authority_development_only(path)
            assert len(got) == 1 and int(got.iloc[0]["season"]) == 2023
    finally:
        r20.EXPECTED_ROWS = original


def main() -> int:
    test_frozen_progression_semantics()
    test_exact_ftn_pbp_merge_and_parity()
    test_receiver_support_and_target_week_exclusion()
    test_last8_games_selected_before_progression_filter()
    test_exact_16_classifiable_support_boundary()
    test_ambiguous_identity_fails_closed()
    test_team_control_support_and_target_week_exclusion()
    test_air_control_independent_window_and_support()
    test_negative_direction_raw_gates_and_zero_denominator_tail()
    test_negative_direction_mediation_gate()
    test_holdout_authority_values_are_never_parsed()
    print("WR-R20 synthetic mechanics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
