#!/usr/bin/env python3
"""Synthetic regression tests for WR-R19 v1b mechanical source/holdout fixes."""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.research import evaluate_wr_r19_receiver_catchability_stage_a_v1 as base
from scripts.research import evaluate_wr_r19_receiver_catchability_stage_a_v1b as v1b


def _ftn_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"nflverse_game_id": "g1", "nflverse_play_id": 10, "season": 2022, "week": 1, "is_catchable_ball": True},
        {"nflverse_game_id": "g1", "nflverse_play_id": 20, "season": 2022, "week": 1, "is_catchable_ball": False},
    ])


def _pbp_rows(week2: int = 1) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "game_id": "g1", "play_id": 10, "season": 2022, "week": 1, "season_type": "REG",
            "posteam": "BUF", "pass_attempt": 1, "sack": 0, "two_point_attempt": 0,
            "receiver_player_id": "00-R1", "receiver_player_name": "Alpha Receiver",
            "air_yards": 7, "complete_pass": 1,
        },
        {
            "game_id": "g1", "play_id": 20, "season": 2022, "week": week2, "season_type": "REG",
            "posteam": "BUF", "pass_attempt": 1, "sack": 0, "two_point_attempt": 0,
            "receiver_player_id": "00-R1", "receiver_player_name": "Alpha Receiver",
            "air_yards": 12, "complete_pass": 0,
        },
    ])


def test_exact_merge_preserves_plain_ftn_season_week_and_all_targets() -> None:
    t, join_rate = v1b._merge_ftn_pbp_targets(_ftn_rows(), _pbp_rows(), 2022)
    assert join_rate == 1.0
    assert list(t["season"]) == [2022, 2022]
    assert list(t["week"]) == [1, 1]
    assert list(t["game_id"]) == ["g1", "g1"]
    assert list(t["catchable"]) == [1.0, 0.0]
    assert list(t["complete_pass_num"]) == [1, 0]
    assert t["receiver_id"].tolist() == ["00-R1", "00-R1"]


def test_season_week_mismatch_fails_closed() -> None:
    try:
        v1b._merge_ftn_pbp_targets(_ftn_rows(), _pbp_rows(week2=2), 2022)
    except RuntimeError as exc:
        assert "season-week parity failed" in str(exc)
    else:
        raise AssertionError("FTN/PBP season-week mismatch did not fail closed")


def test_authority_loader_never_parses_holdout_outcomes() -> None:
    old_expected = base.EXPECTED_ROWS
    base.EXPECTED_ROWS = {2023: 1, 2024: 1}
    try:
        with TemporaryDirectory() as td:
            p = Path(td) / "authority.csv"
            fields = [
                "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
                "entitlement_tgt_share", "mc_rec_yards", "season", "week", "actual_rec_yards",
            ]
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({
                    "variant": base.AUTHORITY_VARIANT, "team": "BUF", "player_clean_key": "alpha receiver",
                    "player": "Alpha Receiver", "wr_rank": "1", "pred_targets": "8.0",
                    "entitlement_tgt_share": "0.30", "mc_rec_yards": "70.0",
                    "season": "2023", "week": "1", "actual_rec_yards": "80.0",
                })
                # These holdout values are deliberately non-numeric. A holdout-safe loader must
                # count/key-audit this row without ever parsing either value.
                w.writerow({
                    "variant": base.AUTHORITY_VARIANT, "team": "BUF", "player_clean_key": "alpha receiver",
                    "player": "Alpha Receiver", "wr_rank": "1", "pred_targets": "DO_NOT_PARSE",
                    "entitlement_tgt_share": "DO_NOT_PARSE", "mc_rec_yards": "DO_NOT_PARSE",
                    "season": "2024", "week": "1", "actual_rec_yards": "DO_NOT_PARSE",
                })
            got = v1b.load_authority_development_only(p)
            assert len(got) == 1
            assert int(got.iloc[0]["season"]) == 2023
            assert float(got.iloc[0]["yard_residual"]) == 10.0
    finally:
        base.EXPECTED_ROWS = old_expected


def main() -> int:
    test_exact_merge_preserves_plain_ftn_season_week_and_all_targets()
    test_season_week_mismatch_fails_closed()
    test_authority_loader_never_parses_holdout_outcomes()
    print("WR-R19 v1b source/holdout synthetic mechanics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
