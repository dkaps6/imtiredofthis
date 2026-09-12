"""Tests for the historical (2024-2025) market Vegas benchmark grader.

Pure/deterministic: no network, no file I/O beyond the CSVs the caller
already loaded into memory.
"""
from __future__ import annotations

import pandas as pd

from scripts.backtest.grade_historical_market_vegas_benchmark_v1 import (
    grade,
    select_one_book_row,
)


def _proj():
    return pd.DataFrame(
        [
            {
                "season": 2024, "week": 1, "team": "KC", "player_clean_key": "patrickmahomes",
                "market": "pass_yards", "mc_proj": 270.0, "actual": 300.0, "game_id": "2024_01_KC_BAL",
            }
        ]
    )


def _props(over_odds=-110, under_odds=-110):
    return pd.DataFrame(
        [
            {
                "game_id": "2024_01_KC_BAL", "player_clean_key": "patrickmahomes", "market": "pass_yards",
                "book": "draftkings", "line": 265.5, "over_odds": over_odds, "under_odds": under_odds,
                "player": "p.mahomes",
            },
            {
                "game_id": "2024_01_KC_BAL", "player_clean_key": "patrickmahomes", "market": "pass_yards",
                "book": "fanduel", "line": 264.5, "over_odds": -105, "under_odds": -115,
                "player": "p.mahomes",
            },
        ]
    )


def test_select_one_book_row_prefers_draftkings():
    picked = select_one_book_row(_props())
    assert len(picked) == 1
    assert picked.iloc[0]["book"] == "draftkings"


def test_grade_picks_model_side_and_wins_when_model_beats_market():
    detail, summary = grade(_proj(), _props())
    assert len(detail) == 1
    row = detail.iloc[0]
    assert row["model_pick_side"] == "OVER"  # 270 > 265.5
    assert row["actual_side"] == "OVER"  # 300 > 265.5
    assert row["bet_result"] == "WIN"
    assert abs(row["model_error"]) == 30.0  # |270 - 300|
    assert abs(row["vegas_error"]) == 34.5  # |265.5 - 300|
    assert row["model_closer_than_vegas"] == True  # noqa: E712

    all_row = summary.loc[summary.market.eq("ALL_MARKETS_2024")].iloc[0]
    assert all_row["decided_bets"] == 1
    assert all_row["wins"] == 1


def test_grade_returns_empty_when_no_join_match():
    props = _props()
    props["game_id"] = "different_game"
    detail, summary = grade(_proj(), props)
    assert detail.empty
    assert summary.empty
